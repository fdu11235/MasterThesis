#!/usr/bin/env python
"""Out-of-sample validation: apply trained model to unseen tickers.

Uses the existing CNN + ES correction network (no retraining).
Downloads new tickers, runs POT diagnostics, predicts k*, computes
VaR/ES with correction, and runs Kupiec + McNeil-Frey.

Usage:
    python run_oos_validation.py --config config/default.yaml
"""

import argparse
import logging
import os
import pickle

import numpy as np
import torch
import yaml
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp

from src.realdata import load_returns, rolling_windows, prepare_real_datasets_signsplit
from src.pot import process_one_dataset
from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.train import predict
from src.evaluate import pot_quantile, pot_es
from src.es_correction import ESCorrectionNet, extract_features, apply_correction
from src.evaluate_real import kupiec_test

logger = logging.getLogger(__name__)

OOS_TICKERS = ["GOOG", "TSLA", "^GSPC"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
                        datefmt='%H:%M:%S')

    with open(args.config) as f:
        config = yaml.safe_load(f)

    p = config['evaluate']['quantile_p']
    backtest_horizon = config['realdata']['backtest_horizon']
    window_size = config['realdata']['window_size']
    step_size = config['realdata']['step_size']
    model_cfg = config['model']
    in_channels = len(config.get('features', {}).get('columns', [0, 1, 2, 3, 4, 5, 6]))

    os.makedirs('outputs/data', exist_ok=True)

    # ── Load trained models ────────────────────────────────────────────────
    logger.info("Loading trained CNN...")
    cnn = ThresholdCNN(
        in_channels=in_channels, channels=model_cfg['channels'],
        kernel_size=model_cfg['kernel_size'], dropout=model_cfg['dropout'],
        pool_sizes=model_cfg.get('pool_sizes'), task='regression')
    cnn.load_state_dict(torch.load('outputs/checkpoints/model_regression.pt', weights_only=True))
    cnn.eval()

    logger.info("Loading ES correction network...")
    ckpt = torch.load('outputs/checkpoints/es_correction_net.pt', weights_only=False)
    es_cfg = config.get('es_correction', {})
    corr_net = ESCorrectionNet(
        in_features=9,
        hidden=es_cfg.get('hidden', 32),
        output_lo=es_cfg.get('output_lo', 0.5),
        output_hi=es_cfg.get('output_hi', 3.0),
        output_mode=es_cfg.get('output_mode', 'softplus'),
    )
    corr_net.load_state_dict(ckpt['state_dict'])
    corr_net.X_mean = ckpt['X_mean']
    corr_net.X_std = ckpt['X_std']
    corr_net.eval()

    # ── Download OOS tickers ───────────────────────────────────────────────
    logger.info("Downloading OOS tickers: %s", OOS_TICKERS)
    ticker_data = load_returns(
        OOS_TICKERS,
        start=config['realdata']['start'],
        end=config['realdata']['end'],
        cache_dir='outputs/data',
    )

    # Build rolling windows + returns lookup
    datasets = []
    returns_lookup = {}
    for ticker in OOS_TICKERS:
        if ticker not in ticker_data:
            logger.warning("Ticker %s not found, skipping", ticker)
            continue
        df = ticker_data[ticker]
        abs_ret = df['abs_return'].values
        dates = df['date'].values
        signed_ret = df['signed_return'].values if 'signed_return' in df.columns else None

        returns_lookup[ticker] = {
            'abs_returns': abs_ret,
            'dates': dates,
            'signed_returns': signed_ret,
        }

        windows = rolling_windows(abs_ret, dates, window_size, step_size, ticker)
        datasets.extend(windows)
        logger.info("  %s: %d windows (%d observations)", ticker, len(windows), len(df))

    logger.info("Total OOS windows: %d", len(datasets))

    # ── Sign-split (loss tail) ─────────────────────────────────────────────
    logger.info("Preparing loss-tail sign-split...")
    loss_datasets = prepare_real_datasets_signsplit(
        config, returns_lookup, datasets, 'loss')

    # ── POT diagnostics ────────────────────────────────────────────────────
    logger.info("Computing POT diagnostics (%d windows)...", len(loss_datasets))
    pot_cfg = config['pot']
    loss_diagnostics = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_one_dataset)(ds, pot_cfg) for ds in loss_datasets
    )

    # ── Build features + predict ───────────────────────────────────────────
    logger.info("Building features and predicting k*...")
    X, y, meta = build_dataset_regression(loss_diagnostics, config)

    # Use ALL windows as test (no fine-tuning — pure OOS)
    y_pred = predict(cnn, X, task='regression')
    k_pred = np.array([
        int(np.clip(round(m['k_min'] + yp * (m['k_max'] - m['k_min'])),
                     m['k_min'], m['k_max']))
        for yp, m in zip(y_pred, meta)
    ])

    # ── Backtest ───────────────────────────────────────────────────────────
    logger.info("Running VaR/ES backtesting...")

    for label, use_correction in [('uncorrected', False), ('corrected', True)]:
        violations_all = []
        residuals_all = []
        n_total = 0
        per_ticker = {}

        for i, (ds, (ds_orig, diag)) in enumerate(zip(loss_datasets, loss_diagnostics)):
            ticker = ds.get('ticker')
            signed = returns_lookup[ticker].get('signed_returns')
            if signed is None:
                continue

            future_start = ds.get('series_end_idx', 0)
            future_end = future_start + backtest_horizon
            if future_end > len(signed):
                continue

            future_signed = signed[future_start:future_end]
            mask = future_signed < 0
            future_mags = np.abs(future_signed[mask])

            k = k_pred[i]
            k_grid = np.asarray(diag['k_grid'])
            k_idx = min(np.searchsorted(k_grid, k), len(diag['params']) - 1)
            xi, beta = diag['params'][k_idx]
            if np.isnan(xi) or np.isnan(beta):
                continue

            sorted_desc = np.sort(ds['samples'])[::-1]
            n = len(sorted_desc)
            var_est = pot_quantile(sorted_desc, k, xi, beta, n, p)
            es_raw = pot_es(sorted_desc, k, xi, beta, n, p)

            if use_correction:
                es_est = apply_correction(corr_net, ds, diag, k, es_raw, p, config)
            else:
                es_est = es_raw

            if len(future_mags) == 0:
                continue

            viols = future_mags > var_est
            n_viols = int(viols.sum())
            violations_all.append(n_viols)
            n_total += len(future_mags)

            for fm in future_mags[viols]:
                residuals_all.append((fm - es_est) / es_est)

            if ticker not in per_ticker:
                per_ticker[ticker] = {'viols': 0, 'total': 0, 'resids': []}
            per_ticker[ticker]['viols'] += n_viols
            per_ticker[ticker]['total'] += len(future_mags)
            for fm in future_mags[viols]:
                per_ticker[ticker]['resids'].append((fm - es_est) / es_est)

        # Overall results
        total_viols = sum(violations_all)
        vr = total_viols / n_total if n_total > 0 else float('nan')

        kup = kupiec_test(vr, n_total, p)
        kup_pass = "PASS" if not kup.get('reject_5pct', True) else "fail"

        if len(residuals_all) >= 2:
            _, mf_p = ttest_1samp(residuals_all, 0)
            mf_pass = "PASS" if mf_p > 0.05 else "fail"
            mean_resid = np.mean(residuals_all)
        else:
            mf_p = float('nan')
            mf_pass = "N/A"
            mean_resid = float('nan')

        logger.info("\n  === %s (loss tail) ===", label.upper())
        logger.info("  Overall: VR=%.4f, Kupiec %s (p=%.4f), MF %s (p=%.4f), mean_resid=%.4f, n_viol=%d/%d",
                     vr, kup_pass, kup.get('p_value', float('nan')),
                     mf_pass, mf_p, mean_resid, total_viols, n_total)

        logger.info("\n  Per-ticker:")
        logger.info("  %-10s %8s %8s %12s %10s", "Ticker", "VR%", "N viol", "Mean Resid", "MF p")
        for ticker in sorted(per_ticker.keys()):
            pt = per_ticker[ticker]
            tvr = pt['viols'] / pt['total'] * 100 if pt['total'] > 0 else 0
            resids = np.array(pt['resids'])
            if len(resids) >= 2:
                _, tp = ttest_1samp(resids, 0)
            else:
                tp = float('nan')
            logger.info("  %-10s %7.2f%% %8d %12.4f %10.4f",
                         ticker, tvr, pt['viols'], np.mean(resids) if len(resids) > 0 else 0, tp)

    # Save
    with open('outputs/oos_validation_results.pkl', 'wb') as f:
        pickle.dump({'tickers': OOS_TICKERS, 'n_windows': len(loss_datasets)}, f)
    logger.info("\nOOS validation complete.")


if __name__ == '__main__':
    main()
