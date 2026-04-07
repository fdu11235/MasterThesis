#!/usr/bin/env python
"""ES bias correction: estimate bias from synthetic data, apply to real data.

Builds a correction table as f(xi) from synthetic data where true ES is known,
then applies the correction to real data ES estimates and re-runs McNeil-Frey.

Usage:
    python run_es_bias_correction.py --config config/default.yaml
"""

import argparse
import logging
import os
import pickle

import numpy as np
import yaml
from scipy.stats import ttest_1samp

from src.evaluate import pot_quantile, pot_es, true_es
from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.train import predict

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
                        datefmt='%H:%M:%S')

    with open(args.config) as f:
        config = yaml.safe_load(f)

    p = config['evaluate']['quantile_p']
    backtest_horizon = config['realdata']['backtest_horizon']
    train_frac = config['realdata']['train_fraction']

    # ── Load model ─────────────────────────────────────────────────────────
    model_cfg = config['model']
    in_channels = len(config.get('features', {}).get('columns', [0,1,2,3,4,5,6]))

    import torch
    model = ThresholdCNN(
        in_channels=in_channels, channels=model_cfg['channels'],
        kernel_size=model_cfg['kernel_size'], dropout=model_cfg['dropout'],
        pool_sizes=model_cfg.get('pool_sizes'), task='regression')
    model.load_state_dict(torch.load('outputs/checkpoints/model_regression.pt', weights_only=True))
    model.eval()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Build bias table from synthetic data
    # ══════════════════════════════════════════════════════════════════════
    logger.info("Step 1: Building bias table from synthetic data...")

    with open('outputs/data/diagnostics.pkl', 'rb') as f:
        all_diagnostics = pickle.load(f)

    X, y, meta = build_dataset_regression(all_diagnostics, config)
    N = len(X)
    torch.manual_seed(42)
    perm = torch.randperm(N)
    test_size = int(N * config['evaluate']['test_fraction'])
    test_idx = perm[:test_size].tolist()

    X_test = X[test_idx]
    test_meta = [meta[i] for i in test_idx]
    test_diags = [all_diagnostics[i] for i in test_idx]

    y_pred = predict(model, X_test, task='regression')
    k_pred = np.array([
        int(np.clip(round(m['k_min'] + yp * (m['k_max'] - m['k_min'])),
                     m['k_min'], m['k_max']))
        for yp, m in zip(y_pred, test_meta)
    ])

    # Compute bias per sample
    xi_vals = []
    bias_vals = []
    for i, (ds, diag) in enumerate(test_diags):
        k = k_pred[i]
        k_grid = np.asarray(diag['k_grid'])
        k_idx = min(np.searchsorted(k_grid, k), len(diag['params']) - 1)
        xi, beta = diag['params'][k_idx]
        if np.isnan(xi) or np.isnan(beta):
            continue

        sorted_desc = np.sort(ds['samples'])[::-1]
        n = len(sorted_desc)
        es_est = pot_es(sorted_desc, k, xi, beta, n, p)

        try:
            es_true = true_es(ds['dist_type'], ds['params'], p)
        except (ValueError, KeyError):
            continue

        if es_true <= 0 or np.isnan(es_est):
            continue

        bias = (es_est - es_true) / es_true
        xi_vals.append(xi)
        bias_vals.append(bias)

    xi_vals = np.array(xi_vals)
    bias_vals = np.array(bias_vals)
    logger.info("  Computed bias for %d samples", len(xi_vals))

    # Build correction table by xi bin
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.5)]
    bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8+']
    bias_table = {}

    logger.info("\n  Bias table (from synthetic data):")
    logger.info("  %-10s %10s %10s %10s %8s", "Xi range", "Median bias", "Mean bias", "Std bias", "Count")
    logger.info("  " + "-" * 55)
    for (lo, hi), lbl in zip(bins, bin_labels):
        mask = (xi_vals >= lo) & (xi_vals < hi)
        if mask.sum() > 0:
            median_bias = float(np.median(bias_vals[mask]))
            mean_bias = float(np.mean(bias_vals[mask]))
            std_bias = float(np.std(bias_vals[mask]))
        else:
            median_bias = mean_bias = std_bias = 0.0
        bias_table[(lo, hi)] = {
            'median_bias': median_bias,
            'mean_bias': mean_bias,
            'std_bias': std_bias,
            'count': int(mask.sum()),
        }
        logger.info("  %-10s %+10.3f %+10.3f %10.3f %8d",
                     lbl, median_bias, mean_bias, std_bias, mask.sum())

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Apply correction to real data
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\nStep 2: Applying bias correction to real data...")

    with open('outputs/data/real_datasets.pkl', 'rb') as f:
        rd = pickle.load(f)
    returns_lookup = rd['returns_lookup']

    with open('outputs/data/real_diagnostics_loss.pkl', 'rb') as f:
        diag_loss = pickle.load(f)
    with open('outputs/data/real_datasets_loss.pkl', 'rb') as f:
        ds_loss = pickle.load(f)

    Xr, yr, metar = build_dataset_regression(diag_loss, config)
    end_dates = [m.get('end_date', '') for m in metar]
    sorted_idx = np.argsort(end_dates)
    n_train = int(len(sorted_idx) * train_frac)
    test_idx_r = sorted_idx[n_train:]

    Xr_test = Xr[test_idx_r]
    test_meta_r = [metar[i] for i in test_idx_r]
    test_diags_r = [diag_loss[i] for i in test_idx_r]
    test_ds_r = [ds_loss[i] for i in test_idx_r]

    yr_pred = predict(model, Xr_test, task='regression')
    kr_pred = np.array([
        int(np.clip(round(m['k_min'] + yp * (m['k_max'] - m['k_min'])),
                     m['k_min'], m['k_max']))
        for yp, m in zip(yr_pred, test_meta_r)
    ])

    def get_correction(xi_val):
        for (lo, hi), stats in bias_table.items():
            if lo <= xi_val < hi:
                return stats['median_bias']
        return 0.0

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Re-run McNeil-Frey with corrected and uncorrected ES
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\nStep 3: Re-running McNeil-Frey test...")

    def run_mf(correction_mode):
        """correction_mode: 'none', 'median_bias', 'mean_bias'"""
        residuals = []
        for i, (ds, (ds_orig, diag)) in enumerate(zip(test_ds_r, test_diags_r)):
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

            k = kr_pred[i]
            k_grid = np.asarray(diag['k_grid'])
            k_idx = min(np.searchsorted(k_grid, k), len(diag['params']) - 1)
            xi, beta = diag['params'][k_idx]
            if np.isnan(xi) or np.isnan(beta):
                continue

            sorted_desc = np.sort(ds['samples'])[::-1]
            n = len(sorted_desc)
            var_est = pot_quantile(sorted_desc, k, xi, beta, n, p)
            es_raw = pot_es(sorted_desc, k, xi, beta, n, p)

            if correction_mode == 'none':
                es_est = es_raw
            else:
                bias = get_correction(xi)
                es_est = es_raw / (1 + bias)

            if len(future_mags) == 0:
                continue
            violations = future_mags > var_est
            for fm in future_mags[violations]:
                residuals.append((fm - es_est) / es_est)

        if len(residuals) < 2:
            return float('nan'), float('nan'), 0
        residuals = np.array(residuals)
        _, pv = ttest_1samp(residuals, 0)
        return pv, float(np.mean(residuals)), len(residuals)

    results = {}
    for mode in ['none', 'median_bias']:
        pv, mr, nv = run_mf(mode)
        label = "PASS" if pv > 0.05 else "fail"
        results[mode] = {'p_value': pv, 'mean_residual': mr, 'n_violations': nv}
        logger.info("  %-15s: MF p=%.4f (%s), mean_resid=%.4f, n_viol=%d",
                     mode, pv, label, mr, nv)

    # Also test on profit tail
    logger.info("\n  Profit tail:")
    with open('outputs/data/real_diagnostics_profit.pkl', 'rb') as f:
        diag_profit = pickle.load(f)
    with open('outputs/data/real_datasets_profit.pkl', 'rb') as f:
        ds_profit = pickle.load(f)

    Xp, yp, metap = build_dataset_regression(diag_profit, config)
    end_dates_p = [m.get('end_date', '') for m in metap]
    sorted_idx_p = np.argsort(end_dates_p)
    n_train_p = int(len(sorted_idx_p) * train_frac)
    test_idx_p = sorted_idx_p[n_train_p:]

    Xp_test = Xp[test_idx_p]
    test_meta_p = [metap[i] for i in test_idx_p]
    test_diags_p = [diag_profit[i] for i in test_idx_p]
    test_ds_p = [ds_profit[i] for i in test_idx_p]

    yp_pred = predict(model, Xp_test, task='regression')
    kp_pred = np.array([
        int(np.clip(round(m['k_min'] + yp_val * (m['k_max'] - m['k_min'])),
                     m['k_min'], m['k_max']))
        for yp_val, m in zip(yp_pred, test_meta_p)
    ])

    def run_mf_profit(correction_mode):
        residuals = []
        for i, (ds, (ds_orig, diag)) in enumerate(zip(test_ds_p, test_diags_p)):
            ticker = ds.get('ticker')
            signed = returns_lookup[ticker].get('signed_returns')
            if signed is None:
                continue
            future_start = ds.get('series_end_idx', 0)
            future_end = future_start + backtest_horizon
            if future_end > len(signed):
                continue

            future_signed = signed[future_start:future_end]
            mask = future_signed > 0
            future_mags = future_signed[mask]

            k = kp_pred[i]
            k_grid = np.asarray(diag['k_grid'])
            k_idx = min(np.searchsorted(k_grid, k), len(diag['params']) - 1)
            xi, beta = diag['params'][k_idx]
            if np.isnan(xi) or np.isnan(beta):
                continue

            sorted_desc = np.sort(ds['samples'])[::-1]
            n = len(sorted_desc)
            var_est = pot_quantile(sorted_desc, k, xi, beta, n, p)
            es_raw = pot_es(sorted_desc, k, xi, beta, n, p)

            if correction_mode == 'none':
                es_est = es_raw
            else:
                bias = get_correction(xi)
                es_est = es_raw / (1 + bias)

            if len(future_mags) == 0:
                continue
            violations = future_mags > var_est
            for fm in future_mags[violations]:
                residuals.append((fm - es_est) / es_est)

        if len(residuals) < 2:
            return float('nan'), float('nan'), 0
        residuals = np.array(residuals)
        _, pv = ttest_1samp(residuals, 0)
        return pv, float(np.mean(residuals)), len(residuals)

    for mode in ['none', 'median_bias']:
        pv, mr, nv = run_mf_profit(mode)
        label = "PASS" if pv > 0.05 else "fail"
        results[f'profit_{mode}'] = {'p_value': pv, 'mean_residual': mr, 'n_violations': nv}
        logger.info("  %-15s: MF p=%.4f (%s), mean_resid=%.4f, n_viol=%d",
                     mode, pv, label, mr, nv)

    # ── Save ───────────────────────────────────────────────────────────────
    save_data = {
        'bias_table': bias_table,
        'results': results,
    }
    with open('outputs/es_bias_correction.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    logger.info("\nResults saved to outputs/es_bias_correction.pkl")


if __name__ == '__main__':
    main()
