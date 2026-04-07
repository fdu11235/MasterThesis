#!/usr/bin/env python
"""Train ES correction network and evaluate on synthetic + real data.

Usage:
    python run_es_correction_net.py --config config/default.yaml
"""

import argparse
import logging
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import ttest_1samp

from src.es_correction import (
    ESCorrectionNet, build_correction_dataset, train_correction_net,
    extract_features, apply_correction,
)
from src.evaluate import pot_quantile, pot_es, true_es
from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.train import predict

plt.style.use('ggplot')
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': '#EBEBEB'})

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
    fig_dir = "docs/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # ── Load CNN model ─────────────────────────────────────────────────────
    model_cfg = config['model']
    in_channels = len(config.get('features', {}).get('columns', [0,1,2,3,4,5,6]))

    cnn = ThresholdCNN(
        in_channels=in_channels, channels=model_cfg['channels'],
        kernel_size=model_cfg['kernel_size'], dropout=model_cfg['dropout'],
        pool_sizes=model_cfg.get('pool_sizes'), task='regression')
    cnn.load_state_dict(torch.load('outputs/checkpoints/model_regression.pt', weights_only=True))
    cnn.eval()

    # ── Load synthetic data ────────────────────────────────────────────────
    with open('outputs/data/diagnostics.pkl', 'rb') as f:
        all_diagnostics = pickle.load(f)

    X, y, meta = build_dataset_regression(all_diagnostics, config)
    N = len(X)
    torch.manual_seed(42)
    perm = torch.randperm(N)
    test_size = int(N * config['evaluate']['test_fraction'])
    test_idx = perm[:test_size].tolist()
    train_idx = perm[test_size:].tolist()

    # Get CNN k* predictions for ALL samples (train + test)
    logger.info("Computing CNN predictions for all %d samples...", N)
    y_pred_all = predict(cnn, X, task='regression')
    k_pred_all = np.array([
        int(np.clip(round(m['k_min'] + yp * (m['k_max'] - m['k_min'])),
                     m['k_min'], m['k_max']))
        for yp, m in zip(y_pred_all, meta)
    ])

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Build correction dataset from TRAINING synthetic data
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 1: Building correction dataset from training data")

    train_diags = [all_diagnostics[i] for i in train_idx]
    train_k = k_pred_all[train_idx]

    X_corr, y_corr = build_correction_dataset(train_diags, train_k, config)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Train correction network
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 2: Training ES correction network")

    corr_model, history = train_correction_net(X_corr, y_corr, config)

    ckpt_path = 'outputs/checkpoints/es_correction_net.pt'
    torch.save({
        'state_dict': corr_model.state_dict(),
        'X_mean': corr_model.X_mean,
        'X_std': corr_model.X_std,
    }, ckpt_path)
    logger.info("Checkpoint saved to %s", ckpt_path)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Evaluate on synthetic TEST data
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 3: Evaluating on synthetic test data")

    test_diags = [all_diagnostics[i] for i in test_idx]
    test_k = k_pred_all[test_idx]

    es_orig_errors = []
    es_corr_errors = []
    xi_vals = []
    corrections_applied = []

    for i, (ds, diag) in enumerate(test_diags):
        k = int(test_k[i])
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
        if es_true <= 0 or np.isnan(es_est) or es_est <= 0:
            continue

        es_corrected = apply_correction(corr_model, ds, diag, k, es_est, p, config)

        es_orig_errors.append(((es_est - es_true) / es_true) ** 2)
        es_corr_errors.append(((es_corrected - es_true) / es_true) ** 2)
        xi_vals.append(xi)
        corrections_applied.append(es_corrected / es_est)

    es_orig_rmse = np.sqrt(np.mean(es_orig_errors)) * 100
    es_corr_rmse = np.sqrt(np.mean(es_corr_errors)) * 100

    logger.info("Synthetic test ES Rel RMSE:")
    logger.info("  Uncorrected: %.2f%%", es_orig_rmse)
    logger.info("  Corrected:   %.2f%%", es_corr_rmse)
    logger.info("  Improvement: %.2f%%", es_orig_rmse - es_corr_rmse)

    # Per xi bin
    xi_arr = np.array(xi_vals)
    orig_arr = np.array(es_orig_errors)
    corr_arr = np.array(es_corr_errors)
    corrections_arr = np.array(corrections_applied)

    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.5)]
    bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8+']

    logger.info("\nPer xi bin:")
    logger.info("  %-10s %12s %12s %12s %8s", "Xi", "Orig RMSE%", "Corr RMSE%", "Improvement", "N")
    for (lo, hi), lbl in zip(bins, bin_labels):
        mask = (xi_arr >= lo) & (xi_arr < hi)
        if mask.sum() == 0:
            continue
        o = np.sqrt(np.mean(orig_arr[mask])) * 100
        c = np.sqrt(np.mean(corr_arr[mask])) * 100
        logger.info("  %-10s %11.1f%% %11.1f%% %+11.1f%% %8d",
                     lbl, o, c, o - c, mask.sum())

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Apply to real data
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 4: Applying to real data")

    with open('outputs/data/real_datasets.pkl', 'rb') as f:
        rd = pickle.load(f)
    returns_lookup = rd['returns_lookup']

    backtest_horizon = config['realdata']['backtest_horizon']
    train_frac = config['realdata']['train_fraction']

    for tail_mode, diag_path, ds_path in [
        ('loss', 'outputs/data/real_diagnostics_loss.pkl', 'outputs/data/real_datasets_loss.pkl'),
        ('profit', 'outputs/data/real_diagnostics_profit.pkl', 'outputs/data/real_datasets_profit.pkl'),
    ]:
        with open(diag_path, 'rb') as f:
            real_diags = pickle.load(f)
        with open(ds_path, 'rb') as f:
            real_ds = pickle.load(f)

        Xr, yr, metar = build_dataset_regression(real_diags, config)
        end_dates = [m.get('end_date', '') for m in metar]
        sorted_idx = np.argsort(end_dates)
        n_train = int(len(sorted_idx) * train_frac)
        r_test_idx = sorted_idx[n_train:]

        Xr_test = Xr[r_test_idx]
        yr_pred = predict(cnn, Xr_test, task='regression')
        r_test_meta = [metar[i] for i in r_test_idx]
        r_test_diags = [real_diags[i] for i in r_test_idx]
        r_test_ds = [real_ds[i] for i in r_test_idx]

        kr_pred = np.array([
            int(np.clip(round(m['k_min'] + yp * (m['k_max'] - m['k_min'])),
                         m['k_min'], m['k_max']))
            for yp, m in zip(yr_pred, r_test_meta)
        ])

        # Compute MF with and without correction
        for label, use_correction in [('uncorrected', False), ('corrected', True)]:
            residuals = []
            for i, (ds, (ds_orig, diag)) in enumerate(zip(r_test_ds, r_test_diags)):
                ticker = ds.get('ticker')
                signed = returns_lookup[ticker].get('signed_returns')
                if signed is None:
                    continue
                future_start = ds.get('series_end_idx', 0)
                future_end = future_start + backtest_horizon
                if future_end > len(signed):
                    continue

                future_signed = signed[future_start:future_end]
                if tail_mode == 'loss':
                    mask = future_signed < 0
                    future_mags = np.abs(future_signed[mask])
                else:
                    mask = future_signed > 0
                    future_mags = future_signed[mask]

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

                if use_correction:
                    es_est = apply_correction(corr_model, ds, diag, k, es_raw, p, config)
                else:
                    es_est = es_raw

                if len(future_mags) == 0:
                    continue
                violations = future_mags > var_est
                for fm in future_mags[violations]:
                    residuals.append((fm - es_est) / es_est)

            if len(residuals) >= 2:
                residuals = np.array(residuals)
                _, pv = ttest_1samp(residuals, 0)
                status = "PASS" if pv > 0.05 else "fail"
                logger.info("  %s %s: MF p=%.4f (%s), mean_resid=%.4f, n=%d",
                             tail_mode, label, pv, status, np.mean(residuals), len(residuals))
            else:
                logger.info("  %s %s: insufficient violations", tail_mode, label)

    # ══════════════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════════════

    # Scatter: predicted vs true correction factor (synthetic)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_corr[:len(corrections_arr)], corrections_arr, s=8, alpha=0.3,
               color='#348ABD', edgecolors='none')
    lims = [0.3, 3.0]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.7)
    ax.set_xlabel('True correction factor (ES_true / ES_est)')
    ax.set_ylabel('Predicted correction factor')
    ax.set_title('ES Correction Network: Predicted vs True')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/es_correction_scatter.png', dpi=150)
    plt.close(fig)

    # Distribution of corrections on real data
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(corrections_arr, bins=50, alpha=0.7, color='#348ABD', edgecolor='none')
    ax.axvline(1.0, color='black', ls='--', lw=1.5, label='No correction (1.0)')
    ax.axvline(np.median(corrections_arr), color='#E24A33', ls='-', lw=1.5,
               label=f'Median = {np.median(corrections_arr):.3f}')
    ax.set_xlabel('Correction factor applied')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of ES Correction Factors (Synthetic Test)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/es_correction_distribution.png', dpi=150)
    plt.close(fig)

    # Save results
    results = {
        'synthetic_es_orig_rmse': es_orig_rmse,
        'synthetic_es_corr_rmse': es_corr_rmse,
        'corrections_applied': corrections_arr,
        'xi_vals': xi_arr,
        'history': history,
    }
    with open('outputs/es_correction_net_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    logger.info("\nResults saved to outputs/es_correction_net_results.pkl")
    logger.info("Plots saved to %s/es_correction_*.png", fig_dir)


if __name__ == '__main__':
    main()
