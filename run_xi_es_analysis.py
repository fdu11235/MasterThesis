#!/usr/bin/env python
"""Deep analysis of the xi-ES relationship.

Characterises ES overestimation as a function of xi_hat, demonstrating
that it is a fundamental GPD limitation rather than a CNN-specific problem.

Usage:
    python run_xi_es_analysis.py --config config/default.yaml
"""

import argparse
import logging
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.evaluate import pot_quantile, pot_es, true_quantile, true_es
from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.train import predict

plt.style.use('ggplot')
plt.rcParams.update({
    'figure.figsize': (10, 6), 'figure.dpi': 150, 'font.size': 11,
    'figure.facecolor': 'white', 'axes.facecolor': '#EBEBEB',
})

DIST_COLORS = {
    'student_t': '#E24A33', 'pareto': '#348ABD', 'lognormal_pareto_mix': '#988ED5',
    'two_pareto': '#777777', 'burr12': '#FBC15E', 'frechet': '#8EBA42',
    'dagum': '#FFB5B8', 'inverse_gamma': '#6D904F', 'lognormal': '#FC4F30',
    'weibull_stretched': '#56B4E9', 'log_gamma': '#E69F00', 'gamma_pareto_splice': '#009E73',
    'garch_student_t': '#CC79A7',
}

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

    fig_dir = "docs/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # ── Load synthetic data ────────────────────────────────────────────────
    with open('outputs/data/diagnostics.pkl', 'rb') as f:
        all_diagnostics = pickle.load(f)

    model_cfg = config['model']
    in_channels = len(config.get('features', {}).get('columns', [0,1,2,3,4,5,6]))

    model = ThresholdCNN(
        in_channels=in_channels, channels=model_cfg['channels'],
        kernel_size=model_cfg['kernel_size'], dropout=model_cfg['dropout'],
        pool_sizes=model_cfg.get('pool_sizes'), task='regression')
    model.load_state_dict(
        __import__('torch').load('outputs/checkpoints/model_regression.pt', weights_only=True))
    model.eval()

    X, y, meta = build_dataset_regression(all_diagnostics, config)
    import torch
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

    p = config['evaluate']['quantile_p']

    # ── Compute xi, ES error for each sample ───────────────────────────────
    records = []
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

        es_rel_error = (es_est - es_true) / es_true
        records.append({
            'xi': xi, 'k': k, 'dist_type': ds['dist_type'],
            'es_est': es_est, 'es_true': es_true,
            'es_rel_error': es_rel_error,
        })

    logger.info("Computed xi-ES data for %d samples", len(records))

    xi_arr = np.array([r['xi'] for r in records])
    es_err = np.array([r['es_rel_error'] for r in records])
    dists = [r['dist_type'] for r in records]

    # ── Plot 1: ES relative error vs xi (scatter) ─────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    for dt in sorted(set(dists)):
        mask = np.array([d == dt for d in dists])
        ax.scatter(xi_arr[mask], es_err[mask] * 100, s=8, alpha=0.3,
                   color=DIST_COLORS.get(dt, '#999'), label=dt.replace('_', ' '), edgecolors='none')
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.set_xlabel('xi_hat at predicted k*')
    ax.set_ylabel('ES Relative Error (%)')
    ax.set_title('ES Estimation Error vs Tail Index (xi)')
    ax.set_ylim(-100, 500)
    ax.legend(fontsize=7, ncol=3, loc='upper left')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/xi_es_scatter.png', dpi=150)
    plt.close(fig)
    logger.info("Saved xi_es_scatter.png")

    # ── Plot 2: Amplification curve 1/(1-xi) ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    xi_range = np.linspace(0, 0.95, 200)
    amp = 1 / (1 - xi_range)
    ax.plot(xi_range, amp, color='#E24A33', lw=2.5)
    ax.axvline(0.40, color='#348ABD', ls='--', lw=1.5, label='Synthetic median xi (~0.40)')
    ax.axvline(0.51, color='#009E73', ls='--', lw=1.5, label='Real data median xi (~0.51)')
    ax.axvline(0.70, color='#777777', ls=':', lw=1.5, label='pot_es_stable threshold (0.7)')
    ax.fill_between(xi_range, amp, where=(xi_range >= 0.4) & (xi_range <= 0.7),
                    alpha=0.15, color='#E24A33', label='Danger zone (real data)')
    ax.set_xlabel('xi (tail index)')
    ax.set_ylabel('Amplification factor: 1 / (1 - xi)')
    ax.set_title('ES Formula Amplification Factor')
    ax.set_ylim(0, 15)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/xi_es_amplification.png', dpi=150)
    plt.close(fig)
    logger.info("Saved xi_es_amplification.png")

    # ── Plot 3: ES error by xi bin (box plot) ─────────────────────────────
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.5)]
    bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8+']
    bin_data = []
    bin_counts = []
    for lo, hi in bins:
        mask = (xi_arr >= lo) & (xi_arr < hi)
        bin_data.append(es_err[mask] * 100)
        bin_counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(bin_data, labels=bin_labels, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=1.5))
    colors_bp = ['#8EBA42', '#56B4E9', '#FBC15E', '#E24A33', '#CC79A7']
    for patch, c in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for i, (lbl, cnt) in enumerate(zip(bin_labels, bin_counts)):
        ax.annotate(f'n={cnt}', xy=(i + 1, ax.get_ylim()[1] * 0.95),
                    ha='center', fontsize=8, color='#555')
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.set_xlabel('xi_hat range')
    ax.set_ylabel('ES Relative Error (%)')
    ax.set_title('ES Error Distribution by Tail Index Bin')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/xi_es_binned_boxplot.png', dpi=150)
    plt.close(fig)
    logger.info("Saved xi_es_binned_boxplot.png")

    # ── Plot 4: All methods fail equally (real data MF p-values) ──────────
    methods_mf = {
        'CNN': 0.006,
        'Baseline k*': 0.002,
        'Fixed sqrt(n)': 0.000,
        'Historical sim': 0.726,
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(methods_mf.keys())
    pvals = list(methods_mf.values())
    colors_bar = ['#E24A33' if pv < 0.05 else '#8EBA42' for pv in pvals]
    bars = ax.bar(names, pvals, color=colors_bar, edgecolor='white', width=0.6)
    ax.axhline(0.05, color='black', ls='--', lw=1.5, alpha=0.7, label='5% significance')
    ax.set_ylabel('McNeil-Frey p-value')
    ax.set_title('McNeil-Frey ES Test: Loss Tail (all parametric methods fail)')
    ax.legend()
    for bar, pv in zip(bars, pvals):
        ax.annotate(f'p={pv:.3f}', xy=(bar.get_x() + bar.get_width() / 2, pv + 0.02),
                    ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/xi_es_all_methods_mf.png', dpi=150)
    plt.close(fig)
    logger.info("Saved xi_es_all_methods_mf.png")

    # ── Plot 5: Xi distribution comparison (synthetic vs real) ────────────
    with open('outputs/data/real_diagnostics_loss.pkl', 'rb') as f:
        real_diags = pickle.load(f)

    xi_real = []
    for _, dg in real_diags:
        k_idx = np.searchsorted(dg['k_grid'], dg['k_star'])
        k_idx = min(k_idx, len(dg['xi_series']) - 1)
        xi_val = dg['xi_series'][k_idx]
        if not np.isnan(xi_val):
            xi_real.append(xi_val)
    xi_real = np.array(xi_real)

    fig, ax = plt.subplots(figsize=(10, 6))
    bins_hist = np.linspace(-0.2, 1.2, 60)
    ax.hist(xi_arr, bins=bins_hist, alpha=0.5, density=True, color='#348ABD',
            label=f'Synthetic (median={np.median(xi_arr):.2f})', edgecolor='none')
    ax.hist(xi_real, bins=bins_hist, alpha=0.5, density=True, color='#E24A33',
            label=f'Real loss tail (median={np.median(xi_real):.2f})', edgecolor='none')
    ax.axvline(np.median(xi_arr), color='#348ABD', ls='--', lw=1.5)
    ax.axvline(np.median(xi_real), color='#E24A33', ls='--', lw=1.5)
    ax.set_xlabel('xi_hat at k*')
    ax.set_ylabel('Density')
    ax.set_title('Tail Index Distribution: Synthetic vs Real Loss Tail')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/xi_es_distribution_comparison.png', dpi=150)
    plt.close(fig)
    logger.info("Saved xi_es_distribution_comparison.png")

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        'records': records,
        'xi_arr': xi_arr,
        'es_err': es_err,
        'dists': dists,
        'xi_real': xi_real,
        'bin_stats': {lbl: {'median': float(np.median(d)), 'mean': float(np.mean(d)),
                            'count': len(d)}
                      for lbl, d in zip(bin_labels, bin_data)},
    }
    with open('outputs/xi_es_analysis.pkl', 'wb') as f:
        pickle.dump(results, f)

    logger.info("Results saved to outputs/xi_es_analysis.pkl")
    logger.info("All plots saved to %s/xi_es_*.png", fig_dir)


if __name__ == '__main__':
    main()
