#!/usr/bin/env python
"""Investigate the two_pareto ES failure.

two_pareto is a composite distribution: a Pareto(alpha1) bulk with the top
`changepoint_frac` (5%) spliced to a heavier Pareto(alpha2). The genuine
heavy tail — the one that sets the true ES — is only the top 5% of the
sample.

Hypothesis: the CNN selects an exceedance count k far larger than the
changepoint, so the top-k window straddles the splice. The Hill / GPD
estimators then average log-spacings across BOTH regimes (light alpha1
bulk + heavy alpha2 tail) and return a blended, too-light tail index,
underestimating ES.

This script tests that by comparing, for every two_pareto dataset in the
cached diagnostics, the ES error at three thresholds:
  - k_pred     : the CNN-selected k
  - k_cp       : the changepoint count (= changepoint_frac * n) — the
                 largest k that stays inside the genuine alpha2 tail
  - k_oracle   : the grid k that minimises |ES(k) - ES_true|

Usage:
    python investigate_two_pareto.py --config config/default.yaml
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

from src.evaluate import pot_es, true_es
from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.train import predict

plt.style.use('ggplot')
plt.rcParams.update({
    'figure.figsize': (10, 6), 'figure.dpi': 150, 'font.size': 11,
    'figure.facecolor': 'white', 'axes.facecolor': '#EBEBEB',
})

logger = logging.getLogger(__name__)

OUT_DIR = 'outputs/two_pareto'
FIG_DIR = f'{OUT_DIR}/figures'
A2_COLORS = {1.05: '#E24A33', 1.1: '#348ABD', 1.5: '#8EBA42'}


def _xi_beta_at_k(diag, k):
    kg = np.asarray(diag['k_grid'])
    i = min(int(np.searchsorted(kg, k)), len(diag['params']) - 1)
    return diag['params'][i]


def es_at_k(ds, diag, k, p):
    """ES estimate at exceedance count k via the (fixed) pot_es."""
    xi, beta = _xi_beta_at_k(diag, k)
    if np.isnan(xi) or np.isnan(beta):
        return np.nan
    sd = np.sort(ds['samples'])[::-1]
    k = int(np.clip(k, diag['k_grid'][0], diag['k_grid'][-1]))
    return pot_es(sd, k, xi, beta, len(sd), p)


def oracle_k(ds, diag, p, es_true):
    """Grid k minimising |ES(k) - ES_true| / ES_true."""
    kg = np.asarray(diag['k_grid'])
    sd = np.sort(ds['samples'])[::-1]
    n = len(sd)
    best_k, best = None, np.inf
    for i, k in enumerate(kg):
        xi, beta = diag['params'][i]
        if np.isnan(xi) or np.isnan(beta):
            continue
        es = pot_es(sd, int(k), xi, beta, n, p)
        if np.isnan(es) or es <= 0:
            continue
        v = abs((es - es_true) / es_true)
        if v < best:
            best, best_k = v, int(k)
    return best_k


def _pad_stack(series_list):
    """Stack unequal-length 1-D arrays, NaN-padding the short ones."""
    L = max(len(s) for s in series_list)
    out = np.full((len(series_list), L), np.nan)
    for i, s in enumerate(series_list):
        out[i, :len(s)] = s
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%H:%M:%S')

    with open(args.config) as f:
        config = yaml.safe_load(f)
    p = config['evaluate']['quantile_p']
    os.makedirs(FIG_DIR, exist_ok=True)

    # ── load cached diagnostics, keep two_pareto only ─────────────────────
    logger.info("Loading cached diagnostics ...")
    with open('outputs/data/diagnostics.pkl', 'rb') as f:
        all_diag = pickle.load(f)
    tp = [(ds, dg) for ds, dg in all_diag if ds['dist_type'] == 'two_pareto']
    logger.info("two_pareto datasets: %d", len(tp))

    # ── CNN k_pred for the two_pareto subset ──────────────────────────────
    mc = config['model']
    model = ThresholdCNN(
        in_channels=len(config['features']['columns']), channels=mc['channels'],
        kernel_size=mc['kernel_size'], dropout=mc['dropout'],
        pool_sizes=mc['pool_sizes'], task='regression')
    model.load_state_dict(torch.load(
        'outputs/checkpoints/model_regression.pt', weights_only=True))
    model.eval()

    X, _, meta = build_dataset_regression(tp, config)
    y_pred = predict(model, X, task='regression')
    k_pred = np.array([
        int(np.clip(round(m['k_min'] + yp * (m['k_max'] - m['k_min'])),
                    m['k_min'], m['k_max']))
        for yp, m in zip(y_pred, meta)
    ])

    # ── per-dataset ES error at the three thresholds ──────────────────────
    records = []
    for i, (ds, diag) in enumerate(tp):
        prm = ds['params']
        es_true = true_es('two_pareto', prm, p)
        if es_true is None or es_true <= 0:
            continue
        n = int(ds['n'])
        cp = prm['changepoint_frac']
        k_cp = int(np.clip(round(cp * n), diag['k_grid'][0], diag['k_grid'][-1]))
        k_or = oracle_k(ds, diag, p, es_true)
        if k_or is None:
            continue

        es_pred = es_at_k(ds, diag, k_pred[i], p)
        es_cp = es_at_k(ds, diag, k_cp, p)
        es_or = es_at_k(ds, diag, k_or, p)
        if np.isnan(es_pred) or np.isnan(es_cp):
            continue

        xi_pred, _ = _xi_beta_at_k(diag, k_pred[i])
        xi_cp, _ = _xi_beta_at_k(diag, k_cp)
        records.append({
            'alpha1': prm['alpha1'], 'alpha2': prm['alpha2'], 'n': n,
            'cp_count': round(cp * n),
            'k_pred': int(k_pred[i]), 'k_cp': k_cp, 'k_oracle': int(k_or),
            'xi_true': 1.0 / prm['alpha2'],
            'xi_pred': float(xi_pred), 'xi_cp': float(xi_cp),
            'E_pred': (es_pred - es_true) / es_true,
            'E_cp': (es_cp - es_true) / es_true,
            'E_oracle': (es_or - es_true) / es_true,
            'k_overshoot': k_pred[i] / max(round(cp * n), 1),
        })
    logger.info("Decomposed %d two_pareto datasets", len(records))

    # ── results table per alpha2 ──────────────────────────────────────────
    a2s = sorted({r['alpha2'] for r in records})
    logger.info("\nTWO_PARETO ES ERROR vs THRESHOLD (median %%, n=1000)\n"
                "| alpha2 | xi_true | E_pred%% (CNN k) | E_cp%% (k<=changepoint) "
                "| E_oracle%% | k_pred | changepoint | k_oracle |\n"
                "|--------|---------|-----------------|------------------------"
                "|-----------|--------|-------------|----------|")
    for a2 in a2s:
        rs = [r for r in records if r['alpha2'] == a2]
        med = lambda key: float(np.median([r[key] for r in rs]))
        logger.info(
            "| %6.2f | %7.3f | %15.1f | %22.1f | %9.1f | %6.0f | %11.0f | %8.0f |",
            a2, 1.0 / a2, med('E_pred') * 100, med('E_cp') * 100,
            med('E_oracle') * 100, med('k_pred'), med('cp_count'),
            med('k_oracle'))

    # ── Plot 1: ES error by threshold strategy, per alpha2 ────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    box_data, box_colors, positions, ticks = [], [], [], []
    palette = {'E_pred': '#348ABD', 'E_cp': '#8EBA42', 'E_oracle': '#E24A33'}
    labels = {'E_pred': 'CNN k', 'E_cp': 'k ≤ changepoint', 'E_oracle': 'oracle k'}
    pos = 0
    for a2 in a2s:
        rs = [r for r in records if r['alpha2'] == a2]
        for term in ('E_pred', 'E_cp', 'E_oracle'):
            box_data.append([r[term] * 100 for r in rs])
            box_colors.append(palette[term])
            positions.append(pos)
            ticks.append(f'{labels[term]}\nα2={a2}')
            pos += 1
        pos += 1
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                    showfliers=False, widths=0.6,
                    medianprops=dict(color='black', lw=1.5))
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(0, color='black', lw=1, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(ticks, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('ES relative error (%)')
    ax.set_title('two_pareto: ES error by threshold choice')
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/two_pareto_es_by_threshold.png', dpi=150)
    plt.close(fig)

    # ── Plot 2: xi_hat / Hill profiles vs k, panel per alpha2 ─────────────
    fig, axes = plt.subplots(1, len(a2s), figsize=(5 * len(a2s), 5),
                             squeeze=False)
    for ax, a2 in zip(axes[0], a2s):
        grp = [(ds, dg) for ds, dg in tp if ds['params']['alpha2'] == a2]
        xi_st = _pad_stack([np.asarray(dg['xi_series']) for _, dg in grp])
        hl_st = _pad_stack([np.asarray(dg['hill_series']) for _, dg in grp])
        kg = max((dg['k_grid'] for _, dg in grp), key=len)
        cp_count = round(grp[0][0]['params']['changepoint_frac']
                         * grp[0][0]['n'])
        ax.plot(kg, np.nanmean(xi_st, axis=0), color='#E24A33', lw=2,
                label='GPD-MLE xi')
        ax.plot(kg, np.nanmean(hl_st, axis=0), color='#348ABD', lw=2,
                label='Hill')
        ax.axhline(1.0 / a2, color='black', ls='--', lw=1.2,
                   label=f'true xi = {1/a2:.2f}')
        ax.axvline(cp_count, color='#8EBA42', ls=':', lw=2,
                   label=f'changepoint (k={cp_count})')
        ax.set_xlabel('k (exceedance count)')
        ax.set_ylabel('tail-index estimate')
        ax.set_title(f'α2 = {a2}  (true xi = {1/a2:.3f})')
        ax.legend(fontsize=8)
    fig.suptitle('two_pareto: tail-index estimate vs threshold — '
                 'the estimate decays once k crosses the splice')
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/two_pareto_xi_profiles.png', dpi=150)
    plt.close(fig)

    # ── Plot 3: ES error vs k overshoot ratio ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for a2 in a2s:
        rs = [r for r in records if r['alpha2'] == a2]
        ax.scatter([r['k_overshoot'] for r in rs],
                   [r['E_pred'] * 100 for r in rs],
                   s=12, alpha=0.4, color=A2_COLORS.get(a2, '#555'),
                   label=f'α2={a2} (xi={1/a2:.2f})', edgecolors='none')
    ax.axvline(1.0, color='#8EBA42', ls=':', lw=2, label='k = changepoint')
    ax.axhline(0, color='black', lw=1, alpha=0.6)
    ax.set_xlabel('k_pred / changepoint count  (>1 = window straddles splice)')
    ax.set_ylabel('ES relative error at CNN k (%)')
    ax.set_title('two_pareto: ES underestimation grows as k overshoots the splice')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/two_pareto_overshoot.png', dpi=150)
    plt.close(fig)

    with open(f'{OUT_DIR}/two_pareto_investigation.pkl', 'wb') as f:
        pickle.dump({'records': records}, f)
    logger.info("Saved results + 3 plots to %s", OUT_DIR)


if __name__ == '__main__':
    main()
