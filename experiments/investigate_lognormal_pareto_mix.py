#!/usr/bin/env python
"""Investigate the lognormal_pareto_mix ES failure.

lognormal_pareto_mix is a composite distribution: each observation is drawn
from Pareto(pareto_alpha) with probability mix_frac and from
Lognormal(lognormal_mu, lognormal_sigma) otherwise. The true ES is set by
the Pareto component; the lognormal observations are the bulk.

Unlike two_pareto there is no hard splice. The expected number of Pareto
draws in a sample of size n is n * mix_frac, and since the Pareto with
alpha in {1.5, 2.0} has a much heavier tail than Lognormal(0, 1), the top
order statistics are dominated by Pareto draws, but the boundary between
"Pareto-dominated tail" and "lognormal-dominated body" is soft.

Hypothesis: as for two_pareto, the CNN selects k far larger than the
expected Pareto count, so the top-k window includes a significant number
of lognormal bulk observations. The resulting GPD fit is a blended-tail
estimate that systematically misses the true Pareto-driven ES.

This script computes, for every lognormal_pareto_mix dataset in the cached
diagnostics, the ES error at three thresholds:
  - k_pred : the CNN-selected k
  - k_mix  : round(mix_frac * n) — the expected number of Pareto draws
  - k_oracle : the grid k that minimises |ES(k) - ES_true| / ES_true

Usage:
    python investigate_lognormal_pareto_mix.py --config config/default.yaml
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

# Make `src` importable when this script is launched from a subfolder.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

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

OUT_DIR = 'outputs/lognormal_pareto_mix'
FIG_DIR = f'{OUT_DIR}/figures'
ALPHA_COLORS = {1.5: '#E24A33', 2.0: '#348ABD'}


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

    # load cached diagnostics, keep lognormal_pareto_mix only
    logger.info("Loading cached diagnostics ...")
    with open('outputs/data/diagnostics.pkl', 'rb') as f:
        all_diag = pickle.load(f)
    lp = [(ds, dg) for ds, dg in all_diag
          if ds['dist_type'] == 'lognormal_pareto_mix']
    logger.info("lognormal_pareto_mix datasets: %d", len(lp))

    # CNN k_pred for the lognormal_pareto_mix subset
    mc = config['model']
    model = ThresholdCNN(
        in_channels=len(config['features']['columns']), channels=mc['channels'],
        kernel_size=mc['kernel_size'], dropout=mc['dropout'],
        pool_sizes=mc['pool_sizes'], task='regression')
    model.load_state_dict(torch.load(
        'outputs/checkpoints/model_regression.pt', weights_only=True))
    model.eval()

    X, _, meta = build_dataset_regression(lp, config)
    y_pred = predict(model, X, task='regression')
    k_pred = np.array([
        int(np.clip(round(m['k_min'] + yp * (m['k_max'] - m['k_min'])),
                    m['k_min'], m['k_max']))
        for yp, m in zip(y_pred, meta)
    ])

    # per-dataset ES error at the three thresholds
    records = []
    for i, (ds, diag) in enumerate(lp):
        prm = ds['params']
        es_true = true_es('lognormal_pareto_mix', prm, p)
        if es_true is None or es_true <= 0:
            continue
        n = int(ds['n'])
        mix_frac = prm['mix_frac']
        k_mix = int(np.clip(round(mix_frac * n),
                            diag['k_grid'][0], diag['k_grid'][-1]))
        k_or = oracle_k(ds, diag, p, es_true)
        if k_or is None:
            continue

        es_pred = es_at_k(ds, diag, k_pred[i], p)
        es_mix = es_at_k(ds, diag, k_mix, p)
        es_or = es_at_k(ds, diag, k_or, p)
        if np.isnan(es_pred) or np.isnan(es_mix):
            continue

        xi_pred, _ = _xi_beta_at_k(diag, k_pred[i])
        xi_mix, _ = _xi_beta_at_k(diag, k_mix)
        records.append({
            'pareto_alpha': prm['pareto_alpha'],
            'mix_frac': mix_frac,
            'n': n,
            'mix_count': round(mix_frac * n),
            'k_pred': int(k_pred[i]),
            'k_mix': k_mix,
            'k_oracle': int(k_or),
            'xi_true': 1.0 / prm['pareto_alpha'],
            'xi_pred': float(xi_pred),
            'xi_mix': float(xi_mix),
            'E_pred': (es_pred - es_true) / es_true,
            'E_mix': (es_mix - es_true) / es_true,
            'E_oracle': (es_or - es_true) / es_true,
            'k_overshoot': k_pred[i] / max(round(mix_frac * n), 1),
        })
    logger.info("Decomposed %d lognormal_pareto_mix datasets", len(records))

    # results table per pareto_alpha
    alphas = sorted({r['pareto_alpha'] for r in records})
    logger.info("\nLOGNORMAL_PARETO_MIX ES ERROR vs THRESHOLD "
                "(median %%, n=1000)\n"
                "| alpha  | xi_true | E_pred%% (CNN k) | E_mix%% (k=mix exp.) "
                "| E_oracle%% | k_pred | mix_count | k_oracle |\n"
                "|--------|---------|-----------------|---------------------"
                "|-----------|--------|-----------|----------|")
    for a in alphas:
        rs = [r for r in records if r['pareto_alpha'] == a]
        med = lambda key: float(np.median([r[key] for r in rs]))
        logger.info(
            "| %6.2f | %7.3f | %15.1f | %19.1f | %9.1f | %6.0f | %9.0f | %8.0f |",
            a, 1.0 / a, med('E_pred') * 100, med('E_mix') * 100,
            med('E_oracle') * 100, med('k_pred'), med('mix_count'),
            med('k_oracle'))

    # Also print pooled RRMSE for direct comparison with the §5.1 number
    def rrmse(arr):
        return float(np.sqrt(np.mean(np.asarray(arr) ** 2)) * 100)

    e_pred = np.array([r['E_pred'] for r in records])
    e_mix = np.array([r['E_mix'] for r in records])
    e_oracle = np.array([r['E_oracle'] for r in records])
    logger.info("\nPooled RRMSE across all lognormal_pareto_mix configs:")
    logger.info("  E_pred   (CNN threshold):    RRMSE = %.1f%%  median = %+.1f%%",
                rrmse(e_pred), float(np.median(e_pred)) * 100)
    logger.info("  E_mix    (mix expected k):   RRMSE = %.1f%%  median = %+.1f%%",
                rrmse(e_mix), float(np.median(e_mix)) * 100)
    logger.info("  E_oracle (oracle threshold): RRMSE = %.1f%%  median = %+.1f%%",
                rrmse(e_oracle), float(np.median(e_oracle)) * 100)
    selection_share = (rrmse(e_pred) - rrmse(e_oracle)) / rrmse(e_pred) * 100
    logger.info("  Selection accounts for ~%.0f%% of pooled RRMSE; "
                "the rest is residual at oracle.", selection_share)

    # Plot 1: ES error by threshold strategy, per pareto_alpha
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data, box_colors, positions, ticks = [], [], [], []
    palette = {'E_pred': '#348ABD', 'E_mix': '#8EBA42', 'E_oracle': '#E24A33'}
    labels = {'E_pred': 'CNN k', 'E_mix': 'k = mix expected',
              'E_oracle': 'oracle k'}
    pos = 0
    for a in alphas:
        rs = [r for r in records if r['pareto_alpha'] == a]
        for term in ('E_pred', 'E_mix', 'E_oracle'):
            box_data.append([r[term] * 100 for r in rs])
            box_colors.append(palette[term])
            positions.append(pos)
            ticks.append(f'{labels[term]}\nα={a}')
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
    ax.set_xticklabels(ticks, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('ES relative error (%)')
    ax.set_title('lognormal_pareto_mix: ES error by threshold choice')
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/lpm_es_by_threshold.png', dpi=150)
    plt.close(fig)

    # Plot 2: xi_hat / Hill profiles vs k, panel per pareto_alpha
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5),
                             squeeze=False)
    for ax, a in zip(axes[0], alphas):
        grp = [(ds, dg) for ds, dg in lp if ds['params']['pareto_alpha'] == a]
        xi_st = _pad_stack([np.asarray(dg['xi_series']) for _, dg in grp])
        hl_st = _pad_stack([np.asarray(dg['hill_series']) for _, dg in grp])
        kg = max((dg['k_grid'] for _, dg in grp), key=len)
        mix_count = round(grp[0][0]['params']['mix_frac']
                          * grp[0][0]['n'])
        ax.plot(kg, np.nanmean(xi_st, axis=0), color='#E24A33', lw=2,
                label='GPD-MLE xi')
        ax.plot(kg, np.nanmean(hl_st, axis=0), color='#348ABD', lw=2,
                label='Hill')
        ax.axhline(1.0 / a, color='black', ls='--', lw=1.2,
                   label=f'true xi = {1/a:.2f}')
        ax.axvline(mix_count, color='#8EBA42', ls=':', lw=2,
                   label=f'mix expected (k={mix_count})')
        ax.set_xlabel('k (exceedance count)')
        ax.set_ylabel('tail-index estimate')
        ax.set_title(f'pareto_alpha = {a}  (true xi = {1/a:.3f})')
        ax.legend(fontsize=8)
    fig.suptitle('lognormal_pareto_mix: tail-index estimate vs threshold — '
                 'the estimate decays as k pulls lognormal bulk into the fit')
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/lpm_xi_profiles.png', dpi=150)
    plt.close(fig)

    # Plot 3: ES error vs k overshoot ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    for a in alphas:
        rs = [r for r in records if r['pareto_alpha'] == a]
        ax.scatter([r['k_overshoot'] for r in rs],
                   [r['E_pred'] * 100 for r in rs],
                   s=14, alpha=0.4,
                   color=ALPHA_COLORS.get(a, '#555'),
                   label=f'α={a} (xi={1/a:.2f})', edgecolors='none')
    ax.axvline(1.0, color='#8EBA42', ls=':', lw=2, label='k = mix expected')
    ax.axhline(0, color='black', lw=1, alpha=0.6)
    ax.set_xlabel('k_pred / mix expected count  '
                  '(>1 = window includes lognormal bulk)')
    ax.set_ylabel('ES relative error at CNN k (%)')
    ax.set_title('lognormal_pareto_mix: ES error vs threshold overshoot')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/lpm_overshoot.png', dpi=150)
    plt.close(fig)

    with open(f'{OUT_DIR}/lognormal_pareto_mix_investigation.pkl', 'wb') as f:
        pickle.dump({'records': records}, f)
    logger.info("Saved results + 3 plots to %s", OUT_DIR)


if __name__ == '__main__':
    main()
