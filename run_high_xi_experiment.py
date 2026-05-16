#!/usr/bin/env python
"""High tail-index stress experiment.

Diagnoses why the network-determined xi produces unreliable Expected
Shortfall for high tail index (xi > 0.5, badly above 0.7). Generates clean
single-family Pareto data at alpha in {1.2, 1.3, 1.4} (xi = 1/alpha =
0.833, 0.769, 0.714), runs it through the *existing* trained pipeline, and
decomposes the ES relative error E = (ES_est - ES_true) / ES_true into:

  E_irr  : irreducible error with an oracle threshold (convexity + GPD-MLE)
  E_sel  : extra error from CNN k-selection vs the oracle  (E_cnn - E_irr)
  E_fb   : effect of the xi>0.7 fallback inside pot_es (Hill/Weissman)
  E_corr : ES error after the deployed ES correction network

Within E_irr the error is further split (at the oracle k) into
  E_convexity : closed-form ES residual when xi is the true 1/alpha
  E_xi_effect : extra error from the GPD-MLE xi vs the true xi

Usage:
    python run_high_xi_experiment.py --config config/high_xi.yaml
    python run_high_xi_experiment.py --config config/high_xi.yaml --smoke
    python run_high_xi_experiment.py --config config/high_xi.yaml --fresh
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
from joblib import Parallel, delayed

from src.es_correction import ESCorrectionNet, apply_correction
from src.evaluate import pot_es, pot_quantile, true_es
from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.pot import process_one_dataset
from src.synthetic import generate_all
from src.train import predict

plt.style.use('ggplot')
plt.rcParams.update({
    'figure.figsize': (10, 6), 'figure.dpi': 150, 'font.size': 11,
    'figure.facecolor': 'white', 'axes.facecolor': '#EBEBEB',
})

ALPHA_COLORS = {1.2: '#E24A33', 1.3: '#348ABD', 1.4: '#8EBA42'}

logger = logging.getLogger(__name__)

OUT_DIR = 'outputs/high_xi'
DATA_DIR = f'{OUT_DIR}/data'
FIG_DIR = f'{OUT_DIR}/figures'


# ──────────────────────────────────────────────────────────────────────────
# ES helpers
# ──────────────────────────────────────────────────────────────────────────
def closed_form_es_at_xi(sorted_desc, k, xi_use, beta, n, p):
    """GPD closed-form ES evaluated with an arbitrary shape parameter.

    ES = (VaR + beta - xi*u) / (1 - xi), always — i.e. the xi<=0.7 branch of
    ``src.evaluate.pot_es`` with the xi>0.7 fallback gate removed. Both VaR
    and the amplification use ``xi_use``.

    With ``xi_use`` = the GPD-MLE xi this is the fallback-free ES estimate;
    with ``xi_use`` = the true 1/alpha it is the "perfect-xi" ES. Differencing
    the two attributes the irreducible error between amplification convexity
    and GPD-MLE xi error.
    """
    var_est = pot_quantile(sorted_desc, k, xi_use, beta, n, p)
    if abs(xi_use) < 1e-8:
        return var_est + beta
    u = sorted_desc[k]
    one_minus_xi = max(1.0 - xi_use, 0.05)  # same stability clamp as pot_es
    return (var_est + beta - xi_use * u) / one_minus_xi


def _xi_beta_at_k(diag, k):
    """GPD (xi, beta) at exceedance count k from the precomputed fit grid."""
    k_grid = np.asarray(diag['k_grid'])
    k_idx = min(int(np.searchsorted(k_grid, k)), len(diag['params']) - 1)
    return diag['params'][k_idx]


def compute_oracle_k(ds, diag, p, es_true, target='es'):
    """Best k over the whole grid: the threshold the CNN *could* have picked.

    target='es' : minimise |pot_es(k) - es_true| / es_true
    target='xi' : minimise |xi_hat(k) - xi_true|

    Returns the oracle k (int) or None if every k has a NaN GPD fit.
    """
    k_grid = np.asarray(diag['k_grid'])
    params = diag['params']
    sorted_desc = np.sort(ds['samples'])[::-1]
    n = len(sorted_desc)
    xi_true = 1.0 / ds['params']['alpha']

    best_k, best_val = None, np.inf
    for i, k in enumerate(k_grid):
        xi, beta = params[i]
        if np.isnan(xi) or np.isnan(beta):
            continue
        if target == 'es':
            es = pot_es(sorted_desc, int(k), xi, beta, n, p)
            if np.isnan(es) or es <= 0:
                continue
            val = abs((es - es_true) / es_true)
        else:
            val = abs(xi - xi_true)
        if val < best_val:
            best_val, best_k = val, int(k)
    return best_k


# ──────────────────────────────────────────────────────────────────────────
# Data + diagnostics
# ──────────────────────────────────────────────────────────────────────────
def load_or_compute_diagnostics(config, n_jobs, fresh):
    """Generate the high-xi Pareto datasets and POT diagnostics (cached)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    syn = config['synthetic']
    cache_meta = {
        'alpha': syn['distributions']['pareto']['alpha'],
        'sample_sizes': syn['sample_sizes'],
        'n_replications': syn['n_replications'],
        'seed': syn['seed'],
    }
    cache_path = f'{DATA_DIR}/diagnostics_high_xi.pkl'

    if os.path.exists(cache_path) and not fresh:
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if cached.get('meta') == cache_meta:
            logger.info("Loaded cached diagnostics: %d datasets",
                        len(cached['all_diagnostics']))
            return cached['all_diagnostics']
        logger.info("Cache config mismatch — recomputing diagnostics")

    logger.info("Generating synthetic datasets ...")
    datasets = generate_all(syn)
    logger.info("Computing POT diagnostics for %d datasets ...", len(datasets))
    all_diagnostics = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_one_dataset)(ds, config['pot']) for ds in datasets
    )
    with open(cache_path, 'wb') as f:
        pickle.dump({'meta': cache_meta, 'all_diagnostics': all_diagnostics}, f)
    logger.info("Cached diagnostics to %s", cache_path)
    return all_diagnostics


def predict_k_per_dataset(all_diagnostics, model, config):
    """CNN-predicted k for every dataset, grouped by sample size.

    The dataset is built per sample size so the zero-padding length L_max
    matches the natural k-grid of that n (the CNN was trained on n=1000).
    Returns an ndarray of k_pred aligned with all_diagnostics.
    """
    k_pred = np.full(len(all_diagnostics), -1, dtype=int)
    sizes = sorted({int(ds['n']) for ds, _ in all_diagnostics})
    for n in sizes:
        idxs = [i for i, (ds, _) in enumerate(all_diagnostics)
                if int(ds['n']) == n]
        sub = [all_diagnostics[i] for i in idxs]
        X, _, meta = build_dataset_regression(sub, config)
        y_pred = predict(model, X, task='regression')
        for j, gi in enumerate(idxs):
            m = meta[j]
            k = round(m['k_min'] + float(y_pred[j]) * (m['k_max'] - m['k_min']))
            k_pred[gi] = int(np.clip(k, m['k_min'], m['k_max']))
        logger.info("CNN k_pred computed for n=%d (%d datasets)", n, len(idxs))
    return k_pred


def load_es_correction_net(config):
    """Reconstruct the trained ES correction network from its checkpoint."""
    path = config['high_xi_experiment']['es_correction_checkpoint']
    ec = config.get('es_correction', {})
    ckpt = torch.load(path, weights_only=False)
    model = ESCorrectionNet(
        in_features=9, hidden=ec.get('hidden', 32),
        output_lo=ec.get('output_lo', 0.5), output_hi=ec.get('output_hi', 3.0),
        output_mode=ec.get('output_mode', 'softplus'))
    model.load_state_dict(ckpt['state_dict'])
    model.X_mean = ckpt['X_mean']
    model.X_std = ckpt['X_std']
    model.eval()
    logger.info("Loaded ES correction net from %s", path)
    return model


# ──────────────────────────────────────────────────────────────────────────
# Per-dataset ES error decomposition
# ──────────────────────────────────────────────────────────────────────────
def decompose_es_error(ds, diag, k_star, k_pred, k_oracle, corr_model, p, config):
    """Full ES error decomposition for one dataset. Returns a record dict."""
    alpha = ds['params']['alpha']
    xi_true = 1.0 / alpha
    es_true = true_es('pareto', ds['params'], p)
    if es_true is None or es_true <= 0:
        return None

    sorted_desc = np.sort(ds['samples'])[::-1]
    n = len(sorted_desc)

    xi_pred, beta_pred = _xi_beta_at_k(diag, k_pred)
    xi_or, beta_or = _xi_beta_at_k(diag, k_oracle)
    xi_ks, beta_ks = _xi_beta_at_k(diag, k_star)
    if any(np.isnan(v) for v in (xi_pred, beta_pred, xi_or, beta_or)):
        return None

    es_cnn = pot_es(sorted_desc, k_pred, xi_pred, beta_pred, n, p)
    es_oracle = pot_es(sorted_desc, k_oracle, xi_or, beta_or, n, p)
    es_kstar = pot_es(sorted_desc, k_star, xi_ks, beta_ks, n, p)
    es_cnn_closed = closed_form_es_at_xi(sorted_desc, k_pred, xi_pred, beta_pred, n, p)
    es_oracle_closed = closed_form_es_at_xi(sorted_desc, k_oracle, xi_or, beta_or, n, p)
    es_oracle_truexi = closed_form_es_at_xi(sorted_desc, k_oracle, xi_true, beta_or, n, p)
    es_corr = apply_correction(corr_model, ds, diag, k_pred, es_cnn, p, config)

    if np.isnan(es_cnn) or np.isnan(es_oracle) or es_cnn <= 0:
        return None

    def E(es):
        return (es - es_true) / es_true

    e_cnn = E(es_cnn)
    e_irr = E(es_oracle)
    return {
        'alpha': alpha, 'xi_true': xi_true, 'n': int(ds['n']),
        'k_star': int(k_star), 'k_pred': int(k_pred), 'k_oracle': int(k_oracle),
        'xi_pred': float(xi_pred), 'xi_oracle': float(xi_or),
        'xi_kstar': float(xi_ks),
        'es_true': float(es_true), 'es_cnn': float(es_cnn),
        'es_oracle': float(es_oracle), 'es_kstar': float(es_kstar),
        'es_corr': float(es_corr),
        # primary decomposition --------------------------------------------
        'E_cnn': float(e_cnn),
        'E_irr': float(e_irr),
        'E_sel': float(e_cnn - e_irr),
        'E_kstar': float(E(es_kstar)),
        'E_fb': float((es_cnn - es_cnn_closed) / es_true),
        'E_corr': float(E(es_corr)),
        # (a) convexity vs (b) xi error, measured at the oracle k ----------
        'E_convexity': float((es_oracle_truexi - es_true) / es_true),
        'E_xi_effect': float((es_oracle_closed - es_oracle_truexi) / es_true),
        'fallback_engaged': bool(xi_pred > 0.7),
    }


# ──────────────────────────────────────────────────────────────────────────
# xi bias / variance vs k
# ──────────────────────────────────────────────────────────────────────────
def _pad_stack(series_list):
    """Stack 1-D arrays of unequal length, NaN-padding the short ones.

    All k-grids start at the same k_min and are contiguous, so index i maps
    to the same k across reps; only the high-k tail length varies (it depends
    on the post-declustering effective sample size)."""
    L_max = max(len(s) for s in series_list)
    out = np.full((len(series_list), L_max), np.nan)
    for i, s in enumerate(series_list):
        out[i, :len(s)] = s
    return out


def xi_bias_variance_by_k(diag_group, xi_true):
    """Per-k GPD-MLE xi bias/variance and Hill bias/variance for one
    (alpha, n) group. k-grid length varies across reps because declustering
    removes a variable number of points, so reps are NaN-padded before
    aggregation; index i still maps to a fixed k = k_min + i.
    """
    xi_stack = _pad_stack([np.asarray(d['xi_series']) for _, d in diag_group])
    hill_stack = _pad_stack([np.asarray(d['hill_series']) for _, d in diag_group])
    longest = max((d['k_grid'] for _, d in diag_group), key=len)
    k_grid = np.asarray(longest)
    return {
        'k_grid': k_grid,
        'xi_bias': np.nanmean(xi_stack, axis=0) - xi_true,
        'xi_std': np.nanstd(xi_stack, axis=0),
        'hill_bias': np.nanmean(hill_stack, axis=0) - xi_true,
        'hill_std': np.nanstd(hill_stack, axis=0),
        'n_reps': xi_stack.shape[0],
    }


# ──────────────────────────────────────────────────────────────────────────
# Aggregation + table
# ──────────────────────────────────────────────────────────────────────────
def _median_iqr(vals):
    a = np.asarray(vals, dtype=float)
    return {
        'median': float(np.median(a)),
        'q25': float(np.percentile(a, 25)),
        'q75': float(np.percentile(a, 75)),
        'mean': float(np.mean(a)),
        'count': int(a.size),
    }


def aggregate(records):
    """Median + IQR of each decomposition term per (alpha, n)."""
    terms = ['E_cnn', 'E_irr', 'E_sel', 'E_kstar', 'E_fb', 'E_corr',
             'E_convexity', 'E_xi_effect']
    decomp = {}
    for r in records:
        key = (r['alpha'], r['n'])
        decomp.setdefault(key, []).append(r)
    out = {}
    for key, recs in sorted(decomp.items()):
        agg = {t: _median_iqr([r[t] for r in recs]) for t in terms}
        agg['k_oracle_median'] = float(np.median([r['k_oracle'] for r in recs]))
        agg['k_pred_median'] = float(np.median([r['k_pred'] for r in recs]))
        agg['k_star_median'] = float(np.median([r['k_star'] for r in recs]))
        agg['xi_true'] = recs[0]['xi_true']
        agg['count'] = len(recs)
        out[key] = agg
    return out


def format_table(decomposition):
    """Markdown table of the decomposition, percentages."""
    hdr = ('| alpha | xi_true |    n | E_cnn% | E_corr% | E_irr% | E_sel% '
           '| E_fb% | E_conv% | E_xi% | k_oracle | k_pred | k* |')
    sep = '|' + '|'.join(['---'] * 13) + '|'
    lines = [hdr, sep]
    for (alpha, n), a in sorted(decomposition.items()):
        lines.append(
            f"| {alpha:.1f} | {a['xi_true']:.3f} | {n:5d} "
            f"| {a['E_cnn']['median']*100:6.1f} "
            f"| {a['E_corr']['median']*100:7.1f} "
            f"| {a['E_irr']['median']*100:6.1f} "
            f"| {a['E_sel']['median']*100:6.1f} "
            f"| {a['E_fb']['median']*100:5.1f} "
            f"| {a['E_convexity']['median']*100:7.1f} "
            f"| {a['E_xi_effect']['median']*100:5.1f} "
            f"| {a['k_oracle_median']:8.0f} "
            f"| {a['k_pred_median']:6.0f} "
            f"| {a['k_star_median']:3.0f} |")
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────
def make_plots(records, decomposition, xi_profiles, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    alphas = sorted({r['alpha'] for r in records})
    ns = sorted({r['n'] for r in records})

    # Plot 1 — decomposition bars (one panel per n) ------------------------
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 5), squeeze=False)
    for ax, n in zip(axes[0], ns):
        x = np.arange(len(alphas))
        w = 0.25
        for off, term, col, lbl in [
                (-w, 'E_irr', '#E24A33', 'E_irr (irreducible)'),
                (0.0, 'E_sel', '#348ABD', 'E_sel (CNN k-selection)'),
                (w, 'E_fb', '#8EBA42', 'E_fb (xi>0.7 fallback)')]:
            vals = [decomposition[(a, n)][term]['median'] * 100 for a in alphas]
            ax.bar(x + off, vals, w, color=col, label=lbl)
        ax.axhline(0, color='black', lw=1, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'a={a}\nxi={1/a:.2f}' for a in alphas])
        ax.set_title(f'n = {n}')
        ax.set_ylabel('Median ES relative error (%)')
    axes[0][0].legend(fontsize=8)
    fig.suptitle('ES error decomposition by tail index')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/high_xi_decomposition_bars.png', dpi=150)
    plt.close(fig)

    # Plot 2 — ES error box plots by strategy, n=1000 ----------------------
    n0 = ns[0]
    fig, ax = plt.subplots(figsize=(12, 6))
    strategies = [('E_irr', 'oracle k'), ('E_cnn', 'CNN k'),
                  ('E_kstar', 'baseline k*'), ('E_corr', 'CNN + correction net')]
    positions, ticklabels, box_data, box_colors = [], [], [], []
    palette = ['#E24A33', '#348ABD', '#777777', '#8EBA42']
    pos = 0
    for ai, a in enumerate(alphas):
        for si, (term, lbl) in enumerate(strategies):
            vals = [r[term] * 100 for r in records
                    if r['alpha'] == a and r['n'] == n0]
            box_data.append(vals)
            box_colors.append(palette[si])
            positions.append(pos)
            ticklabels.append(f'{lbl}\na={a}')
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
    ax.set_xticklabels(ticklabels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('ES relative error (%)')
    ax.set_title(f'ES error by estimation strategy (n={n0})')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/high_xi_es_error_by_alpha.png', dpi=150)
    plt.close(fig)

    # Plot 3 — ES error vs n -----------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for a in alphas:
        col = ALPHA_COLORS.get(a, '#555')
        e_cnn = [decomposition[(a, n)]['E_cnn']['median'] * 100 for n in ns]
        e_irr = [decomposition[(a, n)]['E_irr']['median'] * 100 for n in ns]
        ax.plot(ns, e_cnn, 'o-', color=col, lw=2,
                label=f'E_cnn  a={a} (xi={1/a:.2f})')
        ax.plot(ns, e_irr, 's--', color=col, lw=1.5, alpha=0.7,
                label=f'E_irr  a={a}')
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.set_xlabel('sample size n')
    ax.set_ylabel('Median ES relative error (%)')
    ax.set_title('ES error vs sample size — does the irreducible error shrink?')
    ax.set_xticks(ns)
    ax.legend(fontsize=8, ncol=len(alphas))
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/high_xi_es_error_vs_n.png', dpi=150)
    plt.close(fig)

    # Plot 4 — xi bias vs k, panel per alpha (n=1000) ----------------------
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5),
                             squeeze=False)
    for ax, a in zip(axes[0], alphas):
        prof = xi_profiles.get((a, n0))
        if prof is None:
            continue
        kg = prof['k_grid']
        ax.plot(kg, prof['xi_bias'], color='#E24A33', lw=2, label='GPD-MLE xi bias')
        ax.fill_between(kg, prof['xi_bias'] - prof['xi_std'],
                        prof['xi_bias'] + prof['xi_std'],
                        color='#E24A33', alpha=0.15)
        ax.plot(kg, prof['hill_bias'], color='#348ABD', lw=2, label='Hill bias')
        ax.axhline(0, color='black', lw=1, alpha=0.5)
        ax.set_xlabel('k (exceedance count)')
        ax.set_ylabel('estimator - xi_true')
        ax.set_title(f'a={a}  (xi_true={1/a:.3f}),  n={n0}')
        ax.legend(fontsize=8)
    fig.suptitle('Tail-index estimator bias vs threshold')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/high_xi_xi_bias_vs_k.png', dpi=150)
    plt.close(fig)

    # Plot 5 — k_pred vs k_oracle scatter (n=1000) -------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    for a in alphas:
        ko = [r['k_oracle'] for r in records if r['alpha'] == a and r['n'] == n0]
        kp = [r['k_pred'] for r in records if r['alpha'] == a and r['n'] == n0]
        ax.scatter(ko, kp, s=12, alpha=0.4, color=ALPHA_COLORS.get(a, '#555'),
                   label=f'a={a} (xi={1/a:.2f})', edgecolors='none')
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6, label='k_pred = k_oracle')
    ax.set_xlabel('oracle k (minimises |ES error|)')
    ax.set_ylabel('CNN-predicted k')
    ax.set_title(f'CNN threshold vs oracle threshold (n={n0})')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/high_xi_k_selection.png', dpi=150)
    plt.close(fig)

    logger.info("Saved 5 plots to %s/high_xi_*.png", fig_dir)


# ──────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────
def run_verification(records, model, config):
    """Plan verification checks 1-6. Logs PASS/FAIL, raises on hard failure."""
    logger.info("=" * 60)
    logger.info("VERIFICATION")
    from scipy import stats as sp

    # Check 2 — closed-form Pareto ES sanity
    p = config['evaluate']['quantile_p']
    for a in config['synthetic']['distributions']['pareto']['alpha']:
        manual = sp.pareto.ppf(p, b=a) * a / (a - 1)
        got = true_es('pareto', {'alpha': a}, p)
        assert abs(got - manual) < 1e-9, f"closed-form ES mismatch alpha={a}"
    logger.info("  [PASS] check 2: closed-form Pareto ES matches alpha/(alpha-1)")

    # Check 3 — decomposition identity E_cnn == E_irr + E_sel
    max_err = max(abs(r['E_cnn'] - (r['E_irr'] + r['E_sel'])) for r in records)
    assert max_err < 1e-9, f"decomposition identity broken (max={max_err})"
    logger.info("  [PASS] check 3: E_cnn == E_irr + E_sel (max dev %.1e)", max_err)

    # Check 4 — fallback gate self-test
    eng = [r for r in records if r['fallback_engaged']]
    nofb = [r for r in records if not r['fallback_engaged']]
    if nofb:
        assert all(abs(r['E_fb']) < 1e-9 for r in nofb), \
            "E_fb nonzero where xi_pred <= 0.7"
        logger.info("  [PASS] check 4a: E_fb == 0 for xi_pred <= 0.7 (%d recs)",
                    len(nofb))
    if eng:
        share = np.mean([abs(r['E_fb']) > 1e-9 for r in eng])
        logger.info("  [INFO] check 4b: fallback engaged in %d recs, "
                    "E_fb != 0 in %.0f%%", len(eng), share * 100)

    # Check 5 — oracle optimality
    viol = sum(1 for r in records
               if abs(r['E_irr']) > abs(r['E_cnn']) + 1e-9
               or abs(r['E_irr']) > abs(r['E_kstar']) + 1e-9)
    assert viol == 0, f"oracle not optimal in {viol} records"
    logger.info("  [PASS] check 5: |E_irr| <= |E_cnn| and |E_kstar| (all recs)")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/high_xi.yaml')
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--smoke', action='store_true',
                        help='n=1000 only (fast smoke test)')
    parser.add_argument('--fresh', action='store_true',
                        help='recompute diagnostics, ignore cache')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%H:%M:%S')

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.smoke:
        config['synthetic']['sample_sizes'] = [1000]
        logger.info("SMOKE MODE: n=1000 only")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    p = config['evaluate']['quantile_p']

    # ── data + diagnostics ────────────────────────────────────────────────
    all_diagnostics = load_or_compute_diagnostics(config, args.n_jobs, args.fresh)

    # ── CNN (verification check 1: architecture match) ────────────────────
    mc = config['model']
    in_channels = len(config['features']['columns'])
    model = ThresholdCNN(
        in_channels=in_channels, channels=mc['channels'],
        kernel_size=mc['kernel_size'], dropout=mc['dropout'],
        pool_sizes=mc['pool_sizes'], task='regression')
    ckpt_path = config['high_xi_experiment']['cnn_checkpoint']
    state = torch.load(ckpt_path, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, \
        f"CNN architecture mismatch: missing={missing}, unexpected={unexpected}"
    model.eval()
    logger.info("  [PASS] check 1: CNN checkpoint loaded, architecture matches")

    corr_model = load_es_correction_net(config)

    # ── three k-values + decomposition ────────────────────────────────────
    k_pred_all = predict_k_per_dataset(all_diagnostics, model, config)

    records = []
    for i, (ds, diag) in enumerate(all_diagnostics):
        es_true = true_es('pareto', ds['params'], p)
        if es_true is None or es_true <= 0:
            continue
        k_oracle = compute_oracle_k(
            ds, diag, p, es_true,
            target=config['high_xi_experiment'].get('oracle_target', 'es'))
        if k_oracle is None or k_pred_all[i] < 0:
            continue
        rec = decompose_es_error(ds, diag, int(diag['k_star']),
                                 int(k_pred_all[i]), k_oracle,
                                 corr_model, p, config)
        if rec is not None:
            records.append(rec)
    logger.info("Decomposed ES error for %d / %d datasets",
                len(records), len(all_diagnostics))

    # ── xi bias/variance profiles ─────────────────────────────────────────
    groups = {}
    for d in all_diagnostics:
        ds = d[0]
        groups.setdefault((ds['params']['alpha'], int(ds['n'])), []).append(d)
    xi_profiles = {
        key: xi_bias_variance_by_k(grp, 1.0 / key[0])
        for key, grp in groups.items()
    }

    # ── aggregate, verify, report ─────────────────────────────────────────
    decomposition = aggregate(records)
    run_verification(records, model, config)

    table = format_table(decomposition)
    logger.info("\nES ERROR DECOMPOSITION (median, %%)\n%s", table)

    results = {
        'records': records,
        'decomposition': decomposition,
        'xi_profiles': xi_profiles,
        'table_markdown': table,
        'config_snapshot': config,
    }
    out_pkl = f'{OUT_DIR}/high_xi_experiment.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)
    logger.info("Results saved to %s", out_pkl)

    make_plots(records, decomposition, xi_profiles, FIG_DIR)
    logger.info("Done.")


if __name__ == '__main__':
    main()
