#!/usr/bin/env python
"""Pathway M pipeline — differentiable POT with end-to-end VaR/ES training.

Reuses cached Steps 1-4 from the existing pipeline (synthetic.pkl,
diagnostics.pkl), then trains a DifferentiablePOTModel with pinball +
FZ0 loss and compares against Pathway A.

Usage:
    python run_diff_pipeline.py --config config/default.yaml
    python run_diff_pipeline.py --config config/default.yaml --fresh
"""

import argparse
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

from src.features import build_feature_matrix, normalize_features
from src.evaluate import true_quantile, true_es, pot_quantile, pot_es
from src.diff_pot import DifferentiablePOTModel
from src.train_diff import train_diff_model, predict_diff


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_diff_dataset(all_diagnostics, config):
    """Build grouped dataset for Pathway M training.

    Groups samples by sample size *n*.  For each group, bundles:
    - features:    (N, 4, L_max) zero-padded diagnostic features
    - sorted_desc: (N, n) samples sorted descending
    - true_q:      {p: (N,) tensor} true quantiles at each training level
    - k_min, k_max: ints derived from pot config

    Args:
        all_diagnostics: list of (dataset_dict, diagnostics_dict).
        config: full YAML config dict.

    Returns:
        groups: dict[int, dict]  keyed by sample size.
        meta:   list of dict, one per sample (same order as all_diagnostics).
    """
    pot_cfg = config['pot']
    diff_cfg = config.get('diff_pot', {})
    quantiles = diff_cfg.get('quantiles', [0.95, 0.975, 0.99])

    # -- determine L_max across all samples ---
    L_max = 0
    for _, diag in all_diagnostics:
        L_max = max(L_max, len(diag['k_grid']))

    # -- first pass: build per-sample records --
    records = []
    for ds, diag in all_diagnostics:
        n = int(ds['n'])
        # features
        F = build_feature_matrix(diag)
        F = normalize_features(F)
        L = F.shape[0]
        if L < L_max:
            F = np.vstack([F, np.zeros((L_max - L, F.shape[1]), dtype=F.dtype)])

        # sorted descending samples
        sorted_desc = np.sort(ds['samples'])[::-1].copy()

        # true quantiles at all needed levels
        tq = {}
        for p in quantiles:
            tq[p] = true_quantile(ds['dist_type'], ds['params'], p)

        # true ES at primary level (for evaluation)
        p_primary = config['evaluate'].get('quantile_p', 0.99)
        es_true = true_es(ds['dist_type'], ds['params'], p_primary)

        k_grid = np.asarray(diag['k_grid'])
        k_min_val = int(k_grid[0])
        k_max_val = int(k_grid[-1])

        # normalized k* target for auxiliary supervision
        k_star = diag['k_star']
        if k_max_val > k_min_val:
            k_tilde_true = np.clip((k_star - k_min_val) / (k_max_val - k_min_val), 0.0, 1.0)
        else:
            k_tilde_true = 0.5

        records.append({
            'n': n,
            'features': F,        # (L_max, 4)
            'sorted_desc': sorted_desc,
            'true_q': tq,
            'true_es': es_true,
            'k_tilde_true': k_tilde_true,
            'k_min': k_min_val,
            'k_max': k_max_val,
            'dist_type': ds.get('dist_type', 'unknown'),
            'dist_params': ds.get('params', {}),
        })

    # -- group by n --
    by_n = defaultdict(list)
    for rec in records:
        by_n[rec['n']].append(rec)

    groups = {}
    for n_size, recs in by_n.items():
        N = len(recs)
        feat_np = np.stack([r['features'] for r in recs], axis=0)    # (N, L_max, 4)
        sd_np = np.stack([r['sorted_desc'] for r in recs], axis=0)   # (N, n_size)

        features = torch.tensor(feat_np, dtype=torch.float32).permute(0, 2, 1)  # (N, 4, L_max)
        sorted_desc = torch.tensor(sd_np, dtype=torch.float32)

        tq_tensors = {}
        for p in quantiles:
            tq_tensors[p] = torch.tensor([r['true_q'][p] for r in recs], dtype=torch.float32)

        groups[n_size] = {
            'features': features,
            'sorted_desc': sorted_desc,
            'true_q': tq_tensors,
            'k_min': recs[0]['k_min'],
            'k_max': recs[0]['k_max'],
            'true_es': torch.tensor([r['true_es'] for r in recs], dtype=torch.float32),
            'k_tilde_true': torch.tensor([r['k_tilde_true'] for r in recs], dtype=torch.float32),
            'dist_types': [r['dist_type'] for r in recs],
            'dist_params': [r['dist_params'] for r in recs],
        }

    return groups, records


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_diff(test_groups, model, config):
    """Evaluate Pathway M on test data.

    Returns:
        dict with 'var_rrmse', 'es_rrmse', per-distribution breakdowns,
        and raw arrays for plotting.
    """
    p = config['evaluate'].get('quantile_p', 0.99)
    diff_cfg = config.get('diff_pot', {})
    temp = diff_cfg.get('temperature_end', 20.0)
    temp_u = diff_cfg.get('temperature_u', 50.0)

    all_var_pred, all_var_true = [], []
    all_es_pred, all_es_true = [], []
    all_dist_types = []
    all_n_sizes = []

    model.eval()
    with torch.no_grad():
        for n_size, grp in sorted(test_groups.items()):
            var_hat, es_hat, info = model(
                grp['features'], grp['sorted_desc'],
                grp['k_min'], grp['k_max'], n_size, p,
                temperature=temp, temperature_u=temp_u,
            )
            var_pred = var_hat.numpy()
            es_pred = es_hat.numpy()
            var_true = grp['true_q'][p].numpy()
            es_true_arr = grp['true_es'].numpy()

            all_var_pred.append(var_pred)
            all_var_true.append(var_true)
            all_es_pred.append(es_pred)
            all_es_true.append(es_true_arr)
            all_dist_types.extend(grp['dist_types'])
            all_n_sizes.extend([n_size] * len(var_pred))

    var_pred = np.concatenate(all_var_pred)
    var_true = np.concatenate(all_var_true)
    es_pred = np.concatenate(all_es_pred)
    es_true = np.concatenate(all_es_true)
    dist_types = np.array(all_dist_types)
    n_sizes = np.array(all_n_sizes)

    # overall relative RMSE
    var_rel = (var_pred - var_true) / np.where(var_true != 0, var_true, 1.0)
    es_rel = (es_pred - es_true) / np.where(es_true != 0, es_true, 1.0)

    results = {
        'var_rrmse': np.sqrt(np.mean(var_rel ** 2)),
        'es_rrmse': np.sqrt(np.mean(es_rel ** 2)),
        'var_rrmse_by_dist': {},
        'es_rrmse_by_dist': {},
        'var_rrmse_by_n': {},
        'es_rrmse_by_n': {},
        '_var_pred': var_pred,
        '_var_true': var_true,
        '_es_pred': es_pred,
        '_es_true': es_true,
        '_dist_types': dist_types,
        '_n_sizes': n_sizes,
    }

    for dt in sorted(set(dist_types)):
        mask = dist_types == dt
        results['var_rrmse_by_dist'][dt] = np.sqrt(np.mean(var_rel[mask] ** 2))
        results['es_rrmse_by_dist'][dt] = np.sqrt(np.mean(es_rel[mask] ** 2))

    for ns in sorted(set(n_sizes)):
        mask = n_sizes == ns
        results['var_rrmse_by_n'][ns] = np.sqrt(np.mean(var_rel[mask] ** 2))
        results['es_rrmse_by_n'][ns] = np.sqrt(np.mean(es_rel[mask] ** 2))

    return results


# ---------------------------------------------------------------------------
# Pathway A comparison helper
# ---------------------------------------------------------------------------

def evaluate_pathway_a(test_groups, config):
    """Evaluate Pathway A (regression CNN → scipy GPD → VaR) on same test set.

    Loads the cached regression model, predicts k*, fits GPD per sample with
    scipy, computes VaR and ES.  Returns results dict compatible with
    evaluate_diff output (var_rrmse, es_rrmse, by_dist).

    Returns None if the Pathway A model checkpoint does not exist.
    """
    from src.model import ThresholdCNN
    from src.train import predict
    from src.pot import fit_gpd

    ckpt = "outputs/checkpoints/model_regression.pt"
    if not os.path.exists(ckpt):
        return None

    model_cfg = config['model']
    model_a = ThresholdCNN(
        in_channels=4, channels=model_cfg['channels'],
        kernel_size=model_cfg['kernel_size'],
        dropout=model_cfg['dropout'], task='regression',
    )
    model_a.load_state_dict(torch.load(ckpt, weights_only=True))
    model_a.eval()

    p = config['evaluate'].get('quantile_p', 0.99)

    all_var_pred, all_var_true = [], []
    all_es_pred, all_es_true = [], []
    all_dist_types = []

    for n_size, grp in sorted(test_groups.items()):
        features = grp['features']
        y_pred_norm = predict(model_a, features, task='regression')
        k_min = grp['k_min']
        k_max = grp['k_max']

        for i in range(features.shape[0]):
            k_hat = int(np.clip(round(k_min + y_pred_norm[i] * (k_max - k_min)),
                                k_min, k_max))
            sd_np = grp['sorted_desc'][i].numpy()
            xi, beta = fit_gpd(sd_np, k_hat)
            if np.isnan(xi) or np.isnan(beta):
                continue
            xi = np.clip(xi, -0.5, 0.95)

            var_est = pot_quantile(sd_np, k_hat, xi, beta, n_size, p)
            es_est = pot_es(sd_np, k_hat, xi, beta, n_size, p)

            all_var_pred.append(var_est)
            all_var_true.append(grp['true_q'][p][i].item())
            all_es_pred.append(es_est)
            all_es_true.append(grp['true_es'][i].item())
            all_dist_types.append(grp['dist_types'][i])

    if not all_var_pred:
        return None

    var_pred = np.array(all_var_pred)
    var_true = np.array(all_var_true)
    es_pred = np.array(all_es_pred)
    es_true = np.array(all_es_true)
    dist_types = np.array(all_dist_types)

    var_rel = (var_pred - var_true) / np.where(var_true != 0, var_true, 1.0)
    es_rel = (es_pred - es_true) / np.where(es_true != 0, es_true, 1.0)

    results = {
        'var_rrmse': np.sqrt(np.mean(var_rel ** 2)),
        'es_rrmse': np.sqrt(np.mean(es_rel ** 2)),
        'var_rrmse_by_dist': {},
        'es_rrmse_by_dist': {},
    }
    for dt in sorted(set(dist_types)):
        mask = dist_types == dt
        results['var_rrmse_by_dist'][dt] = np.sqrt(np.mean(var_rel[mask] ** 2))
        results['es_rrmse_by_dist'][dt] = np.sqrt(np.mean(es_rel[mask] ** 2))

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_diff_results(res_m, res_a, save_dir):
    """Side-by-side bar charts: Pathway M vs Pathway A."""
    os.makedirs(save_dir, exist_ok=True)

    dist_types = sorted(res_m['var_rrmse_by_dist'].keys())
    has_a = res_a is not None

    for metric, label in [('var_rrmse_by_dist', 'VaR'), ('es_rrmse_by_dist', 'ES')]:
        fig, ax = plt.subplots(figsize=(max(8, len(dist_types) * 1.5), 5))
        x = np.arange(len(dist_types))
        width = 0.35

        vals_m = [res_m[metric].get(d, 0) * 100 for d in dist_types]
        bars_m = ax.bar(x - width / 2 if has_a else x, vals_m, width if has_a else 0.6,
                        label='Pathway M (diff)', color='tab:blue')

        if has_a:
            vals_a = [res_a[metric].get(d, 0) * 100 for d in dist_types]
            bars_a = ax.bar(x + width / 2, vals_a, width,
                            label='Pathway A (baseline)', color='tab:orange')

        ax.set_ylabel('Relative RMSE (%)')
        ax.set_title(f'{label} Relative RMSE by Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(dist_types, rotation=25, ha='right', fontsize=9)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{label.lower()}_rrmse_by_dist.png'), dpi=150)
        plt.close(fig)

    # by sample size
    if 'var_rrmse_by_n' in res_m:
        n_sizes = sorted(res_m['var_rrmse_by_n'].keys())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (metric, label) in zip(axes, [('var_rrmse_by_n', 'VaR'), ('es_rrmse_by_n', 'ES')]):
            vals = [res_m[metric][ns] * 100 for ns in n_sizes]
            ax.bar([str(ns) for ns in n_sizes], vals, color='tab:blue')
            ax.set_ylabel('Relative RMSE (%)')
            ax.set_xlabel('Sample size n')
            ax.set_title(f'Pathway M: {label} RRMSE by n')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'rrmse_by_n.png'), dpi=150)
        plt.close(fig)

    logging.getLogger(__name__).info("Figures saved to %s", save_dir)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pathway M: differentiable POT pipeline for VaR/ES.")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML configuration file.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore cached diff model and retrain.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("diff_pipeline")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", args.config)

    os.makedirs("outputs/data", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/figures/diff", exist_ok=True)

    # ── Steps 1-4: load cached data ────────────────────────────────────
    synthetic_path = "outputs/data/synthetic.pkl"
    diagnostics_path = "outputs/data/diagnostics.pkl"

    if not os.path.exists(diagnostics_path):
        logger.error("Cached diagnostics not found at %s. "
                      "Run run_pipeline.py first to generate Steps 1-4.",
                      diagnostics_path)
        return

    with open(diagnostics_path, "rb") as f:
        all_diagnostics = pickle.load(f)
    logger.info("[Steps 1-4] Loaded %d cached diagnostics", len(all_diagnostics))

    # ── Step 5d: build diff dataset ────────────────────────────────────
    logger.info("[Step 5d] Building differentiable dataset …")
    groups, records = build_diff_dataset(all_diagnostics, config)
    for n_size, grp in sorted(groups.items()):
        logger.info("  n=%d: %d samples, features %s, sorted_desc %s",
                     n_size, grp['features'].shape[0],
                     tuple(grp['features'].shape), tuple(grp['sorted_desc'].shape))

    # ── Train/test split (deterministic) ───────────────────────────────
    test_frac = config['evaluate'].get('test_fraction', 0.2)
    train_groups = {}
    test_groups = {}

    torch.manual_seed(42)
    for n_size, grp in groups.items():
        N = grp['features'].shape[0]
        perm = torch.randperm(N)
        test_size = int(N * test_frac)
        test_idx = perm[:test_size]
        train_idx = perm[test_size:]

        def _split(grp, idx):
            quantiles = config.get('diff_pot', {}).get('quantiles', [0.95, 0.975, 0.99])
            return {
                'features': grp['features'][idx],
                'sorted_desc': grp['sorted_desc'][idx],
                'true_q': {p: grp['true_q'][p][idx] for p in quantiles},
                'true_es': grp['true_es'][idx],
                'k_tilde_true': grp['k_tilde_true'][idx],
                'k_min': grp['k_min'],
                'k_max': grp['k_max'],
                'dist_types': [grp['dist_types'][i] for i in idx.tolist()],
                'dist_params': [grp['dist_params'][i] for i in idx.tolist()],
            }

        train_groups[n_size] = _split(grp, train_idx)
        test_groups[n_size] = _split(grp, test_idx)

    n_train = sum(g['features'].shape[0] for g in train_groups.values())
    n_test = sum(g['features'].shape[0] for g in test_groups.values())
    logger.info("  Train: %d, Test: %d", n_train, n_test)

    # ── Step 6d: train model ───────────────────────────────────────────
    diff_cfg = config.get('diff_pot', {})
    model_cfg = config.get('model', {})
    ckpt_path = "outputs/checkpoints/model_diff.pt"

    model = DifferentiablePOTModel(
        in_channels=4,
        channels=model_cfg.get('channels', [16, 32]),
        kernel_size=model_cfg.get('kernel_size', 5),
        dropout=model_cfg.get('dropout', 0.2),
    )

    # optional warm start
    warm_start = diff_cfg.get('warm_start_from')
    if warm_start and os.path.exists(warm_start):
        sd = torch.load(warm_start, weights_only=True)
        n_loaded = model.load_backbone_from(sd)
        logger.info("Warm-started %d parameters from %s", n_loaded, warm_start)

    if not args.fresh and os.path.exists(ckpt_path):
        logger.info("[Step 6d] Loading cached Pathway M model from %s", ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()
    else:
        logger.info("[Step 6d] Training Pathway M model …")
        model = train_diff_model(model, train_groups, test_groups, config)
        torch.save(model.state_dict(), ckpt_path)
        logger.info("  → checkpoint saved to %s", ckpt_path)

    # ── Step 7d: evaluate and compare ──────────────────────────────────
    logger.info("[Step 7d] Evaluating Pathway M …")
    res_m = evaluate_diff(test_groups, model, config)

    logger.info("  Pathway M — VaR Relative RMSE: %.2f%%", res_m['var_rrmse'] * 100)
    logger.info("  Pathway M — ES  Relative RMSE: %.2f%%", res_m['es_rrmse'] * 100)
    for dt in sorted(res_m['var_rrmse_by_dist']):
        logger.info("    %s: VaR=%.2f%%  ES=%.2f%%",
                     dt,
                     res_m['var_rrmse_by_dist'][dt] * 100,
                     res_m['es_rrmse_by_dist'][dt] * 100)
    for ns in sorted(res_m.get('var_rrmse_by_n', {})):
        logger.info("    n=%d: VaR=%.2f%%  ES=%.2f%%",
                     ns,
                     res_m['var_rrmse_by_n'][ns] * 100,
                     res_m['es_rrmse_by_n'][ns] * 100)

    # Pathway A comparison
    logger.info("Evaluating Pathway A for comparison …")
    res_a = evaluate_pathway_a(test_groups, config)
    if res_a is not None:
        logger.info("  Pathway A — VaR Relative RMSE: %.2f%%", res_a['var_rrmse'] * 100)
        logger.info("  Pathway A — ES  Relative RMSE: %.2f%%", res_a['es_rrmse'] * 100)
        for dt in sorted(res_a['var_rrmse_by_dist']):
            logger.info("    %s: VaR=%.2f%%  ES=%.2f%%",
                         dt,
                         res_a['var_rrmse_by_dist'][dt] * 100,
                         res_a['es_rrmse_by_dist'][dt] * 100)
    else:
        logger.warning("  Pathway A model not found — skipping comparison")

    plot_diff_results(res_m, res_a, "outputs/figures/diff")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger("diff_pipeline").warning(
            "Interrupted by user. Re-run to resume from last checkpoint."
        )
