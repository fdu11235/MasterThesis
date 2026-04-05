"""Feature matrix construction for the CNN model."""

import logging
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def build_feature_matrix(diagnostics: dict, columns=None) -> np.ndarray:
    """Stack diagnostic columns into a feature matrix.

    Parameters
    ----------
    diagnostics : dict
        Must contain:
        - 'params'              : ndarray of shape (L, 2) with columns [xi_hat, beta_hat]
        - 'score_gof'           : ndarray of shape (L,)
        - 'score_mean_excess'   : ndarray of shape (L,)
        - 'hill_series'         : ndarray of shape (L,)
        - 'qq_residual_series'  : ndarray of shape (L,)
        - 'mean_excess_values'  : ndarray of shape (L,)
    columns : list of int or None
        Column indices to select from the full 7-column matrix.
        If None, all 7 columns are returned.

    Returns
    -------
    F : ndarray of shape (L, C)
        Columns are a subset of [xi_hat, beta_hat, S_gof, S_me, hill, qq_resid, me_values].
    """
    params = np.asarray(diagnostics["params"])                    # (L, 2)
    score_gof = np.asarray(diagnostics["score_gof"])              # (L,)
    score_me = np.asarray(diagnostics["score_mean_excess"])       # (L,)
    hill = np.asarray(diagnostics["hill_series"])                 # (L,)
    qq_resid = np.asarray(diagnostics["qq_residual_series"])      # (L,)
    me_vals = np.asarray(diagnostics["mean_excess_values"])       # (L,)

    xi_hat = params[:, 0]
    beta_hat = params[:, 1]

    # Replace NaN values with 0 for CNN input stability
    hill = np.nan_to_num(hill, nan=0.0)
    qq_resid = np.nan_to_num(qq_resid, nan=0.0)
    me_vals = np.nan_to_num(me_vals, nan=0.0)

    F = np.column_stack([xi_hat, beta_hat, score_gof, score_me,
                         hill, qq_resid, me_vals])                # (L, 7)

    if columns is not None:
        F = F[:, columns]

    return F


def normalize_features(F: np.ndarray) -> np.ndarray:
    """Z-score normalize each column of the feature matrix.

    Parameters
    ----------
    F : ndarray of shape (L, C)

    Returns
    -------
    F_norm : ndarray of shape (L, C)
        Each column transformed as (col - mean) / (std + 1e-10).
    """
    mean = F.mean(axis=0)
    std = F.std(axis=0)
    F_norm = (F - mean) / (std + 1e-10)
    return F_norm


def build_dataset(
    all_diagnostics: list,
    config: dict,
) -> dict:
    """Build per-sample-size datasets of feature tensors and labels.

    Parameters
    ----------
    all_diagnostics : list of (dataset_dict, diagnostics_dict)
        - dataset_dict    : contains key 'n' (sample size)
        - diagnostics_dict: contains 'k_grid', 'params', 'score_gof', 'k_star'
    config : dict
        Reserved for future use (e.g. device selection).

    Returns
    -------
    datasets : dict[int, tuple[Tensor, Tensor]]
        Keyed by sample size *n*.  Each value is (X, y) where
        - X has shape (N, C, L)  — channels-first for Conv1d (C=7)
        - y has shape (N,)       — class indices into k_grid
    """
    # Group entries by sample size
    groups: dict[int, list] = defaultdict(list)
    for dataset_dict, diagnostics_dict in all_diagnostics:
        n = int(dataset_dict["n"])
        groups[n].append(diagnostics_dict)

    datasets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for n, diag_list in groups.items():
        feature_matrices = []
        labels = []

        for diag in diag_list:
            k_grid = np.asarray(diag["k_grid"])
            k_star = diag["k_star"]

            # Build and normalize feature matrix  (L, C)
            columns = config.get("features", {}).get("columns")
            F = build_feature_matrix(diag, columns=columns)
            F = normalize_features(F)
            feature_matrices.append(F)

            # Label: index into k_grid where k_star falls
            label = int(np.searchsorted(k_grid, k_star))
            label = min(label, len(k_grid) - 1)
            labels.append(label)

        # Stack into (N, L, C) then transpose to (N, C, L) for Conv1d
        X_np = np.stack(feature_matrices, axis=0)   # (N, L, 3)
        X = torch.tensor(X_np, dtype=torch.float32)  # (N, L, 3)
        X = X.permute(0, 2, 1)                       # (N, 3, L)

        y = torch.tensor(labels, dtype=torch.long)    # (N,)

        datasets[n] = (X, y)
        logger.info(
            "Sample size n=%d: %d samples, k_grid length=%d, X shape=%s",
            n, len(labels), X_np.shape[1], tuple(X.shape),
        )
        label_arr = np.array(labels)
        logger.debug(
            "  Label distribution: min=%d, max=%d, mean=%.1f, std=%.1f",
            label_arr.min(), label_arr.max(), label_arr.mean(), label_arr.std(),
        )

    return datasets


def build_dataset_regression(
    all_diagnostics: list,
    config: dict,
) -> tuple:
    """Build a unified dataset with normalized [0, 1] regression labels.

    Parameters
    ----------
    all_diagnostics : list of (dataset_dict, diagnostics_dict)
    config : dict

    Returns
    -------
    X : Tensor of shape (N, C, L_max) — zero-padded feature matrices (C=7 channels)
    y : Tensor of shape (N,) float32 — normalized k* in [0, 1]
    meta : list of dict — per-sample metadata with k_min, k_max, n
    """
    # First pass: determine L_max
    L_max = 0
    for _, diag in all_diagnostics:
        k_grid = np.asarray(diag["k_grid"])
        L_max = max(L_max, len(k_grid))

    feature_matrices = []
    labels = []
    meta = []

    for ds, diag in all_diagnostics:
        n = int(ds["n"])
        k_grid = np.asarray(diag["k_grid"])
        k_star = diag["k_star"]
        k_min = int(k_grid[0])
        k_max = int(k_grid[-1])

        # Build and normalize feature matrix (L, C)
        columns = config.get("features", {}).get("columns")
        F = build_feature_matrix(diag, columns=columns)
        F = normalize_features(F)

        # Pad to L_max with zeros
        L = F.shape[0]
        if L < L_max:
            pad = np.zeros((L_max - L, F.shape[1]), dtype=F.dtype)
            F = np.vstack([F, pad])

        feature_matrices.append(F)

        # Normalized label in [0, 1]
        if k_max > k_min:
            y_val = (k_star - k_min) / (k_max - k_min)
        else:
            y_val = 0.5
        y_val = np.clip(y_val, 0.0, 1.0)
        labels.append(y_val)

        meta.append({
            "k_min": k_min, "k_max": k_max, "n": n,
            "window_idx": ds.get("window_idx"),
            "end_date": ds.get("end_date"),
            "dist_type": ds.get("dist_type", "unknown"),
        })

    # Stack into (N, L_max, C) then transpose to (N, C, L_max)
    X_np = np.stack(feature_matrices, axis=0)
    X = torch.tensor(X_np, dtype=torch.float32).permute(0, 2, 1)
    y = torch.tensor(labels, dtype=torch.float32)

    logger.info(
        "Regression dataset: %d samples, L_max=%d, X shape=%s, y range=[%.3f, %.3f]",
        len(labels), L_max, tuple(X.shape), y.min().item(), y.max().item(),
    )

    return X, y, meta


def build_var_es_curves(
    all_diagnostics: list,
    config: dict,
    L_max: int,
) -> tuple:
    """Precompute normalised VaR and ES curves for VaR-aware training.

    For each dataset, computes VaR(k) and ES(k) at every k in k_grid,
    normalises by the true quantile/ES (ratio: 1.0 = perfect), clips
    extremes, and zero-pads to L_max.

    Parameters
    ----------
    all_diagnostics : list of (dataset_dict, diagnostics_dict)
    config : dict with 'evaluate.quantile_p'
    L_max : int, maximum k_grid length (for padding)

    Returns
    -------
    var_curves : Tensor (N, L_max) — normalised VaR ratios, 0 in padded region
    es_curves : Tensor (N, L_max) — normalised ES ratios, 0 in padded region
    """
    from src.evaluate import pot_quantile, pot_es, true_quantile, true_es

    p = config.get("evaluate", {}).get("quantile_p", 0.99)
    CLIP_LO, CLIP_HI = 0.1, 10.0

    var_curves = []
    es_curves = []

    for ds, diag in all_diagnostics:
        k_grid = np.asarray(diag["k_grid"])
        params = np.asarray(diag["params"])
        sorted_desc = np.sort(ds["samples"])[::-1]
        n = len(sorted_desc)
        L = len(k_grid)

        # Compute true quantile/ES
        dist_type = ds.get("dist_type", "unknown")
        dist_params = ds.get("params", {})
        try:
            var_true = true_quantile(dist_type, dist_params, p)
            es_true = true_es(dist_type, dist_params, p)
        except (ValueError, KeyError):
            var_true, es_true = 1.0, 1.0  # fallback for real data

        if var_true <= 0 or np.isnan(var_true):
            var_true = 1.0
        if es_true <= 0 or np.isnan(es_true):
            es_true = 1.0

        # VaR/ES at every k
        var_k = np.zeros(L, dtype=np.float64)
        es_k = np.zeros(L, dtype=np.float64)
        for i, k in enumerate(k_grid):
            xi, beta = params[i]
            if np.isnan(xi) or np.isnan(beta):
                var_k[i] = 1.0  # neutral
                es_k[i] = 1.0
            else:
                var_k[i] = pot_quantile(sorted_desc, int(k), xi, beta, n, p)
                es_k[i] = pot_es(sorted_desc, int(k), xi, beta, n, p)

        # Normalise to ratio (1.0 = perfect)
        var_ratio = np.clip(var_k / var_true, CLIP_LO, CLIP_HI)
        es_ratio = np.clip(es_k / es_true, CLIP_LO, CLIP_HI)

        # Replace any remaining NaN/Inf
        var_ratio = np.nan_to_num(var_ratio, nan=1.0, posinf=CLIP_HI, neginf=CLIP_LO)
        es_ratio = np.nan_to_num(es_ratio, nan=1.0, posinf=CLIP_HI, neginf=CLIP_LO)

        # Pad to L_max
        if L < L_max:
            var_ratio = np.concatenate([var_ratio, np.zeros(L_max - L)])
            es_ratio = np.concatenate([es_ratio, np.zeros(L_max - L)])

        var_curves.append(var_ratio)
        es_curves.append(es_ratio)

    var_t = torch.tensor(np.stack(var_curves), dtype=torch.float32)
    es_t = torch.tensor(np.stack(es_curves), dtype=torch.float32)

    logger.info("VaR/ES curves: %d samples, L_max=%d, var range=[%.2f, %.2f], es range=[%.2f, %.2f]",
                len(var_curves), L_max,
                var_t[var_t > 0].min().item(), var_t.max().item(),
                es_t[es_t > 0].min().item(), es_t.max().item())

    return var_t, es_t
