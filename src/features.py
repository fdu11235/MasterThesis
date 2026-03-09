"""Feature matrix construction for the CNN model."""

import logging
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def build_feature_matrix(diagnostics: dict) -> np.ndarray:
    """Stack diagnostic columns into a feature matrix.

    Parameters
    ----------
    diagnostics : dict
        Must contain:
        - 'params'    : ndarray of shape (L, 2) with columns [xi_hat, beta_hat]
        - 'score_gof' : ndarray of shape (L,)

    Returns
    -------
    F : ndarray of shape (L, 3)
        Columns are [xi_hat(k), beta_hat(k), S_gof(k)].
    """
    params = np.asarray(diagnostics["params"])        # (L, 2)
    score_gof = np.asarray(diagnostics["score_gof"])  # (L,)

    xi_hat = params[:, 0]
    beta_hat = params[:, 1]

    F = np.column_stack([xi_hat, beta_hat, score_gof])  # (L, 3)
    return F


def normalize_features(F: np.ndarray) -> np.ndarray:
    """Z-score normalize each column of the feature matrix.

    Parameters
    ----------
    F : ndarray of shape (L, 3)

    Returns
    -------
    F_norm : ndarray of shape (L, 3)
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
        - X has shape (N, 3, L)  — channels-first for Conv1d
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

            # Build and normalize feature matrix  (L, 3)
            F = build_feature_matrix(diag)
            F = normalize_features(F)
            feature_matrices.append(F)

            # Label: index into k_grid where k_star falls
            label = int(np.searchsorted(k_grid, k_star))
            label = min(label, len(k_grid) - 1)
            labels.append(label)

        # Stack into (N, L, 3) then transpose to (N, 3, L) for Conv1d
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
