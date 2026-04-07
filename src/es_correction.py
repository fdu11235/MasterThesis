"""ES correction network: learns to correct GPD-based ES estimates.

The GPD closed-form ES formula ES = (VaR + beta - xi*u) / (1-xi)
overestimates ES when xi is moderate-to-high due to the 1/(1-xi)
amplification. This module trains a small MLP to predict a correction
factor from diagnostic features, using synthetic data where true ES
is known.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sp_stats
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import pot_quantile, pot_es, true_es

logger = logging.getLogger(__name__)


class ESCorrectionNet(nn.Module):
    """Small MLP that predicts ES correction factor from scalar features."""

    def __init__(self, in_features=9, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        # Output > 0.5: correction factor can reduce ES by at most half
        return self.net(x) + 0.5


def extract_features(ds, diag, k, p=0.99):
    """Extract 9 scalar features for the ES correction network.

    Parameters
    ----------
    ds : dict
        Dataset dict with 'samples'.
    diag : dict
        Diagnostics dict with 'k_grid', 'params', 'hill_series',
        'score_gof', 'mean_excess_values'.
    k : int
        Predicted k value.
    p : float
        Quantile probability.

    Returns
    -------
    ndarray of shape (9,) or None if extraction fails.
    """
    k_grid = np.asarray(diag['k_grid'])
    k_idx = min(np.searchsorted(k_grid, k), len(diag['params']) - 1)
    xi, beta = diag['params'][k_idx]

    if np.isnan(xi) or np.isnan(beta):
        return None

    samples = ds['samples']
    sorted_desc = np.sort(samples)[::-1]
    n = len(sorted_desc)

    var_est = pot_quantile(sorted_desc, k, xi, beta, n, p)
    median_samples = np.median(samples)

    hill_at_k = diag['hill_series'][k_idx] if k_idx < len(diag['hill_series']) else xi
    gof_at_k = diag['score_gof'][k_idx] if k_idx < len(diag['score_gof']) else 1.0
    me_at_k = diag['mean_excess_values'][k_idx] if k_idx < len(diag['mean_excess_values']) else 0.0

    kurtosis = float(sp_stats.kurtosis(samples))
    amplification = 1.0 / max(1.0 - xi, 0.05)

    features = np.array([
        xi,                                          # 0: tail index
        beta,                                        # 1: scale
        k / n,                                       # 2: exceedance fraction
        var_est / max(median_samples, 1e-10),        # 3: VaR extremeness
        hill_at_k,                                   # 4: Hill estimator
        gof_at_k,                                    # 5: GoF quality
        me_at_k,                                     # 6: mean excess
        kurtosis,                                    # 7: global tail heaviness
        amplification,                               # 8: 1/(1-xi)
    ], dtype=np.float64)

    # Replace NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0)
    return features


def build_correction_dataset(all_diagnostics, k_pred, config):
    """Build training dataset for the ES correction network.

    Parameters
    ----------
    all_diagnostics : list of (dataset_dict, diagnostics_dict)
    k_pred : ndarray of predicted k values
    config : dict

    Returns
    -------
    X : ndarray (N, 9) — features
    y : ndarray (N,) — correction factors (ES_true / ES_est)
    """
    p = config.get('evaluate', {}).get('quantile_p', 0.99)

    X_list = []
    y_list = []

    for i, (ds, diag) in enumerate(all_diagnostics):
        k = int(k_pred[i])
        feats = extract_features(ds, diag, k, p)
        if feats is None:
            continue

        k_grid = np.asarray(diag['k_grid'])
        k_idx = min(np.searchsorted(k_grid, k), len(diag['params']) - 1)
        xi, beta = diag['params'][k_idx]
        sorted_desc = np.sort(ds['samples'])[::-1]
        n = len(sorted_desc)

        es_est = pot_es(sorted_desc, k, xi, beta, n, p)
        try:
            es_true = true_es(ds['dist_type'], ds['params'], p)
        except (ValueError, KeyError):
            continue

        if es_true <= 0 or es_est <= 0 or np.isnan(es_est):
            continue

        correction = es_true / es_est
        # Clip extreme corrections to avoid training instability
        correction = np.clip(correction, 0.1, 5.0)

        X_list.append(feats)
        y_list.append(correction)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info("ES correction dataset: %d samples, correction range [%.3f, %.3f], "
                "mean=%.3f, median=%.3f",
                len(y), y.min(), y.max(), y.mean(), np.median(y))
    return X, y


def train_correction_net(X, y, config=None, val_frac=0.2, max_epochs=200, patience=20):
    """Train the ES correction network.

    Returns
    -------
    model : ESCorrectionNet (on CPU, best weights loaded)
    history : dict with train_loss, val_loss
    """
    n = len(X)
    perm = np.random.RandomState(42).permutation(n)
    val_size = int(n * val_frac)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    # Normalise features (z-score)
    X_mean = X[train_idx].mean(axis=0)
    X_std = X[train_idx].std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std

    X_train = torch.tensor(X_norm[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X_norm[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32)

    model = ESCorrectionNet(in_features=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6)

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    best_val = float('inf')
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        history["train_loss"].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze(-1)
            val_loss = criterion(val_pred, y_val).item()
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info("  Epoch %3d: train=%.4f, val=%.4f, best=%.4f",
                        epoch + 1, epoch_loss, val_loss, best_val)

    if best_state:
        model.load_state_dict(best_state)
    logger.info("  Training complete: best_val=%.4f, epochs=%d", best_val, epoch + 1)

    # Store normalisation params on the model for inference
    model.X_mean = X_mean
    model.X_std = X_std

    return model, history


def apply_correction(model, ds, diag, k, es_raw, p=0.99):
    """Apply the correction network to a single ES estimate.

    Returns corrected ES value.
    """
    feats = extract_features(ds, diag, k, p)
    if feats is None:
        return es_raw

    # Normalise using training stats
    feats_norm = (feats - model.X_mean) / model.X_std
    feats_t = torch.tensor(feats_norm, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        correction = model(feats_t).item()

    return es_raw * correction
