import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AsymmetricSmoothL1Loss(nn.Module):
    """SmoothL1Loss that penalizes under-prediction more than over-prediction.

    When pred < target (k* too small → VaR too low → too many violations),
    the loss is scaled by *under_weight*. Over-predictions keep weight 1.0.
    """

    def __init__(self, under_weight=2.0):
        super().__init__()
        self.under_weight = under_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def forward(self, pred, target):
        loss = self.smooth_l1(pred, target)
        weight = torch.where(pred < target, self.under_weight, 1.0)
        return (loss * weight).mean()


def _differentiable_interpolate(curve, idx):
    """Linearly interpolate a 1D curve at continuous index positions.

    Parameters
    ----------
    curve : Tensor (batch, L)
    idx : Tensor (batch,) — continuous indices in [0, L-1]

    Returns
    -------
    Tensor (batch,) — interpolated values
    """
    L = curve.shape[1]
    idx = idx.clamp(0, L - 1)
    lo = idx.long()
    hi = (lo + 1).clamp(max=L - 1)
    frac = idx - lo.float()
    return curve[torch.arange(len(idx)), lo] * (1 - frac) + curve[torch.arange(len(idx)), hi] * frac


class VaRAwareLoss(nn.Module):
    """Combined loss: k* prediction + VaR quality + ES quality.

    L = alpha * L_k + beta * L_var + gamma * L_es

    L_k: AsymmetricSmoothL1 on normalised k*
    L_var: SmoothL1 on VaR ratio (target = 1.0)
    L_es: SmoothL1 on ES ratio (target = 1.0)
    """

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3, under_weight=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k_loss = AsymmetricSmoothL1Loss(under_weight=under_weight)

    def forward(self, k_pred, k_true, var_curve=None, es_curve=None):
        """Compute combined loss.

        Parameters
        ----------
        k_pred : Tensor (batch,) — predicted normalised k in [0, 1]
        k_true : Tensor (batch,) — target normalised k in [0, 1]
        var_curve : Tensor (batch, L) — normalised VaR ratios (1.0 = perfect)
        es_curve : Tensor (batch, L) — normalised ES ratios (1.0 = perfect)

        Returns
        -------
        total_loss : scalar Tensor
        components : dict with L_k, L_var, L_es as floats
        """
        L_k = self.k_loss(k_pred, k_true)

        components = {"L_k": L_k.item(), "L_var": 0.0, "L_es": 0.0}
        total = self.alpha * L_k

        if var_curve is not None and self.beta > 0:
            L = var_curve.shape[1]
            idx = k_pred * (L - 1)
            var_at_pred = _differentiable_interpolate(var_curve, idx)
            # Only penalise where curve is valid (non-zero = non-padded)
            mask = var_at_pred > 0
            if mask.any():
                L_var = nn.functional.smooth_l1_loss(
                    var_at_pred[mask], torch.ones_like(var_at_pred[mask]))
            else:
                L_var = torch.tensor(0.0, device=k_pred.device)
            components["L_var"] = L_var.item()
            total = total + self.beta * L_var

        if es_curve is not None and self.gamma > 0:
            L = es_curve.shape[1]
            idx = k_pred * (L - 1)
            es_at_pred = _differentiable_interpolate(es_curve, idx)
            mask = es_at_pred > 0
            if mask.any():
                L_es = nn.functional.smooth_l1_loss(
                    es_at_pred[mask], torch.ones_like(es_at_pred[mask]))
            else:
                L_es = torch.tensor(0.0, device=k_pred.device)
            components["L_es"] = L_es.item()
            total = total + self.gamma * L_es

        return total, components


def train_model(X, y, model, config, task="classification",
                var_curves=None, es_curves=None):
    """Train model with early stopping on validation loss.

    Args:
        X: Tensor of shape (N, C, L)
        y: Tensor of shape (N,) — class indices (classification) or float targets (regression)
        model: ThresholdCNN instance
        config: dict with keys 'lr', 'batch_size', 'max_epochs', 'patience'
        task: 'classification' or 'regression'
        var_curves: optional Tensor (N, L_max) — precomputed normalised VaR ratios
        es_curves: optional Tensor (N, L_max) — precomputed normalised ES ratios

    Returns:
        tuple of (trained model with best weights loaded, history dict)
    """
    test_frac = config.get('test_fraction', 0.2)
    n = len(X)
    perm = torch.randperm(n)
    val_size = int(n * test_frac)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Split VaR/ES curves if provided
    has_var_es = var_curves is not None and es_curves is not None
    if has_var_es:
        vc_train, vc_val = var_curves[train_idx], var_curves[val_idx]
        ec_train, ec_val = es_curves[train_idx], es_curves[val_idx]
    else:
        vc_train = vc_val = ec_train = ec_val = None

    logger.info("Train/val split: %d train, %d val (device=%s)", len(X_train), len(X_val), _device)

    model = model.to(_device)

    if has_var_es:
        train_loader = DataLoader(
            TensorDataset(X_train, y_train, vc_train, ec_train),
            batch_size=config.get('batch_size', 64),
            shuffle=True,
        )
    else:
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=config.get('batch_size', 64),
            shuffle=True,
        )

    # Transfer learning: discriminative LR and backbone freezing
    freeze_backbone_epochs = config.get('freeze_backbone_epochs', 0)
    backbone_lr_factor = config.get('backbone_lr_factor', 1.0)
    base_lr = config.get('lr', 1e-3)

    # Freeze backbone if requested
    if freeze_backbone_epochs > 0 and hasattr(model, 'conv'):
        for p in model.conv.parameters():
            p.requires_grad = False
        logger.info("Froze backbone for first %d epochs", freeze_backbone_epochs)

    # Discriminative LR: lower LR for backbone
    if backbone_lr_factor < 1.0 and hasattr(model, 'conv') and hasattr(model, 'head'):
        param_groups = [
            {'params': model.conv.parameters(), 'lr': base_lr * backbone_lr_factor},
            {'params': model.head.parameters(), 'lr': base_lr},
        ]
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
    )

    use_var_aware = False
    if task == "regression":
        loss_type = config.get("loss_type", "smooth_l1")
        if loss_type == "var_aware" and has_var_es:
            alpha = config.get("loss_alpha", 1.0)
            beta = config.get("loss_beta", 0.5)
            gamma = config.get("loss_gamma", 0.3)
            under_weight = config.get("asymmetric_weight", 2.0)
            criterion = VaRAwareLoss(alpha=alpha, beta=beta, gamma=gamma,
                                     under_weight=under_weight)
            use_var_aware = True
            logger.info("Using VaRAwareLoss (alpha=%.1f, beta=%.1f, gamma=%.1f, uw=%.1f)",
                        alpha, beta, gamma, under_weight)
        elif loss_type in ("asymmetric", "var_aware"):
            # var_aware without curves falls back to asymmetric (e.g., real data)
            under_weight = config.get("asymmetric_weight", 2.0)
            criterion = AsymmetricSmoothL1Loss(under_weight=under_weight)
            if loss_type == "var_aware":
                logger.info("VaR-aware requested but no VaR/ES curves — falling back to AsymmetricSmoothL1Loss")
            else:
                logger.info("Using AsymmetricSmoothL1Loss (under_weight=%.1f)", under_weight)
        else:
            criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 10)
    best_state = None
    history = {"train_loss": [], "val_loss": [], "lr": [],
               "train_L_k": [], "train_L_var": [], "train_L_es": [],
               "val_L_k": [], "val_L_var": [], "val_L_es": []}

    for epoch in range(config.get('max_epochs', 100)):
        # Unfreeze backbone after freeze period
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs and hasattr(model, 'conv'):
            for p in model.conv.parameters():
                p.requires_grad = True
            logger.info("Unfreezing backbone at epoch %d", epoch + 1)
            # Rebuild optimizer with all parameters
            if backbone_lr_factor < 1.0:
                param_groups = [
                    {'params': model.conv.parameters(), 'lr': base_lr * backbone_lr_factor},
                    {'params': model.head.parameters(), 'lr': base_lr},
                ]
                optimizer = torch.optim.Adam(param_groups)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
            )

        model.train()
        epoch_train_loss = 0.0
        epoch_comps = {"L_k": 0.0, "L_var": 0.0, "L_es": 0.0}
        n_batches = 0
        for batch in train_loader:
            if use_var_aware:
                xb, yb, vcb, ecb = [t.to(_device) for t in batch]
            else:
                xb, yb = batch[0].to(_device), batch[1].to(_device)
                vcb = ecb = None
            optimizer.zero_grad()
            pred = model(xb)
            if use_var_aware:
                loss, comps = criterion(pred, yb, vcb, ecb)
                for k in epoch_comps:
                    epoch_comps[k] += comps.get(k, 0.0)
            else:
                loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_train_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)
        for k in epoch_comps:
            history[f"train_{k}"].append(epoch_comps[k] / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            if has_var_es:
                val_loader = DataLoader(TensorDataset(X_val, y_val, vc_val, ec_val),
                                        batch_size=config.get('batch_size', 64),
                                        shuffle=False)
            else:
                val_loader = DataLoader(TensorDataset(X_val, y_val),
                                        batch_size=config.get('batch_size', 64),
                                        shuffle=False)
            val_loss_sum = 0.0
            val_comps = {"L_k": 0.0, "L_var": 0.0, "L_es": 0.0}
            val_n = 0
            for batch in val_loader:
                if use_var_aware:
                    xv, yv, vcv, ecv = [t.to(_device) for t in batch]
                    vl, vc = criterion(model(xv), yv, vcv, ecv)
                    for k in val_comps:
                        val_comps[k] += vc.get(k, 0.0) * len(xv)
                else:
                    xv, yv = batch[0].to(_device), batch[1].to(_device)
                    vl = criterion(model(xv), yv)
                val_loss_sum += vl.item() * len(xv)
                val_n += len(xv)
            val_loss = val_loss_sum / max(val_n, 1)
            for k in val_comps:
                history[f"val_{k}"].append(val_comps[k] / max(val_n, 1))

        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            logger.info(
                "Epoch %3d: val_loss=%.4f, best=%.4f, patience=%d/%d, lr=%.1e",
                epoch + 1, val_loss, best_val_loss, patience_counter, patience, cur_lr,
            )

    logger.info("Training complete — best val_loss=%.4f", best_val_loss)
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()
    return model, history


def predict(model, X, task="classification"):
    """Return predictions.

    Args:
        model: trained ThresholdCNN
        X: Tensor of shape (N, 3, L)
        task: 'classification' or 'regression'

    Returns:
        ndarray of shape (N,) — class indices (classification) or float values in [0,1] (regression)
    """
    model = model.to(_device)
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = X[i:i+64].to(_device)
            out = model(batch)
            results.append(out.cpu())
    out = torch.cat(results, dim=0)
    if task == "regression":
        return out.numpy()
    return out.argmax(dim=1).numpy()
