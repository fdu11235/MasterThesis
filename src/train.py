import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(X, y, model, config, task="classification"):
    """Train model with early stopping on validation loss.

    Args:
        X: Tensor of shape (N, 3, L)
        y: Tensor of shape (N,) — class indices (classification) or float targets (regression)
        model: ThresholdCNN instance
        config: dict with keys 'lr', 'batch_size', 'max_epochs', 'patience'
        task: 'classification' or 'regression'

    Returns:
        tuple of (trained model with best weights loaded, history dict)
    """
    model = model.to(device)
    logger.info("Training on device: %s", device)

    test_frac = config.get('test_fraction', 0.2)
    n = len(X)
    perm = torch.randperm(n)
    val_size = int(n * test_frac)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
    X_val, y_val = X[val_idx].to(device), y[val_idx].to(device)
    logger.info("Train/val split: %d train, %d val", len(X_train), len(X_val))

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

    if task == "regression":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 10)
    best_state = None
    history = {"train_loss": [], "val_loss": [], "lr": []}

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
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_train_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()

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
    model = model.to(device)
    X = X.to(device)
    model.eval()
    with torch.no_grad():
        out = model(X)
        if task == "regression":
            return out.cpu().numpy()
        return out.cpu().argmax(dim=1).numpy()
