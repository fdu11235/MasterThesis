import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def train_model(X, y, model, config):
    """Train model with early stopping on validation loss.

    Args:
        X: Tensor of shape (N, 3, L)
        y: Tensor of shape (N,) class indices
        model: ThresholdCNN instance
        config: dict with keys 'lr', 'batch_size', 'max_epochs', 'patience'

    Returns:
        trained model (with best weights loaded)
    """
    test_frac = config.get('test_fraction', 0.2)
    n = len(X)
    perm = torch.randperm(n)
    val_size = int(n * test_frac)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    logger.info("Train/val split: %d train, %d val", len(X_train), len(X_val))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config.get('batch_size', 64),
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 10)
    best_state = None

    for epoch in range(config.get('max_epochs', 100)):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()

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
            logger.info(
                "Epoch %3d: val_loss=%.4f, best=%.4f, patience=%d/%d",
                epoch + 1, val_loss, best_val_loss, patience_counter, patience,
            )

    logger.info("Training complete — best val_loss=%.4f", best_val_loss)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, X):
    """Return predicted class indices.

    Args:
        model: trained ThresholdCNN
        X: Tensor of shape (N, 3, L)

    Returns:
        ndarray of shape (N,) with predicted class indices
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)
        return logits.argmax(dim=1).numpy()
