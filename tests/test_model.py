"""Tests for src.model module."""

import torch
import torch.nn as nn
import pytest

from src.model import ThresholdCNN


class TestThresholdCNN:
    """Tests for ThresholdCNN."""

    def test_forward_output_shape(self):
        batch, length, n_classes = 8, 40, 10
        model = ThresholdCNN(in_channels=4, n_classes=n_classes)
        x = torch.randn(batch, 4, length)
        out = model(x)
        assert out.shape == (batch, n_classes)

    def test_overfit_tiny_dataset(self):
        """Model should be able to overfit a tiny dataset (loss decreases)."""
        n_classes = 4
        batch = 16
        length = 20

        model = ThresholdCNN(
            in_channels=4, channels=[8, 16], kernel_size=3,
            dropout=0.0, n_classes=n_classes,
        )
        X = torch.randn(batch, 4, length)
        y = torch.randint(0, n_classes, (batch,))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        model.train()
        initial_loss = None
        final_loss = None
        for step in range(50):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            if step == 49:
                final_loss = loss.item()

        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )
