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


class TestTransferLearning:
    """Tests for load_pretrained_backbone."""

    def test_all_params_load_matching_architecture(self):
        """All params transfer when architectures match exactly."""
        model_a = ThresholdCNN(in_channels=7, channels=[32, 64], task="regression")
        model_b = ThresholdCNN(in_channels=7, channels=[32, 64], task="regression")

        n_loaded, n_skipped = model_b.load_pretrained_backbone(model_a.state_dict())
        assert n_loaded == len(list(model_a.state_dict().keys()))
        assert n_skipped == 0

    def test_graceful_skip_on_shape_mismatch(self):
        """Mismatched shapes are skipped gracefully."""
        model_a = ThresholdCNN(in_channels=7, channels=[32, 64], task="regression")
        model_b = ThresholdCNN(in_channels=4, channels=[16, 32], task="regression")

        n_loaded, n_skipped = model_b.load_pretrained_backbone(model_a.state_dict())
        assert n_skipped > 0

    def test_weights_actually_change(self):
        """After loading, target model weights match source."""
        model_a = ThresholdCNN(in_channels=7, channels=[32, 64], task="regression")
        model_b = ThresholdCNN(in_channels=7, channels=[32, 64], task="regression")

        # Capture weights before
        before = {k: v.clone() for k, v in model_b.state_dict().items()}

        model_b.load_pretrained_backbone(model_a.state_dict())

        # After loading, weights should match model_a
        for k in model_a.state_dict():
            assert torch.equal(model_b.state_dict()[k], model_a.state_dict()[k]), \
                f"Param {k} did not transfer correctly"
