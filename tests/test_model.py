"""Tests for src.model module."""

import torch
import torch.nn as nn
import pytest

from src.model import ThresholdCNN, ResBlock1d


class TestResBlock1d:
    """Tests for residual blocks."""

    def test_same_channels_identity_skip(self):
        block = ResBlock1d(32, 32, kernel_size=5)
        x = torch.randn(4, 32, 50)
        out = block(x)
        assert out.shape == x.shape

    def test_different_channels_projection(self):
        block = ResBlock1d(16, 64, kernel_size=3)
        x = torch.randn(4, 16, 50)
        out = block(x)
        assert out.shape == (4, 64, 50)


class TestThresholdCNN:
    """Tests for ThresholdCNN."""

    def test_forward_output_shape_classification(self):
        batch, length, n_classes = 8, 40, 10
        model = ThresholdCNN(in_channels=4, channels=[16, 32],
                             n_classes=n_classes, pool_sizes=[1, 4])
        x = torch.randn(batch, 4, length)
        out = model(x)
        assert out.shape == (batch, n_classes)

    def test_forward_output_shape_regression(self):
        batch, length = 8, 100
        model = ThresholdCNN(in_channels=7, channels=[64, 128, 256, 256],
                             pool_sizes=[1, 4, 16], task="regression")
        x = torch.randn(batch, 7, length)
        out = model(x)
        assert out.shape == (batch,)
        assert (out >= 0).all() and (out <= 1).all()  # sigmoid output

    def test_variable_length_input(self):
        model = ThresholdCNN(in_channels=7, channels=[32, 64],
                             pool_sizes=[1, 4], task="regression")
        for length in [20, 50, 200, 721]:
            x = torch.randn(4, 7, length)
            out = model(x)
            assert out.shape == (4,)

    def test_overfit_tiny_dataset(self):
        """Model should be able to overfit a tiny dataset (loss decreases)."""
        batch = 16
        length = 40

        model = ThresholdCNN(
            in_channels=4, channels=[16, 32], kernel_size=3,
            dropout=0.0, pool_sizes=[1, 4], task="regression",
        )
        X = torch.randn(batch, 4, length)
        y = torch.rand(batch)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.SmoothL1Loss()

        model.train()
        initial_loss = None
        final_loss = None
        for step in range(100):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            if step == 99:
                final_loss = loss.item()

        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )


class TestTransferLearning:
    """Tests for load_pretrained_backbone."""

    def test_all_params_load_matching_architecture(self):
        """All params transfer when architectures match exactly."""
        model_a = ThresholdCNN(in_channels=7, channels=[64, 128],
                               pool_sizes=[1, 4], task="regression")
        model_b = ThresholdCNN(in_channels=7, channels=[64, 128],
                               pool_sizes=[1, 4], task="regression")

        n_loaded, n_skipped = model_b.load_pretrained_backbone(model_a.state_dict())
        assert n_loaded == len(list(model_a.state_dict().keys()))
        assert n_skipped == 0

    def test_graceful_skip_on_shape_mismatch(self):
        """Mismatched shapes are skipped gracefully."""
        model_a = ThresholdCNN(in_channels=7, channels=[64, 128],
                               pool_sizes=[1, 4], task="regression")
        model_b = ThresholdCNN(in_channels=4, channels=[16, 32],
                               pool_sizes=[1, 4], task="regression")

        n_loaded, n_skipped = model_b.load_pretrained_backbone(model_a.state_dict())
        assert n_skipped > 0

    def test_weights_actually_change(self):
        """After loading, target model weights match source."""
        model_a = ThresholdCNN(in_channels=7, channels=[64, 128],
                               pool_sizes=[1, 4], task="regression")
        model_b = ThresholdCNN(in_channels=7, channels=[64, 128],
                               pool_sizes=[1, 4], task="regression")

        model_b.load_pretrained_backbone(model_a.state_dict())

        for k in model_a.state_dict():
            assert torch.equal(model_b.state_dict()[k], model_a.state_dict()[k]), \
                f"Param {k} did not transfer correctly"
