"""Tests for src.features module."""

import numpy as np
import torch
import pytest

from src.features import build_feature_matrix, normalize_features, build_dataset


class TestBuildFeatureMatrix:
    """Tests for build_feature_matrix."""

    def test_correct_shape(self):
        L = 50
        diagnostics = {
            "params": np.random.randn(L, 2),
            "score_gof": np.random.rand(L),
        }
        F = build_feature_matrix(diagnostics)
        assert F.shape == (L, 3)


class TestNormalizeFeatures:
    """Tests for normalize_features."""

    def test_normalized_columns(self):
        rng = np.random.RandomState(0)
        F = rng.randn(100, 3) * 5 + 3  # offset and scaled
        F_norm = normalize_features(F)
        for col in range(3):
            assert abs(F_norm[:, col].mean()) < 0.1, (
                f"Column {col} mean = {F_norm[:, col].mean()}"
            )
            assert abs(F_norm[:, col].std() - 1.0) < 0.1, (
                f"Column {col} std = {F_norm[:, col].std()}"
            )


class TestBuildDataset:
    """Tests for build_dataset."""

    def test_returns_correct_keys_and_shapes(self):
        L = 40
        k_grid = np.arange(10, 10 + L)
        n_samples = 5
        sample_size = 200

        all_diagnostics = []
        for _ in range(n_samples):
            dataset_dict = {"n": sample_size}
            diagnostics_dict = {
                "k_grid": k_grid,
                "params": np.random.randn(L, 2),
                "score_gof": np.random.rand(L),
                "k_star": int(k_grid[L // 2]),
            }
            all_diagnostics.append((dataset_dict, diagnostics_dict))

        datasets = build_dataset(all_diagnostics, config={})
        assert sample_size in datasets
        X, y = datasets[sample_size]
        assert X.shape == (n_samples, 3, L)
        assert y.shape == (n_samples,)
        assert X.dtype == torch.float32
        assert y.dtype == torch.long
