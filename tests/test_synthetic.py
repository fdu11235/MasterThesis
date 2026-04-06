"""Tests for src.synthetic module."""

import numpy as np
import pytest

from src.synthetic import generate_dataset, generate_all


class TestGenerateDataset:
    """Tests for generate_dataset."""

    def test_returns_correct_keys_and_shapes(self):
        result = generate_dataset("pareto", {"alpha": 2.0}, n=200, seed=42)
        assert set(result.keys()) == {"samples", "params", "dist_type", "n"}
        assert result["samples"].shape == (200,)
        assert result["dist_type"] == "pareto"
        assert result["n"] == 200
        assert result["params"] == {"alpha": 2.0}

    def test_reproducibility_with_same_seed(self):
        a = generate_dataset("pareto", {"alpha": 2.0}, n=200, seed=7)
        b = generate_dataset("pareto", {"alpha": 2.0}, n=200, seed=7)
        np.testing.assert_array_equal(a["samples"], b["samples"])

    def test_pareto_values_positive(self):
        result = generate_dataset("pareto", {"alpha": 1.5}, n=200, seed=0)
        assert np.all(result["samples"] > 0)

    def test_student_t_absolute_values_positive(self):
        result = generate_dataset("student_t", {"df": 3.0}, n=200, seed=0)
        assert np.all(result["samples"] >= 0)
        # At least some should be strictly positive
        assert np.any(result["samples"] > 0)


class TestNewDistributions:
    """Tests for newly added distribution generators."""

    @pytest.mark.parametrize("dist_type,params", [
        ("burr12", {"c": 2, "d": 1}),
        ("frechet", {"c": 3.0}),
        ("dagum", {"c": 2, "d": 1}),
        ("inverse_gamma", {"a": 3}),
        ("lognormal", {"sigma": 1.0}),
        ("weibull_stretched", {"c": 0.6}),
        ("log_gamma", {"b": 2.0, "p": 2.0}),
        ("gamma_pareto_splice", {"gamma_shape": 2, "pareto_alpha": 1.5, "splice_quantile": 0.9}),
    ])
    def test_values_positive(self, dist_type, params):
        result = generate_dataset(dist_type, params, n=500, seed=42)
        assert result["samples"].shape == (500,)
        assert np.all(result["samples"] > 0)

    @pytest.mark.parametrize("dist_type,params", [
        ("burr12", {"c": 5, "d": 2}),
        ("frechet", {"c": 2.0}),
        ("dagum", {"c": 5, "d": 2}),
        ("inverse_gamma", {"a": 5}),
        ("lognormal", {"sigma": 0.5}),
        ("weibull_stretched", {"c": 0.4}),
        ("log_gamma", {"b": 1.5, "p": 2.0}),
        ("gamma_pareto_splice", {"gamma_shape": 3, "pareto_alpha": 2.0, "splice_quantile": 0.9}),
    ])
    def test_reproducibility(self, dist_type, params):
        a = generate_dataset(dist_type, params, n=200, seed=7)
        b = generate_dataset(dist_type, params, n=200, seed=7)
        np.testing.assert_array_equal(a["samples"], b["samples"])


class TestGarchWrapped:
    """Tests for GARCH-wrapped distribution generators."""

    @pytest.mark.parametrize("dist_type,params", [
        ("garch_student_t", {"df": 3, "garch_alpha": 0.1, "garch_beta": 0.85, "garch_omega": 0.01}),
        ("garch_pareto", {"alpha": 2.0, "garch_alpha": 0.1, "garch_beta": 0.85, "garch_omega": 0.01}),
    ])
    def test_output_shape_and_positive(self, dist_type, params):
        result = generate_dataset(dist_type, params, n=500, seed=42)
        assert result["samples"].shape == (500,)
        assert np.all(result["samples"] >= 0), "GARCH abs residuals must be non-negative"

    @pytest.mark.parametrize("dist_type,params", [
        ("garch_student_t", {"df": 4, "garch_alpha": 0.1, "garch_beta": 0.85, "garch_omega": 0.01}),
        ("garch_pareto", {"alpha": 1.5, "garch_alpha": 0.1, "garch_beta": 0.85, "garch_omega": 0.01}),
    ])
    def test_garch_filtering_produces_different_data(self, dist_type, params):
        """GARCH-filtered data should differ from plain i.i.d. data."""
        garch_result = generate_dataset(dist_type, params, n=1000, seed=42)
        # Compare with the base distribution (same seed won't match due to GARCH)
        base_type = dist_type.replace("garch_", "")
        base_params = {k: v for k, v in params.items()
                       if k not in ("garch_alpha", "garch_beta", "garch_omega")}
        base_result = generate_dataset(base_type, base_params, n=1000, seed=42)
        # Shapes match but values should differ
        assert garch_result["samples"].shape == base_result["samples"].shape
        assert not np.allclose(garch_result["samples"], base_result["samples"])


class TestGenerateAll:
    """Tests for generate_all."""

    def test_returns_correct_number_of_datasets(self):
        config = {
            "sample_sizes": [200],
            "n_replications": 1,
            "seed": 0,
            "distributions": {
                "pareto": {"alpha": [1.5]},
                "student_t": {"df": [3.0]},
            },
        }
        datasets = generate_all(config)
        # 2 distributions x 1 param combo each x 1 sample size x 1 replication = 2
        assert len(datasets) == 2
        for ds in datasets:
            assert set(ds.keys()) == {"samples", "params", "dist_type", "n"}

    def test_all_distributions_count(self):
        """All 12 distribution families with minimal params produce correct count."""
        config = {
            "sample_sizes": [100],
            "n_replications": 1,
            "seed": 0,
            "distributions": {
                "student_t": {"df": [3]},
                "pareto": {"alpha": [2.0]},
                "lognormal_pareto_mix": {
                    "lognormal_mu": 0.0, "lognormal_sigma": 1.0,
                    "pareto_alpha": [1.5], "mix_frac": 0.1,
                },
                "two_pareto": {
                    "alpha1": [2.0], "alpha2": [1.0], "changepoint_frac": 0.05,
                },
                "burr12": {"c": [2], "d": [1]},
                "frechet": {"c": [3.0]},
                "dagum": {"c": [2], "d": [1]},
                "inverse_gamma": {"a": [3]},
                "lognormal": {"sigma": [1.0]},
                "weibull_stretched": {"c": [0.6]},
                "log_gamma": {"b": [2.0], "p": 2.0},
                "gamma_pareto_splice": {
                    "gamma_shape": [2], "pareto_alpha": [1.5],
                    "splice_quantile": 0.9,
                },
            },
        }
        datasets = generate_all(config)
        # 12 distributions x 1 combo each x 1 size x 1 rep = 12
        assert len(datasets) == 12
