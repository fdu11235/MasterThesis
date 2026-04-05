"""Tests for src.perturbation module."""

import numpy as np
import pytest

from src.perturbation import perturb_random_deletion, perturb_bootstrap


def _make_dataset(n=1000, seed=42):
    """Create a simple test dataset dict."""
    rng = np.random.RandomState(seed)
    return {
        "samples": rng.exponential(size=n),
        "n": n,
        "dist_type": "test",
        "params": {},
    }


class TestRandomDeletion:
    def test_reduces_sample_count(self):
        ds = _make_dataset(1000)
        result = perturb_random_deletion(ds, fraction=0.2, seed=0)
        assert result["n"] == 800
        assert len(result["samples"]) == 800

    def test_fraction_zero_returns_all(self):
        ds = _make_dataset(100)
        result = perturb_random_deletion(ds, fraction=0.0, seed=0)
        assert result["n"] == 100
        assert len(result["samples"]) == 100

    def test_preserves_positive_values(self):
        ds = _make_dataset(500)
        result = perturb_random_deletion(ds, fraction=0.1, seed=0)
        assert np.all(result["samples"] > 0)

    def test_reproducibility(self):
        ds = _make_dataset(500)
        a = perturb_random_deletion(ds, fraction=0.15, seed=7)
        b = perturb_random_deletion(ds, fraction=0.15, seed=7)
        np.testing.assert_array_equal(a["samples"], b["samples"])

    def test_different_seeds_differ(self):
        ds = _make_dataset(500)
        a = perturb_random_deletion(ds, fraction=0.15, seed=0)
        b = perturb_random_deletion(ds, fraction=0.15, seed=1)
        assert not np.array_equal(a["samples"], b["samples"])

    def test_metadata_stored(self):
        ds = _make_dataset(100)
        result = perturb_random_deletion(ds, fraction=0.1, seed=0)
        assert result["perturbation"] == "deletion"
        assert result["deletion_fraction"] == 0.1

    def test_does_not_mutate_original(self):
        ds = _make_dataset(100)
        original_samples = ds["samples"].copy()
        perturb_random_deletion(ds, fraction=0.2, seed=0)
        np.testing.assert_array_equal(ds["samples"], original_samples)

    def test_invalid_fraction_raises(self):
        ds = _make_dataset(100)
        with pytest.raises(ValueError):
            perturb_random_deletion(ds, fraction=1.0, seed=0)
        with pytest.raises(ValueError):
            perturb_random_deletion(ds, fraction=-0.1, seed=0)


class TestBootstrap:
    def test_same_length(self):
        ds = _make_dataset(500)
        result = perturb_bootstrap(ds, seed=0)
        assert len(result["samples"]) == 500
        assert result["n"] == 500

    def test_produces_duplicates(self):
        ds = _make_dataset(500)
        result = perturb_bootstrap(ds, seed=0)
        unique_count = len(np.unique(result["samples"]))
        assert unique_count < 500  # bootstrap should produce some duplicates

    def test_reproducibility(self):
        ds = _make_dataset(500)
        a = perturb_bootstrap(ds, seed=7)
        b = perturb_bootstrap(ds, seed=7)
        np.testing.assert_array_equal(a["samples"], b["samples"])

    def test_different_seeds_differ(self):
        ds = _make_dataset(500)
        a = perturb_bootstrap(ds, seed=0)
        b = perturb_bootstrap(ds, seed=1)
        assert not np.array_equal(a["samples"], b["samples"])

    def test_metadata_stored(self):
        ds = _make_dataset(100)
        result = perturb_bootstrap(ds, seed=0)
        assert result["perturbation"] == "bootstrap"

    def test_preserves_positive_values(self):
        ds = _make_dataset(500)
        result = perturb_bootstrap(ds, seed=0)
        assert np.all(result["samples"] > 0)

    def test_does_not_mutate_original(self):
        ds = _make_dataset(100)
        original_samples = ds["samples"].copy()
        perturb_bootstrap(ds, seed=0)
        np.testing.assert_array_equal(ds["samples"], original_samples)


class TestPOTCompatibility:
    """Test that perturbed datasets work with POT diagnostics."""

    def test_deletion_compatible_with_pot(self):
        from src.pot import process_one_dataset
        ds = _make_dataset(1000)
        perturbed = perturb_random_deletion(ds, fraction=0.1, seed=0)
        pot_cfg = {"k_min": 30, "k_max_frac": 0.15, "delta": 5,
                   "weights": [1.0, 1.0, 1.0], "decluster": False}
        _ds_out, diag = process_one_dataset(perturbed, pot_cfg)
        assert "k_star" in diag
        assert diag["k_star"] >= 30

    def test_bootstrap_compatible_with_pot(self):
        from src.pot import process_one_dataset
        ds = _make_dataset(1000)
        perturbed = perturb_bootstrap(ds, seed=0)
        pot_cfg = {"k_min": 30, "k_max_frac": 0.15, "delta": 5,
                   "weights": [1.0, 1.0, 1.0], "decluster": False}
        _ds_out, diag = process_one_dataset(perturbed, pot_cfg)
        assert "k_star" in diag
        assert diag["k_star"] >= 30
