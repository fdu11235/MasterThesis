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
