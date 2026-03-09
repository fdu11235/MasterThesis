"""Tests for src.pot module."""

import numpy as np
import pytest
from scipy.stats import pareto

from src.pot import (
    candidate_k_grid,
    decluster_runs,
    fit_gpd,
    score_penalty,
    compute_baseline_k_star,
    process_one_dataset,
    fit_all_k,
    score_stability,
    score_gof,
)


class TestCandidateKGrid:
    """Tests for candidate_k_grid."""

    def test_bounds(self):
        n = 200
        k_min = 10
        k_max_frac = 0.5
        grid = candidate_k_grid(n, k_min, k_max_frac)
        assert grid[0] == k_min
        assert grid[-1] <= int(np.floor(k_max_frac * n))


class TestFitGPD:
    """Tests for fit_gpd on known Pareto data."""

    def test_recovers_pareto_params_approximately(self):
        # Pareto(alpha=2) has GPD shape xi = 1/alpha = 0.5
        rng = np.random.RandomState(42)
        samples = pareto.rvs(b=2.0, size=2000, random_state=rng)
        sorted_desc = np.sort(samples)[::-1]
        k = 200
        xi_hat, beta_hat = fit_gpd(sorted_desc, k)
        # xi should be near 0.5 for Pareto(alpha=2)
        assert abs(xi_hat - 0.5) < 0.3, f"xi_hat={xi_hat}, expected ~0.5"
        assert beta_hat > 0


class TestScorePenalty:
    """Tests for score_penalty."""

    def test_monotonically_decreasing(self):
        k_grid = np.arange(10, 101)
        pen = score_penalty(k_grid)
        assert np.all(np.diff(pen) < 0)


class TestComputeBaselineKStar:
    """Tests for compute_baseline_k_star."""

    def test_k_star_within_grid(self):
        rng = np.random.RandomState(99)
        samples = pareto.rvs(b=2.0, size=500, random_state=rng)
        sorted_desc = np.sort(samples)[::-1]
        k_grid = candidate_k_grid(len(samples), k_min=10, k_max_frac=0.3)
        k_star, diagnostics = compute_baseline_k_star(
            sorted_desc, k_grid, delta=5, weights=(1.0, 1.0, 1.0)
        )
        assert k_star >= k_grid[0]
        assert k_star <= k_grid[-1]

    def test_score_arrays_correct_length(self):
        rng = np.random.RandomState(12)
        samples = pareto.rvs(b=1.5, size=300, random_state=rng)
        sorted_desc = np.sort(samples)[::-1]
        k_grid = candidate_k_grid(len(samples), k_min=10, k_max_frac=0.25)
        _, diagnostics = compute_baseline_k_star(
            sorted_desc, k_grid, delta=5, weights=(1.0, 1.0, 1.0)
        )
        L = len(k_grid)
        assert len(diagnostics["score_stability"]) == L
        assert len(diagnostics["score_gof"]) == L
        assert len(diagnostics["score_penalty"]) == L
        assert len(diagnostics["total_score"]) == L
        assert diagnostics["params"].shape == (L, 2)


class TestDeclusterRuns:
    """Tests for decluster_runs."""

    def test_no_clusters_when_all_below(self):
        """If no values exceed threshold, nothing is removed."""
        samples = np.ones(100)  # all identical → none exceed 90th pctile
        filtered, n_clusters, n_removed = decluster_runs(samples, run_length=10)
        assert n_removed == 0
        assert len(filtered) == len(samples)

    def test_single_cluster_keeps_max(self):
        """A single cluster of exceedances keeps only the maximum."""
        rng = np.random.RandomState(42)
        samples = rng.uniform(0.0, 0.5, size=100)
        # Inject a cluster at indices 50-54
        samples[50] = 5.0
        samples[51] = 10.0  # max
        samples[52] = 7.0
        samples[53] = 6.0
        samples[54] = 4.0
        filtered, n_clusters, n_removed = decluster_runs(samples, run_length=10,
                                                          quantile_threshold=0.90)
        assert n_clusters >= 1
        assert 10.0 in filtered  # max is kept
        assert len(filtered) < len(samples)

    def test_two_clusters_separated_by_long_gap(self):
        """Two clusters separated by a gap >= run_length produce two maxima."""
        samples = np.zeros(100)
        # Cluster 1: indices 10-12
        samples[10] = 5.0
        samples[11] = 8.0  # max of cluster 1
        samples[12] = 6.0
        # Cluster 2: indices 30-32 (gap of 17 zeros > run_length=10)
        samples[30] = 4.0
        samples[31] = 9.0  # max of cluster 2
        samples[32] = 3.0
        filtered, n_clusters, n_removed = decluster_runs(samples, run_length=10,
                                                          quantile_threshold=0.90)
        assert n_clusters == 2
        assert 8.0 in filtered
        assert 9.0 in filtered
        assert n_removed == 4  # 6 exceedances - 2 maxima

    def test_short_gap_merges_clusters(self):
        """Clusters separated by gap < run_length merge into one."""
        samples = np.zeros(100)
        # Group 1: indices 10-12
        samples[10] = 5.0
        samples[11] = 8.0
        samples[12] = 6.0
        # Gap of only 2 (indices 13-14 are 0)
        # Group 2: indices 15-17
        samples[15] = 4.0
        samples[16] = 9.0  # overall max
        samples[17] = 3.0
        filtered, n_clusters, n_removed = decluster_runs(samples, run_length=10,
                                                          quantile_threshold=0.90)
        # With run_length=10, a gap of 2 should NOT split clusters
        assert n_clusters == 1
        assert 9.0 in filtered  # overall max kept
        assert n_removed == 5  # 6 exceedances - 1 max

    def test_integration_with_process_one_dataset(self):
        """process_one_dataset with decluster=True runs without error."""
        rng = np.random.RandomState(42)
        samples = pareto.rvs(b=2.0, size=500, random_state=rng)
        ds = {"samples": samples, "n": len(samples)}
        pot_cfg = {
            "k_min": 10,
            "k_max_frac": 0.15,
            "delta": 5,
            "weights": [1.0, 1.0, 1.0],
            "decluster": True,
            "decluster_run_length": 5,
        }
        ds_out, diagnostics = process_one_dataset(ds, pot_cfg)
        assert diagnostics["declustered"] is True
        assert diagnostics["n_original"] == 500
        assert diagnostics["n_effective"] <= 500
        assert "k_star" in diagnostics
