"""Tests for src.evaluate module."""

import numpy as np
import pytest
from scipy.stats import genpareto

from src.evaluate import agreement_rate, pot_quantile


class TestAgreementRate:
    """Tests for agreement_rate."""

    def test_identical_arrays(self):
        k = np.array([10, 20, 30, 40, 50])
        assert agreement_rate(k, k, radius=0) == 1.0

    def test_very_distant_arrays(self):
        k_pred = np.array([0, 0, 0, 0, 0])
        k_true = np.array([1000, 2000, 3000, 4000, 5000])
        assert agreement_rate(k_pred, k_true, radius=5) == 0.0


class TestPotQuantile:
    """Tests for pot_quantile."""

    def test_returns_finite_float(self):
        rng = np.random.RandomState(42)
        samples = np.sort(rng.pareto(2.0, size=500))[::-1]
        k = 50
        xi, beta = 0.5, 1.0
        q = pot_quantile(samples, k, xi, beta, n=500, p=0.99)
        assert np.isfinite(q)
        assert isinstance(q, (float, np.floating))

    def test_known_gpd_params(self):
        """With known GPD parameters, pot_quantile should approximate
        scipy.stats.genpareto.ppf for the exceedance distribution."""
        xi = 0.3
        beta = 2.0
        n = 1000
        k = 100
        p = 0.99

        # Build a synthetic sorted_desc where sorted_desc[k] = u (threshold)
        u = 5.0
        # Generate exceedances from GPD and add threshold
        rng = np.random.RandomState(7)
        exceedances = genpareto.rvs(c=xi, scale=beta, size=k, random_state=rng)
        top_k = u + exceedances
        # Fill the rest with values <= u
        rest = np.linspace(0, u, n - k)
        sorted_desc = np.sort(np.concatenate([top_k, rest]))[::-1]

        q_pot = pot_quantile(sorted_desc, k, xi, beta, n, p)

        # Analytical: Q(p) = u + (beta/xi) * ((n/k * (1-p))^(-xi) - 1)
        q_analytical = u + (beta / xi) * ((n / k * (1 - p)) ** (-xi) - 1)

        assert abs(q_pot - q_analytical) < 1e-10, (
            f"pot_quantile={q_pot}, analytical={q_analytical}"
        )
