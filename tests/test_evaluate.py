"""Tests for src.evaluate module."""

import numpy as np
import pytest
from scipy.stats import genpareto

from src.evaluate import agreement_rate, pot_quantile, pot_es, true_es, true_quantile


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


class TestPotES:
    """Tests for pot_es."""

    def test_returns_finite_float(self):
        rng = np.random.RandomState(42)
        samples = np.sort(rng.pareto(2.0, size=500))[::-1]
        k = 50
        xi, beta = 0.5, 1.0
        es = pot_es(samples, k, xi, beta, n=500, p=0.99)
        assert np.isfinite(es)
        assert isinstance(es, (float, np.floating))

    def test_es_greater_than_var(self):
        """ES should always be >= VaR for the same quantile level."""
        rng = np.random.RandomState(42)
        samples = np.sort(rng.pareto(2.0, size=500))[::-1]
        k = 50
        xi, beta = 0.5, 1.0
        var = pot_quantile(samples, k, xi, beta, n=500, p=0.99)
        es = pot_es(samples, k, xi, beta, n=500, p=0.99)
        assert es >= var, f"ES={es} should be >= VaR={var}"

    def test_known_params_match_analytical(self):
        """Test ES formula against direct computation."""
        xi = 0.3
        beta = 2.0
        n = 1000
        k = 100
        p = 0.99
        u = 5.0

        rng = np.random.RandomState(7)
        exceedances = genpareto.rvs(c=xi, scale=beta, size=k, random_state=rng)
        top_k = u + exceedances
        rest = np.linspace(0, u, n - k)
        sorted_desc = np.sort(np.concatenate([top_k, rest]))[::-1]

        es_pot = pot_es(sorted_desc, k, xi, beta, n, p)

        # Analytical: ES = (VaR + beta - xi * u) / (1 - xi)
        var_est = pot_quantile(sorted_desc, k, xi, beta, n, p)
        es_analytical = (var_est + beta - xi * u) / (1 - xi)

        assert abs(es_pot - es_analytical) < 1e-10, (
            f"pot_es={es_pot}, analytical={es_analytical}"
        )

    def test_xi_near_zero(self):
        """When xi ~ 0, ES should be VaR + beta."""
        xi = 0.0
        beta = 2.0
        n = 1000
        k = 100
        p = 0.99
        u = 5.0

        rest = np.linspace(0, u, n)
        sorted_desc = np.sort(rest)[::-1]

        var_est = pot_quantile(sorted_desc, k, xi, beta, n, p)
        es = pot_es(sorted_desc, k, xi, beta, n, p)
        assert abs(es - (var_est + beta)) < 1e-10


class TestTrueES:
    """Tests for true_es."""

    @pytest.mark.parametrize("dist_type,dist_params", [
        ("student_t", {"df": 5}),
        ("pareto", {"alpha": 2.0}),
        ("pareto", {"alpha": 3.0}),
    ])
    def test_positive_finite_result(self, dist_type, dist_params):
        es = true_es(dist_type, dist_params, 0.99)
        assert np.isfinite(es)
        assert es > 0

    @pytest.mark.parametrize("dist_type,dist_params", [
        ("student_t", {"df": 5}),
        ("pareto", {"alpha": 2.0}),
        ("pareto", {"alpha": 3.0}),
    ])
    def test_es_greater_than_quantile(self, dist_type, dist_params):
        """ES should be >= VaR for all dist types."""
        p = 0.99
        var = true_quantile(dist_type, dist_params, p)
        es = true_es(dist_type, dist_params, p)
        assert es >= var, f"ES={es} should be >= VaR={var} for {dist_type}"

    def test_pareto_close_to_analytical(self):
        """For Pareto with alpha>1, MC ES should be close to alpha/(alpha-1) * VaR."""
        alpha = 3.0
        from scipy.stats import pareto as pareto_dist
        var_p = pareto_dist.ppf(0.99, b=alpha)
        expected_es = alpha / (alpha - 1) * var_p
        es = true_es("pareto", {"alpha": alpha}, 0.99)
        # MC with 10M samples should be within ~1% of analytical
        assert abs(es - expected_es) / expected_es < 0.02, (
            f"MC ES={es}, analytical={expected_es}"
        )
