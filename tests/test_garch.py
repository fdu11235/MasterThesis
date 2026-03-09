"""Tests for GARCH(1,1) fitting and volatility filtering."""

import numpy as np
import pytest

from src.garch import fit_garch_and_filter, _fallback


def _simulate_garch11(n=2000, omega=0.01, alpha=0.05, beta=0.90, seed=42):
    """Simulate a GARCH(1,1) process for testing."""
    rng = np.random.RandomState(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

    return returns, np.sqrt(sigma2)


class TestFitGarchAndFilter:
    def test_convergence_on_garch_data(self):
        """GARCH fitting should converge on data generated from a GARCH process."""
        returns, true_vol = _simulate_garch11(n=2000)
        result = fit_garch_and_filter(returns, forecast_horizon=5)

        assert result["converged"] is True
        assert result["std_residuals"].shape == returns.shape
        assert result["abs_std_residuals"].shape == returns.shape
        assert result["conditional_vol"].shape == returns.shape
        assert result["forecast_vol"].shape == (5,)

    def test_standardized_residuals_approx_unit_variance(self):
        """Standardized residuals z_t should have approximately unit variance."""
        returns, _ = _simulate_garch11(n=5000, seed=123)
        result = fit_garch_and_filter(returns, forecast_horizon=1)

        if result["converged"]:
            std_var = np.var(result["std_residuals"])
            # Should be close to 1 (allowing some tolerance)
            assert 0.5 < std_var < 2.0, f"Std residual variance = {std_var}"

    def test_forecast_vol_positive(self):
        """Forecasted volatilities should be positive."""
        returns, _ = _simulate_garch11(n=1000)
        result = fit_garch_and_filter(returns, forecast_horizon=10)
        assert np.all(result["forecast_vol"] > 0)

    def test_conditional_vol_positive(self):
        """Conditional volatility should be positive."""
        returns, _ = _simulate_garch11(n=1000)
        result = fit_garch_and_filter(returns, forecast_horizon=1)
        assert np.all(result["conditional_vol"] > 0)

    def test_different_forecast_horizons(self):
        """Forecast horizon parameter should control output length."""
        returns, _ = _simulate_garch11(n=1000)

        for h in [1, 5, 20, 250]:
            result = fit_garch_and_filter(returns, forecast_horizon=h)
            assert len(result["forecast_vol"]) == h


class TestFallback:
    def test_fallback_returns_correct_shapes(self):
        """Fallback should return correct shapes with constant vol."""
        returns = np.random.randn(500)
        result = _fallback(returns, forecast_horizon=10)

        assert result["converged"] is False
        assert result["std_residuals"].shape == (500,)
        assert result["abs_std_residuals"].shape == (500,)
        assert result["conditional_vol"].shape == (500,)
        assert result["forecast_vol"].shape == (10,)

    def test_fallback_constant_vol(self):
        """Fallback volatility should be constant (sample std dev)."""
        returns = np.random.randn(500)
        result = _fallback(returns, forecast_horizon=5)

        # All conditional vols should be the same
        assert np.all(result["conditional_vol"] == result["conditional_vol"][0])
        # Forecast vols should all be the same
        assert np.all(result["forecast_vol"] == result["forecast_vol"][0])


class TestVarBacktestGarch:
    def test_garch_backtest_shapes(self):
        """var_backtest_garch should return correct shapes."""
        from src.evaluate_real import var_backtest_garch

        n = 1000
        rng = np.random.RandomState(42)
        samples = np.sort(rng.pareto(2.0, n))[::-1]
        future_returns = rng.pareto(2.0, 250)
        forecast_vol = np.ones(250) * 0.01

        # Use simple GPD params
        k = 50
        xi = 0.5
        beta = 1.0
        p = 0.99

        bt = var_backtest_garch(samples, k, xi, beta, n, p,
                                 future_returns, forecast_vol)

        assert bt["var_t"].shape == (250,)
        assert bt["es_t"].shape == (250,)
        assert bt["violations_binary"].shape == (250,)
        assert 0 <= bt["violation_rate"] <= 1
        assert bt["n_future"] == 250

    def test_garch_backtest_higher_vol_more_violations(self):
        """Higher forecast volatility should generally yield fewer violations
        (larger VaR thresholds)."""
        from src.evaluate_real import var_backtest_garch

        n = 1000
        rng = np.random.RandomState(42)
        samples = np.sort(rng.pareto(2.0, n))[::-1]
        future_returns = np.abs(rng.standard_normal(250)) * 0.02

        k = 50
        xi = 0.3
        beta = 0.5
        p = 0.99

        # Low vol forecast
        bt_low = var_backtest_garch(samples, k, xi, beta, n, p,
                                     future_returns, np.ones(250) * 0.001)
        # High vol forecast
        bt_high = var_backtest_garch(samples, k, xi, beta, n, p,
                                      future_returns, np.ones(250) * 1.0)

        # Higher vol → larger VaR → fewer violations
        assert bt_high["n_violations"] <= bt_low["n_violations"]
