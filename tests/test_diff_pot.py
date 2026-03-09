"""Tests for the differentiable POT pipeline (Pathway M)."""

import numpy as np
import pytest
import torch
from scipy.stats import genpareto

from src.diff_pot import (
    DifferentiablePOTModel,
    compute_var_es_from_params,
    differentiable_es,
    differentiable_gpd_pwm,
    differentiable_var,
    soft_threshold_mask,
)
from src.train_diff import es_direct_loss, fz0_loss, pinball_loss


# ---------------------------------------------------------------------------
# soft_threshold_mask
# ---------------------------------------------------------------------------

class TestSoftThresholdMask:

    def test_shape(self):
        k = torch.tensor([50.0, 75.0])
        w = soft_threshold_mask(k, n=200, temperature=10.0)
        assert w.shape == (2, 200)

    def test_values_between_0_and_1(self):
        k = torch.tensor([50.0])
        w = soft_threshold_mask(k, n=200, temperature=10.0)
        assert (w >= 0).all() and (w <= 1).all()

    def test_monotonically_decreasing(self):
        k = torch.tensor([50.0])
        w = soft_threshold_mask(k, n=200, temperature=10.0)
        diffs = w[0, 1:] - w[0, :-1]
        assert (diffs <= 1e-7).all(), "Weights should be monotonically decreasing"

    def test_high_temperature_approaches_hard_cutoff(self):
        k = torch.tensor([50.0])
        w = soft_threshold_mask(k, n=200, temperature=200.0)
        # positions well below k should be ~1, well above should be ~0
        assert w[0, 30].item() > 0.999
        assert w[0, 70].item() < 0.001


# ---------------------------------------------------------------------------
# differentiable_gpd_pwm
# ---------------------------------------------------------------------------

class TestDifferentiableGPDPWM:

    def test_recovers_known_gpd_params(self):
        """GPD(xi=0.5, sigma=2.0) exceedances should give xi ~ 0.5."""
        torch.manual_seed(42)
        rng = np.random.RandomState(42)

        n = 5000
        k_true = 300
        xi_true, sigma_true = 0.5, 2.0
        u_val = 10.0

        # Generate exceedances from GPD and place above threshold
        exc = genpareto.rvs(c=xi_true, scale=sigma_true, size=k_true,
                            random_state=rng)
        top = u_val + exc
        rest = np.linspace(0.0, u_val, n - k_true, endpoint=False)
        samples = np.concatenate([top, rest])
        sorted_desc = np.sort(samples)[::-1].copy()

        sd = torch.tensor(sorted_desc, dtype=torch.float64).unsqueeze(0)
        k_cont = torch.tensor([float(k_true)], dtype=torch.float64)

        # High temperature for near-hard cutoff
        weights = soft_threshold_mask(k_cont, n, temperature=100.0)
        xi_hat, sigma_hat, u_hat = differentiable_gpd_pwm(
            sd, weights, k_cont, temperature_u=100.0,
        )

        assert abs(xi_hat.item() - xi_true) < 0.25, \
            f"xi_hat={xi_hat.item():.3f}, expected ~{xi_true}"
        assert u_hat.item() == pytest.approx(u_val, abs=1.0), \
            f"u_hat={u_hat.item():.2f}, expected ~{u_val}"

    def test_gradients_flow(self):
        """Gradients should propagate through k_cont and sorted_desc."""
        n = 200
        sd = torch.randn(2, n, dtype=torch.float64).abs().sort(dim=1, descending=True).values
        sd.requires_grad_(True)
        k_cont = torch.tensor([50.0, 60.0], dtype=torch.float64, requires_grad=True)

        weights = soft_threshold_mask(k_cont, n, temperature=10.0)
        xi, sigma, u = differentiable_gpd_pwm(sd, weights, k_cont, temperature_u=50.0)

        loss = xi.sum() + sigma.sum() + u.sum()
        loss.backward()

        assert sd.grad is not None and sd.grad.abs().sum() > 0
        assert k_cont.grad is not None and k_cont.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# differentiable_var / differentiable_es
# ---------------------------------------------------------------------------

class TestDifferentiableVarES:

    def _make_params(self):
        xi = torch.tensor([0.3], dtype=torch.float64)
        sigma = torch.tensor([2.0], dtype=torch.float64)
        u = torch.tensor([5.0], dtype=torch.float64)
        k_eff = torch.tensor([100.0], dtype=torch.float64)
        return xi, sigma, u, k_eff

    def test_var_matches_analytical(self):
        xi, sigma, u, k_eff = self._make_params()
        n, p = 1000.0, 0.99
        var = differentiable_var(xi, sigma, u, k_eff, n, p)
        ratio = (n / k_eff.item()) * (1 - p)
        expected = u.item() + (sigma.item() / xi.item()) * (ratio ** (-xi.item()) - 1)
        assert var.item() == pytest.approx(expected, rel=1e-6)

    def test_es_greater_than_var(self):
        xi, sigma, u, k_eff = self._make_params()
        var = differentiable_var(xi, sigma, u, k_eff, 1000.0, 0.99)
        es = differentiable_es(xi, sigma, u, var)
        assert es.item() > var.item(), "ES should exceed VaR"

    def test_es_xi_zero(self):
        """For xi ~ 0, ES = VaR + sigma (exponential tail)."""
        sigma = torch.tensor([2.0], dtype=torch.float64)
        u = torch.tensor([5.0], dtype=torch.float64)
        xi = torch.tensor([0.0], dtype=torch.float64)
        k_eff = torch.tensor([100.0], dtype=torch.float64)
        var = differentiable_var(xi, sigma, u, k_eff, 1000.0, 0.99)
        es = differentiable_es(xi, sigma, u, var)
        assert es.item() == pytest.approx(var.item() + sigma.item(), rel=1e-5)

    def test_es_analytical_gpd(self):
        """Check ES formula against analytical GPD ES."""
        xi_val, sigma_val, u_val = 0.3, 2.0, 5.0
        xi = torch.tensor([xi_val], dtype=torch.float64)
        sigma = torch.tensor([sigma_val], dtype=torch.float64)
        u = torch.tensor([u_val], dtype=torch.float64)
        k_eff = torch.tensor([100.0], dtype=torch.float64)

        var = differentiable_var(xi, sigma, u, k_eff, 1000.0, 0.99)
        es = differentiable_es(xi, sigma, u, var)

        expected_es = (var.item() + sigma_val - xi_val * u_val) / (1 - xi_val)
        assert es.item() == pytest.approx(expected_es, rel=1e-6)

    def test_compute_var_es_from_params(self):
        xi = torch.tensor([0.3])
        sigma = torch.tensor([2.0])
        u = torch.tensor([5.0])
        k_eff = torch.tensor([100.0])
        var, es = compute_var_es_from_params(xi, sigma, u, k_eff, 1000, 0.99)
        assert var.dtype == torch.float32
        assert es.item() > var.item()


# ---------------------------------------------------------------------------
# pinball_loss
# ---------------------------------------------------------------------------

class TestPinballLoss:

    def test_zero_when_perfect(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        loss = pinball_loss(y, y, tau=0.99)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_asymmetry(self):
        """Underprediction should be penalized ~tau/(1-tau) times more."""
        y_true = torch.tensor([10.0])
        q_under = torch.tensor([9.0])   # underprediction
        q_over = torch.tensor([11.0])   # overprediction

        loss_under = pinball_loss(y_true, q_under, tau=0.99)
        loss_over = pinball_loss(y_true, q_over, tau=0.99)

        # underprediction: tau * (10 - 9) = 0.99
        # overprediction: (1-tau) * (11 - 10) = 0.01
        assert loss_under.item() == pytest.approx(0.99, rel=1e-5)
        assert loss_over.item() == pytest.approx(0.01, rel=1e-5)
        assert loss_under.item() > 90 * loss_over.item()


# ---------------------------------------------------------------------------
# fz0_loss (ES loss)
# ---------------------------------------------------------------------------

class TestFZ0Loss:

    def test_finite_for_valid_inputs(self):
        y_samples = torch.rand(4, 100) * 10
        var_hat = torch.tensor([5.0, 5.0, 5.0, 5.0])
        es_hat = torch.tensor([7.0, 7.0, 7.0, 7.0])
        loss = fz0_loss(y_samples, var_hat, es_hat, tau=0.99)
        assert torch.isfinite(loss)

    def test_gradients_flow(self):
        y_samples = torch.rand(4, 100) * 10
        var_hat = torch.tensor([5.0, 5.0, 5.0, 5.0], requires_grad=True)
        es_hat = torch.tensor([7.0, 7.0, 7.0, 7.0], requires_grad=True)
        loss = fz0_loss(y_samples, var_hat, es_hat, tau=0.99)
        loss.backward()
        assert var_hat.grad is not None
        assert es_hat.grad is not None


# ---------------------------------------------------------------------------
# DifferentiablePOTModel
# ---------------------------------------------------------------------------

class TestDifferentiablePOTModel:

    def _make_inputs(self, batch=4, n=200, L=50):
        features = torch.randn(batch, 4, L)
        sorted_desc = torch.randn(batch, n).abs().sort(dim=1, descending=True).values
        return features, sorted_desc

    def test_forward_returns_correct_types(self):
        model = DifferentiablePOTModel(channels=[8, 16], kernel_size=3)
        feat, sd = self._make_inputs()
        var_hat, es_hat, info = model(feat, sd, k_min=10, k_max=50, n=200, p=0.99)

        assert var_hat.shape == (4,)
        assert es_hat.shape == (4,)
        assert 'xi' in info and 'sigma' in info and 'u' in info
        assert info['xi'].shape == (4,)

    def test_backward_completes(self):
        model = DifferentiablePOTModel(channels=[8, 16], kernel_size=3)
        feat, sd = self._make_inputs()
        var_hat, es_hat, info = model(feat, sd, k_min=10, k_max=50, n=200, p=0.99)

        loss = var_hat.sum() + es_hat.sum()
        loss.backward()

        # check all CNN params have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            # at least some gradients should be non-zero
        total_grad = sum(p.grad.abs().sum().item() for p in model.parameters())
        assert total_grad > 0, "All gradients are zero"

    def test_var_and_es_finite(self):
        model = DifferentiablePOTModel(channels=[8, 16], kernel_size=3)
        feat, sd = self._make_inputs(batch=8, n=500)
        var_hat, es_hat, _ = model(feat, sd, k_min=10, k_max=75, n=500, p=0.99)
        assert torch.isfinite(var_hat).all()
        assert torch.isfinite(es_hat).all()

    def test_load_backbone_from(self):
        """Warm-start from a compatible state dict."""
        from src.model import ThresholdCNN
        cnn = ThresholdCNN(in_channels=4, channels=[8, 16], kernel_size=3,
                           dropout=0.0, task="regression")
        diff = DifferentiablePOTModel(in_channels=4, channels=[8, 16],
                                      kernel_size=3, dropout=0.0)
        n_loaded = diff.load_backbone_from(cnn.state_dict())
        assert n_loaded > 0


# ---------------------------------------------------------------------------
# ES stability near xi=1
# ---------------------------------------------------------------------------

class TestESStableNearXiOne:

    @pytest.mark.parametrize("xi_val", [0.9, 0.95, 1.0, 1.5])
    def test_es_finite_and_positive(self, xi_val):
        """ES should stay finite and positive even when xi >= 1."""
        xi = torch.tensor([xi_val], dtype=torch.float64)
        sigma = torch.tensor([2.0], dtype=torch.float64)
        u = torch.tensor([5.0], dtype=torch.float64)
        var = torch.tensor([20.0], dtype=torch.float64)

        es = differentiable_es(xi, sigma, u, var)
        assert torch.isfinite(es).all(), f"ES not finite for xi={xi_val}"
        assert es.item() > 0, f"ES not positive for xi={xi_val}"


# ---------------------------------------------------------------------------
# es_direct_loss
# ---------------------------------------------------------------------------

class TestESDirectLoss:

    def test_zero_when_perfect(self):
        es_true = torch.tensor([5.0, 10.0, 20.0])
        es_hat = es_true.clone()
        loss = es_direct_loss(es_true, es_hat)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_scale_independent(self):
        """Loss should be similar regardless of ES magnitude."""
        es_true_small = torch.tensor([1.0, 2.0])
        es_hat_small = torch.tensor([1.5, 2.5])
        loss_small = es_direct_loss(es_true_small, es_hat_small)

        es_true_big = torch.tensor([1000.0, 2000.0])
        es_hat_big = torch.tensor([1500.0, 2500.0])
        loss_big = es_direct_loss(es_true_big, es_hat_big)

        assert loss_small.item() == pytest.approx(loss_big.item(), rel=1e-4)

    def test_gradients_flow(self):
        es_true = torch.tensor([5.0, 10.0])
        es_hat = torch.tensor([6.0, 12.0], requires_grad=True)
        loss = es_direct_loss(es_true, es_hat)
        loss.backward()
        assert es_hat.grad is not None
        assert es_hat.grad.abs().sum() > 0

    def test_magnitude_order_one(self):
        """Loss should be O(1) for reasonable prediction errors."""
        es_true = torch.tensor([10.0, 20.0, 50.0])
        es_hat = torch.tensor([12.0, 18.0, 55.0])
        loss = es_direct_loss(es_true, es_hat)
        assert 0.0 < loss.item() < 10.0


# ---------------------------------------------------------------------------
# xi upper clamp
# ---------------------------------------------------------------------------

class TestXiUpperClamp:

    def test_gpd_pwm_clamps_high_xi(self):
        """GPD PWM with true xi=1.25 should produce estimated xi <= 0.95."""
        rng = np.random.RandomState(123)
        n = 5000
        k_true = 300
        xi_true = 1.25
        sigma_true = 2.0
        u_val = 10.0

        exc = genpareto.rvs(c=xi_true, scale=sigma_true, size=k_true,
                            random_state=rng)
        top = u_val + exc
        rest = np.linspace(0.0, u_val, n - k_true, endpoint=False)
        samples = np.concatenate([top, rest])
        sorted_desc = np.sort(samples)[::-1].copy()

        sd = torch.tensor(sorted_desc, dtype=torch.float64).unsqueeze(0)
        k_cont = torch.tensor([float(k_true)], dtype=torch.float64)

        weights = soft_threshold_mask(k_cont, n, temperature=100.0)
        xi_hat, _, _ = differentiable_gpd_pwm(sd, weights, k_cont, temperature_u=100.0)

        assert xi_hat.item() <= 0.95 + 1e-6, \
            f"xi_hat={xi_hat.item():.3f} should be clamped to <= 0.95"
