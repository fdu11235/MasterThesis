"""Differentiable POT pipeline components (Pathway M).

Provides soft-threshold masking, differentiable GPD estimation via
probability weighted moments, and differentiable VaR / ES computation
for end-to-end training with pinball loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core differentiable functions
# ---------------------------------------------------------------------------

def soft_threshold_mask(k_cont, n, temperature=10.0):
    """Sigmoid-based soft threshold mask.

    For positions i = 0 .. n-1 (descending-sorted data), returns
    weights_i = sigmoid(temperature * (k_cont - i)).
    Monotonically decreasing; high temperature approaches a hard cutoff.

    Args:
        k_cont: (batch,) continuous k values (float64 recommended).
        n: int, sequence length.
        temperature: sharpness of the sigmoid transition.

    Returns:
        (batch, n) weight tensor in (0, 1).
    """
    i = torch.arange(n, dtype=k_cont.dtype, device=k_cont.device)  # (n,)
    # k_cont: (batch, 1)  -  i: (1, n)
    weights = torch.sigmoid(temperature * (k_cont.unsqueeze(1) - i.unsqueeze(0)))
    return weights


def differentiable_gpd_pwm(sorted_desc, weights, k_cont, temperature_u=50.0):
    """GPD parameter estimation via differentiable probability weighted moments.

    Computes soft threshold *u* via Gaussian-kernel soft-indexing around
    *k_cont*, then estimates (xi, sigma) from the weighted exceedances using
    the closed-form PWM estimator (Hosking & Wallis, 1987).

    Args:
        sorted_desc: (batch, n) samples sorted descending (float64).
        weights: (batch, n) soft threshold mask.
        k_cont: (batch,) continuous k values (float64).
        temperature_u: bandwidth for Gaussian kernel indexing of threshold.

    Returns:
        xi:    (batch,) GPD shape parameter, clamped to [-0.5, 0.95].
        sigma: (batch,) GPD scale parameter (positive, via softplus).
        u:     (batch,) soft threshold value.
    """
    batch, n = sorted_desc.shape
    dtype, device = sorted_desc.dtype, sorted_desc.device

    # --- soft threshold u via Gaussian kernel around k_cont ---
    i = torch.arange(n, dtype=dtype, device=device)              # (n,)
    u_logits = -temperature_u * (i.unsqueeze(0) - k_cont.unsqueeze(1)) ** 2
    u_weights = torch.softmax(u_logits, dim=1)                   # (batch, n)
    u = (u_weights * sorted_desc).sum(dim=1)                     # (batch,)

    # --- soft exceedances ---
    exceedances = F.relu(sorted_desc - u.unsqueeze(1))           # (batch, n)

    # effective number of exceedances (floor for stability)
    w_sum = weights.sum(dim=1).clamp(min=10.0)                   # (batch,)

    # --- PWM: M_0 and M_1 ---
    we = weights * exceedances                                   # (batch, n)
    M_0 = we.sum(dim=1) / w_sum                                 # (batch,)

    # (1-F) rank for descending data: exclusive cumsum / total ≈ i/k
    # Position 0 (largest) gets weight ≈ 0, position k-1 (smallest exc.) ≈ (k-1)/k
    cumw = torch.cumsum(weights, dim=1)                          # (batch, n)
    excl_cumw = cumw - weights                                   # (batch, n)
    one_minus_F = excl_cumw / w_sum.unsqueeze(1)                 # (batch, n)

    M_1 = (we * one_minus_F).sum(dim=1) / w_sum                 # (batch,)

    # --- estimators ---
    denom = (M_0 - 2.0 * M_1).clamp(min=1e-8)

    xi = (2.0 - M_0 / denom).clamp(-0.5, 0.95)
    sigma = F.softplus(2.0 * M_0 * M_1 / denom)

    return xi, sigma, u


def differentiable_var(xi, sigma, u, k_eff, n, p):
    """POT quantile (VaR) formula, differentiable w.r.t. all inputs.

    Q(p) = u + (sigma / xi) * ((n / k_eff * (1-p))^(-xi) - 1)    if |xi| > eps
    Q(p) = u - sigma * log(n / k_eff * (1-p))                      if |xi| ~ 0

    Args:
        xi, sigma, u: (batch,) GPD parameters.
        k_eff: (batch,) effective number of exceedances.
        n: scalar or (batch,) sample size.
        p: float, quantile probability (e.g. 0.99).

    Returns:
        (batch,) VaR estimates.
    """
    ratio = ((n / k_eff) * (1.0 - p)).clamp(min=1e-8)

    var_standard = u + (sigma / xi) * (ratio.pow(-xi) - 1.0)
    var_zero_xi = u - sigma * ratio.log()

    return torch.where(xi.abs() < 1e-6, var_zero_xi, var_standard)


def differentiable_es(xi, sigma, u, var_hat):
    """GPD closed-form Expected Shortfall.

    ES(p) = (VaR(p) + sigma - xi * u) / (1 - xi)     if |xi| > eps
    ES(p) = VaR(p) + sigma                             if xi ~ 0

    Valid for xi < 1.

    Args:
        xi, sigma, u: (batch,) GPD parameters.
        var_hat: (batch,) VaR estimates.

    Returns:
        (batch,) ES (CVaR) estimates.
    """
    one_minus_xi = (1.0 - xi).clamp(min=0.05)
    es_standard = (var_hat + sigma - xi * u) / one_minus_xi
    es_zero_xi = var_hat + sigma

    return torch.where(xi.abs() < 1e-6, es_zero_xi, es_standard)


def compute_var_es_from_params(xi, sigma, u, k_eff, n, p):
    """Compute VaR and ES from pre-computed GPD parameters at quantile *p*.

    Useful for multi-quantile loss: estimate GPD once, then evaluate at
    several quantile levels without re-running the CNN backbone.

    Args:
        xi, sigma, u, k_eff: (batch,) tensors (float32 ok; upcast internally).
        n: scalar or (batch,) sample size.
        p: quantile probability.

    Returns:
        var_hat, es_hat: (batch,) float32 tensors.
    """
    xi64 = xi.double()
    sigma64 = sigma.double()
    u64 = u.double()
    keff64 = k_eff.double()
    n_f = n.double() if torch.is_tensor(n) else float(n)

    var64 = differentiable_var(xi64, sigma64, u64, keff64, n_f, p)
    es64 = differentiable_es(xi64, sigma64, u64, var64)

    return var64.float(), es64.float()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DifferentiablePOTModel(nn.Module):
    """End-to-end differentiable POT model for VaR and ES estimation.

    Architecture: CNN backbone (identical to ThresholdCNN regression)
    predicts k_tilde in (0, 1), which is denormalized to k_cont in
    [k_min, k_max].  Then: sigmoid mask -> weighted exceedances ->
    PWM GPD estimation -> POT VaR -> closed-form ES.
    """

    def __init__(self, in_channels=4, channels=None, kernel_size=5, dropout=0.2):
        super().__init__()
        if channels is None:
            channels = [16, 32]

        # --- CNN backbone (same arch as ThresholdCNN regression) ---
        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.BatchNorm1d(ch),
            ])
            prev_ch = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(prev_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, features, sorted_desc, k_min, k_max, n, p,
                temperature=10.0, temperature_u=50.0):
        """Full forward pass: features -> k_tilde -> GPD -> VaR, ES.

        Args:
            features:    (batch, 4, L_max) diagnostic feature tensor.
            sorted_desc: (batch, n_samples) samples sorted descending.
            k_min:       scalar or (batch,), minimum k value.
            k_max:       scalar or (batch,), maximum k value.
            n:           scalar or (batch,), sample size.
            p:           float, quantile probability (e.g. 0.99).
            temperature: sigmoid mask sharpness.
            temperature_u: Gaussian kernel bandwidth for threshold.

        Returns:
            var_hat: (batch,) VaR estimates (float32).
            es_hat:  (batch,) ES estimates (float32).
            info:    dict with intermediate values (all float32).
        """
        # CNN backbone -> k_tilde in (0, 1)
        x = self.conv(features)
        x = self.pool(x).squeeze(-1)
        k_tilde = self.head(x).squeeze(-1)                       # (batch,)

        # Denormalize to [k_min, k_max]
        k_cont = k_min + k_tilde * (k_max - k_min)               # (batch,)

        # --- float64 for GPD layer ---
        sorted64 = sorted_desc.double()
        k64 = k_cont.double()
        n_f = n.double() if torch.is_tensor(n) else float(n)

        n_samples = sorted_desc.shape[1]
        weights = soft_threshold_mask(k64, n_samples, temperature)

        xi, sigma, u = differentiable_gpd_pwm(sorted64, weights, k64, temperature_u)

        k_eff = weights.sum(dim=1).clamp(min=10.0)

        var_hat = differentiable_var(xi, sigma, u, k_eff, n_f, p)
        es_hat = differentiable_es(xi, sigma, u, var_hat)

        # back to float32
        var_hat = var_hat.float()
        es_hat = es_hat.float()

        info = {
            'k_tilde': k_tilde,
            'k_cont': k_cont,
            'k_eff': k_eff.float(),
            'xi': xi.float(),
            'sigma': sigma.float(),
            'u': u.float(),
        }
        return var_hat, es_hat, info

    def load_backbone_from(self, state_dict):
        """Warm-start CNN backbone from a pre-trained ThresholdCNN checkpoint.

        Copies all parameters whose names and shapes match.

        Args:
            state_dict: state dict from a ThresholdCNN (Pathway A).

        Returns:
            int — number of parameters loaded.
        """
        own = self.state_dict()
        loaded = 0
        for name, param in state_dict.items():
            if name in own and own[name].shape == param.shape:
                own[name].copy_(param)
                loaded += 1
        self.load_state_dict(own)
        return loaded
