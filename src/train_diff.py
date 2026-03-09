"""Training utilities for the differentiable POT pipeline (Pathway M).

Provides loss functions (pinball, FZ0, combined), the training loop with
temperature annealing and gradient clipping, and a predict helper.
"""

import logging

import torch
import torch.nn.functional as F

from src.diff_pot import compute_var_es_from_params

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def pinball_loss(y_true, y_pred, tau):
    """Pinball (quantile-regression) loss.

    L = tau * max(y - q, 0) + (1-tau) * max(q - y, 0)

    Args:
        y_true: (batch,) true quantile values.
        y_pred: (batch,) predicted quantile values.
        tau: float in (0, 1), quantile level.

    Returns:
        Scalar mean loss.
    """
    err = y_true - y_pred
    return torch.mean(torch.max(tau * err, (tau - 1.0) * err))


def fz0_loss(y_samples, var_hat, es_hat, tau):
    """Fissler-Ziegel FZ0 loss for joint (VaR, ES) elicitation.

    L_i = (1/ES) * (relu(y_i - VaR) / (1-tau) - ES + VaR) + log(ES)

    Uses actual sample observations *y_samples* as realisations.

    Args:
        y_samples: (batch, m) actual observations (or full sorted_desc).
        var_hat:   (batch,) predicted VaR.
        es_hat:    (batch,) predicted ES (must be > 0).
        tau:       float, quantile level.

    Returns:
        Scalar mean loss.
    """
    es_safe = es_hat.clamp(min=1e-4)
    exceedance = torch.relu(y_samples - var_hat.unsqueeze(1))     # (batch, m)
    term = exceedance / (1.0 - tau)                               # (batch, m)
    inner = term - es_safe.unsqueeze(1) + var_hat.unsqueeze(1)    # (batch, m)
    loss = inner / es_safe.unsqueeze(1) + torch.log(es_safe).unsqueeze(1)
    return loss.mean()


def es_direct_loss(es_true, es_hat):
    """Scale-free ES loss: Smooth L1 on ratio es_hat/es_true."""
    ratio = es_hat / es_true.clamp(min=1e-4)
    return F.smooth_l1_loss(ratio, torch.ones_like(ratio))


def multi_quantile_loss(true_q, info, n, quantiles, quantile_weights, var_primary, tau):
    """Weighted pinball loss at multiple quantile levels.

    Re-uses GPD parameters from *info* (computed once by the forward pass)
    to evaluate VaR at additional quantile levels, giving denser gradients
    than a single tau=0.99 pinball term.

    Args:
        true_q:           dict {p: (batch,) tensor} of true quantile values.
        info:             dict from model.forward() with xi, sigma, u, k_eff.
        n:                scalar, sample size.
        quantiles:        list of float, quantile levels.
        quantile_weights: list of float, corresponding weights (should sum ~1).
        var_primary:      (batch,) VaR at the primary tau (avoids re-computation).
        tau:              float, primary quantile level (typically 0.99).

    Returns:
        Scalar mean multi-quantile pinball loss.
    """
    loss = torch.tensor(0.0, device=var_primary.device)
    for q, w in zip(quantiles, quantile_weights):
        if abs(q - tau) < 1e-8:
            var_q = var_primary
        else:
            var_q, _ = compute_var_es_from_params(
                info['xi'], info['sigma'], info['u'], info['k_eff'], n, q,
            )
        tq = true_q[q]
        loss = loss + w * pinball_loss(tq, var_q, q)
    return loss


def combined_loss(true_q, true_es, var_hat, es_hat, info, n, config,
                  k_tilde_true=None):
    """Full training loss: multi-quantile pinball + direct ES + auxiliary k.

    Args:
        true_q:        dict {p: (batch,) tensor} of true quantile values.
        true_es:       (batch,) true Expected Shortfall values.
        var_hat:       (batch,) predicted VaR at primary tau.
        es_hat:        (batch,) predicted ES at primary tau.
        info:          dict from model.forward().
        n:             scalar, sample size.
        config:        dict with keys pinball_weight, es_weight, quantiles,
                       quantile_weights, quantile_p, k_aux_weight.
        k_tilde_true:  (batch,) normalized target k from Pathway A (optional).

    Returns:
        Scalar total loss.
    """
    tau = config.get('quantile_p', 0.99)
    w_pb = config.get('pinball_weight', 0.7)
    w_es = config.get('es_weight', 0.3)
    w_k = config.get('k_aux_weight', 0.0)
    quantiles = config.get('quantiles', [0.95, 0.975, 0.99])
    quantile_weights = config.get('quantile_weights', [0.2, 0.3, 0.5])

    loss_mq = multi_quantile_loss(
        true_q, info, n, quantiles, quantile_weights, var_hat, tau,
    )
    loss_es = es_direct_loss(true_es, es_hat)

    total = w_pb * loss_mq + w_es * loss_es

    if k_tilde_true is not None and w_k > 0:
        loss_k = F.mse_loss(info['k_tilde'], k_tilde_true)
        total = total + w_k * loss_k

    return total


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_diff_model(model, train_groups, val_groups, config):
    """Train differentiable POT model with combined loss and early stopping.

    Args:
        model:        DifferentiablePOTModel instance.
        train_groups: dict[int, dict] keyed by sample size *n*, each with
                      'features', 'sorted_desc', 'true_q', 'k_min', 'k_max'.
        val_groups:   same structure as train_groups.
        config:       dict with diff_pot + evaluate settings.

    Returns:
        Trained model (best validation weights loaded).
    """
    diff_cfg = config.get('diff_pot', {})
    eval_cfg = config.get('evaluate', {})

    lr = diff_cfg.get('lr', 0.0005)
    batch_size = diff_cfg.get('batch_size', 32)
    max_epochs = diff_cfg.get('max_epochs', 300)
    patience = diff_cfg.get('patience', 20)
    temp_start = diff_cfg.get('temperature_start', 5.0)
    temp_end = diff_cfg.get('temperature_end', 20.0)
    temp_anneal = diff_cfg.get('temperature_anneal', True)
    temp_u = diff_cfg.get('temperature_u', 50.0)
    tau = eval_cfg.get('quantile_p', 0.99)

    k_aux_weight = diff_cfg.get('k_aux_weight', 0.0)

    # merge diff_pot config with quantile_p for combined_loss
    loss_cfg = dict(diff_cfg)
    loss_cfg['quantile_p'] = tau

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6,
    )
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        # temperature annealing
        if temp_anneal:
            frac = min(epoch / max(max_epochs // 2, 1), 1.0)
            temperature = temp_start + frac * (temp_end - temp_start)
        else:
            temperature = temp_end

        # anneal k_aux weight: full strength first half, decay to 0 second half
        if k_aux_weight > 0:
            aux_frac = max(1.0 - 2.0 * epoch / max(max_epochs, 1), 0.0)
            loss_cfg['k_aux_weight'] = k_aux_weight * aux_frac
        else:
            loss_cfg['k_aux_weight'] = 0.0

        # ---- train --------------------------------------------------------
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for n_size, grp in train_groups.items():
            feat = grp['features']
            sd = grp['sorted_desc']
            tq = grp['true_q']       # {p: (N,) tensor}
            te = grp['true_es']      # (N,) tensor
            kt = grp.get('k_tilde_true')  # (N,) tensor or None
            k_min = grp['k_min']
            k_max = grp['k_max']
            N = feat.shape[0]
            perm = torch.randperm(N)

            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                feat_b = feat[idx]
                sd_b = sd[idx]
                tq_b = {p: v[idx] for p, v in tq.items()}
                te_b = te[idx]
                kt_b = kt[idx] if kt is not None else None

                optimizer.zero_grad()
                var_hat, es_hat, info = model(
                    feat_b, sd_b, k_min, k_max, n_size, tau,
                    temperature=temperature, temperature_u=temp_u,
                )

                loss = combined_loss(
                    tq_b, te_b, var_hat, es_hat, info, n_size, loss_cfg,
                    k_tilde_true=kt_b,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)

        # ---- validation ---------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for n_size, grp in val_groups.items():
                var_hat, es_hat, info = model(
                    grp['features'], grp['sorted_desc'],
                    grp['k_min'], grp['k_max'], n_size, tau,
                    temperature=temp_end, temperature_u=temp_u,
                )
                vloss = combined_loss(
                    grp['true_q'], grp['true_es'], var_hat, es_hat,
                    info, n_size, loss_cfg,
                )
                val_loss_sum += vloss.item() * grp['features'].shape[0]
                val_count += grp['features'].shape[0]

        val_loss = val_loss_sum / max(val_count, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            cur_k_aux = loss_cfg.get('k_aux_weight', 0.0)
            logger.info(
                "Epoch %3d: train=%.6f  val=%.6f  best=%.6f  patience=%d/%d  temp=%.1f  lr=%.1e  k_aux=%.2f",
                epoch + 1, avg_train, val_loss, best_val_loss,
                patience_counter, patience, temperature, cur_lr, cur_k_aux,
            )

    logger.info("Training complete - best val_loss=%.6f", best_val_loss)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_diff(model, features, sorted_desc, k_min, k_max, n, p,
                 temperature=20.0, temperature_u=50.0):
    """Run inference and return VaR, ES, and GPD diagnostics.

    Args:
        model:       trained DifferentiablePOTModel.
        features:    (N, 4, L_max) tensor.
        sorted_desc: (N, n_samples) tensor.
        k_min, k_max: scalars.
        n: scalar, sample size.
        p: quantile probability.
        temperature, temperature_u: sigmoid / Gaussian bandwidth.

    Returns:
        var_est: (N,) ndarray of VaR estimates.
        es_est:  (N,) ndarray of ES estimates.
        k_vals:  (N,) ndarray of predicted k values.
        gpd:     dict with 'xi', 'sigma', 'u' arrays.
    """
    model.eval()
    with torch.no_grad():
        var_hat, es_hat, info = model(
            features, sorted_desc, k_min, k_max, n, p,
            temperature=temperature, temperature_u=temperature_u,
        )
    return (
        var_hat.numpy(),
        es_hat.numpy(),
        info['k_cont'].numpy(),
        {k: v.numpy() for k, v in info.items() if k not in ('k_tilde', 'k_cont', 'k_eff')},
    )
