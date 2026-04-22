import logging
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from scipy.stats import genpareto, norm
from scipy.special import gamma as gamma_fn, beta as beta_fn, betainc, gammainc, gammaincc

from src.synthetic import (
    _generate_student_t, _generate_pareto,
    _generate_lognormal_pareto_mix, _generate_two_pareto,
    _generate_log_gamma, _generate_gamma_pareto_splice,
)

logger = logging.getLogger(__name__)

_MC_CACHE_PATH = "outputs/data/mc_es_cache.pkl"
_mc_quantile_cache = {}
_mc_es_cache = {}
_cache_loaded = False


def _params_key(dist_type, dist_params, p):
    return (dist_type, tuple(sorted(dist_params.items())), p)


def _load_cache():
    """Load persistent MC caches from disk on first use."""
    global _mc_quantile_cache, _mc_es_cache, _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    if os.path.exists(_MC_CACHE_PATH):
        try:
            with open(_MC_CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            _mc_quantile_cache.update(data.get("quantile", {}))
            _mc_es_cache.update(data.get("es", {}))
            logger.info("Loaded MC cache from %s (%d quantile, %d es entries)",
                        _MC_CACHE_PATH, len(_mc_quantile_cache), len(_mc_es_cache))
        except Exception as e:
            logger.warning("Failed to load MC cache: %s", e)


def _save_cache():
    """Persist the MC caches to disk."""
    os.makedirs(os.path.dirname(_MC_CACHE_PATH), exist_ok=True)
    try:
        with open(_MC_CACHE_PATH, "wb") as f:
            pickle.dump({"quantile": _mc_quantile_cache, "es": _mc_es_cache}, f)
    except Exception as e:
        logger.warning("Failed to save MC cache: %s", e)


def _mc_quantile(dist_type, dist_params, p, n_mc=10_000_000, seed=99999):
    rng = np.random.RandomState(seed)
    if dist_type == 'lognormal_pareto_mix':
        samples = _generate_lognormal_pareto_mix(rng, n_mc, **dist_params)
    elif dist_type == 'two_pareto':
        samples = _generate_two_pareto(rng, n_mc, **dist_params)
    elif dist_type == 'log_gamma':
        samples = _generate_log_gamma(rng, n_mc, **dist_params)
    elif dist_type == 'gamma_pareto_splice':
        samples = _generate_gamma_pareto_splice(rng, n_mc, **dist_params)
    else:
        raise ValueError(f"No MC quantile for {dist_type}")
    return float(np.quantile(samples, p))


def agreement_rate(k_pred, k_true, radius):
    """Fraction of predictions within radius of true k_star.

    Args:
        k_pred: ndarray of predicted k values
        k_true: ndarray of true (baseline) k values
        radius: int tolerance
    Returns:
        float in [0, 1]
    """
    return np.mean(np.abs(k_pred - k_true) <= radius)


def pot_quantile(sorted_desc, k, xi, beta, n, p):
    """Compute POT quantile estimate.

    Q(p) = u + (beta/xi) * ((n/k * (1-p))^(-xi) - 1)  if xi != 0
    Q(p) = u - beta * log(n/k * (1-p))                   if xi == 0

    Args:
        sorted_desc: samples sorted descending
        k: number of exceedances
        xi: GPD shape parameter
        beta: GPD scale parameter
        n: total sample size
        p: quantile probability (e.g. 0.99)
    Returns:
        float quantile estimate
    """
    u = sorted_desc[k]  # threshold
    if abs(xi) < 1e-8:
        return u - beta * np.log(n / k * (1 - p))
    return u + (beta / xi) * ((n / k * (1 - p)) ** (-xi) - 1)


def pot_es(sorted_desc, k, xi, beta, n, p):
    """GPD closed-form Expected Shortfall.

    ES(p) = (VaR(p) + beta - xi * u) / (1 - xi)   if xi != 0
    ES(p) = VaR(p) + beta                           if xi ~ 0
    """
    var_est = pot_quantile(sorted_desc, k, xi, beta, n, p)
    u = sorted_desc[k]
    if abs(xi) < 1e-8:
        return var_est + beta
    one_minus_xi = max(1 - xi, 0.05)  # stability clamp (mirrors diff_pot.py:130)
    return (var_est + beta - xi * u) / one_minus_xi


def pot_es_stable(sorted_desc, k, xi, beta, n, p):
    """Expected Shortfall with semi-parametric fallback for high xi.

    When xi > 0.7, the closed-form ES formula has 1/(1-xi) which amplifies
    errors dramatically. Falls back to the empirical mean of observations
    exceeding VaR (a legitimate Hill-type ES estimator).
    """
    var_est = pot_quantile(sorted_desc, k, xi, beta, n, p)
    if xi <= 0.7:
        return pot_es(sorted_desc, k, xi, beta, n, p)
    # Semi-parametric: mean of observations exceeding VaR
    tail = sorted_desc[sorted_desc > var_est]
    if len(tail) >= 2:
        return float(tail.mean())
    return var_est + beta  # exponential tail fallback


def _resolve_garch_type(dist_type, dist_params):
    """Map GARCH-wrapped types to their base distribution for true quantile/ES.

    GARCH standardised residuals have the same theoretical tail as the base
    innovation distribution, so we delegate to the base type.
    """
    if dist_type.startswith('garch_'):
        base_type = dist_type[len('garch_'):]
        base_params = {k: v for k, v in dist_params.items()
                       if k not in ('garch_omega', 'garch_alpha', 'garch_beta')}
        return base_type, base_params
    return dist_type, dist_params


def true_quantile(dist_type, dist_params, p):
    """Analytical quantile from known distribution parameters.

    Args:
        dist_type: str, one of 'student_t', 'pareto', 'lognormal_pareto_mix', 'two_pareto'
        dist_params: dict of distribution parameters
        p: quantile probability
    Returns:
        float true quantile
    """
    dist_type, dist_params = _resolve_garch_type(dist_type, dist_params)
    if dist_type == 'student_t':
        return abs(stats.t.ppf(p, df=dist_params['df']))
    elif dist_type == 'pareto':
        return stats.pareto.ppf(p, b=dist_params['alpha'])
    elif dist_type == 'burr12':
        return stats.burr12.ppf(p, c=dist_params['c'], d=dist_params['d'])
    elif dist_type == 'frechet':
        return stats.invweibull.ppf(p, c=dist_params['c'])
    elif dist_type == 'dagum':
        return stats.burr.ppf(p, c=dist_params['c'], d=dist_params['d'])
    elif dist_type == 'inverse_gamma':
        return stats.invgamma.ppf(p, a=dist_params['a'])
    elif dist_type == 'lognormal':
        return stats.lognorm.ppf(p, s=dist_params['sigma'])
    elif dist_type == 'weibull_stretched':
        return stats.weibull_min.ppf(p, c=dist_params['c'])
    elif dist_type in ('lognormal_pareto_mix', 'two_pareto',
                        'log_gamma', 'gamma_pareto_splice'):
        _load_cache()
        key = _params_key(dist_type, dist_params, p)
        if key not in _mc_quantile_cache:
            logger.info("Computing MC quantile for %s (p=%s) ...", dist_type, p)
            _mc_quantile_cache[key] = _mc_quantile(dist_type, dist_params, p)
            _save_cache()
        return _mc_quantile_cache[key]
    raise ValueError(f"Unknown dist_type: {dist_type}")


def _mc_es(dist_type, dist_params, p, n_mc=10_000_000, seed=99999):
    """Monte Carlo Expected Shortfall: E[X | X > VaR(p)]."""
    dist_type, dist_params = _resolve_garch_type(dist_type, dist_params)
    rng = np.random.RandomState(seed)
    generators = {
        'student_t': lambda: np.abs(stats.t.rvs(df=dist_params['df'], size=n_mc, random_state=rng)),
        'pareto': lambda: stats.pareto.rvs(b=dist_params['alpha'], size=n_mc, random_state=rng),
        'lognormal_pareto_mix': lambda: _generate_lognormal_pareto_mix(rng, n_mc, **dist_params),
        'two_pareto': lambda: _generate_two_pareto(rng, n_mc, **dist_params),
        'burr12': lambda: stats.burr12.rvs(c=dist_params['c'], d=dist_params['d'], size=n_mc, random_state=rng),
        'frechet': lambda: stats.invweibull.rvs(c=dist_params['c'], size=n_mc, random_state=rng),
        'dagum': lambda: stats.burr.rvs(c=dist_params['c'], d=dist_params['d'], size=n_mc, random_state=rng),
        'inverse_gamma': lambda: stats.invgamma.rvs(a=dist_params['a'], size=n_mc, random_state=rng),
        'lognormal': lambda: stats.lognorm.rvs(s=dist_params['sigma'], size=n_mc, random_state=rng),
        'weibull_stretched': lambda: stats.weibull_min.rvs(c=dist_params['c'], size=n_mc, random_state=rng),
        'log_gamma': lambda: _generate_log_gamma(rng, n_mc, **dist_params),
        'gamma_pareto_splice': lambda: _generate_gamma_pareto_splice(rng, n_mc, **dist_params),
    }
    if dist_type not in generators:
        raise ValueError(f"No MC ES for {dist_type}")
    samples = generators[dist_type]()
    q = np.quantile(samples, p)
    tail = samples[samples > q]
    return float(tail.mean()) if len(tail) > 0 else float(q)


def _analytical_es(dist_type, dist_params, p):
    """Closed-form or numerically-integrated ES for tractable distributions.

    Returns None for distributions without a practical analytical form
    (composites like mixtures/splices, Log-Gamma). For Student-t the
    distribution of |T| is used (matches the abs-return convention in
    the synthetic generator).
    """
    if dist_type == 'pareto':
        alpha = dist_params['alpha']
        if alpha <= 1:
            return None  # ES undefined (infinite mean)
        var = stats.pareto.ppf(p, b=alpha)
        return var * alpha / (alpha - 1)

    if dist_type == 'student_t':
        df = dist_params['df']
        if df <= 1:
            return None
        # Synthetic samples are |T|, so VaR at prob p of |T| equals the
        # t-quantile at (1+p)/2. By symmetry, E[|T| | |T|>v] = E[T | T>v].
        upper_p = (p + 1.0) / 2.0
        v = stats.t.ppf(upper_p, df=df)
        fv = stats.t.pdf(v, df=df)
        # E[T | T>v] = (df + v^2)/(df-1) * f(v) / (1 - F(v)),
        # and 1 - F(v) = (1-p)/2 on the upper tail.
        return (df + v * v) / (df - 1.0) * fv / ((1.0 - p) / 2.0)

    if dist_type == 'lognormal':
        sigma = dist_params['sigma']
        phi_inv_p = norm.ppf(p)
        return np.exp(sigma * sigma / 2.0) * norm.cdf(sigma - phi_inv_p) / (1.0 - p)

    if dist_type == 'burr12':
        c, d = dist_params['c'], dist_params['d']
        if c * d <= 1:
            return None
        a1, a2 = d - 1.0 / c, 1.0 + 1.0 / c
        x = (1.0 - p) ** (1.0 / d)
        return float(d * beta_fn(a1, a2) * betainc(a1, a2, x) / (1.0 - p))

    if dist_type == 'frechet':
        c = dist_params['c']
        if c <= 1:
            return None
        s = 1.0 - 1.0 / c
        return float(gamma_fn(s) * gammainc(s, -np.log(p)) / (1.0 - p))

    if dist_type == 'dagum':
        c, d = dist_params['c'], dist_params['d']
        if c <= 1:
            return None
        a1, a2 = 1.0 - 1.0 / c, d + 1.0 / c
        x = 1.0 - p ** (1.0 / d)
        return float(d * beta_fn(a1, a2) * betainc(a1, a2, x) / (1.0 - p))

    if dist_type == 'inverse_gamma':
        a = dist_params['a']
        if a <= 1:
            return None
        var = stats.invgamma.ppf(p, a=a)
        return float(gamma_fn(a - 1) * gammainc(a - 1, 1.0 / var) / (gamma_fn(a) * (1.0 - p)))

    if dist_type == 'weibull_stretched':
        c = dist_params['c']
        s = 1.0 + 1.0 / c
        return float(gamma_fn(s) * gammaincc(s, -np.log(1.0 - p)) / (1.0 - p))

    return None


def true_es(dist_type, dist_params, p):
    """True Expected Shortfall: E[X | X > VaR(p)].

    Uses a closed-form / numerically integrated value where available,
    falling back to Monte Carlo (10M samples) for composite distributions.
    Both paths are cached by (dist_type, params, p), with persistence to
    ``outputs/data/mc_es_cache.pkl`` so results are reused across runs.
    """
    dist_type, dist_params = _resolve_garch_type(dist_type, dist_params)
    _load_cache()
    key = _params_key(dist_type, dist_params, p)
    if key in _mc_es_cache:
        return _mc_es_cache[key]

    val = _analytical_es(dist_type, dist_params, p)
    if val is None:
        logger.info("Computing MC ES for %s (p=%s) ...", dist_type, p)
        val = _mc_es(dist_type, dist_params, p)
    _mc_es_cache[key] = float(val)
    _save_cache()
    return _mc_es_cache[key]


def evaluate_all(test_data, k_pred, k_true, config):
    """Compute all evaluation metrics.

    Args:
        test_data: list of (dataset_dict, diagnostics_dict) for test set
        k_pred: ndarray of predicted k values (mapped back to actual k values, not indices)
        k_true: ndarray of true baseline k values
        config: eval config dict with 'agreement_radii', 'quantile_p'
    Returns:
        dict with 'agreement' (dict radius->rate), 'quantile_rmse' (float)
    """
    results = {}

    # Agreement rates
    results['agreement'] = {}
    for r in config.get('agreement_radii', [5, 10]):
        results['agreement'][r] = agreement_rate(k_pred, k_true, r)

    # Quantile and ES RMSE
    p = config.get('quantile_p', 0.99)
    q_est = []
    q_true = []
    es_est_list = []
    es_true_list = []
    dist_types = []
    dist_params = []
    for i, (ds, diag) in enumerate(test_data):
        sorted_desc = np.sort(ds['samples'])[::-1]
        k = k_pred[i]
        k_idx = np.searchsorted(diag['k_grid'], k)
        k_idx = min(k_idx, len(diag['params']) - 1)
        xi, beta = diag['params'][k_idx]
        if np.isnan(xi) or np.isnan(beta):
            continue
        xi = np.clip(xi, -0.5, 0.95)  # stability clamp (mirrors diff_pot pipeline)
        n = ds['n']
        q_est.append(pot_quantile(sorted_desc, k, xi, beta, n, p))
        q_true.append(true_quantile(ds['dist_type'], ds['params'], p))
        es_est_list.append(pot_es_stable(sorted_desc, k, xi, beta, n, p))
        es_true_list.append(true_es(ds['dist_type'], ds['params'], p))
        dist_types.append(ds['dist_type'])
        dist_params.append(ds['params'])

    if q_est:
        q_est_arr = np.array(q_est)
        q_true_arr = np.array(q_true)
        es_est_arr = np.array(es_est_list)
        es_true_arr = np.array(es_true_list)

        results['quantile_rmse'] = np.sqrt(np.mean((q_est_arr - q_true_arr) ** 2))

        # Relative RMSE (normalized by true quantile)
        rel_errors = (q_est_arr - q_true_arr) / q_true_arr
        results['relative_rmse'] = np.sqrt(np.mean(rel_errors ** 2))

        # ES RMSE
        results['es_rmse'] = np.sqrt(np.mean((es_est_arr - es_true_arr) ** 2))
        es_rel_errors = (es_est_arr - es_true_arr) / es_true_arr
        results['es_relative_rmse'] = np.sqrt(np.mean(es_rel_errors ** 2))

        # Per-distribution RMSE breakdown
        results['rmse_by_dist'] = {}
        for dist_type in sorted(set(dist_types)):
            mask = np.array([d == dist_type for d in dist_types])
            q_e = q_est_arr[mask]
            q_t = q_true_arr[mask]
            es_e = es_est_arr[mask]
            es_t = es_true_arr[mask]
            results['rmse_by_dist'][dist_type] = {
                'rmse': np.sqrt(np.mean((q_e - q_t) ** 2)),
                'relative_rmse': np.sqrt(np.mean(((q_e - q_t) / q_t) ** 2)),
                'es_rmse': np.sqrt(np.mean((es_e - es_t) ** 2)),
                'es_relative_rmse': np.sqrt(np.mean(((es_e - es_t) / es_t) ** 2)),
                'count': int(mask.sum()),
            }

        # Per-distribution MAE
        for dist_type in sorted(set(dist_types)):
            mask = np.array([d == dist_type for d in dist_types])
            q_e = q_est_arr[mask]
            q_t = q_true_arr[mask]
            results['rmse_by_dist'][dist_type]['mae'] = float(np.mean(np.abs(q_e - q_t)))

        # Quantile and ES MAE
        results['quantile_mae'] = float(np.mean(np.abs(q_est_arr - q_true_arr)))
        results['es_mae'] = float(np.mean(np.abs(es_est_arr - es_true_arr)))

        # Bootstrap 95% CIs on relative_rmse and es_relative_rmse
        n_boot = 1000
        rng = np.random.RandomState(42)
        n_samples = len(rel_errors)

        boot_rel_rmse = np.empty(n_boot)
        boot_es_rel_rmse = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.randint(0, n_samples, size=n_samples)
            boot_rel_rmse[b] = np.sqrt(np.mean(rel_errors[idx] ** 2))
            boot_es_rel_rmse[b] = np.sqrt(np.mean(es_rel_errors[idx] ** 2))

        results['relative_rmse_ci'] = (
            float(np.percentile(boot_rel_rmse, 2.5)),
            float(np.percentile(boot_rel_rmse, 97.5)),
        )
        results['es_relative_rmse_ci'] = (
            float(np.percentile(boot_es_rel_rmse, 2.5)),
            float(np.percentile(boot_es_rel_rmse, 97.5)),
        )

        # Store raw data for plotting
        results['_q_est'] = q_est
        results['_q_true'] = q_true
        results['_es_est'] = es_est_list
        results['_es_true'] = es_true_list
        results['_dist_types'] = dist_types
        results['_dist_params'] = dist_params
        results['_quantile_p'] = p
        results['_rel_errors'] = rel_errors.tolist()
    else:
        results['quantile_rmse'] = float('nan')
        results['relative_rmse'] = float('nan')
        results['es_rmse'] = float('nan')
        results['es_relative_rmse'] = float('nan')
        results['rmse_by_dist'] = {}
        results['quantile_mae'] = float('nan')
        results['es_mae'] = float('nan')
        results['relative_rmse_ci'] = (float('nan'), float('nan'))
        results['es_relative_rmse_ci'] = (float('nan'), float('nan'))

    # k prediction metrics
    results['k_pred'] = k_pred
    results['k_true'] = k_true
    k_errors = k_pred.astype(float) - k_true.astype(float)
    results['k_mae'] = float(np.mean(np.abs(k_errors)))
    results['k_median_ae'] = float(np.median(np.abs(k_errors)))

    # R² between k_pred and k_true
    ss_res = np.sum(k_errors ** 2)
    ss_tot = np.sum((k_true.astype(float) - np.mean(k_true)) ** 2)
    results['k_r2'] = float(1 - ss_res / (ss_tot + 1e-10))

    logger.debug("Quantile RMSE: %.4f, Relative RMSE: %.2f%% (from %d / %d valid samples)",
                 results['quantile_rmse'], results['relative_rmse'] * 100,
                 len(q_est), len(test_data))

    return results


def plot_training_curves(history, save_dir):
    """Plot training/validation loss and learning rate curves.

    Args:
        history: dict with 'train_loss', 'val_loss', 'lr' lists
        save_dir: directory to save figures
    """
    if history is None:
        return
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(epochs, history["train_loss"], label="Train loss", lw=1.2)
    ax1.plot(epochs, history["val_loss"], label="Val loss", lw=1.2)
    # Mark best epoch (early stop point)
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    ax1.axvline(best_epoch, color='r', ls='--', lw=0.8,
                label=f'Best epoch={best_epoch}')
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9)
    ax1.set_title("Training Curves")

    ax2.plot(epochs, history["lr"], color='tab:green', lw=1.2)
    ax2.set_ylabel("Learning Rate")
    ax2.set_xlabel("Epoch")
    ax2.set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)

    # Loss decomposition plot (if VaR-aware components are tracked)
    has_components = (history.get("train_L_k") and
                      any(v > 0 for v in history["train_L_k"]))
    if has_components:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

        # Top-left: total loss
        axes[0, 0].plot(epochs, history["train_loss"], label="Train", lw=1.2)
        axes[0, 0].plot(epochs, history["val_loss"], label="Val", lw=1.2)
        axes[0, 0].axvline(best_epoch, color='r', ls='--', lw=0.8)
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].legend(fontsize=8)

        # Top-right: L_k (k* prediction)
        axes[0, 1].plot(epochs, history["train_L_k"], label="Train L_k", lw=1.2)
        axes[0, 1].plot(epochs, history["val_L_k"], label="Val L_k", lw=1.2)
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("L_k (k* prediction)")
        axes[0, 1].legend(fontsize=8)

        # Bottom-left: L_var (VaR quality)
        axes[1, 0].plot(epochs, history["train_L_var"], label="Train L_var", lw=1.2, color='tab:orange')
        axes[1, 0].plot(epochs, history["val_L_var"], label="Val L_var", lw=1.2, color='tab:red')
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_title("L_var (VaR quality)")
        axes[1, 0].legend(fontsize=8)

        # Bottom-right: L_es (ES quality)
        axes[1, 1].plot(epochs, history["train_L_es"], label="Train L_es", lw=1.2, color='tab:green')
        axes[1, 1].plot(epochs, history["val_L_es"], label="Val L_es", lw=1.2, color='tab:purple')
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_title("L_es (ES quality)")
        axes[1, 1].legend(fontsize=8)

        fig.suptitle("Loss Decomposition: k* + VaR + ES", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "loss_decomposition.png"), dpi=150)
        plt.close(fig)


def plot_pred_vs_true(k_pred, k_true, dist_types, save_dir):
    """Scatter plot of predicted vs true k, colored by distribution type.

    Args:
        k_pred, k_true: arrays of predicted and true k values
        dist_types: list of distribution type strings
        save_dir: directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    unique_dists = sorted(set(dist_types))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_dists), 1)))

    for j, dt in enumerate(unique_dists):
        mask = np.array([d == dt for d in dist_types])
        ax.scatter(k_true[mask], k_pred[mask], s=15, alpha=0.5,
                   color=colors[j], label=dt)

    lims = [min(k_true.min(), k_pred.min()), max(k_true.max(), k_pred.max())]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.7, label='y=x')

    # R²
    ss_res = np.sum((k_pred.astype(float) - k_true.astype(float)) ** 2)
    ss_tot = np.sum((k_true.astype(float) - np.mean(k_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    ax.set_title(f"Predicted vs True k  (R²={r2:.3f})")
    ax.set_xlabel("k_true")
    ax.set_ylabel("k_pred")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "pred_vs_true.png"), dpi=150)
    plt.close(fig)


def plot_residuals(k_pred, k_true, rel_errors, save_dir):
    """Histogram of k prediction errors and quantile relative errors.

    Args:
        k_pred, k_true: arrays of predicted and true k values
        rel_errors: list/array of relative quantile errors (fraction)
        save_dir: directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    k_errors = k_pred.astype(float) - k_true.astype(float)
    ax1.hist(k_errors, bins=40, edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='r', ls='--', lw=1)
    ax1.set_xlabel("k_pred - k_true")
    ax1.set_ylabel("Count")
    ax1.set_title(f"k Prediction Error (MAE={np.mean(np.abs(k_errors)):.1f})")

    rel_err = np.array(rel_errors) * 100
    ax2.hist(rel_err, bins=40, edgecolor='black', alpha=0.7, color='tab:orange')
    ax2.axvline(0, color='r', ls='--', lw=1)
    ax2.set_xlabel("Relative Quantile Error (%)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Quantile Estimation Error (RRMSE={np.sqrt(np.mean(np.array(rel_errors)**2))*100:.1f}%)")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "residuals.png"), dpi=150)
    plt.close(fig)


def plot_gpd_qq(test_data, k_pred, save_dir):
    """QQ-plot of exceedances vs fitted GPD for representative samples.

    Args:
        test_data: list of (dataset_dict, diagnostics_dict)
        k_pred: array of predicted k values
        save_dir: directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    n_examples = min(4, len(test_data))
    if n_examples == 0:
        return

    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))
    if n_examples == 1:
        axes = [axes]

    # Pick evenly spaced samples
    indices = np.linspace(0, len(test_data) - 1, n_examples, dtype=int)

    for ax_idx, i in enumerate(indices):
        ds, diag = test_data[i]
        sorted_desc = np.sort(ds['samples'])[::-1]
        k = int(k_pred[i])
        k_idx = np.searchsorted(diag['k_grid'], k)
        k_idx = min(k_idx, len(diag['params']) - 1)
        xi, beta = diag['params'][k_idx]

        if np.isnan(xi) or np.isnan(beta):
            axes[ax_idx].set_title("GPD fit failed")
            continue

        exceedances = sorted_desc[:k] - sorted_desc[k]
        exc_sorted = np.sort(exceedances)
        n_exc = len(exc_sorted)
        probs = (np.arange(1, n_exc + 1) - 0.5) / n_exc
        theoretical_q = genpareto.ppf(probs, xi, loc=0, scale=beta)

        axes[ax_idx].scatter(theoretical_q, exc_sorted, s=8, alpha=0.6)
        lims = [0, max(exc_sorted.max(), theoretical_q.max())]
        axes[ax_idx].plot(lims, lims, 'r--', lw=1)
        axes[ax_idx].set_xlabel("GPD quantiles")
        axes[ax_idx].set_ylabel("Empirical quantiles")
        dist_label = ds.get('dist_type', 'unknown')
        axes[ax_idx].set_title(f"{dist_label} (k={k})")

    fig.suptitle("GPD QQ-plots (CNN threshold)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "gpd_qq.png"), dpi=150)
    plt.close(fig)


def plot_tail_fit(test_data, k_pred, save_dir):
    """Log-log plot of empirical vs fitted GPD survival function.

    Args:
        test_data: list of (dataset_dict, diagnostics_dict)
        k_pred: array of predicted k values
        save_dir: directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    n_examples = min(4, len(test_data))
    if n_examples == 0:
        return

    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))
    if n_examples == 1:
        axes = [axes]

    indices = np.linspace(0, len(test_data) - 1, n_examples, dtype=int)

    for ax_idx, i in enumerate(indices):
        ds, diag = test_data[i]
        sorted_desc = np.sort(ds['samples'])[::-1]
        k = int(k_pred[i])
        k_idx = np.searchsorted(diag['k_grid'], k)
        k_idx = min(k_idx, len(diag['params']) - 1)
        xi, beta = diag['params'][k_idx]

        if np.isnan(xi) or np.isnan(beta):
            axes[ax_idx].set_title("GPD fit failed")
            continue

        exceedances = sorted_desc[:k] - sorted_desc[k]
        exc_sorted = np.sort(exceedances)[::-1]
        n_exc = len(exc_sorted)

        # Empirical survival
        emp_surv = np.arange(1, n_exc + 1) / n_exc
        # Fitted GPD survival
        fitted_surv = 1 - genpareto.cdf(exc_sorted, xi, loc=0, scale=beta)

        axes[ax_idx].loglog(exc_sorted, emp_surv, '.', ms=4, alpha=0.5, label='Empirical')
        axes[ax_idx].loglog(exc_sorted, np.clip(fitted_surv, 1e-10, 1), '-', lw=1.2,
                            color='red', label='GPD fit')
        axes[ax_idx].set_xlabel("Exceedance")
        axes[ax_idx].set_ylabel("Survival P(X > x)")
        dist_label = ds.get('dist_type', 'unknown')
        axes[ax_idx].set_title(f"{dist_label} (k={k})")
        axes[ax_idx].legend(fontsize=8)

    fig.suptitle("Tail Fit: Empirical vs GPD Survival", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "tail_fit.png"), dpi=150)
    plt.close(fig)


def plot_mean_excess(test_data, k_pred, save_dir):
    """Mean excess function plot with CNN-selected threshold marked.

    Args:
        test_data: list of (dataset_dict, diagnostics_dict)
        k_pred: array of predicted k values
        save_dir: directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    n_examples = min(4, len(test_data))
    if n_examples == 0:
        return

    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4))
    if n_examples == 1:
        axes = [axes]

    indices = np.linspace(0, len(test_data) - 1, n_examples, dtype=int)

    for ax_idx, i in enumerate(indices):
        ds, diag = test_data[i]
        sorted_desc = np.sort(ds['samples'])[::-1]
        k = int(k_pred[i])
        n = len(sorted_desc)

        # Compute mean excess function over a range of thresholds
        n_points = min(200, n - 2)
        thresholds = np.sort(ds['samples'])[int(n * 0.5):]  # upper 50%
        step = max(1, len(thresholds) // n_points)
        thresholds = thresholds[::step]

        me_vals = []
        me_thresh = []
        for u in thresholds:
            above = ds['samples'][ds['samples'] > u]
            if len(above) < 5:
                continue
            me_vals.append(np.mean(above - u))
            me_thresh.append(u)

        if me_vals:
            axes[ax_idx].plot(me_thresh, me_vals, lw=1.2)
            # Mark CNN threshold
            cnn_threshold = sorted_desc[k] if k < len(sorted_desc) else sorted_desc[-1]
            axes[ax_idx].axvline(cnn_threshold, color='r', ls='--', lw=1,
                                label=f'CNN u(k={k})')
            axes[ax_idx].set_xlabel("Threshold u")
            axes[ax_idx].set_ylabel("Mean Excess e(u)")
            dist_label = ds.get('dist_type', 'unknown')
            axes[ax_idx].set_title(f"{dist_label}")
            axes[ax_idx].legend(fontsize=8)

    fig.suptitle("Mean Excess Function", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "mean_excess.png"), dpi=150)
    plt.close(fig)


def plot_results(results, all_diagnostics, save_dir,
                 k_pred=None, k_true=None, history=None):
    """Generate and save evaluation plots.

    Args:
        results: dict from evaluate_all
        all_diagnostics: list of (dataset_dict, diagnostics_dict)
        save_dir: directory to save figures
        k_pred: optional ndarray of predicted k values
        k_true: optional ndarray of true k values
        history: optional training history dict
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Agreement rate bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    radii = sorted(results['agreement'].keys())
    rates = [results['agreement'][r] for r in radii]
    ax.bar([f'r={r}' for r in radii], rates)
    ax.set_ylabel('Agreement Rate')
    ax.set_title('Agreement Rate by Radius')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'agreement_rates.png'), dpi=150)
    plt.close(fig)

    # 2. Example diagnostic curves: xi(k), GOF(k), Score(k) (first 4 datasets)
    n_examples = min(4, len(all_diagnostics))
    fig, axes = plt.subplots(n_examples, 3, figsize=(16, 3 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]
    for i in range(n_examples):
        ds, diag = all_diagnostics[i]
        k_grid = diag['k_grid']

        axes[i, 0].plot(k_grid, diag['xi_series'])
        axes[i, 0].axvline(diag['k_star'], color='r', ls='--', label=f'k*={diag["k_star"]}')
        axes[i, 0].set_ylabel('xi_hat(k)')
        axes[i, 0].set_title(f'{ds["dist_type"]} n={ds["n"]}')
        axes[i, 0].legend(fontsize=8)

        axes[i, 1].plot(k_grid, diag['score_gof'], color='tab:orange')
        axes[i, 1].axvline(diag['k_star'], color='r', ls='--', label=f'k*={diag["k_star"]}')
        axes[i, 1].set_ylabel('AD statistic')
        axes[i, 1].set_title('GOF(k) — Anderson-Darling')
        axes[i, 1].legend(fontsize=8)

        axes[i, 2].plot(k_grid, diag['total_score'], color='tab:green')
        axes[i, 2].axvline(diag['k_star'], color='r', ls='--', label=f'k*={diag["k_star"]}')
        axes[i, 2].set_ylabel('Total Score')
        axes[i, 2].set_title('Score(k)')
        axes[i, 2].legend(fontsize=8)

    for col in range(3):
        axes[-1, col].set_xlabel('k')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'diagnostic_curves.png'), dpi=150)
    plt.close(fig)

    # 3. Quantile error: predicted vs true scatter + per-distribution RMSE bars
    p = results.get('_quantile_p', 0.99)
    if 'rmse_by_dist' in results and results['rmse_by_dist']:
        # 3a. Per-distribution relative RMSE bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        dist_names = sorted(results['rmse_by_dist'].keys())
        rel_rmses = [results['rmse_by_dist'][d]['relative_rmse'] * 100 for d in dist_names]
        counts = [results['rmse_by_dist'][d]['count'] for d in dist_names]
        bars = ax.bar(dist_names, rel_rmses, color='tab:blue')
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'n={c}', ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('Relative RMSE (%)')
        ax.set_title(f'Quantile Estimation Error by Distribution (p={p})')
        ax.axhline(results['relative_rmse'] * 100, color='r', ls='--', lw=1,
                    label=f'Overall: {results["relative_rmse"]*100:.1f}%')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'quantile_rmse_by_dist.png'), dpi=150)
        plt.close(fig)

    if '_q_est' in results and '_q_true' in results:
        # 3b. Relative error box plot grouped by distribution + parameters
        q_est = np.array(results['_q_est'])
        q_true = np.array(results['_q_true'])
        dist_types = results.get('_dist_types', [])
        dist_params_list = results.get('_dist_params', [])
        rel_errors = (q_est - q_true) / q_true * 100  # percentage

        # Build group labels from dist_type + key params
        def _make_label(dt, dp):
            if dt == 'student_t':
                return f't(df={dp.get("df", "?")})'
            elif dt == 'pareto':
                return f'Pareto(a={dp.get("alpha", "?")})'
            elif dt == 'lognormal_pareto_mix':
                return f'LN-Par(a={dp.get("pareto_alpha", "?")})'
            elif dt == 'two_pareto':
                return f'2Par({dp.get("alpha1", "?")},{dp.get("alpha2", "?")})'
            elif dt == 'burr12':
                return f'Burr12(c={dp.get("c", "?")},d={dp.get("d", "?")})'
            elif dt == 'frechet':
                return f'Frechet(c={dp.get("c", "?")})'
            elif dt == 'dagum':
                return f'Dagum(c={dp.get("c", "?")},d={dp.get("d", "?")})'
            elif dt == 'inverse_gamma':
                return f'InvGa(a={dp.get("a", "?")})'
            elif dt == 'lognormal':
                return f'LN(s={dp.get("sigma", "?")})'
            elif dt == 'weibull_stretched':
                return f'Weib(c={dp.get("c", "?")})'
            elif dt == 'log_gamma':
                return f'LogGa(b={dp.get("b", "?")},p={dp.get("p", "?")})'
            elif dt == 'gamma_pareto_splice':
                return f'Ga-Par(k={dp.get("gamma_shape", "?")},a={dp.get("pareto_alpha", "?")})'
            return dt

        labels = [_make_label(dt, dp) for dt, dp in zip(dist_types, dist_params_list)]

        # Group by label, sorted by median true quantile
        from collections import OrderedDict
        groups = {}
        for i, lab in enumerate(labels):
            groups.setdefault(lab, {'rel_err': [], 'q_true': []})
            groups[lab]['rel_err'].append(rel_errors[i])
            groups[lab]['q_true'].append(q_true[i])

        sorted_labels = sorted(groups.keys(),
                                key=lambda l: np.median(groups[l]['q_true']))

        box_data = [groups[l]['rel_err'] for l in sorted_labels]
        tick_labels = [f'{l}\nQ={np.median(groups[l]["q_true"]):.1f}' for l in sorted_labels]

        fig, ax = plt.subplots(figsize=(max(8, len(sorted_labels) * 1.2), 5))
        bp = ax.boxplot(box_data, tick_labels=tick_labels, patch_artist=True,
                        medianprops=dict(color='red', lw=1.5))
        for patch in bp['boxes']:
            patch.set_facecolor('tab:blue')
            patch.set_alpha(0.6)
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set_ylabel('Relative Error (%)')
        ax.set_xlabel('Distribution (sorted by true quantile)')
        ax.set_title(f'POT Quantile Estimation Error by Distribution (p={p})')
        plt.xticks(rotation=30, ha='right', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'quantile_error_boxplot.png'), dpi=150)
        plt.close(fig)

    # New plots: training curves, pred vs true, residuals, GPD QQ, tail fit, mean excess
    if history is not None:
        plot_training_curves(history, save_dir)

    if k_pred is not None and k_true is not None:
        dist_types_plot = results.get('_dist_types', ['unknown'] * len(k_pred))
        plot_pred_vs_true(k_pred, k_true, dist_types_plot, save_dir)

        rel_errors = results.get('_rel_errors', [])
        if rel_errors:
            plot_residuals(k_pred, k_true, rel_errors, save_dir)

        plot_gpd_qq(all_diagnostics, k_pred, save_dir)
        plot_tail_fit(all_diagnostics, k_pred, save_dir)
        plot_mean_excess(all_diagnostics, k_pred, save_dir)

    logger.info("Figures saved to %s", save_dir)
