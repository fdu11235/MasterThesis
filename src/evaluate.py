import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from src.synthetic import _generate_lognormal_pareto_mix, _generate_two_pareto

logger = logging.getLogger(__name__)

_mc_quantile_cache = {}


def _params_key(dist_type, dist_params, p):
    return (dist_type, tuple(sorted(dist_params.items())), p)


def _mc_quantile(dist_type, dist_params, p, n_mc=10_000_000, seed=99999):
    rng = np.random.RandomState(seed)
    if dist_type == 'lognormal_pareto_mix':
        samples = _generate_lognormal_pareto_mix(rng, n_mc, **dist_params)
    elif dist_type == 'two_pareto':
        samples = _generate_two_pareto(rng, n_mc, **dist_params)
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


def true_quantile(dist_type, dist_params, p):
    """Analytical quantile from known distribution parameters.

    Args:
        dist_type: str, one of 'student_t', 'pareto', 'lognormal_pareto_mix', 'two_pareto'
        dist_params: dict of distribution parameters
        p: quantile probability
    Returns:
        float true quantile
    """
    if dist_type == 'student_t':
        return abs(stats.t.ppf(p, df=dist_params['df']))
    elif dist_type == 'pareto':
        return stats.pareto.ppf(p, b=dist_params['alpha'])
    elif dist_type in ('lognormal_pareto_mix', 'two_pareto'):
        key = _params_key(dist_type, dist_params, p)
        if key not in _mc_quantile_cache:
            logger.info("Computing MC quantile for %s (p=%s) ...", dist_type, p)
            _mc_quantile_cache[key] = _mc_quantile(dist_type, dist_params, p)
        return _mc_quantile_cache[key]
    raise ValueError(f"Unknown dist_type: {dist_type}")


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

    # Quantile RMSE
    p = config.get('quantile_p', 0.99)
    q_est = []
    q_true = []
    dist_types = []
    for i, (ds, diag) in enumerate(test_data):
        sorted_desc = np.sort(ds['samples'])[::-1]
        k = k_pred[i]
        k_idx = np.searchsorted(diag['k_grid'], k)
        k_idx = min(k_idx, len(diag['params']) - 1)
        xi, beta = diag['params'][k_idx]
        if np.isnan(xi) or np.isnan(beta):
            continue
        n = ds['n']
        q_est.append(pot_quantile(sorted_desc, k, xi, beta, n, p))
        q_true.append(true_quantile(ds['dist_type'], ds['params'], p))
        dist_types.append(ds['dist_type'])

    if q_est:
        q_est_arr = np.array(q_est)
        q_true_arr = np.array(q_true)
        results['quantile_rmse'] = np.sqrt(np.mean((q_est_arr - q_true_arr) ** 2))

        # Relative RMSE (normalized by true quantile)
        rel_errors = (q_est_arr - q_true_arr) / q_true_arr
        results['relative_rmse'] = np.sqrt(np.mean(rel_errors ** 2))

        # Per-distribution RMSE breakdown
        results['rmse_by_dist'] = {}
        for dist_type in sorted(set(dist_types)):
            mask = np.array([d == dist_type for d in dist_types])
            q_e = q_est_arr[mask]
            q_t = q_true_arr[mask]
            results['rmse_by_dist'][dist_type] = {
                'rmse': np.sqrt(np.mean((q_e - q_t) ** 2)),
                'relative_rmse': np.sqrt(np.mean(((q_e - q_t) / q_t) ** 2)),
                'count': int(mask.sum()),
            }
    else:
        results['quantile_rmse'] = float('nan')
        results['relative_rmse'] = float('nan')
        results['rmse_by_dist'] = {}

    logger.debug("Quantile RMSE: %.4f, Relative RMSE: %.2f%% (from %d / %d valid samples)",
                 results['quantile_rmse'], results['relative_rmse'] * 100,
                 len(q_est), len(test_data))

    return results


def plot_results(results, all_diagnostics, save_dir):
    """Generate and save evaluation plots.

    Args:
        results: dict from evaluate_all
        all_diagnostics: list of (dataset_dict, diagnostics_dict)
        save_dir: directory to save figures
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

    # 2. Example xi(k) and Score(k) curves (first 4 datasets)
    n_examples = min(4, len(all_diagnostics))
    fig, axes = plt.subplots(n_examples, 2, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]
    for i in range(n_examples):
        ds, diag = all_diagnostics[i]
        k_grid = diag['k_grid']

        axes[i, 0].plot(k_grid, diag['xi_series'])
        axes[i, 0].axvline(diag['k_star'], color='r', ls='--', label=f'k*={diag["k_star"]}')
        axes[i, 0].set_ylabel('xi_hat')
        axes[i, 0].set_title(f'{ds["dist_type"]} n={ds["n"]}')
        axes[i, 0].legend()

        axes[i, 1].plot(k_grid, diag['total_score'])
        axes[i, 1].axvline(diag['k_star'], color='r', ls='--', label=f'k*={diag["k_star"]}')
        axes[i, 1].set_ylabel('Total Score')
        axes[i, 1].legend()

    axes[-1, 0].set_xlabel('k')
    axes[-1, 1].set_xlabel('k')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'xi_and_score_curves.png'), dpi=150)
    plt.close(fig)

    logger.info("Figures saved to %s", save_dir)
