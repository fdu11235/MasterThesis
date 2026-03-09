import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from src.synthetic import (
    _generate_student_t, _generate_pareto,
    _generate_lognormal_pareto_mix, _generate_two_pareto,
)

logger = logging.getLogger(__name__)

_mc_quantile_cache = {}
_mc_es_cache = {}


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


def _mc_es(dist_type, dist_params, p, n_mc=10_000_000, seed=99999):
    """Monte Carlo Expected Shortfall: E[X | X > VaR(p)]."""
    rng = np.random.RandomState(seed)
    generators = {
        'student_t': lambda: np.abs(stats.t.rvs(df=dist_params['df'], size=n_mc, random_state=rng)),
        'pareto': lambda: stats.pareto.rvs(b=dist_params['alpha'], size=n_mc, random_state=rng),
        'lognormal_pareto_mix': lambda: _generate_lognormal_pareto_mix(rng, n_mc, **dist_params),
        'two_pareto': lambda: _generate_two_pareto(rng, n_mc, **dist_params),
    }
    if dist_type not in generators:
        raise ValueError(f"No MC ES for {dist_type}")
    samples = generators[dist_type]()
    q = np.quantile(samples, p)
    tail = samples[samples > q]
    return float(tail.mean()) if len(tail) > 0 else float(q)


def true_es(dist_type, dist_params, p):
    """True Expected Shortfall via Monte Carlo: E[X | X > VaR(p)].

    Uses 10M samples, cached by (dist_type, params, p).
    Matches the MC approach in run_diff_pipeline.py for consistency.
    """
    key = _params_key(dist_type, dist_params, p)
    if key not in _mc_es_cache:
        logger.info("Computing MC ES for %s (p=%s) ...", dist_type, p)
        _mc_es_cache[key] = _mc_es(dist_type, dist_params, p)
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
        n = ds['n']
        q_est.append(pot_quantile(sorted_desc, k, xi, beta, n, p))
        q_true.append(true_quantile(ds['dist_type'], ds['params'], p))
        es_est_list.append(pot_es(sorted_desc, k, xi, beta, n, p))
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

        # Store raw data for plotting
        results['_q_est'] = q_est
        results['_q_true'] = q_true
        results['_es_est'] = es_est_list
        results['_es_true'] = es_true_list
        results['_dist_types'] = dist_types
        results['_dist_params'] = dist_params
        results['_quantile_p'] = p
    else:
        results['quantile_rmse'] = float('nan')
        results['relative_rmse'] = float('nan')
        results['es_rmse'] = float('nan')
        results['es_relative_rmse'] = float('nan')
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
                return f'Pareto(α={dp.get("alpha", "?")})'
            elif dt == 'lognormal_pareto_mix':
                return f'LN-Par(α={dp.get("pareto_alpha", "?")})'
            elif dt == 'two_pareto':
                return f'2Par({dp.get("alpha1", "?")},{dp.get("alpha2", "?")})'
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

    logger.info("Figures saved to %s", save_dir)
