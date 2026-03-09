"""VaR backtesting evaluation for real financial data.

Implements VaR computation from GPD fits, violation counting,
and Kupiec (1995) proportion-of-failures test.
"""

import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2

from src.evaluate import pot_quantile

logger = logging.getLogger(__name__)


def var_backtest(sorted_desc, k, xi, beta, n, p, future_returns):
    """Compute VaR(p) from GPD fit and count violations in future returns.

    Parameters
    ----------
    sorted_desc : ndarray
        Window samples sorted descending.
    k : int
        Number of exceedances.
    xi, beta : float
        GPD shape and scale parameters.
    n : int
        Sample size.
    p : float
        Quantile probability (e.g. 0.99).
    future_returns : ndarray
        Absolute log-returns in the backtest horizon.

    Returns
    -------
    dict with var_estimate, n_violations, violation_rate, n_future.
    """
    var_est = pot_quantile(sorted_desc, k, xi, beta, n, p)
    violations = future_returns > var_est
    n_violations = int(violations.sum())
    n_future = len(future_returns)
    violation_rate = n_violations / n_future if n_future > 0 else float('nan')

    return {
        "var_estimate": var_est,
        "n_violations": n_violations,
        "violation_rate": violation_rate,
        "n_future": n_future,
    }


def kupiec_test(violation_rate, n, p):
    """Kupiec (1995) proportion-of-failures (POF) test.

    Tests H0: true violation rate = 1-p against H1: != 1-p.

    Parameters
    ----------
    violation_rate : float
        Observed violation rate.
    n : int
        Number of future observations.
    p : float
        VaR confidence level (e.g. 0.99).

    Returns
    -------
    dict with statistic, p_value, reject_5pct.
    """
    expected_rate = 1 - p
    x = round(violation_rate * n)

    # Avoid log(0)
    eps = 1e-10
    vr = np.clip(violation_rate, eps, 1 - eps)
    er = np.clip(expected_rate, eps, 1 - eps)

    # Log-likelihood ratio statistic
    lr = 2 * (x * np.log(vr / er) + (n - x) * np.log((1 - vr) / (1 - er)))
    lr = max(lr, 0.0)  # numerical safety

    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        "statistic": lr,
        "p_value": p_value,
        "reject_5pct": p_value < 0.05,
    }


def _get_k_and_params(diag, k_value):
    """Look up GPD params for a given k in the diagnostics grid."""
    k_grid = np.asarray(diag["k_grid"])
    k_idx = np.searchsorted(k_grid, k_value)
    k_idx = min(k_idx, len(diag["params"]) - 1)
    xi, beta = diag["params"][k_idx]
    actual_k = int(k_grid[k_idx])
    return actual_k, xi, beta


def evaluate_real(test_data, diagnostics_list, k_pred, k_baseline,
                  returns_lookup, config):
    """Full VaR backtesting evaluation.

    Parameters
    ----------
    test_data : list[dict]
        Test-set window dicts (with ticker, series_end_idx).
    diagnostics_list : list[dict]
        Corresponding diagnostics dicts.
    k_pred : ndarray
        CNN-predicted k values.
    k_baseline : ndarray
        Baseline k* values from scoring function.
    returns_lookup : dict
        Per-ticker dict with 'abs_returns' array.
    config : dict
        Must have 'realdata.backtest_horizon' and 'evaluate.quantile_p'.

    Returns
    -------
    dict with per-method violation rates, Kupiec test results, and details.
    """
    backtest_horizon = config["realdata"]["backtest_horizon"]
    p = config["evaluate"]["quantile_p"]

    methods = {
        "cnn": k_pred,
        "baseline_k_star": k_baseline,
    }

    # Add fixed k=sqrt(n) baseline
    sqrt_k = np.array([int(np.sqrt(ds["n"])) for ds in test_data])
    methods["fixed_sqrt_n"] = sqrt_k

    results = {m: {"violations": [], "var_estimates": [], "n_future_list": []}
               for m in methods}
    results["historical_sim"] = {"violations": [], "var_estimates": [], "n_future_list": []}

    n_skipped = 0

    for i, (ds, diag) in enumerate(zip(test_data, diagnostics_list)):
        ticker = ds["ticker"]
        series_end_idx = ds["series_end_idx"]
        full_returns = returns_lookup[ticker]["abs_returns"]

        # Extract future returns
        future_start = series_end_idx
        future_end = series_end_idx + backtest_horizon
        if future_end > len(full_returns):
            n_skipped += 1
            continue

        future_returns = full_returns[future_start:future_end]
        sorted_desc = np.sort(ds["samples"])[::-1]
        n = ds["n"]

        # GPD-based methods
        for method_name, k_values in methods.items():
            k = int(k_values[i])
            _, xi, beta = _get_k_and_params(diag, k)

            if np.isnan(xi) or np.isnan(beta):
                continue

            bt = var_backtest(sorted_desc, k, xi, beta, n, p, future_returns)
            results[method_name]["violations"].append(bt["violation_rate"])
            results[method_name]["var_estimates"].append(bt["var_estimate"])
            results[method_name]["n_future_list"].append(bt["n_future"])

        # Historical simulation baseline (empirical quantile)
        hist_var = np.quantile(ds["samples"], p)
        hist_violations = future_returns > hist_var
        hist_vr = hist_violations.sum() / len(future_returns)
        results["historical_sim"]["violations"].append(hist_vr)
        results["historical_sim"]["var_estimates"].append(hist_var)
        results["historical_sim"]["n_future_list"].append(len(future_returns))

    if n_skipped > 0:
        logger.info("Skipped %d windows with insufficient future data", n_skipped)

    # Aggregate
    summary = {}
    expected_rate = 1 - p
    for method_name, data in results.items():
        vr_arr = np.array(data["violations"])
        if len(vr_arr) == 0:
            summary[method_name] = {"mean_violation_rate": float('nan'), "n_windows": 0}
            continue

        mean_vr = float(vr_arr.mean())
        total_violations = sum(
            round(vr * nf) for vr, nf in zip(data["violations"], data["n_future_list"])
        )
        total_obs = sum(data["n_future_list"])
        overall_vr = total_violations / total_obs if total_obs > 0 else float('nan')

        kup = kupiec_test(overall_vr, total_obs, p)

        summary[method_name] = {
            "mean_violation_rate": mean_vr,
            "overall_violation_rate": overall_vr,
            "expected_rate": expected_rate,
            "n_windows": len(vr_arr),
            "kupiec": kup,
        }

    return {"summary": summary, "details": results}


def plot_real_results(results, save_dir):
    """Generate bar chart of violation rates and save to disk.

    Parameters
    ----------
    results : dict
        Output of evaluate_real().
    save_dir : str
        Directory for figures.
    """
    os.makedirs(save_dir, exist_ok=True)
    summary = results["summary"]

    # 1. Violation rate bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = sorted(summary.keys())
    vr_values = [summary[m].get("overall_violation_rate", float('nan')) for m in methods]
    expected = summary[methods[0]].get("expected_rate", 0.01) if methods else 0.01

    colors = []
    for m in methods:
        if m == "cnn":
            colors.append("#2196F3")
        elif m == "baseline_k_star":
            colors.append("#4CAF50")
        else:
            colors.append("#9E9E9E")

    bars = ax.bar(methods, vr_values, color=colors)
    ax.axhline(expected, color='red', ls='--', lw=1.5, label=f'Expected = {expected:.3f}')
    ax.set_ylabel('Violation Rate')
    ax.set_title('VaR Backtest: Violation Rates by Method')
    ax.legend()

    for bar, val in zip(bars, vr_values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'violation_rates.png'), dpi=150)
    plt.close(fig)

    # 2. VaR time series for CNN method (if available)
    details = results.get("details", {})
    if "cnn" in details and details["cnn"]["var_estimates"]:
        fig, ax = plt.subplots(figsize=(12, 4))
        var_est = details["cnn"]["var_estimates"]
        ax.plot(var_est, lw=0.8, label='CNN VaR estimate')
        if "baseline_k_star" in details and details["baseline_k_star"]["var_estimates"]:
            ax.plot(details["baseline_k_star"]["var_estimates"], lw=0.8,
                    alpha=0.7, label='Baseline k* VaR')
        ax.set_xlabel('Test Window Index')
        ax.set_ylabel('VaR Estimate')
        ax.set_title('VaR Estimates Over Time')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'var_time_series.png'), dpi=150)
        plt.close(fig)

    logger.info("Real-data figures saved to %s", save_dir)
