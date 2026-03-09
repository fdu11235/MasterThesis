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


def christoffersen_test(violations_binary):
    """Christoffersen (1998) conditional coverage test.

    Tests H0: violations are independent (Markov property) using the
    independence LR statistic based on transition counts between
    consecutive days.

    Parameters
    ----------
    violations_binary : array-like
        Binary array (1 = VaR exceeded, 0 = not).

    Returns
    -------
    dict with lr_ind (independence LR), lr_cc (joint conditional coverage LR),
    p_value_ind, p_value_cc, reject_ind_5pct, reject_cc_5pct.
    """
    v = np.asarray(violations_binary, dtype=int)
    if len(v) < 2:
        return {"lr_ind": float('nan'), "lr_cc": float('nan'),
                "p_value_ind": float('nan'), "p_value_cc": float('nan'),
                "reject_ind_5pct": False, "reject_cc_5pct": False}

    # Count transitions
    n00 = n01 = n10 = n11 = 0
    for t in range(len(v) - 1):
        if v[t] == 0 and v[t + 1] == 0:
            n00 += 1
        elif v[t] == 0 and v[t + 1] == 1:
            n01 += 1
        elif v[t] == 1 and v[t + 1] == 0:
            n10 += 1
        else:
            n11 += 1

    eps = 1e-10

    # Transition probabilities
    pi01 = n01 / (n00 + n01 + eps)
    pi11 = n11 / (n10 + n11 + eps)

    # Unconditional probability
    n1 = n01 + n11
    n0 = n00 + n10
    pi = n1 / (n0 + n1 + eps)

    pi01 = np.clip(pi01, eps, 1 - eps)
    pi11 = np.clip(pi11, eps, 1 - eps)
    pi = np.clip(pi, eps, 1 - eps)

    # Independence LR statistic
    lr_ind = 2 * (
        n00 * np.log((1 - pi01) / (1 - pi + eps) + eps)
        + n01 * np.log(pi01 / (pi + eps) + eps)
        + n10 * np.log((1 - pi11) / (1 - pi + eps) + eps)
        + n11 * np.log(pi11 / (pi + eps) + eps)
    )
    lr_ind = max(lr_ind, 0.0)

    # Kupiec (POF) LR for the joint test
    n_total = len(v)
    x = v.sum()
    vr = np.clip(x / n_total, eps, 1 - eps)
    # We don't have p here, so joint test = lr_ind + lr_pof
    # But we just report independence separately
    p_value_ind = 1 - chi2.cdf(lr_ind, df=1)

    return {
        "lr_ind": lr_ind,
        "p_value_ind": p_value_ind,
        "reject_ind_5pct": p_value_ind < 0.05,
        "n_transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
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

    results = {m: {"violations": [], "var_estimates": [], "n_future_list": [], "tickers": [],
                    "violations_binary": []}
               for m in methods}
    results["historical_sim"] = {"violations": [], "var_estimates": [], "n_future_list": [], "tickers": [],
                                  "violations_binary": []}

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
            results[method_name]["tickers"].append(ticker)
            results[method_name]["violations_binary"].extend(
                (future_returns > bt["var_estimate"]).astype(int).tolist()
            )

        # Historical simulation baseline (empirical quantile)
        hist_var = np.quantile(ds["samples"], p)
        hist_violations = future_returns > hist_var
        hist_vr = hist_violations.sum() / len(future_returns)
        results["historical_sim"]["violations"].append(hist_vr)
        results["historical_sim"]["var_estimates"].append(hist_var)
        results["historical_sim"]["n_future_list"].append(len(future_returns))
        results["historical_sim"]["tickers"].append(ticker)
        results["historical_sim"]["violations_binary"].extend(
            (future_returns > hist_var).astype(int).tolist()
        )

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

        # Christoffersen independence test on binary violation sequence
        chris = christoffersen_test(data["violations_binary"])

        summary[method_name] = {
            "mean_violation_rate": mean_vr,
            "overall_violation_rate": overall_vr,
            "expected_rate": expected_rate,
            "n_windows": len(vr_arr),
            "kupiec": kup,
            "christoffersen": chris,
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

    # 3. Per-ticker violation rate box plot across methods
    details = results.get("details", {})
    plot_methods = ["cnn", "baseline_k_star", "fixed_sqrt_n", "historical_sim"]
    plot_methods = [m for m in plot_methods if m in details and details[m]["tickers"]]

    if plot_methods:
        # Collect all unique tickers
        all_tickers = sorted(set(details[plot_methods[0]]["tickers"]))

        if len(all_tickers) > 1:
            fig, axes = plt.subplots(1, len(plot_methods), figsize=(4 * len(plot_methods), 5),
                                     sharey=True)
            if len(plot_methods) == 1:
                axes = [axes]

            expected = results["summary"][plot_methods[0]].get("expected_rate", 0.01)

            for ax, method in zip(axes, plot_methods):
                tickers_list = details[method]["tickers"]
                vr_list = details[method]["violations"]

                # Group violations by ticker
                ticker_groups = {}
                for t, vr in zip(tickers_list, vr_list):
                    ticker_groups.setdefault(t, []).append(vr)

                sorted_tickers = sorted(ticker_groups.keys())
                box_data = [ticker_groups[t] for t in sorted_tickers]
                tick_labels = [f'{t}\n(n={len(ticker_groups[t])})' for t in sorted_tickers]

                bp = ax.boxplot(box_data, tick_labels=tick_labels, patch_artist=True,
                                medianprops=dict(color='red', lw=1.5))
                for patch in bp['boxes']:
                    patch.set_facecolor('#2196F3' if method == 'cnn' else
                                        '#4CAF50' if method == 'baseline_k_star' else '#9E9E9E')
                    patch.set_alpha(0.6)
                ax.axhline(expected, color='red', ls='--', lw=1, alpha=0.7)
                ax.set_title(method.replace('_', ' ').title(), fontsize=10)
                ax.tick_params(axis='x', rotation=30)

            axes[0].set_ylabel('Violation Rate')
            fig.suptitle('Per-Ticker Violation Rates by Method', fontsize=12, y=1.02)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, 'violation_rates_by_ticker.png'), dpi=150,
                        bbox_inches='tight')
            plt.close(fig)

    logger.info("Real-data figures saved to %s", save_dir)
