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
from scipy.stats import chi2, ttest_1samp

from src.evaluate import pot_quantile, pot_es

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
    es_est = pot_es(sorted_desc, k, xi, beta, n, p)
    violations = future_returns > var_est
    n_violations = int(violations.sum())
    n_future = len(future_returns)
    violation_rate = n_violations / n_future if n_future > 0 else float('nan')

    return {
        "var_estimate": var_est,
        "es_estimate": es_est,
        "n_violations": n_violations,
        "violation_rate": violation_rate,
        "n_future": n_future,
    }


def var_backtest_signsplit(sorted_desc, k, xi, beta, n, p,
                          future_signed, tail_mode, forecast_vol=None):
    """Compute VaR from GPD fit and count violations for a specific tail.

    Only future returns matching the requested sign are considered.
    When *forecast_vol* is given (GARCH path), VaR/ES on standardised
    residuals are scaled back to return units via ``sigma_t * VaR_z``.

    Parameters
    ----------
    sorted_desc : ndarray
        Window samples sorted descending (positive, sign-filtered).
    k : int
        Number of exceedances.
    xi, beta : float
        GPD shape and scale parameters.
    n : int
        Sample size of the sign-filtered window.
    p : float
        Quantile probability (e.g. 0.99).
    future_signed : ndarray
        Signed log-returns in the backtest horizon.
    tail_mode : str
        ``"loss"`` or ``"profit"``.
    forecast_vol : ndarray or None
        GARCH-forecasted sigma for each future day. If given, VaR/ES are
        multiplied by forecast_vol to convert from z-score to return units.

    Returns
    -------
    dict with var_estimate, es_estimate, n_violations, violation_rate,
    n_future (count of same-sign future days).
    """
    var_z = pot_quantile(sorted_desc, k, xi, beta, n, p)
    es_z = pot_es(sorted_desc, k, xi, beta, n, p)

    if tail_mode == "loss":
        mask = future_signed < 0
        future_magnitudes = np.abs(future_signed[mask])
    else:  # profit
        mask = future_signed > 0
        future_magnitudes = future_signed[mask]

    if forecast_vol is not None:
        horizon = min(len(future_signed), len(forecast_vol))
        fvol = forecast_vol[:horizon]
        # Scale z-score VaR/ES by per-day forecast volatility,
        # then filter to same-sign days
        fvol_filtered = fvol[mask[:horizon]]
        var_per_day = fvol_filtered * var_z
        es_per_day = fvol_filtered * es_z
        var_est = float(np.mean(var_per_day)) if len(var_per_day) > 0 else var_z
        es_est = float(np.mean(es_per_day)) if len(es_per_day) > 0 else es_z
    else:
        var_per_day = None
        var_est = var_z
        es_est = es_z

    n_future = len(future_magnitudes)
    if n_future > 0:
        if var_per_day is not None:
            violations = future_magnitudes[:len(var_per_day)] > var_per_day
        else:
            violations = future_magnitudes > var_est
        n_violations = int(violations.sum())
        violation_rate = n_violations / n_future
    else:
        n_violations = 0
        violation_rate = float("nan")

    return {
        "var_estimate": var_est,
        "es_estimate": es_est,
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


def mcneil_frey_test(future_returns, var_est, es_est):
    """McNeil & Frey (2000) exceedance residual test.

    When VaR is violated, test if mean excess over ES is zero.

    Parameters
    ----------
    future_returns : ndarray
        Absolute returns in the backtest horizon.
    var_est : float or ndarray
        VaR estimate(s).
    es_est : float or ndarray
        ES estimate(s).

    Returns
    -------
    dict with t_stat, p_value, reject_5pct, n_violations, mean_residual.
    """
    future_returns = np.asarray(future_returns)
    var_est = np.asarray(var_est)
    es_est = np.asarray(es_est)

    mask = future_returns > var_est
    n_violations = int(mask.sum())
    if n_violations < 2:
        return {
            "t_stat": float('nan'), "p_value": float('nan'),
            "reject_5pct": False, "n_violations": n_violations,
        }
    residuals = (future_returns[mask] - es_est[mask] if es_est.ndim > 0
                 else future_returns[mask] - es_est) / (
                 es_est[mask] if es_est.ndim > 0 else es_est)
    t_stat, p_value = ttest_1samp(residuals, 0)
    return {
        "t_stat": float(t_stat), "p_value": float(p_value),
        "reject_5pct": p_value < 0.05,
        "n_violations": n_violations,
        "mean_residual": float(residuals.mean()),
    }


def var_backtest_garch(sorted_desc, k, xi, beta, n, p, future_returns, forecast_vol):
    """GARCH-conditional VaR/ES backtest.

    Computes VaR/ES on standardized residuals, then scales by GARCH-forecasted
    volatility to get time-varying VaR_t and ES_t.

    Parameters
    ----------
    sorted_desc : ndarray
        Standardized residuals sorted descending.
    k : int
        Number of exceedances.
    xi, beta : float
        GPD shape and scale parameters.
    n : int
        Sample size.
    p : float
        Quantile probability (e.g. 0.99).
    future_returns : ndarray
        Raw absolute returns in the backtest horizon.
    forecast_vol : ndarray
        GARCH-forecasted sigma for each future day.

    Returns
    -------
    dict with var_z, es_z (residual-level), var_t, es_t (arrays),
    n_violations, violation_rate, n_future, violations_binary.
    """
    var_z = pot_quantile(sorted_desc, k, xi, beta, n, p)
    es_z = pot_es(sorted_desc, k, xi, beta, n, p)

    # Truncate forecast_vol to match future_returns length
    horizon = min(len(future_returns), len(forecast_vol))
    future_returns = future_returns[:horizon]
    fvol = forecast_vol[:horizon]

    var_t = fvol * var_z  # time-varying VaR
    es_t = fvol * es_z    # time-varying ES

    violations = future_returns > var_t
    n_violations = int(violations.sum())
    n_future = len(future_returns)
    violation_rate = n_violations / n_future if n_future > 0 else float('nan')

    return {
        "var_z": var_z,
        "es_z": es_z,
        "var_t": var_t,
        "es_t": es_t,
        "n_violations": n_violations,
        "violation_rate": violation_rate,
        "n_future": n_future,
        "violations_binary": violations.astype(int),
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
                  returns_lookup, config,
                  garch_test_data=None, garch_diagnostics_list=None,
                  garch_k_pred=None, garch_k_baseline=None):
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
    garch_test_data : list[dict], optional
        GARCH-filtered test-set window dicts.
    garch_diagnostics_list : list[dict], optional
        Corresponding GARCH diagnostics dicts.
    garch_k_pred : ndarray, optional
        CNN-predicted k values for GARCH-filtered data.
    garch_k_baseline : ndarray, optional
        Baseline k* values for GARCH-filtered data.

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

    def _empty_bucket():
        return {"violations": [], "var_estimates": [], "es_estimates": [],
                "n_future_list": [], "tickers": [], "violations_binary": [],
                "future_returns_all": [], "var_all": [], "es_all": []}

    results = {m: _empty_bucket() for m in methods}
    results["historical_sim"] = _empty_bucket()

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
            results[method_name]["es_estimates"].append(bt["es_estimate"])
            results[method_name]["n_future_list"].append(bt["n_future"])
            results[method_name]["tickers"].append(ticker)
            results[method_name]["violations_binary"].extend(
                (future_returns > bt["var_estimate"]).astype(int).tolist()
            )
            results[method_name]["future_returns_all"].extend(future_returns.tolist())
            results[method_name]["var_all"].extend([bt["var_estimate"]] * len(future_returns))
            results[method_name]["es_all"].extend([bt["es_estimate"]] * len(future_returns))

        # Historical simulation baseline (empirical quantile)
        hist_var = np.quantile(ds["samples"], p)
        samples_above = ds["samples"][ds["samples"] > hist_var]
        hist_es = float(samples_above.mean()) if len(samples_above) > 0 else hist_var
        hist_violations = future_returns > hist_var
        hist_vr = hist_violations.sum() / len(future_returns)
        results["historical_sim"]["violations"].append(hist_vr)
        results["historical_sim"]["var_estimates"].append(hist_var)
        results["historical_sim"]["es_estimates"].append(hist_es)
        results["historical_sim"]["n_future_list"].append(len(future_returns))
        results["historical_sim"]["tickers"].append(ticker)
        results["historical_sim"]["violations_binary"].extend(
            (future_returns > hist_var).astype(int).tolist()
        )
        results["historical_sim"]["future_returns_all"].extend(future_returns.tolist())
        results["historical_sim"]["var_all"].extend([hist_var] * len(future_returns))
        results["historical_sim"]["es_all"].extend([hist_es] * len(future_returns))

    # ── GARCH-conditional methods ──────────────────────────────────────────
    if garch_test_data is not None and garch_diagnostics_list is not None:
        garch_methods = {}
        if garch_k_baseline is not None:
            garch_methods["baseline_garch"] = garch_k_baseline
        if garch_k_pred is not None:
            garch_methods["cnn_garch"] = garch_k_pred

        for m in garch_methods:
            results[m] = _empty_bucket()

        n_garch_skipped = 0
        for i, (ds, diag) in enumerate(zip(garch_test_data, garch_diagnostics_list)):
            ticker = ds["ticker"]
            series_end_idx = ds["series_end_idx"]
            full_returns = returns_lookup[ticker]["abs_returns"]
            forecast_vol = ds.get("garch_forecast_vol")

            future_start = series_end_idx
            future_end = series_end_idx + backtest_horizon
            if future_end > len(full_returns) or forecast_vol is None:
                n_garch_skipped += 1
                continue

            future_returns = full_returns[future_start:future_end]
            sorted_desc = np.sort(ds["samples"])[::-1]
            n = ds["n"]

            for method_name, k_values in garch_methods.items():
                k = int(k_values[i])
                _, xi, beta = _get_k_and_params(diag, k)

                if np.isnan(xi) or np.isnan(beta):
                    continue

                bt = var_backtest_garch(sorted_desc, k, xi, beta, n, p,
                                        future_returns, forecast_vol)
                results[method_name]["violations"].append(bt["violation_rate"])
                results[method_name]["var_estimates"].append(float(np.mean(bt["var_t"])))
                results[method_name]["es_estimates"].append(float(np.mean(bt["es_t"])))
                results[method_name]["n_future_list"].append(bt["n_future"])
                results[method_name]["tickers"].append(ticker)
                results[method_name]["violations_binary"].extend(
                    bt["violations_binary"].tolist()
                )
                results[method_name]["future_returns_all"].extend(future_returns[:bt["n_future"]].tolist())
                results[method_name]["var_all"].extend(bt["var_t"].tolist())
                results[method_name]["es_all"].extend(bt["es_t"].tolist())

        if n_garch_skipped > 0:
            logger.info("Skipped %d GARCH windows with insufficient future data",
                        n_garch_skipped)

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

        # ES: mean estimate and McNeil-Frey test
        es_arr = np.array(data["es_estimates"])
        mean_es = float(es_arr.mean()) if len(es_arr) > 0 else float('nan')

        # Pooled McNeil-Frey test across all windows
        mf = mcneil_frey_test(
            np.array(data["future_returns_all"]),
            np.array(data["var_all"]),
            np.array(data["es_all"]),
        ) if data["future_returns_all"] else {
            "t_stat": float('nan'), "p_value": float('nan'),
            "reject_5pct": False, "n_violations": 0,
        }

        summary[method_name] = {
            "mean_violation_rate": mean_vr,
            "overall_violation_rate": overall_vr,
            "expected_rate": expected_rate,
            "n_windows": len(vr_arr),
            "kupiec": kup,
            "christoffersen": chris,
            "mean_es_estimate": mean_es,
            "mcneil_frey": mf,
        }

    # ── Multi-level VaR coverage ────────────────────────────────────────────
    p_levels = [0.95, 0.975, 0.99, 0.995]
    multi_level = {}
    for method_name, k_values in methods.items():
        multi_level[method_name] = {}
        for pl in p_levels:
            viol_count = 0
            total_count = 0
            for i, (ds, diag) in enumerate(zip(test_data, diagnostics_list)):
                ticker = ds["ticker"]
                series_end_idx = ds["series_end_idx"]
                full_returns = returns_lookup[ticker]["abs_returns"]
                future_start = series_end_idx
                future_end = series_end_idx + backtest_horizon
                if future_end > len(full_returns):
                    continue
                future_returns = full_returns[future_start:future_end]
                sorted_desc = np.sort(ds["samples"])[::-1]
                n = ds["n"]
                k = int(k_values[i])
                _, xi, beta = _get_k_and_params(diag, k)
                if np.isnan(xi) or np.isnan(beta):
                    continue
                var_est = pot_quantile(sorted_desc, k, xi, beta, n, pl)
                n_viol = int((future_returns > var_est).sum())
                viol_count += n_viol
                total_count += len(future_returns)
            if total_count > 0:
                obs_rate = viol_count / total_count
                kup = kupiec_test(obs_rate, total_count, pl)
                multi_level[method_name][pl] = {
                    "violation_rate": obs_rate,
                    "expected": 1 - pl,
                    "kupiec": kup,
                }
            else:
                multi_level[method_name][pl] = {
                    "violation_rate": float('nan'),
                    "expected": 1 - pl,
                    "kupiec": {},
                }

    # Historical simulation multi-level
    multi_level["historical_sim"] = {}
    for pl in p_levels:
        viol_count = 0
        total_count = 0
        for i, (ds, diag) in enumerate(zip(test_data, diagnostics_list)):
            ticker = ds["ticker"]
            series_end_idx = ds["series_end_idx"]
            full_returns = returns_lookup[ticker]["abs_returns"]
            future_start = series_end_idx
            future_end = series_end_idx + backtest_horizon
            if future_end > len(full_returns):
                continue
            future_returns = full_returns[future_start:future_end]
            hist_var = np.quantile(ds["samples"], pl)
            n_viol = int((future_returns > hist_var).sum())
            viol_count += n_viol
            total_count += len(future_returns)
        if total_count > 0:
            obs_rate = viol_count / total_count
            kup = kupiec_test(obs_rate, total_count, pl)
            multi_level["historical_sim"][pl] = {
                "violation_rate": obs_rate,
                "expected": 1 - pl,
                "kupiec": kup,
            }
        else:
            multi_level["historical_sim"][pl] = {
                "violation_rate": float('nan'),
                "expected": 1 - pl,
                "kupiec": {},
            }

    return {"summary": summary, "details": results, "multi_level": multi_level}


def plot_rolling_violations(results, save_dir):
    """Plot rolling average violation rate per method over test windows.

    Parameters
    ----------
    results : dict
        Output of evaluate_real().
    save_dir : str
        Directory for figures.
    """
    os.makedirs(save_dir, exist_ok=True)
    details = results.get("details", {})
    summary = results.get("summary", {})

    fig, ax = plt.subplots(figsize=(10, 5))
    window_size = 20  # rolling window

    for method in sorted(details.keys()):
        violations = details[method].get("violations", [])
        if len(violations) < window_size:
            continue
        vr_arr = np.array(violations)
        rolling_avg = np.convolve(vr_arr, np.ones(window_size) / window_size, mode='valid')
        ax.plot(rolling_avg, lw=1.2, alpha=0.8, label=method)

    expected = list(summary.values())[0].get("expected_rate", 0.01) if summary else 0.01
    ax.axhline(expected, color='red', ls='--', lw=1, label=f'Expected={expected:.3f}')
    ax.set_xlabel("Window Index")
    ax.set_ylabel(f"Rolling Violation Rate (window={window_size})")
    ax.set_title("Rolling Average Violation Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "rolling_violations.png"), dpi=150)
    plt.close(fig)


def plot_multi_level_coverage(results, save_dir):
    """Grouped bar chart: observed vs expected violation rate at each p-level.

    Parameters
    ----------
    results : dict
        Output of evaluate_real() (must contain 'multi_level').
    save_dir : str
        Directory for figures.
    """
    multi_level = results.get("multi_level")
    if not multi_level:
        return
    os.makedirs(save_dir, exist_ok=True)

    methods = sorted(multi_level.keys())
    p_levels = sorted(next(iter(multi_level.values())).keys())
    n_methods = len(methods)
    n_levels = len(p_levels)

    fig, ax = plt.subplots(figsize=(max(8, n_levels * 2.5), 5))
    x = np.arange(n_levels)
    width = 0.8 / (n_methods + 1)  # +1 for expected bars

    # Expected bars
    expected_rates = [1 - pl for pl in p_levels]
    ax.bar(x - 0.4 + width / 2, expected_rates, width, label="Expected",
           color='red', alpha=0.3, edgecolor='red')

    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
    for j, method in enumerate(methods):
        obs_rates = []
        for pl in p_levels:
            info = multi_level[method].get(pl, {})
            obs_rates.append(info.get("violation_rate", float('nan')))
        ax.bar(x - 0.4 + (j + 1.5) * width, obs_rates, width,
               label=method, color=colors[j], edgecolor='black', lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"p={pl}" for pl in p_levels])
    ax.set_ylabel("Violation Rate")
    ax.set_title("Multi-Level VaR Coverage")
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "multi_level_coverage.png"), dpi=150)
    plt.close(fig)


def evaluate_real_signsplit(test_data, diagnostics_list, k_pred, k_baseline,
                           returns_lookup, config, tail_mode):
    """VaR backtesting for sign-split (loss or profit) tails.

    Parameters
    ----------
    test_data : list[dict]
        Sign-split test-set window dicts (with ticker, series_end_idx, tail_mode).
    diagnostics_list : list[dict]
        Corresponding diagnostics dicts.
    k_pred : ndarray
        CNN-predicted k values.
    k_baseline : ndarray
        Baseline k* values from scoring function.
    returns_lookup : dict
        Per-ticker dict with 'signed_returns' array.
    config : dict
        Must have 'realdata.backtest_horizon' and 'evaluate.quantile_p'.
    tail_mode : str
        ``"loss"`` or ``"profit"``.

    Returns
    -------
    dict with per-method violation rates and details.
    """
    backtest_horizon = config["realdata"]["backtest_horizon"]
    p = config["evaluate"]["quantile_p"]

    methods = {
        "cnn": k_pred,
        "baseline_k_star": k_baseline,
    }

    sqrt_k = np.array([int(np.sqrt(ds["n"])) for ds in test_data])
    methods["fixed_sqrt_n"] = sqrt_k

    def _empty_bucket():
        return {"violations": [], "var_estimates": [], "es_estimates": [],
                "n_future_list": [], "tickers": [],
                "violations_binary": [], "future_returns_all": [],
                "var_all": [], "es_all": []}

    results = {m: _empty_bucket() for m in methods}
    results["historical_sim"] = _empty_bucket()
    n_skipped = 0

    for i, (ds, diag) in enumerate(zip(test_data, diagnostics_list)):
        ticker = ds["ticker"]
        series_end_idx = ds.get("series_end_idx", 0)
        signed_returns = returns_lookup[ticker].get("signed_returns")

        if signed_returns is None:
            n_skipped += 1
            continue

        future_start = series_end_idx
        future_end = series_end_idx + backtest_horizon
        if future_end > len(signed_returns):
            n_skipped += 1
            continue

        future_signed = signed_returns[future_start:future_end]

        # Extract future magnitudes for this tail
        if tail_mode == "loss":
            sign_mask = future_signed < 0
            future_mags = np.abs(future_signed[sign_mask])
        else:
            sign_mask = future_signed > 0
            future_mags = future_signed[sign_mask]

        sorted_desc = np.sort(ds["samples"])[::-1]
        n = ds["n"]

        for method_name, k_values in methods.items():
            k = int(k_values[i])
            _, xi, beta = _get_k_and_params(diag, k)

            if np.isnan(xi) or np.isnan(beta):
                continue

            fvol = ds.get("garch_forecast_vol")
            bt = var_backtest_signsplit(sorted_desc, k, xi, beta, n, p,
                                        future_signed, tail_mode,
                                        forecast_vol=fvol)
            results[method_name]["violations"].append(bt["violation_rate"])
            results[method_name]["var_estimates"].append(bt["var_estimate"])
            results[method_name]["es_estimates"].append(bt["es_estimate"])
            results[method_name]["n_future_list"].append(bt["n_future"])
            results[method_name]["tickers"].append(ticker)
            if len(future_mags) > 0:
                viol_binary = (future_mags > bt["var_estimate"]).astype(int)
                results[method_name]["violations_binary"].extend(viol_binary.tolist())
                results[method_name]["future_returns_all"].extend(future_mags.tolist())
                results[method_name]["var_all"].extend(
                    [bt["var_estimate"]] * len(future_mags))
                results[method_name]["es_all"].extend(
                    [bt["es_estimate"]] * len(future_mags))

        # Historical simulation
        hist_var = np.quantile(ds["samples"], p)
        samples_above = ds["samples"][ds["samples"] > hist_var]
        hist_es = float(samples_above.mean()) if len(samples_above) > 0 else hist_var

        if len(future_mags) > 0:
            hist_vr = (future_mags > hist_var).sum() / len(future_mags)
            hist_viol_binary = (future_mags > hist_var).astype(int)
            results["historical_sim"]["violations_binary"].extend(hist_viol_binary.tolist())
            results["historical_sim"]["future_returns_all"].extend(future_mags.tolist())
            results["historical_sim"]["var_all"].extend([hist_var] * len(future_mags))
            results["historical_sim"]["es_all"].extend([hist_es] * len(future_mags))
        else:
            hist_vr = float("nan")
        results["historical_sim"]["violations"].append(hist_vr)
        results["historical_sim"]["var_estimates"].append(hist_var)
        results["historical_sim"]["es_estimates"].append(hist_es)
        results["historical_sim"]["n_future_list"].append(len(future_mags))
        results["historical_sim"]["tickers"].append(ticker)

    # Summary statistics with full statistical tests
    summary = {}
    expected_rate = 1 - p
    for method_name, data in results.items():
        vr_arr = np.array([v for v in data["violations"] if not np.isnan(v)])
        if len(vr_arr) == 0:
            summary[method_name] = {"mean_violation_rate": float("nan"), "n_windows": 0}
            continue

        mean_vr = float(vr_arr.mean())
        total_violations = sum(
            round(vr * nf) for vr, nf in zip(data["violations"], data["n_future_list"])
            if not np.isnan(vr)
        )
        total_obs = sum(nf for vr, nf in zip(data["violations"], data["n_future_list"])
                        if not np.isnan(vr))
        overall_vr = total_violations / total_obs if total_obs > 0 else float("nan")

        kup = kupiec_test(overall_vr, total_obs, p)
        chris = christoffersen_test(data["violations_binary"])

        es_arr = np.array(data["es_estimates"])
        mean_es = float(es_arr.mean()) if len(es_arr) > 0 else float("nan")

        mf = mcneil_frey_test(
            np.array(data["future_returns_all"]),
            np.array(data["var_all"]),
            np.array(data["es_all"]),
        ) if data["future_returns_all"] else {
            "t_stat": float("nan"), "p_value": float("nan"),
            "reject_5pct": False, "n_violations": 0,
        }

        summary[method_name] = {
            "mean_violation_rate": mean_vr,
            "overall_violation_rate": overall_vr,
            "expected_rate": expected_rate,
            "n_windows": len(vr_arr),
            "mean_var": float(np.nanmean(data["var_estimates"])) if data["var_estimates"] else float("nan"),
            "kupiec": kup,
            "christoffersen": chris,
            "mean_es_estimate": mean_es,
            "mcneil_frey": mf,
        }

    logger.info("Sign-split (%s) backtesting: %d windows evaluated, %d skipped",
                tail_mode, len(test_data) - n_skipped, n_skipped)
    for method, s in summary.items():
        kup = s.get("kupiec", {})
        mf = s.get("mcneil_frey", {})
        logger.info("  %-20s: VR=%.4f, mean_VaR=%.4f, Kupiec reject=%s, MF reject=%s (p=%.4f)",
                     method, s.get("overall_violation_rate", float("nan")),
                     s.get("mean_var", float("nan")),
                     kup.get("reject_5pct", "N/A"),
                     mf.get("reject_5pct", "N/A"),
                     mf.get("p_value", float("nan")))

    return {"methods": results, "summary": summary, "tail_mode": tail_mode}


def plot_real_results(results, save_dir, history=None, garch_history=None):
    """Generate bar chart of violation rates and save to disk.

    Parameters
    ----------
    results : dict
        Output of evaluate_real().
    save_dir : str
        Directory for figures.
    history : dict, optional
        Training history for unconditional model.
    garch_history : dict, optional
        Training history for GARCH model.
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
        elif m == "cnn_garch":
            colors.append("#1565C0")
        elif m == "baseline_k_star":
            colors.append("#4CAF50")
        elif m == "baseline_garch":
            colors.append("#2E7D32")
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
    plot_methods = ["cnn", "cnn_garch", "baseline_k_star", "baseline_garch",
                    "fixed_sqrt_n", "historical_sim"]
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

    # New plots
    plot_rolling_violations(results, save_dir)
    plot_multi_level_coverage(results, save_dir)

    # Training curves (import from evaluate.py)
    from src.evaluate import plot_training_curves
    if history is not None:
        plot_training_curves(history, save_dir)
    if garch_history is not None:
        plot_training_curves(garch_history, os.path.join(save_dir, "garch"))

    logger.info("Real-data figures saved to %s", save_dir)
