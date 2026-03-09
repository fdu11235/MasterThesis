"""POT diagnostics and baseline scoring for GPD threshold selection."""

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.stats import genpareto

logger = logging.getLogger(__name__)


def candidate_k_grid(n: int, k_min: int, k_max_frac: float) -> NDArray:
    """Return candidate k values as an integer array."""
    return np.arange(k_min, int(np.floor(k_max_frac * n)) + 1)


def fit_gpd(sorted_desc: NDArray, k: int) -> tuple[float, float]:
    """Fit GPD to the top-k exceedances above the (k+1)-th order statistic.

    Returns (xi_hat, beta_hat) where xi is shape and beta is scale.
    On convergence failure returns (nan, nan).
    """
    exceedances = sorted_desc[:k] - sorted_desc[k]
    try:
        xi_hat, _, beta_hat = genpareto.fit(exceedances, floc=0)
    except Exception:
        logger.debug("GPD fit failed for k=%d", k)
        return (np.nan, np.nan)
    return (xi_hat, beta_hat)


def fit_all_k(sorted_desc: NDArray, k_grid: NDArray) -> NDArray:
    """Fit GPD for every k in k_grid.

    Returns an array of shape (len(k_grid), 2) with columns [xi, beta].
    """
    params = np.empty((len(k_grid), 2), dtype=float)
    for i, k in enumerate(k_grid):
        params[i] = fit_gpd(sorted_desc, k)
    return params


def score_stability(xi_series: NDArray, k_grid: NDArray, delta: int) -> NDArray:
    """Rolling variance of xi_hat in a window [i-delta, i+delta], truncated at boundaries."""
    n = len(xi_series)
    scores = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - delta)
        hi = min(n, i + delta + 1)
        scores[i] = np.nanvar(xi_series[lo:hi])
    return scores


def _anderson_darling_gpd(exceedances: NDArray, xi: float, beta: float) -> float:
    """Anderson-Darling statistic for GPD fit.

    AD = -n - (1/n) * sum_{i=1}^{n} (2i-1) * [ln(F(z_i)) + ln(1 - F(z_{n+1-i}))]
    where z is sorted ascending and F is the fitted GPD CDF.
    """
    n = len(exceedances)
    z = np.sort(exceedances)
    F = genpareto.cdf(z, xi, loc=0, scale=beta)
    # Clamp to avoid log(0)
    F = np.clip(F, 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    ad = -n - (1.0 / n) * np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1])))
    return ad


def score_gof(sorted_desc: NDArray, k_grid: NDArray, params: NDArray) -> NDArray:
    """Anderson-Darling statistic for each k. Returns large value when params are NaN."""
    scores = np.empty(len(k_grid), dtype=float)
    for i, k in enumerate(k_grid):
        xi, beta = params[i]
        if np.isnan(xi) or np.isnan(beta):
            scores[i] = 100.0
            continue
        exceedances = sorted_desc[:k] - sorted_desc[k]
        scores[i] = _anderson_darling_gpd(exceedances, xi, beta)
    return scores


def score_mean_excess(sorted_desc: NDArray, k_grid: NDArray) -> NDArray:
    """Mean excess linearity score for each k.

    For GPD-distributed exceedances, the mean excess function e(u) = E[X-u | X>u]
    is linear in u. We measure 1 - R² of a linear fit to the empirical mean excess
    at sub-thresholds within the exceedances. Lower = more linear = better GPD fit.
    """
    scores = np.empty(len(k_grid), dtype=float)
    for i, k in enumerate(k_grid):
        exceedances = sorted_desc[:k] - sorted_desc[k]
        if k < 10:
            scores[i] = 1.0
            continue
        # Compute mean excess at ~10 sub-thresholds within exceedances
        n_points = min(10, k // 2)
        thresholds = np.linspace(0, np.percentile(exceedances, 80), n_points + 1)[:-1]
        me_values = []
        me_thresholds = []
        for u in thresholds:
            above = exceedances[exceedances > u]
            if len(above) < 5:
                continue
            me_values.append(np.mean(above - u))
            me_thresholds.append(u)
        if len(me_values) < 3:
            scores[i] = 1.0
            continue
        # Fit linear regression, compute R²
        x = np.array(me_thresholds)
        y = np.array(me_values)
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        scores[i] = 1 - max(r2, 0.0)
    return scores


def score_penalty(k_grid: NDArray) -> NDArray:
    """Penalty score: 1 / sqrt(k)."""
    return 1.0 / np.sqrt(k_grid.astype(float))


def _min_max_normalize(s: NDArray) -> NDArray:
    """Min-max normalize an array to [0, 1]."""
    return (s - s.min()) / (s.max() - s.min() + 1e-10)


def hill_estimator(sorted_desc: NDArray, k_grid: NDArray) -> NDArray:
    """Hill estimator of the tail index for each k.

    H(k) = (1/k) * sum_{i=1}^{k} ln(X_{(i)} / X_{(k+1)})
    Non-parametric estimate, independent from GPD-based xi.
    """
    result = np.empty(len(k_grid), dtype=float)
    for i, k in enumerate(k_grid):
        threshold = sorted_desc[k]
        if threshold <= 0:
            result[i] = np.nan
            continue
        log_ratios = np.log(sorted_desc[:k] / threshold)
        result[i] = np.mean(log_ratios)
    return result


def qq_residual(sorted_desc: NDArray, k_grid: NDArray, params: NDArray) -> NDArray:
    """QQ-plot residual RMSE for each k.

    Compares empirical quantiles of exceedances vs fitted GPD quantiles.
    Returns RMSE of the QQ deviation, capturing fit quality differently from AD.
    """
    result = np.empty(len(k_grid), dtype=float)
    for i, k in enumerate(k_grid):
        xi, beta = params[i]
        if np.isnan(xi) or np.isnan(beta) or k < 5:
            result[i] = np.nan
            continue
        exceedances = sorted_desc[:k] - sorted_desc[k]
        exceedances_sorted = np.sort(exceedances)
        n_exc = len(exceedances_sorted)
        # Empirical quantiles
        empirical_q = exceedances_sorted
        # Theoretical GPD quantiles at the same probability points
        probs = (np.arange(1, n_exc + 1) - 0.5) / n_exc
        theoretical_q = genpareto.ppf(probs, xi, loc=0, scale=beta)
        # RMSE of QQ deviation
        result[i] = np.sqrt(np.mean((empirical_q - theoretical_q) ** 2))
    return result


def mean_excess_values(sorted_desc: NDArray, k_grid: NDArray) -> NDArray:
    """Raw mean of exceedances at each threshold k.

    e(k) = (1/k) * sum_{i=1}^{k} (X_{(i)} - X_{(k+1)})
    """
    result = np.empty(len(k_grid), dtype=float)
    for i, k in enumerate(k_grid):
        exceedances = sorted_desc[:k] - sorted_desc[k]
        result[i] = np.mean(exceedances)
    return result


def compute_baseline_k_star(
    sorted_desc: NDArray,
    k_grid: NDArray,
    delta: int,
    weights: tuple[float, float, float],
) -> tuple[int, dict]:
    """Combine stability, GoF, and penalty scores to select k*.

    Parameters
    ----------
    sorted_desc : array, sample sorted in descending order
    k_grid : candidate k values
    delta : half-window size for stability scoring
    weights : (w_stability, w_gof, w_penalty)

    Returns
    -------
    k_star : optimal k value
    diagnostics : dict with k_grid, params, xi_series, individual scores,
                  total_score, and k_star
    """
    params = fit_all_k(sorted_desc, k_grid)
    xi_series = params[:, 0]

    s_stab = score_stability(xi_series, k_grid, delta)
    s_gof = score_gof(sorted_desc, k_grid, params)
    s_me = score_mean_excess(sorted_desc, k_grid)
    s_pen = score_penalty(k_grid)

    # Min-max normalize each score to [0, 1]
    s_stab_n = _min_max_normalize(s_stab)
    s_gof_n = _min_max_normalize(s_gof)
    s_pen_n = _min_max_normalize(s_pen)

    w_stab, w_gof, w_pen = weights
    total = w_stab * s_stab_n + w_gof * s_gof_n + w_pen * s_pen_n

    idx = int(np.argmin(total))
    k_star = int(k_grid[idx])

    n_nan = int(np.isnan(xi_series).sum())
    if n_nan > 0:
        logger.warning("GPD fit produced NaN for %d / %d k values", n_nan, len(k_grid))
    logger.debug(
        "Baseline k*=%d (index %d/%d), total_score_min=%.4f",
        k_star, idx, len(k_grid), total[idx],
    )

    # Compute additional diagnostic series for CNN channels
    hill_series = hill_estimator(sorted_desc, k_grid)
    qq_resid_series = qq_residual(sorted_desc, k_grid, params)
    me_values = mean_excess_values(sorted_desc, k_grid)

    diagnostics = {
        "k_grid": k_grid,
        "params": params,
        "xi_series": xi_series,
        "score_stability": s_stab,
        "score_gof": s_gof,
        "score_mean_excess": s_me,
        "score_penalty": s_pen,
        "total_score": total,
        "k_star": k_star,
        "hill_series": hill_series,
        "qq_residual_series": qq_resid_series,
        "mean_excess_values": me_values,
    }
    return k_star, diagnostics


def process_one_dataset(ds, pot_cfg):
    """Process a single dataset through POT diagnostics (Steps 2-4).

    Parameters
    ----------
    ds : dict
        Dataset dict with keys 'samples', 'n' (and optional metadata).
    pot_cfg : dict
        POT configuration with keys 'k_min', 'k_max_frac', 'delta', 'weights'.

    Returns
    -------
    tuple of (ds, diagnostics)
    """
    samples = ds["samples"]
    sorted_desc = np.sort(samples)[::-1]

    k_grid = candidate_k_grid(
        n=ds["n"],
        k_min=pot_cfg["k_min"],
        k_max_frac=pot_cfg["k_max_frac"],
    )

    _, diagnostics = compute_baseline_k_star(
        sorted_desc=sorted_desc,
        k_grid=k_grid,
        delta=pot_cfg["delta"],
        weights=tuple(pot_cfg["weights"]),
    )

    return (ds, diagnostics)
