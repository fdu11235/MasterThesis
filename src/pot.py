"""POT diagnostics and baseline scoring for GPD threshold selection."""

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.stats import genpareto, kstest

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


def score_gof(sorted_desc: NDArray, k_grid: NDArray, params: NDArray) -> NDArray:
    """KS statistic for each k. Returns 1.0 when params are NaN."""
    scores = np.empty(len(k_grid), dtype=float)
    for i, k in enumerate(k_grid):
        xi, beta = params[i]
        if np.isnan(xi) or np.isnan(beta):
            scores[i] = 1.0
            continue
        exceedances = sorted_desc[:k] - sorted_desc[k]
        stat, _ = kstest(exceedances, "genpareto", args=(xi, 0, beta))
        scores[i] = stat
    return scores


def score_penalty(k_grid: NDArray) -> NDArray:
    """Penalty score: 1 / sqrt(k)."""
    return 1.0 / np.sqrt(k_grid.astype(float))


def _min_max_normalize(s: NDArray) -> NDArray:
    """Min-max normalize an array to [0, 1]."""
    return (s - s.min()) / (s.max() - s.min() + 1e-10)


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

    diagnostics = {
        "k_grid": k_grid,
        "params": params,
        "xi_series": xi_series,
        "score_stability": s_stab,
        "score_gof": s_gof,
        "score_penalty": s_pen,
        "total_score": total,
        "k_star": k_star,
    }
    return k_star, diagnostics
