"""Synthetic data generation for POT/GPD threshold selection experiments.

Provides twelve distribution families with diverse tail behaviour:

  Fréchet MDA (xi > 0, power-law tails):
  - Student-t (absolute values)
  - Pareto
  - Lognormal-Pareto mixture
  - Two-Pareto (regime change)
  - Burr XII (Singh-Maddala)
  - Fréchet (inverse Weibull)
  - Dagum (Burr III)
  - Inverse Gamma (Vinci)
  - Log-Gamma

  Gumbel MDA (xi = 0, subexponential / stretched-exponential):
  - Lognormal (pure)
  - Weibull (stretched, c < 1)

  Mixture / splice:
  - Gamma-Pareto splice

Usage:
    from src.synthetic import generate_all
    import yaml

    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)
    datasets = generate_all(config["synthetic"])
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-sample generators
# ---------------------------------------------------------------------------

def _generate_student_t(
    rng: np.random.RandomState, n: int, df: float
) -> np.ndarray:
    """Absolute Student-t samples (positive, heavy-tailed)."""
    return np.abs(stats.t.rvs(df=df, size=n, random_state=rng))


def _generate_pareto(
    rng: np.random.RandomState, n: int, alpha: float
) -> np.ndarray:
    """Standard Pareto(alpha) samples via scipy (support [1, inf))."""
    return stats.pareto.rvs(b=alpha, size=n, random_state=rng)


def _generate_lognormal_pareto_mix(
    rng: np.random.RandomState,
    n: int,
    lognormal_mu: float,
    lognormal_sigma: float,
    pareto_alpha: float,
    mix_frac: float,
) -> np.ndarray:
    """Lognormal body with Pareto tail contamination.

    Each observation is drawn from Pareto(pareto_alpha) with probability
    *mix_frac* and from Lognormal(lognormal_mu, lognormal_sigma) otherwise.
    The Bernoulli mask is drawn with *rng* for reproducibility.
    """
    mask = rng.random_sample(n) < mix_frac  # Bernoulli via RandomState
    samples = np.empty(n)
    n_pareto = int(mask.sum())
    n_lognormal = n - n_pareto
    samples[mask] = stats.pareto.rvs(
        b=pareto_alpha, size=n_pareto, random_state=rng
    )
    samples[~mask] = stats.lognorm.rvs(
        s=lognormal_sigma, scale=np.exp(lognormal_mu),
        size=n_lognormal, random_state=rng,
    )
    return samples


def _generate_two_pareto(
    rng: np.random.RandomState,
    n: int,
    alpha1: float,
    alpha2: float,
    changepoint_frac: float,
) -> np.ndarray:
    """Two-regime Pareto: bulk ~ Pareto(alpha1), top fraction ~ Pareto(alpha2).

    The top *changepoint_frac* of the bulk sample is replaced with
    Pareto(alpha2) draws that are scaled so the new tail starts at the
    empirical changepoint value.
    """
    bulk = stats.pareto.rvs(b=alpha1, size=n, random_state=rng)
    bulk.sort()
    cp_idx = int(np.ceil(n * (1 - changepoint_frac)))
    cp_value = bulk[cp_idx - 1]  # value at the changepoint boundary
    n_tail = n - cp_idx
    # Pareto(alpha2) has support [1, inf); scale so minimum equals cp_value
    tail = stats.pareto.rvs(
        b=alpha2, size=n_tail, random_state=rng
    ) * cp_value
    bulk[cp_idx:] = tail
    # Shuffle so the ordering is not baked in
    rng.shuffle(bulk)
    return bulk


def _generate_burr12(
    rng: np.random.RandomState, n: int, c: float, d: float
) -> np.ndarray:
    """Burr XII (Singh-Maddala) samples. Tail index xi = 1/(c*d)."""
    return stats.burr12.rvs(c=c, d=d, size=n, random_state=rng)


def _generate_frechet(
    rng: np.random.RandomState, n: int, c: float
) -> np.ndarray:
    """Fréchet (inverse Weibull) samples. Tail index xi = 1/c."""
    return stats.invweibull.rvs(c=c, size=n, random_state=rng)


def _generate_dagum(
    rng: np.random.RandomState, n: int, c: float, d: float
) -> np.ndarray:
    """Dagum (Burr III) samples. Tail index xi = 1/c."""
    return stats.burr.rvs(c=c, d=d, size=n, random_state=rng)


def _generate_inverse_gamma(
    rng: np.random.RandomState, n: int, a: float
) -> np.ndarray:
    """Inverse Gamma (Vinci) samples. Tail index xi = 1/a."""
    return stats.invgamma.rvs(a=a, size=n, random_state=rng)


def _generate_lognormal(
    rng: np.random.RandomState, n: int, sigma: float
) -> np.ndarray:
    """Lognormal samples (Gumbel MDA, xi = 0)."""
    return stats.lognorm.rvs(s=sigma, size=n, random_state=rng)


def _generate_weibull_stretched(
    rng: np.random.RandomState, n: int, c: float
) -> np.ndarray:
    """Stretched Weibull samples with c < 1 (Gumbel MDA, xi = 0)."""
    return stats.weibull_min.rvs(c=c, size=n, random_state=rng)


def _generate_log_gamma(
    rng: np.random.RandomState, n: int, b: float, p: float
) -> np.ndarray:
    """Log-Gamma samples: X = exp(Y), Y ~ Gamma(shape=p, scale=1/b).

    Generalises Pareto Type I (p=1 recovers Pareto). Tail index xi = 1/b.
    """
    y = stats.gamma.rvs(a=p, scale=1.0 / b, size=n, random_state=rng)
    return np.exp(y)


def _generate_gamma_pareto_splice(
    rng: np.random.RandomState,
    n: int,
    gamma_shape: float,
    pareto_alpha: float,
    splice_quantile: float,
) -> np.ndarray:
    """Gamma body with Pareto tail spliced at a high quantile.

    Samples below *splice_quantile* come from Gamma(gamma_shape);
    those above are replaced with Pareto(pareto_alpha) draws scaled
    so the splice is continuous at the threshold.
    """
    samples = stats.gamma.rvs(a=gamma_shape, size=n, random_state=rng)
    threshold = np.quantile(samples, splice_quantile)
    above = samples > threshold
    n_tail = int(above.sum())
    if n_tail > 0:
        tail = stats.pareto.rvs(
            b=pareto_alpha, size=n_tail, random_state=rng
        ) * threshold
        samples[above] = tail
    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# GARCH-wrapped generator
# ---------------------------------------------------------------------------

def _generate_garch_wrapped(
    rng: np.random.RandomState,
    n: int,
    base_dist_type: str,
    base_params: dict[str, Any],
    garch_omega: float = 0.01,
    garch_alpha: float = 0.10,
    garch_beta: float = 0.85,
) -> np.ndarray:
    """Generate GARCH-filtered absolute standardised residuals.

    1. Draw i.i.d. innovations from *base_dist_type*.
    2. Simulate a GARCH(1,1) return series  r_t = sigma_t * z_t.
    3. Fit GARCH(1,1) to recover estimated residuals  z_hat_t = r_t / sigma_hat_t.
    4. Return |z_hat_t|.

    This produces training data whose diagnostic curves (xi, AD, mean excess)
    exhibit the same noise characteristics as real data processed through the
    GARCH-POT pipeline.
    """
    from src.garch import fit_garch_and_filter

    # Need extra samples for burn-in; generate 20% more, then trim
    n_gen = n + max(200, n // 5)

    # Step 1: i.i.d. innovations from base distribution
    base_generators = {
        "student_t": _generate_student_t,
        "pareto": _generate_pareto,
    }
    if base_dist_type not in base_generators:
        raise ValueError(
            f"GARCH wrapper only supports base distributions: "
            f"{list(base_generators)}, got {base_dist_type!r}"
        )

    # Generate raw innovations (positive); convert to signed for GARCH
    raw = base_generators[base_dist_type](rng, n_gen, **base_params)
    # Assign random signs so GARCH sees realistic signed returns
    signs = rng.choice([-1.0, 1.0], size=n_gen)
    innovations = raw * signs

    # Standardize innovations to unit variance (GARCH assumes E[z^2] = 1)
    innovations = innovations / (np.std(innovations) + 1e-10)

    # Step 2: simulate GARCH(1,1) returns
    sigma2 = np.empty(n_gen)
    returns = np.empty(n_gen)
    sigma2[0] = garch_omega / (1.0 - garch_alpha - garch_beta)  # unconditional var
    returns[0] = np.sqrt(sigma2[0]) * innovations[0]

    for t in range(1, n_gen):
        sigma2[t] = (garch_omega
                     + garch_alpha * returns[t - 1] ** 2
                     + garch_beta * sigma2[t - 1])
        returns[t] = np.sqrt(sigma2[t]) * innovations[t]

    # Trim burn-in
    returns = returns[-n:]

    # Step 3: fit GARCH to recover standardized residuals
    result = fit_garch_and_filter(returns, forecast_horizon=1)

    # Step 4: return absolute standardized residuals
    return result['abs_std_residuals']


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(
    dist_type: str,
    dist_params: dict[str, Any],
    n: int,
    seed: int,
) -> dict[str, Any]:
    """Generate a single synthetic dataset.

    Parameters
    ----------
    dist_type : str
        One of ``"student_t"``, ``"pareto"``, ``"lognormal_pareto_mix"``,
        ``"two_pareto"``.
    dist_params : dict
        Distribution-specific parameters (scalar values, not lists).
    n : int
        Sample size.
    seed : int
        Random seed for full reproducibility.

    Returns
    -------
    dict with keys ``"samples"`` (ndarray), ``"params"`` (dict),
    ``"dist_type"`` (str), ``"n"`` (int).
    """
    rng = np.random.RandomState(seed)

    generators = {
        "student_t": lambda: _generate_student_t(rng, n, **dist_params),
        "pareto": lambda: _generate_pareto(rng, n, **dist_params),
        "lognormal_pareto_mix": lambda: _generate_lognormal_pareto_mix(
            rng, n, **dist_params
        ),
        "two_pareto": lambda: _generate_two_pareto(rng, n, **dist_params),
        "burr12": lambda: _generate_burr12(rng, n, **dist_params),
        "frechet": lambda: _generate_frechet(rng, n, **dist_params),
        "dagum": lambda: _generate_dagum(rng, n, **dist_params),
        "inverse_gamma": lambda: _generate_inverse_gamma(rng, n, **dist_params),
        "lognormal": lambda: _generate_lognormal(rng, n, **dist_params),
        "weibull_stretched": lambda: _generate_weibull_stretched(rng, n, **dist_params),
        "log_gamma": lambda: _generate_log_gamma(rng, n, **dist_params),
        "gamma_pareto_splice": lambda: _generate_gamma_pareto_splice(
            rng, n, **dist_params
        ),
        "garch_student_t": lambda: _generate_garch_wrapped(
            rng, n, "student_t",
            {k: v for k, v in dist_params.items()
             if k not in ("garch_omega", "garch_alpha", "garch_beta")},
            garch_omega=dist_params.get("garch_omega", 0.01),
            garch_alpha=dist_params.get("garch_alpha", 0.10),
            garch_beta=dist_params.get("garch_beta", 0.85),
        ),
        "garch_pareto": lambda: _generate_garch_wrapped(
            rng, n, "pareto",
            {k: v for k, v in dist_params.items()
             if k not in ("garch_omega", "garch_alpha", "garch_beta")},
            garch_omega=dist_params.get("garch_omega", 0.01),
            garch_alpha=dist_params.get("garch_alpha", 0.10),
            garch_beta=dist_params.get("garch_beta", 0.85),
        ),
    }

    if dist_type not in generators:
        raise ValueError(
            f"Unknown dist_type {dist_type!r}. "
            f"Choose from {list(generators)}"
        )

    samples = generators[dist_type]()
    return {
        "samples": samples,
        "params": dict(dist_params),
        "dist_type": dist_type,
        "n": n,
    }


def _param_combos(dist_type: str, dist_cfg: dict) -> list[dict[str, Any]]:
    """Expand a distribution config block into a list of scalar param dicts."""
    if dist_type == "student_t":
        return [{"df": df} for df in dist_cfg["df"]]

    if dist_type == "pareto":
        return [{"alpha": a} for a in dist_cfg["alpha"]]

    if dist_type == "lognormal_pareto_mix":
        mu = dist_cfg["lognormal_mu"]
        sigma = dist_cfg["lognormal_sigma"]
        mix = dist_cfg["mix_frac"]
        return [
            {
                "lognormal_mu": mu,
                "lognormal_sigma": sigma,
                "pareto_alpha": pa,
                "mix_frac": mix,
            }
            for pa in dist_cfg["pareto_alpha"]
        ]

    if dist_type == "two_pareto":
        cp = dist_cfg["changepoint_frac"]
        return [
            {"alpha1": a1, "alpha2": a2, "changepoint_frac": cp}
            for a1, a2 in itertools.product(
                dist_cfg["alpha1"], dist_cfg["alpha2"]
            )
        ]

    if dist_type == "burr12":
        return [
            {"c": c, "d": d}
            for c, d in itertools.product(dist_cfg["c"], dist_cfg["d"])
        ]

    if dist_type == "frechet":
        return [{"c": c} for c in dist_cfg["c"]]

    if dist_type == "dagum":
        return [
            {"c": c, "d": d}
            for c, d in itertools.product(dist_cfg["c"], dist_cfg["d"])
        ]

    if dist_type == "inverse_gamma":
        return [{"a": a} for a in dist_cfg["a"]]

    if dist_type == "lognormal":
        return [{"sigma": s} for s in dist_cfg["sigma"]]

    if dist_type == "weibull_stretched":
        return [{"c": c} for c in dist_cfg["c"]]

    if dist_type == "log_gamma":
        p = dist_cfg["p"]
        return [{"b": b, "p": p} for b in dist_cfg["b"]]

    if dist_type == "gamma_pareto_splice":
        sq = dist_cfg["splice_quantile"]
        return [
            {"gamma_shape": gs, "pareto_alpha": pa, "splice_quantile": sq}
            for gs, pa in itertools.product(
                dist_cfg["gamma_shape"], dist_cfg["pareto_alpha"]
            )
        ]

    if dist_type == "garch_student_t":
        ga = dist_cfg.get("garch_alpha", 0.1)
        gb = dist_cfg.get("garch_beta", 0.85)
        go = dist_cfg.get("garch_omega", 0.01)
        return [
            {"df": df, "garch_alpha": ga, "garch_beta": gb, "garch_omega": go}
            for df in dist_cfg["df"]
        ]

    if dist_type == "garch_pareto":
        ga = dist_cfg.get("garch_alpha", 0.1)
        gb = dist_cfg.get("garch_beta", 0.85)
        go = dist_cfg.get("garch_omega", 0.01)
        return [
            {"alpha": a, "garch_alpha": ga, "garch_beta": gb, "garch_omega": go}
            for a in dist_cfg["alpha"]
        ]

    raise ValueError(f"Unknown dist_type {dist_type!r}")


def generate_all(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all synthetic datasets specified by *config*.

    Parameters
    ----------
    config : dict
        The ``synthetic`` section of the YAML config (i.e. with keys
        ``sample_sizes``, ``n_replications``, ``seed``, ``distributions``).

    Returns
    -------
    list of dicts, each produced by :func:`generate_dataset`.
    """
    sample_sizes: list[int] = config["sample_sizes"]
    n_replications: int = config["n_replications"]
    base_seed: int = config["seed"]
    distributions: dict = config["distributions"]

    results: list[dict[str, Any]] = []
    seed_counter = 0

    for dist_type, dist_cfg in distributions.items():
        combos = _param_combos(dist_type, dist_cfg)
        logger.info("Distribution %s: %d parameter combos", dist_type, len(combos))
        for params in combos:
            for n in sample_sizes:
                for _rep in range(n_replications):
                    seed = base_seed + seed_counter
                    seed_counter += 1
                    results.append(
                        generate_dataset(dist_type, params, n, seed)
                    )
        logger.debug(
            "  %s done — %d datasets so far", dist_type, len(results)
        )

    logger.info("Generated %d total datasets", len(results))
    return results
