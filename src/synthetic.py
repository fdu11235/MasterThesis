"""Synthetic data generation for POT/GPD threshold selection experiments.

Provides four distribution families with heavy-tailed behaviour:
  - Student-t (absolute values)
  - Pareto
  - Lognormal-Pareto mixture
  - Two-Pareto (regime change)

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
