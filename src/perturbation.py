"""Data perturbation utilities for robustness experiments.

Provides two perturbation strategies for testing CNN threshold selection
stability:
  - Random deletion: remove a fraction of observations
  - Bootstrap resampling: resample with replacement to same size
"""

from __future__ import annotations

import numpy as np


def perturb_random_deletion(
    dataset: dict, fraction: float, seed: int
) -> dict:
    """Remove a random fraction of observations from a dataset.

    Parameters
    ----------
    dataset : dict
        Dataset dict with at least ``"samples"`` (ndarray) and ``"n"`` (int).
    fraction : float
        Fraction of observations to remove, in [0, 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        New dataset dict with reduced ``"samples"`` and updated ``"n"``.
        Includes metadata: ``"perturbation"`` and ``"deletion_fraction"``.
    """
    if not 0 <= fraction < 1:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")

    samples = dataset["samples"]
    n = len(samples)
    n_keep = max(1, round(n * (1 - fraction)))

    rng = np.random.RandomState(seed)
    keep_idx = np.sort(rng.choice(n, n_keep, replace=False))

    result = dict(dataset)
    result["samples"] = samples[keep_idx].copy()
    result["n"] = n_keep
    result["perturbation"] = "deletion"
    result["deletion_fraction"] = fraction
    return result


def perturb_bootstrap(dataset: dict, seed: int) -> dict:
    """Resample observations with replacement to the same size.

    Parameters
    ----------
    dataset : dict
        Dataset dict with at least ``"samples"`` (ndarray).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        New dataset dict with bootstrapped ``"samples"`` (same length).
        Includes metadata: ``"perturbation"``.
    """
    samples = dataset["samples"]
    n = len(samples)

    rng = np.random.RandomState(seed)
    boot_idx = rng.choice(n, n, replace=True)

    result = dict(dataset)
    result["samples"] = samples[boot_idx].copy()
    result["perturbation"] = "bootstrap"
    return result
