#!/usr/bin/env python
"""Scoring function weight grid search.

Evaluates different weight combinations for the k* scoring function
(stability, GoF, penalty, mean_excess) by computing downstream VaR/ES
RMSE against true quantiles. Uses cached diagnostics — no re-running POT.

Usage:
    python run_scoring_experiment.py --config config/default.yaml
"""

import argparse
import logging
import os
import pickle

import numpy as np
import yaml

from src.evaluate import pot_quantile, pot_es, true_quantile, true_es

logger = logging.getLogger(__name__)


def _min_max_normalize(s):
    smin, smax = np.nanmin(s), np.nanmax(s)
    return (s - smin) / (smax - smin + 1e-10)


WEIGHT_GRID = [
    # (stability, gof, penalty, mean_excess) — label
    ((1, 1, 1, 0),     "baseline [1,1,1,0]"),
    ((2, 1, 1, 0),     "stab-heavy [2,1,1,0]"),
    ((1, 2, 1, 0),     "gof-heavy [1,2,1,0]"),
    ((1, 1, 2, 0),     "pen-heavy [1,1,2,0]"),
    ((0.5, 1, 1, 0),   "low-stab [0.5,1,1,0]"),
    ((1, 1, 0.5, 0),   "low-pen [1,1,0.5,0]"),
    ((1, 1, 1, 1),     "+ME equal [1,1,1,1]"),
    ((1, 1, 1, 2),     "+ME heavy [1,1,1,2]"),
    ((2, 1, 0.5, 1),   "stab+ME [2,1,0.5,1]"),
    ((1, 2, 0.5, 1),   "gof+ME [1,2,0.5,1]"),
    ((2, 2, 1, 1),     "stab+gof [2,2,1,1]"),
    ((1, 1, 0, 1),     "no-pen [1,1,0,1]"),
    ((1, 1, 0, 0),     "stab+gof only [1,1,0,0]"),
    ((0, 1, 1, 0),     "gof+pen [0,1,1,0]"),
    ((0, 0, 0, 1),     "ME only [0,0,0,1]"),
    ((1, 0, 0, 0),     "stab only [1,0,0,0]"),
    ((0, 1, 0, 0),     "gof only [0,1,0,0]"),
]


def evaluate_weights(weights, all_diagnostics, p):
    """Recompute k* with given weights and evaluate VaR/ES quality."""
    w_stab, w_gof, w_pen, w_me = weights

    var_errors = []
    es_errors = []
    var_rel_errors = []
    es_rel_errors = []

    for ds, diag in all_diagnostics:
        k_grid = np.asarray(diag["k_grid"])
        params = np.asarray(diag["params"])

        # Recompute k* with new weights
        s_stab = _min_max_normalize(diag["score_stability"])
        s_gof = _min_max_normalize(diag["score_gof"])
        s_pen = _min_max_normalize(diag["score_penalty"])
        s_me = _min_max_normalize(diag["score_mean_excess"])
        total = w_stab * s_stab + w_gof * s_gof + w_pen * s_pen + w_me * s_me
        k_idx = int(np.argmin(total))
        k_star = int(k_grid[k_idx])

        xi, beta = params[k_idx]
        if np.isnan(xi) or np.isnan(beta):
            continue

        sorted_desc = np.sort(ds["samples"])[::-1]
        n = len(sorted_desc)

        # Compute VaR/ES at this k*
        var_est = pot_quantile(sorted_desc, k_star, xi, beta, n, p)
        es_est = pot_es(sorted_desc, k_star, xi, beta, n, p)

        # True values
        try:
            var_true = true_quantile(ds["dist_type"], ds["params"], p)
            es_true = true_es(ds["dist_type"], ds["params"], p)
        except (ValueError, KeyError):
            continue

        if var_true <= 0 or es_true <= 0 or np.isnan(var_est) or np.isnan(es_est):
            continue

        var_errors.append((var_est - var_true) ** 2)
        es_errors.append((es_est - es_true) ** 2)
        var_rel_errors.append(((var_est - var_true) / var_true) ** 2)
        es_rel_errors.append(((es_est - es_true) / es_true) ** 2)

    n_eval = len(var_errors)
    if n_eval == 0:
        return {"var_rmse": float("nan"), "es_rmse": float("nan"),
                "var_rel_rmse": float("nan"), "es_rel_rmse": float("nan"),
                "n_eval": 0}

    return {
        "var_rmse": np.sqrt(np.mean(var_errors)),
        "es_rmse": np.sqrt(np.mean(es_errors)),
        "var_rel_rmse": np.sqrt(np.mean(var_rel_errors)),
        "es_rel_rmse": np.sqrt(np.mean(es_rel_errors)),
        "n_eval": n_eval,
    }


def main():
    parser = argparse.ArgumentParser(description="Scoring function weight grid search.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    p = config.get("evaluate", {}).get("quantile_p", 0.99)

    # Load cached diagnostics
    diag_path = "outputs/data/diagnostics.pkl"
    logger.info("Loading diagnostics from %s …", diag_path)
    with open(diag_path, "rb") as f:
        all_diagnostics = pickle.load(f)
    logger.info("Loaded %d diagnostics", len(all_diagnostics))

    # Filter to n=5000 for cleaner evaluation (optional: use all)
    diags_5k = [(ds, dg) for ds, dg in all_diagnostics if ds["n"] == 5000]
    logger.info("Using %d datasets (n=5000) for evaluation", len(diags_5k))

    # Run grid search
    results = []
    for weights, label in WEIGHT_GRID:
        metrics = evaluate_weights(weights, diags_5k, p)
        results.append({
            "label": label,
            "weights": weights,
            **metrics,
        })
        logger.info("  %-28s VaR_RelRMSE=%.4f  ES_RelRMSE=%.4f  (n=%d)",
                     label, metrics["var_rel_rmse"], metrics["es_rel_rmse"], metrics["n_eval"])

    # Sort by VaR relative RMSE
    results.sort(key=lambda r: r["var_rel_rmse"])

    logger.info("\n" + "=" * 80)
    logger.info("RESULTS (sorted by VaR Relative RMSE)")
    logger.info("=" * 80)
    logger.info("%-30s %-12s %-12s %-12s %-12s", "Weights", "VaR_RelRMSE", "ES_RelRMSE", "VaR_RMSE", "ES_RMSE")
    logger.info("-" * 80)
    for r in results:
        logger.info("%-30s %-12.4f %-12.4f %-12.4f %-12.4f",
                     r["label"], r["var_rel_rmse"], r["es_rel_rmse"],
                     r["var_rmse"], r["es_rmse"])

    best = results[0]
    logger.info("\nBest weights: %s (%s)", best["weights"], best["label"])
    logger.info("  VaR Relative RMSE: %.4f", best["var_rel_rmse"])
    logger.info("  ES Relative RMSE:  %.4f", best["es_rel_rmse"])

    # Save results
    out_path = "outputs/scoring_experiment_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
