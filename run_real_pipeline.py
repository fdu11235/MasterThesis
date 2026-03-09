#!/usr/bin/env python
"""Step 8 — Real financial data pipeline with pseudo-labels and VaR backtesting.

Usage:
    python run_real_pipeline.py --config config/default.yaml
    python run_real_pipeline.py --config config/default.yaml --fresh
    python run_real_pipeline.py --config config/default.yaml --n-jobs 4
"""

import argparse
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import yaml
from joblib import Parallel, delayed

from src.realdata import prepare_real_datasets
from src.pot import process_one_dataset
from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.train import train_model, predict
from src.evaluate_real import evaluate_real, plot_real_results


def main():
    # ── CLI ───────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Run the real-data POT pipeline (Step 8).")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML configuration file.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO).")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoints and run from scratch.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs for diagnostics (default: -1 = all cores).")
    args = parser.parse_args()

    # ── Logging ────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("real_pipeline")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", args.config)

    if args.fresh:
        logger.info("--fresh flag set: ignoring existing checkpoints")

    # ── Output directories ────────────────────────────────────────────────
    os.makedirs("outputs/data", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/figures/real", exist_ok=True)

    # ── Step 1: Load real data ────────────────────────────────────────────
    real_ds_path = "outputs/data/real_datasets.pkl"
    if not args.fresh and os.path.exists(real_ds_path):
        logger.info("[Step 1] Loading cached real datasets from %s", real_ds_path)
        with open(real_ds_path, "rb") as f:
            saved = pickle.load(f)
        datasets = saved["datasets"]
        returns_lookup = saved["returns_lookup"]
        logger.info("  → %d windows loaded (skipped download)", len(datasets))
    else:
        logger.info("[Step 1] Loading real financial data …")
        datasets, returns_lookup = prepare_real_datasets(config)
        with open(real_ds_path, "wb") as f:
            pickle.dump({"datasets": datasets, "returns_lookup": returns_lookup}, f)
        logger.info("  → %d windows saved to %s", len(datasets), real_ds_path)

    # ── Steps 2-4: POT diagnostics ───────────────────────────────────────
    diag_path = "outputs/data/real_diagnostics.pkl"
    if not args.fresh and os.path.exists(diag_path):
        logger.info("[Steps 2-4] Loading cached diagnostics from %s", diag_path)
        with open(diag_path, "rb") as f:
            all_diagnostics = pickle.load(f)
        logger.info("  → %d diagnostics loaded (skipped computation)", len(all_diagnostics))
    else:
        logger.info("[Steps 2-4] Computing POT diagnostics (%d windows, n_jobs=%d) …",
                     len(datasets), args.n_jobs)
        pot_cfg = config["pot"]

        all_diagnostics = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_one_dataset)(ds, pot_cfg) for ds in datasets
        )

        with open(diag_path, "wb") as f:
            pickle.dump(all_diagnostics, f)
        logger.info("  → diagnostics saved to %s", diag_path)

    # ── Step 5: Build regression features ─────────────────────────────────
    logger.info("[Step 5] Building regression dataset …")
    X, y, meta = build_dataset_regression(all_diagnostics, config)
    logger.info("  → X %s, y %s", tuple(X.shape), tuple(y.shape))

    # ── Time-ordered split (by end_date) ──────────────────────────────────
    # Sort indices by end_date for temporal ordering
    end_dates = [m.get("end_date", "") for m in meta]
    sorted_indices = np.argsort(end_dates)

    train_frac = config["realdata"]["train_fraction"]
    n_train = int(len(sorted_indices) * train_frac)

    train_idx = sorted_indices[:n_train]
    test_idx = sorted_indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    test_meta = [meta[i] for i in test_idx]
    test_diags = [all_diagnostics[i] for i in test_idx]

    logger.info("  Time-ordered split: %d train, %d test", len(train_idx), len(test_idx))
    if len(test_meta) > 0:
        logger.info("  Train period ends: %s", end_dates[train_idx[-1]])
        logger.info("  Test period starts: %s", end_dates[test_idx[0]])

    # ── Step 6: Train CNN ─────────────────────────────────────────────────
    model_cfg = config["model"]
    ckpt_path = "outputs/checkpoints/model_real.pt"

    if not args.fresh and os.path.exists(ckpt_path):
        logger.info("[Step 6] Loading cached model from %s", ckpt_path)
        model = ThresholdCNN(
            in_channels=4,
            channels=model_cfg["channels"],
            kernel_size=model_cfg["kernel_size"],
            dropout=model_cfg["dropout"],
            task="regression",
        )
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()
    else:
        logger.info("[Step 6] Training regression model (train=%d, test=%d) …",
                     len(X_train), len(X_test))
        model = ThresholdCNN(
            in_channels=4,
            channels=model_cfg["channels"],
            kernel_size=model_cfg["kernel_size"],
            dropout=model_cfg["dropout"],
            task="regression",
        )
        train_config = {
            "lr": model_cfg["lr"],
            "batch_size": model_cfg["batch_size"],
            "max_epochs": model_cfg["max_epochs"],
            "patience": model_cfg["patience"],
            "test_fraction": config["evaluate"]["test_fraction"],
        }
        model = train_model(X_train, y_train, model, train_config, task="regression")
        torch.save(model.state_dict(), ckpt_path)
        logger.info("  → checkpoint saved to %s", ckpt_path)

    # ── Step 7: Predict, denormalize, evaluate ────────────────────────────
    logger.info("[Step 7] Evaluating with VaR backtesting …")
    y_pred_norm = predict(model, X_test, task="regression")

    # Denormalize predictions to actual k values
    k_pred = np.array([
        int(np.clip(round(m["k_min"] + yp * (m["k_max"] - m["k_min"])),
                     m["k_min"], m["k_max"]))
        for yp, m in zip(y_pred_norm, test_meta)
    ])

    # Baseline k* from diagnostics
    k_baseline = np.array([diag["k_star"] for _, diag in test_diags])

    # Extract test-set window dicts and diagnostics dicts
    test_ds = [ds for ds, _ in test_diags]
    test_diag_dicts = [diag for _, diag in test_diags]

    # Run VaR backtesting
    results = evaluate_real(
        test_data=test_ds,
        diagnostics_list=test_diag_dicts,
        k_pred=k_pred,
        k_baseline=k_baseline,
        returns_lookup=returns_lookup,
        config=config,
    )

    # Log results
    logger.info("=" * 60)
    logger.info("VaR Backtest Results")
    logger.info("=" * 60)
    for method, stats in results["summary"].items():
        vr = stats.get("overall_violation_rate", float('nan'))
        expected = stats.get("expected_rate", float('nan'))
        n_win = stats.get("n_windows", 0)
        kup = stats.get("kupiec", {})
        kup_str = (f"LR={kup.get('statistic', 'N/A'):.2f}, "
                   f"p={kup.get('p_value', 'N/A'):.4f}, "
                   f"reject={kup.get('reject_5pct', 'N/A')}"
                   if kup else "N/A")
        chris = stats.get("christoffersen", {})
        chris_str = (f"LR={chris.get('lr_ind', 'N/A'):.2f}, "
                     f"p={chris.get('p_value_ind', 'N/A'):.4f}, "
                     f"reject={chris.get('reject_ind_5pct', 'N/A')}"
                     if chris else "N/A")
        mf = stats.get("mcneil_frey", {})
        mf_str = (f"t={mf.get('t_stat', float('nan')):.2f}, "
                  f"p={mf.get('p_value', float('nan')):.4f}, "
                  f"reject={mf.get('reject_5pct', 'N/A')}"
                  if mf else "N/A")
        logger.info("  %-20s: VR=%.4f (expected=%.4f), n=%d", method, vr, expected, n_win)
        logger.info("    Kupiec:          %s", kup_str)
        logger.info("    Christoffersen:  %s", chris_str)
        logger.info("    Mean ES:         %.4f", stats.get("mean_es_estimate", float('nan')))
        logger.info("    McNeil-Frey:     %s", mf_str)
    logger.info("=" * 60)

    # Plot
    plot_real_results(results, "outputs/figures/real")

    logger.info("Real-data pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger("real_pipeline").warning(
            "Interrupted by user. Previously completed steps are saved — "
            "re-run to resume from last checkpoint."
        )
