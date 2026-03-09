#!/usr/bin/env python
"""Main pipeline entry point — orchestrates Steps 1-7.

Usage:
    python run_pipeline.py --config config/default.yaml
"""

import argparse
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import yaml

from src.synthetic import generate_all
from src.pot import candidate_k_grid, compute_baseline_k_star
from src.features import build_dataset
from src.model import ThresholdCNN
from src.train import train_model, predict
from src.evaluate import evaluate_all, plot_results


def main():
    # ── CLI ───────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Run the full POT threshold-selection pipeline.")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML configuration file.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO).")
    args = parser.parse_args()

    # ── Logging ────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("pipeline")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", args.config)

    # ── Output directories ────────────────────────────────────────────────
    os.makedirs("outputs/data", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # ── Step 1: Generate synthetic data ───────────────────────────────────
    logger.info("[Step 1] Generating synthetic data …")
    datasets = generate_all(config["synthetic"])
    with open("outputs/data/synthetic.pkl", "wb") as f:
        pickle.dump(datasets, f)
    logger.info("  → %d datasets saved to outputs/data/synthetic.pkl", len(datasets))

    # ── Steps 2-4: Sort, build k-grid, compute baseline k* ───────────────
    logger.info("[Steps 2-4] Computing POT diagnostics …")
    pot_cfg = config["pot"]
    all_diagnostics = []  # list of (dataset_dict, diagnostics_dict)

    for i, ds in enumerate(datasets):
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

        all_diagnostics.append((ds, diagnostics))

        if (i + 1) % 500 == 0 or (i + 1) == len(datasets):
            logger.info("  → %d/%d datasets processed", i + 1, len(datasets))

    with open("outputs/data/diagnostics.pkl", "wb") as f:
        pickle.dump(all_diagnostics, f)
    logger.info("  → diagnostics saved to outputs/data/diagnostics.pkl")

    # ── Step 5: Build feature tensors (grouped by sample size) ────────────
    logger.info("[Step 5] Building feature tensors …")
    grouped_datasets = build_dataset(all_diagnostics, config)
    for n_size, (X, y) in sorted(grouped_datasets.items()):
        logger.info("  → n=%d: X %s, y %s", n_size, tuple(X.shape), tuple(y.shape))

    # ── Pre-split: group diagnostics by sample size for train/test ────────
    test_frac = config["evaluate"]["test_fraction"]
    diag_by_size = defaultdict(list)
    for ds, diag in all_diagnostics:
        diag_by_size[int(ds["n"])].append((ds, diag))

    # ── Steps 6-7: Train, predict, evaluate per sample-size group ─────────
    for n_size in sorted(grouped_datasets.keys()):
        X, y = grouped_datasets[n_size]
        group_diags = diag_by_size[n_size]
        k_grid = np.asarray(group_diags[0][1]["k_grid"])
        n_classes = len(k_grid)
        N = len(X)

        # Train / test split (deterministic per group)
        perm = torch.randperm(N)
        test_size = int(N * test_frac)
        test_idx = perm[:test_size]
        train_idx = perm[test_size:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        test_diags = [group_diags[i] for i in test_idx.tolist()]

        # ── Step 6: Create & train model ──────────────────────────────────
        logger.info(
            "[Step 6] Training model for n=%d (train=%d, test=%d, classes=%d) …",
            n_size, len(X_train), len(X_test), n_classes,
        )

        model_cfg = config["model"]
        model = ThresholdCNN(
            in_channels=3,
            channels=model_cfg["channels"],
            kernel_size=model_cfg["kernel_size"],
            dropout=model_cfg["dropout"],
            n_classes=n_classes,
        )

        train_config = {
            "lr": model_cfg["lr"],
            "batch_size": model_cfg["batch_size"],
            "max_epochs": model_cfg["max_epochs"],
            "patience": model_cfg["patience"],
            "test_fraction": config["evaluate"]["test_fraction"],
        }
        model = train_model(X_train, y_train, model, train_config)

        ckpt_path = f"outputs/checkpoints/model_n{n_size}.pt"
        torch.save(model.state_dict(), ckpt_path)
        logger.info("  → checkpoint saved to %s", ckpt_path)

        # ── Step 7: Predict, map indices → k values, evaluate, plot ───────
        logger.info("[Step 7] Evaluating model for n=%d …", n_size)
        pred_indices = predict(model, X_test)
        true_indices = y_test.numpy()

        # Map class indices back to actual k values
        k_pred_values = k_grid[np.clip(pred_indices, 0, n_classes - 1)]
        k_true_values = k_grid[np.clip(true_indices, 0, n_classes - 1)]

        results = evaluate_all(
            test_data=test_diags,
            k_pred=k_pred_values,
            k_true=k_true_values,
            config=config["evaluate"],
        )

        logger.info("  → Agreement rates: %s", results['agreement'])
        logger.info("  → Quantile RMSE:   %.4f", results['quantile_rmse'])

        fig_dir = f"outputs/figures/n{n_size}"
        plot_results(results, test_diags, fig_dir)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
