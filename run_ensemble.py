#!/usr/bin/env python
"""Ensemble training: train N models with different seeds, average predictions.

Usage:
    python run_ensemble.py --config config/default.yaml
    python run_ensemble.py --config config/default.yaml --n-seeds 5
"""

import argparse
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import yaml

from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.train import train_model, predict
from src.evaluate import evaluate_all, plot_results


SEEDS = [42, 123, 456, 789, 1024]


def set_all_seeds(seed):
    """Set seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train ensemble of N models with different seeds.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of ensemble members (default: 5)")
    args = parser.parse_args()

    # ── Logging ────────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/pipeline_ensemble.log"

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w"),
        ],
    )
    logger = logging.getLogger("ensemble")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", args.config)

    # ── Feature config ─────────────────────────────────────────────────
    feat_cfg = config.get("features", {})
    in_channels = len(feat_cfg.get("columns", [0, 1, 2, 3, 4, 5, 6]))
    tag = feat_cfg.get("tag", "")
    out_base = f"outputs/{tag}" if tag else "outputs"

    os.makedirs(f"{out_base}/checkpoints", exist_ok=True)
    os.makedirs(f"{out_base}/figures/ensemble", exist_ok=True)

    # ── Load cached data ─────────────────────────────────────────────────
    diagnostics_path = "outputs/data/diagnostics.pkl"
    if not os.path.exists(diagnostics_path):
        logger.error("No cached diagnostics found at %s. Run run_pipeline.py first.", diagnostics_path)
        return

    logger.info("Loading cached diagnostics from %s", diagnostics_path)
    with open(diagnostics_path, "rb") as f:
        all_diagnostics = pickle.load(f)
    logger.info("  → %d diagnostics loaded", len(all_diagnostics))

    # ── Build dataset ─────────────────────────────────────────────────────
    model_cfg = config["model"]
    test_frac = config["evaluate"]["test_fraction"]

    logger.info("Building unified regression dataset …")
    X, y, meta = build_dataset_regression(all_diagnostics, config)
    logger.info("  → X %s, y %s", tuple(X.shape), tuple(y.shape))

    # ── Fixed train/test split (same as run_pipeline.py) ──────────────────
    N = len(X)
    torch.manual_seed(42)
    perm = torch.randperm(N)
    test_size = int(N * test_frac)
    test_idx = perm[:test_size]
    train_idx = perm[test_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    test_meta = [meta[i] for i in test_idx.tolist()]
    test_diags_all = [all_diagnostics[i] for i in test_idx.tolist()]

    logger.info("Train/test split: %d train, %d test", len(X_train), len(X_test))

    # ── Train ensemble ────────────────────────────────────────────────────
    seeds = SEEDS[:args.n_seeds]
    all_y_pred_norm = []
    all_histories = []

    for i, seed in enumerate(seeds):
        logger.info("═══ Training model %d/%d (seed=%d) ═══", i + 1, len(seeds), seed)
        set_all_seeds(seed)

        model = ThresholdCNN(
            in_channels=in_channels,
            channels=model_cfg["channels"],
            kernel_size=model_cfg["kernel_size"],
            dropout=model_cfg["dropout"],
            pool_sizes=model_cfg.get("pool_sizes"),
            task="regression",
        )

        train_config = {
            "lr": model_cfg["lr"],
            "batch_size": model_cfg["batch_size"],
            "max_epochs": model_cfg["max_epochs"],
            "patience": model_cfg["patience"],
            "test_fraction": config["evaluate"]["test_fraction"],
            "loss_type": model_cfg.get("loss_type", "smooth_l1"),
            "asymmetric_weight": model_cfg.get("asymmetric_weight", 2.0),
        }

        model, history = train_model(X_train, y_train, model, train_config,
                                      task="regression")
        all_histories.append(history)

        # Save individual checkpoint
        ckpt_path = f"{out_base}/checkpoints/model_ensemble_seed{seed}.pt"
        torch.save(model.state_dict(), ckpt_path)
        logger.info("  → checkpoint saved to %s", ckpt_path)

        # Predict
        y_pred_norm = predict(model, X_test, task="regression")
        all_y_pred_norm.append(y_pred_norm)

        best_val = min(history["val_loss"])
        n_epochs = len(history["val_loss"])
        logger.info("  → seed=%d: best_val_loss=%.4f, epochs=%d", seed, best_val, n_epochs)

    # ── Ensemble predictions ──────────────────────────────────────────────
    all_y_pred_norm = np.array(all_y_pred_norm)  # (n_seeds, N_test)
    ensemble_pred_norm = all_y_pred_norm.mean(axis=0)
    ensemble_std_norm = all_y_pred_norm.std(axis=0)

    logger.info("Ensemble prediction std: mean=%.4f, median=%.4f, max=%.4f",
                ensemble_std_norm.mean(), np.median(ensemble_std_norm), ensemble_std_norm.max())

    # ── Denormalize & evaluate ────────────────────────────────────────────
    def denormalize_k(y_pred_norm, meta_list):
        return np.array([
            np.clip(round(m["k_min"] + yp * (m["k_max"] - m["k_min"])),
                    m["k_min"], m["k_max"])
            for yp, m in zip(y_pred_norm, meta_list)
        ], dtype=int)

    k_true_values = denormalize_k(y_test.numpy(), test_meta)

    # Evaluate ensemble
    k_ensemble = denormalize_k(ensemble_pred_norm, test_meta)

    # Also evaluate each individual model for comparison
    individual_results = []
    for i, y_pred in enumerate(all_y_pred_norm):
        k_pred = denormalize_k(y_pred, test_meta)
        groups = defaultdict(lambda: {"idx": []})
        for j, m in enumerate(test_meta):
            groups[m["n"]]["idx"].append(j)

        seed_results = {}
        for n_size in sorted(groups.keys()):
            idx = groups[n_size]["idx"]
            res = evaluate_all(
                test_data=[test_diags_all[j] for j in idx],
                k_pred=k_pred[idx],
                k_true=k_true_values[idx],
                config=config["evaluate"],
            )
            seed_results[n_size] = res
        individual_results.append(seed_results)

    # Evaluate ensemble
    groups = defaultdict(lambda: {"idx": []})
    for j, m in enumerate(test_meta):
        groups[m["n"]]["idx"].append(j)

    ensemble_results = {}
    for n_size in sorted(groups.keys()):
        idx = groups[n_size]["idx"]
        grp_k_pred = k_ensemble[idx]
        grp_k_true = k_true_values[idx]
        grp_diags = [test_diags_all[j] for j in idx]

        results = evaluate_all(
            test_data=grp_diags,
            k_pred=grp_k_pred,
            k_true=grp_k_true,
            config=config["evaluate"],
        )
        ensemble_results[n_size] = results

        logger.info("  n=%d (%d samples):", n_size, len(idx))
        logger.info("    [ENSEMBLE] Rel. RMSE: %.2f%%, ES Rel. RMSE: %.2f%%",
                     results['relative_rmse'] * 100, results['es_relative_rmse'] * 100)

        # Compare with individual models
        indiv_rel_rmses = [individual_results[s][n_size]['relative_rmse'] for s in range(len(seeds))]
        indiv_es_rmses = [individual_results[s][n_size]['es_relative_rmse'] for s in range(len(seeds))]
        logger.info("    [SINGLE]   Rel. RMSE: %.2f%% ± %.2f%% (mean ± std of %d seeds)",
                     np.mean(indiv_rel_rmses) * 100, np.std(indiv_rel_rmses) * 100, len(seeds))
        logger.info("    [SINGLE]   ES Rel. RMSE: %.2f%% ± %.2f%%",
                     np.mean(indiv_es_rmses) * 100, np.std(indiv_es_rmses) * 100)

        # Per-distribution breakdown
        for dist_type, metrics in results.get('rmse_by_dist', {}).items():
            logger.info("    %s: RelRMSE=%.2f%%, ES_RelRMSE=%.2f%%",
                         dist_type, metrics['relative_rmse'] * 100,
                         metrics['es_relative_rmse'] * 100)

        plot_results(results, grp_diags, f"{out_base}/figures/ensemble/n{n_size}",
                     k_pred=grp_k_pred, k_true=grp_k_true, history=all_histories[0])

    # ── Uncertainty analysis: ensemble std vs actual error ─────────────────
    k_errors = np.abs(k_ensemble - k_true_values).astype(float)
    k_std_denorm = np.array([
        ensemble_std_norm[j] * (m["k_max"] - m["k_min"])
        for j, m in enumerate(test_meta)
    ])
    correlation = np.corrcoef(k_std_denorm, k_errors)[0, 1]
    logger.info("Ensemble std vs |k_error| correlation: %.3f", correlation)

    # ── Save results ──────────────────────────────────────────────────────
    results_path = f"{out_base}/ensemble_results.pkl"
    save_data = {
        "seeds": seeds,
        "ensemble_results": ensemble_results,
        "individual_results": individual_results,
        "ensemble_pred_norm": ensemble_pred_norm,
        "ensemble_std_norm": ensemble_std_norm,
        "all_y_pred_norm": all_y_pred_norm,
        "std_error_correlation": correlation,
        "histories": all_histories,
    }
    with open(results_path, "wb") as f:
        pickle.dump(save_data, f)
    logger.info("Ensemble results saved to %s", results_path)
    logger.info("Logs saved to %s", log_file)
    logger.info("Ensemble pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger("ensemble").warning("Interrupted by user.")
