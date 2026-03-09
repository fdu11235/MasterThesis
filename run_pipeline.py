#!/usr/bin/env python
"""Main pipeline entry point — orchestrates Steps 1-7.

Usage:
    python run_pipeline.py --config config/default.yaml
    python run_pipeline.py --config config/default.yaml --fresh
    python run_pipeline.py --config config/default.yaml --n-jobs 4
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

from src.synthetic import generate_all
from src.pot import process_one_dataset
from src.features import build_dataset, build_dataset_regression
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
    logger = logging.getLogger("pipeline")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", args.config)

    if args.fresh:
        logger.info("--fresh flag set: ignoring existing checkpoints")

    # ── Output directories ────────────────────────────────────────────────
    os.makedirs("outputs/data", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # ── Step 1: Generate synthetic data ───────────────────────────────────
    synthetic_path = "outputs/data/synthetic.pkl"
    if not args.fresh and os.path.exists(synthetic_path):
        logger.info("[Step 1] Loading cached synthetic data from %s", synthetic_path)
        with open(synthetic_path, "rb") as f:
            datasets = pickle.load(f)
        logger.info("  → %d datasets loaded (skipped generation)", len(datasets))
    else:
        logger.info("[Step 1] Generating synthetic data …")
        datasets = generate_all(config["synthetic"])
        with open(synthetic_path, "wb") as f:
            pickle.dump(datasets, f)
        logger.info("  → %d datasets saved to %s", len(datasets), synthetic_path)

    # ── Steps 2-4: Sort, build k-grid, compute baseline k* ───────────────
    diagnostics_path = "outputs/data/diagnostics.pkl"
    if not args.fresh and os.path.exists(diagnostics_path):
        logger.info("[Steps 2-4] Loading cached diagnostics from %s", diagnostics_path)
        with open(diagnostics_path, "rb") as f:
            all_diagnostics = pickle.load(f)
        logger.info("  → %d diagnostics loaded (skipped computation)", len(all_diagnostics))
    else:
        logger.info("[Steps 2-4] Computing POT diagnostics (%d datasets, n_jobs=%d) …",
                     len(datasets), args.n_jobs)
        pot_cfg = config["pot"]

        all_diagnostics = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_one_dataset)(ds, pot_cfg) for ds in datasets
        )

        with open(diagnostics_path, "wb") as f:
            pickle.dump(all_diagnostics, f)
        logger.info("  → diagnostics saved to %s", diagnostics_path)

    # ── Determine task mode ─────────────────────────────────────────────
    task = config["model"].get("task", "classification")
    model_cfg = config["model"]
    test_frac = config["evaluate"]["test_fraction"]

    if task == "regression":
        # ── Step 5: Build unified regression dataset ──────────────────────
        logger.info("[Step 5] Building unified regression dataset …")
        X, y, meta = build_dataset_regression(all_diagnostics, config)
        logger.info("  → X %s, y %s", tuple(X.shape), tuple(y.shape))

        # ── Step 6: Single train/test split, train one model ─────────────
        ckpt_path = "outputs/checkpoints/model_regression.pt"
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

        if not args.fresh and os.path.exists(ckpt_path):
            logger.info("[Step 6] Loading cached regression model from %s", ckpt_path)
            model = ThresholdCNN(
                in_channels=7,
                channels=model_cfg["channels"],
                kernel_size=model_cfg["kernel_size"],
                dropout=model_cfg["dropout"],
                task="regression",
            )
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            model.eval()
            history = None
        else:
            logger.info(
                "[Step 6] Training unified regression model (train=%d, test=%d) …",
                len(X_train), len(X_test),
            )
            model = ThresholdCNN(
                in_channels=7,
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
            model, history = train_model(X_train, y_train, model, train_config, task="regression")
            torch.save(model.state_dict(), ckpt_path)
            logger.info("  → checkpoint saved to %s", ckpt_path)

        # ── Step 7: Predict, denormalize, group by n, evaluate ───────────
        logger.info("[Step 7] Evaluating regression model …")
        y_pred_norm = predict(model, X_test, task="regression")

        # Denormalize predictions and true values per sample
        k_pred_values = np.array([
            np.clip(round(m["k_min"] + yp * (m["k_max"] - m["k_min"])),
                    m["k_min"], m["k_max"])
            for yp, m in zip(y_pred_norm, test_meta)
        ], dtype=int)
        k_true_values = np.array([
            np.clip(round(m["k_min"] + yt * (m["k_max"] - m["k_min"])),
                    m["k_min"], m["k_max"])
            for yt, m in zip(y_test.numpy(), test_meta)
        ], dtype=int)

        # Group by sample size for per-n evaluation
        groups = defaultdict(lambda: {"idx": []})
        for i, m in enumerate(test_meta):
            groups[m["n"]]["idx"].append(i)

        for n_size in sorted(groups.keys()):
            idx = groups[n_size]["idx"]
            grp_k_pred = k_pred_values[idx]
            grp_k_true = k_true_values[idx]
            grp_diags = [test_diags_all[i] for i in idx]

            results = evaluate_all(
                test_data=grp_diags,
                k_pred=grp_k_pred,
                k_true=grp_k_true,
                config=config["evaluate"],
            )

            logger.info("  n=%d (%d samples):", n_size, len(idx))
            logger.info("    Agreement rates: %s", results['agreement'])
            logger.info("    Quantile RMSE:   %.4f", results['quantile_rmse'])
            logger.info("    Relative RMSE:   %.2f%%", results['relative_rmse'] * 100)
            ci = results.get('relative_rmse_ci', (float('nan'), float('nan')))
            logger.info("    Rel. RMSE 95%% CI: [%.2f%%, %.2f%%]", ci[0] * 100, ci[1] * 100)
            logger.info("    ES RMSE:         %.4f", results['es_rmse'])
            logger.info("    ES Rel. RMSE:    %.2f%%", results['es_relative_rmse'] * 100)
            es_ci = results.get('es_relative_rmse_ci', (float('nan'), float('nan')))
            logger.info("    ES Rel. RMSE 95%% CI: [%.2f%%, %.2f%%]", es_ci[0] * 100, es_ci[1] * 100)
            logger.info("    k R²:            %.4f", results.get('k_r2', float('nan')))
            logger.info("    k MAE:           %.2f", results.get('k_mae', float('nan')))
            logger.info("    k Median AE:     %.2f", results.get('k_median_ae', float('nan')))
            logger.info("    Quantile MAE:    %.4f", results.get('quantile_mae', float('nan')))
            logger.info("    ES MAE:          %.4f", results.get('es_mae', float('nan')))
            for dist_type, metrics in results.get('rmse_by_dist', {}).items():
                logger.info("    %s: RMSE=%.4f, RelRMSE=%.2f%%, MAE=%.4f, ES_RMSE=%.4f, ES_RelRMSE=%.2f%%, n=%d",
                             dist_type, metrics['rmse'], metrics['relative_rmse'] * 100,
                             metrics.get('mae', float('nan')),
                             metrics['es_rmse'], metrics['es_relative_rmse'] * 100, metrics['count'])

            fig_dir = f"outputs/figures/n{n_size}"
            plot_results(results, grp_diags, fig_dir,
                         k_pred=grp_k_pred, k_true=grp_k_true, history=history)

    else:
        # ── Classification path (original) ────────────────────────────────
        logger.info("[Step 5] Building feature tensors (classification) …")
        grouped_datasets = build_dataset(all_diagnostics, config)
        for n_size, (X, y) in sorted(grouped_datasets.items()):
            logger.info("  → n=%d: X %s, y %s", n_size, tuple(X.shape), tuple(y.shape))

        diag_by_size = defaultdict(list)
        for ds, diag in all_diagnostics:
            diag_by_size[int(ds["n"])].append((ds, diag))

        for n_size in sorted(grouped_datasets.keys()):
            X, y = grouped_datasets[n_size]
            group_diags = diag_by_size[n_size]
            k_grid = np.asarray(group_diags[0][1]["k_grid"])
            n_classes = len(k_grid)
            N = len(X)

            torch.manual_seed(42)
            perm = torch.randperm(N)
            test_size = int(N * test_frac)
            test_idx = perm[:test_size]
            train_idx = perm[test_size:]

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            test_diags = [group_diags[i] for i in test_idx.tolist()]

            ckpt_path = f"outputs/checkpoints/model_n{n_size}.pt"

            if not args.fresh and os.path.exists(ckpt_path):
                logger.info("[Step 6] Loading cached model for n=%d from %s", n_size, ckpt_path)
                model = ThresholdCNN(
                    in_channels=7,
                    channels=model_cfg["channels"],
                    kernel_size=model_cfg["kernel_size"],
                    dropout=model_cfg["dropout"],
                    n_classes=n_classes,
                )
                model.load_state_dict(torch.load(ckpt_path, weights_only=True))
                model.eval()
            else:
                logger.info(
                    "[Step 6] Training model for n=%d (train=%d, test=%d, classes=%d) …",
                    n_size, len(X_train), len(X_test), n_classes,
                )
                model = ThresholdCNN(
                    in_channels=7,
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
                model, _ = train_model(X_train, y_train, model, train_config)
                torch.save(model.state_dict(), ckpt_path)
                logger.info("  → checkpoint saved to %s", ckpt_path)

            logger.info("[Step 7] Evaluating model for n=%d …", n_size)
            pred_indices = predict(model, X_test)
            true_indices = y_test.numpy()

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
            logger.info("  → Relative RMSE:   %.2f%%", results['relative_rmse'] * 100)
            logger.info("  → ES RMSE:         %.4f", results['es_rmse'])
            logger.info("  → ES Rel. RMSE:    %.2f%%", results['es_relative_rmse'] * 100)
            for dist_type, metrics in results.get('rmse_by_dist', {}).items():
                logger.info("    %s: RMSE=%.4f, RelRMSE=%.2f%%, ES_RMSE=%.4f, ES_RelRMSE=%.2f%%, n=%d",
                             dist_type, metrics['rmse'], metrics['relative_rmse'] * 100,
                             metrics['es_rmse'], metrics['es_relative_rmse'] * 100, metrics['count'])

            fig_dir = f"outputs/figures/n{n_size}"
            plot_results(results, test_diags, fig_dir)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger("pipeline").warning(
            "Interrupted by user. Previously completed steps are saved — "
            "re-run to resume from last checkpoint."
        )
