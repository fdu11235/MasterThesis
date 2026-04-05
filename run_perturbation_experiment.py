#!/usr/bin/env python
"""Perturbation robustness experiment for CNN threshold selection.

Loads existing pipeline outputs (synthetic datasets, trained CNN), applies
perturbations (random deletion, bootstrap) to test datasets, re-runs POT
diagnostics and CNN prediction, and measures stability of k* predictions.

Usage:
    python run_perturbation_experiment.py --config config/default.yaml
    python run_perturbation_experiment.py --config config/default.yaml --n-jobs 4
"""

import argparse
import logging
import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from joblib import Parallel, delayed

from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.perturbation import perturb_bootstrap, perturb_random_deletion
from src.pot import process_one_dataset
from src.train import predict


def main():
    parser = argparse.ArgumentParser(description="Perturbation robustness experiment.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("perturbation")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", args.config)

    feat_cfg = config.get("features", {})
    in_channels = len(feat_cfg.get("columns", [0, 1, 2, 3, 4, 5, 6]))
    tag = feat_cfg.get("tag", "")
    out_base = f"outputs/{tag}" if tag else "outputs"

    perturb_cfg = config.get("perturbation", {})
    deletion_fractions = perturb_cfg.get("deletion_fractions", [0.05, 0.10, 0.20])
    n_bootstrap_reps = perturb_cfg.get("n_bootstrap_replications", 5)
    perturb_seed = perturb_cfg.get("seed", 12345)

    os.makedirs(f"{out_base}/figures/perturbation", exist_ok=True)

    # ── Load prerequisites ─────────────────────────────────────────────────
    synthetic_path = "outputs/data/synthetic.pkl"
    diagnostics_path = "outputs/data/diagnostics.pkl"
    ckpt_path = f"{out_base}/checkpoints/model_regression.pt"

    for path in [synthetic_path, diagnostics_path, ckpt_path]:
        if not os.path.exists(path):
            logger.error("Required file not found: %s. Run run_pipeline.py first.", path)
            return

    with open(synthetic_path, "rb") as f:
        datasets = pickle.load(f)
    with open(diagnostics_path, "rb") as f:
        all_diagnostics = pickle.load(f)
    logger.info("Loaded %d datasets and %d diagnostics", len(datasets), len(all_diagnostics))

    # ── Load trained CNN ───────────────────────────────────────────────────
    model_cfg = config["model"]
    model = ThresholdCNN(
        in_channels=in_channels,
        channels=model_cfg["channels"],
        kernel_size=model_cfg["kernel_size"],
        dropout=model_cfg["dropout"],
        pool_sizes=model_cfg.get("pool_sizes"),
        task="regression",
    )
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    logger.info("Loaded trained CNN from %s", ckpt_path)

    # ── Reproduce test split (same logic as run_pipeline.py) ──────────────
    X, y, meta = build_dataset_regression(all_diagnostics, config)
    N = len(X)
    test_frac = config["evaluate"]["test_fraction"]

    torch.manual_seed(42)
    perm = torch.randperm(N)
    test_size = int(N * test_frac)
    test_idx = perm[:test_size].tolist()

    logger.info("Test set: %d samples (reproduced from run_pipeline.py split)", len(test_idx))

    # ── Baseline predictions on unperturbed test data ─────────────────────
    X_test = X[test_idx]
    test_meta = [meta[i] for i in test_idx]
    test_datasets = [datasets[i] for i in test_idx]

    y_pred_baseline = predict(model, X_test, task="regression")
    k_pred_baseline = np.array([
        np.clip(round(m["k_min"] + yp * (m["k_max"] - m["k_min"])),
                m["k_min"], m["k_max"])
        for yp, m in zip(y_pred_baseline, test_meta)
    ], dtype=int)

    logger.info("Baseline k* predictions computed (median=%d)", np.median(k_pred_baseline))

    # ── Run perturbation experiments ───────────────────────────────────────
    pot_cfg = dict(config["pot"])
    pot_cfg["decluster"] = False  # synthetic data is i.i.d.

    experiments = []

    # Deletion experiments
    for frac in deletion_fractions:
        experiments.append({
            "name": f"delete_{int(frac*100)}pct",
            "label": f"Delete {int(frac*100)}%",
            "perturb_fn": lambda ds, seed, f=frac: perturb_random_deletion(ds, f, seed),
            "n_reps": 1,
        })

    # Bootstrap experiments
    experiments.append({
        "name": "bootstrap",
        "label": "Bootstrap",
        "perturb_fn": lambda ds, seed: perturb_bootstrap(ds, seed),
        "n_reps": n_bootstrap_reps,
    })

    all_results = {}

    for exp in experiments:
        name = exp["name"]
        logger.info("Running experiment: %s (%d reps) …", name, exp["n_reps"])

        k_deviations = []
        all_k_pred_perturbed = []  # per-rep k* predictions

        for rep in range(exp["n_reps"]):
            # Perturb test datasets
            rep_seed = perturb_seed + rep * 100000
            perturbed_datasets = [
                exp["perturb_fn"](ds, rep_seed + i)
                for i, ds in enumerate(test_datasets)
            ]

            # Re-run POT diagnostics
            perturbed_diags = Parallel(n_jobs=args.n_jobs, verbose=0)(
                delayed(process_one_dataset)(ds, pot_cfg) for ds in perturbed_datasets
            )

            # Save perturbed diagnostics
            diag_save_path = f"outputs/data/perturbed_diags_{name}_rep{rep}.pkl"
            with open(diag_save_path, "wb") as f:
                pickle.dump(perturbed_diags, f)

            # Build features and predict
            X_pert, y_pert, meta_pert = build_dataset_regression(perturbed_diags, config)
            y_pred_pert = predict(model, X_pert, task="regression")
            k_pred_pert = np.array([
                np.clip(round(m["k_min"] + yp * (m["k_max"] - m["k_min"])),
                        m["k_min"], m["k_max"])
                for yp, m in zip(y_pred_pert, meta_pert)
            ], dtype=int)

            deviations = np.abs(k_pred_pert - k_pred_baseline)
            k_deviations.append(deviations)
            all_k_pred_perturbed.append(k_pred_pert)

        k_deviations = np.concatenate(k_deviations)

        # Compute metrics
        agreement_5 = np.mean(k_deviations <= 5)
        agreement_10 = np.mean(k_deviations <= 10)
        mad = np.mean(k_deviations)
        median_dev = np.median(k_deviations)

        all_results[name] = {
            "label": exp["label"],
            "k_deviations": k_deviations,
            "k_pred_baseline": k_pred_baseline,
            "k_pred_perturbed": all_k_pred_perturbed,  # list of arrays, one per rep
            "agreement_5": agreement_5,
            "agreement_10": agreement_10,
            "mad": mad,
            "median_deviation": median_dev,
        }

        logger.info("  %s: agree@5=%.3f  agree@10=%.3f  MAD=%.1f  median_dev=%.1f",
                     name, agreement_5, agreement_10, mad, median_dev)

    # ── Save results ───────────────────────────────────────────────────────
    results_path = f"{out_base}/perturbation_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(all_results, f)
    logger.info("Results saved to %s", results_path)

    # ── Plots ──────────────────────────────────────────────────────────────
    fig_dir = f"{out_base}/figures/perturbation"

    # Box plot of |k* deviations|
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["label"] for r in all_results.values()]
    data = [r["k_deviations"] for r in all_results.values()]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("|k*_perturbed - k*_original|")
    ax.set_title("CNN Threshold Stability Under Perturbation")
    ax.axhline(5, color="green", ls="--", alpha=0.5, label="radius=5")
    ax.axhline(10, color="orange", ls="--", alpha=0.5, label="radius=10")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/k_deviation_boxplot.png", dpi=150)
    plt.close(fig)

    # Agreement rate bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.35
    agree5 = [r["agreement_5"] for r in all_results.values()]
    agree10 = [r["agreement_10"] for r in all_results.values()]
    ax.bar(x - width / 2, agree5, width, label="radius=5")
    ax.bar(x + width / 2, agree10, width, label="radius=10")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Agreement Rate")
    ax.set_title("k* Agreement Rate: Perturbed vs Original")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/agreement_rate_bar.png", dpi=150)
    plt.close(fig)

    logger.info("Plots saved to %s", fig_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
