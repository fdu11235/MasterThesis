#!/usr/bin/env python
"""Step 8 — Real financial data pipeline with pseudo-labels and VaR backtesting.

Runs both unconditional POT and GARCH-conditional POT (McNeil & Frey, 2000).

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

from src.realdata import (
    prepare_real_datasets, prepare_real_datasets_garch,
    prepare_real_datasets_signsplit, prepare_real_datasets_garch_signsplit,
)
from src.pot import process_one_dataset
from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.train import train_model, predict
from src.evaluate_real import evaluate_real, evaluate_real_signsplit, plot_real_results


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

    # ── Feature config ─────────────────────────────────────────────────
    feat_cfg = config.get("features", {})
    in_channels = len(feat_cfg.get("columns", [0, 1, 2, 3, 4, 5, 6]))
    tag = feat_cfg.get("tag", "")
    out_base = f"outputs/{tag}" if tag else "outputs"

    if args.fresh:
        logger.info("--fresh flag set: ignoring existing checkpoints")

    # ── Output directories ────────────────────────────────────────────────
    os.makedirs("outputs/data", exist_ok=True)
    os.makedirs(f"{out_base}/checkpoints", exist_ok=True)
    os.makedirs(f"{out_base}/figures/real", exist_ok=True)

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

    # ── Steps 2-4: POT diagnostics (unconditional) ───────────────────────
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

    # ── GARCH-filtered datasets ───────────────────────────────────────────
    garch_ds_path = "outputs/data/real_garch_datasets.pkl"
    if not args.fresh and os.path.exists(garch_ds_path):
        logger.info("[GARCH] Loading cached GARCH-filtered datasets from %s", garch_ds_path)
        with open(garch_ds_path, "rb") as f:
            garch_datasets = pickle.load(f)
        logger.info("  → %d GARCH windows loaded", len(garch_datasets))
    else:
        logger.info("[GARCH] Fitting GARCH(1,1) to each window …")
        garch_datasets = prepare_real_datasets_garch(config, returns_lookup, datasets)
        with open(garch_ds_path, "wb") as f:
            pickle.dump(garch_datasets, f)
        logger.info("  → %d GARCH windows saved to %s", len(garch_datasets), garch_ds_path)

    # ── GARCH POT diagnostics ────────────────────────────────────────────
    garch_diag_path = "outputs/data/real_garch_diagnostics.pkl"
    if not args.fresh and os.path.exists(garch_diag_path):
        logger.info("[GARCH] Loading cached GARCH diagnostics from %s", garch_diag_path)
        with open(garch_diag_path, "rb") as f:
            garch_diagnostics = pickle.load(f)
        logger.info("  → %d GARCH diagnostics loaded", len(garch_diagnostics))
    else:
        logger.info("[GARCH] Computing POT diagnostics on standardized residuals (%d windows) …",
                     len(garch_datasets))
        pot_cfg = config["pot"]
        garch_diagnostics = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_one_dataset)(ds, pot_cfg) for ds in garch_datasets
        )
        with open(garch_diag_path, "wb") as f:
            pickle.dump(garch_diagnostics, f)
        logger.info("  → GARCH diagnostics saved to %s", garch_diag_path)

    # ── Step 5: Build regression features (unconditional) ─────────────────
    logger.info("[Step 5] Building regression dataset …")
    X, y, meta = build_dataset_regression(all_diagnostics, config)
    logger.info("  → X %s, y %s", tuple(X.shape), tuple(y.shape))

    # ── Time-ordered split (by end_date) ──────────────────────────────────
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

    # ── Step 5b: Build GARCH regression features ─────────────────────────
    logger.info("[Step 5b] Building GARCH regression dataset …")
    gX, gy, gmeta = build_dataset_regression(garch_diagnostics, config)
    logger.info("  → GARCH X %s, y %s", tuple(gX.shape), tuple(gy.shape))

    # Time-ordered split for GARCH
    garch_end_dates = [m.get("end_date", "") for m in gmeta]
    garch_sorted_indices = np.argsort(garch_end_dates)
    gn_train = int(len(garch_sorted_indices) * train_frac)
    garch_train_idx = garch_sorted_indices[:gn_train]
    garch_test_idx = garch_sorted_indices[gn_train:]

    gX_train, gy_train = gX[garch_train_idx], gy[garch_train_idx]
    gX_test, gy_test = gX[garch_test_idx], gy[garch_test_idx]
    garch_test_meta = [gmeta[i] for i in garch_test_idx]
    garch_test_diags = [garch_diagnostics[i] for i in garch_test_idx]

    logger.info("  GARCH split: %d train, %d test", len(garch_train_idx), len(garch_test_idx))

    # ── Transfer learning config ─────────────────────────────────────────
    model_cfg = config["model"]
    tl_cfg = config.get("transfer_learning", {})
    tl_enabled = tl_cfg.get("enabled", False)

    # ── Step 6: Train CNN (unconditional) ─────────────────────────────────
    ckpt_path = f"{out_base}/checkpoints/model_real.pt"
    if tl_enabled:
        ckpt_path = f"{out_base}/checkpoints/model_real_transfer.pt"

    if not args.fresh and os.path.exists(ckpt_path):
        logger.info("[Step 6] Loading cached model from %s", ckpt_path)
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
        history = None
    else:
        logger.info("[Step 6] Training regression model (train=%d, test=%d) …",
                     len(X_train), len(X_test))
        model = ThresholdCNN(
            in_channels=in_channels,
            channels=model_cfg["channels"],
            kernel_size=model_cfg["kernel_size"],
            dropout=model_cfg["dropout"],
            pool_sizes=model_cfg.get("pool_sizes"),
            task="regression",
        )

        # Transfer learning: load pretrained weights
        if tl_enabled:
            pretrained_path = tl_cfg.get("pretrained_path", "outputs/checkpoints/model_regression.pt")
            if os.path.exists(pretrained_path):
                pretrained_state = torch.load(pretrained_path, weights_only=True)
                n_loaded, n_skipped = model.load_pretrained_backbone(pretrained_state)
                logger.info("Transfer learning: loaded %d params, skipped %d from %s",
                            n_loaded, n_skipped, pretrained_path)
            else:
                logger.warning("Transfer learning enabled but pretrained not found: %s", pretrained_path)

        train_config = {
            "lr": model_cfg["lr"],
            "batch_size": model_cfg["batch_size"],
            "max_epochs": model_cfg["max_epochs"],
            "patience": model_cfg["patience"],
            "test_fraction": config["evaluate"]["test_fraction"],
            "loss_type": model_cfg.get("loss_type", "smooth_l1"),
            "asymmetric_weight": model_cfg.get("asymmetric_weight", 2.0),
        }
        if tl_enabled:
            train_config["freeze_backbone_epochs"] = tl_cfg.get("freeze_backbone_epochs", 0)
            train_config["backbone_lr_factor"] = tl_cfg.get("backbone_lr_factor", 1.0)

        model, history = train_model(X_train, y_train, model, train_config, task="regression")
        torch.save(model.state_dict(), ckpt_path)
        logger.info("  → checkpoint saved to %s", ckpt_path)

    # ── Step 6b: Train CNN on GARCH-filtered features ─────────────────────
    garch_ckpt_path = f"{out_base}/checkpoints/model_real_garch.pt"
    if tl_enabled:
        garch_ckpt_path = f"{out_base}/checkpoints/model_real_garch_transfer.pt"

    if not args.fresh and os.path.exists(garch_ckpt_path):
        logger.info("[Step 6b] Loading cached GARCH model from %s", garch_ckpt_path)
        garch_model = ThresholdCNN(
            in_channels=in_channels,
            channels=model_cfg["channels"],
            kernel_size=model_cfg["kernel_size"],
            dropout=model_cfg["dropout"],
            pool_sizes=model_cfg.get("pool_sizes"),
            task="regression",
        )
        garch_model.load_state_dict(torch.load(garch_ckpt_path, weights_only=True))
        garch_model.eval()
        garch_history = None
    else:
        logger.info("[Step 6b] Training GARCH regression model (train=%d, test=%d) …",
                     len(gX_train), len(gX_test))
        garch_model = ThresholdCNN(
            in_channels=in_channels,
            channels=model_cfg["channels"],
            kernel_size=model_cfg["kernel_size"],
            dropout=model_cfg["dropout"],
            pool_sizes=model_cfg.get("pool_sizes"),
            task="regression",
        )

        # Transfer learning: load pretrained weights
        if tl_enabled:
            pretrained_path = tl_cfg.get("pretrained_path", "outputs/checkpoints/model_regression.pt")
            if os.path.exists(pretrained_path):
                pretrained_state = torch.load(pretrained_path, weights_only=True)
                n_loaded, n_skipped = garch_model.load_pretrained_backbone(pretrained_state)
                logger.info("Transfer learning (GARCH): loaded %d params, skipped %d from %s",
                            n_loaded, n_skipped, pretrained_path)
            else:
                logger.warning("Transfer learning enabled but pretrained not found: %s", pretrained_path)

        train_config = {
            "lr": model_cfg["lr"],
            "batch_size": model_cfg["batch_size"],
            "max_epochs": model_cfg["max_epochs"],
            "patience": model_cfg["patience"],
            "test_fraction": config["evaluate"]["test_fraction"],
            "loss_type": model_cfg.get("loss_type", "smooth_l1"),
            "asymmetric_weight": model_cfg.get("asymmetric_weight", 2.0),
        }
        if tl_enabled:
            train_config["freeze_backbone_epochs"] = tl_cfg.get("freeze_backbone_epochs", 0)
            train_config["backbone_lr_factor"] = tl_cfg.get("backbone_lr_factor", 1.0)

        garch_model, garch_history = train_model(gX_train, gy_train, garch_model, train_config, task="regression")
        torch.save(garch_model.state_dict(), garch_ckpt_path)
        logger.info("  → GARCH checkpoint saved to %s", garch_ckpt_path)

    # ── Step 7: Predict, denormalize, evaluate ────────────────────────────
    logger.info("[Step 7] Evaluating with VaR backtesting …")

    # Unconditional predictions
    y_pred_norm = predict(model, X_test, task="regression")
    k_pred = np.array([
        int(np.clip(round(m["k_min"] + yp * (m["k_max"] - m["k_min"])),
                     m["k_min"], m["k_max"]))
        for yp, m in zip(y_pred_norm, test_meta)
    ])
    k_baseline = np.array([diag["k_star"] for _, diag in test_diags])

    # GARCH predictions
    gy_pred_norm = predict(garch_model, gX_test, task="regression")
    garch_k_pred = np.array([
        int(np.clip(round(m["k_min"] + yp * (m["k_max"] - m["k_min"])),
                     m["k_min"], m["k_max"]))
        for yp, m in zip(gy_pred_norm, garch_test_meta)
    ])
    garch_k_baseline = np.array([diag["k_star"] for _, diag in garch_test_diags])

    # Extract test-set window dicts and diagnostics dicts
    test_ds = [ds for ds, _ in test_diags]
    test_diag_dicts = [diag for _, diag in test_diags]

    garch_test_ds = [ds for ds, _ in garch_test_diags]
    garch_test_diag_dicts = [diag for _, diag in garch_test_diags]

    # Run VaR backtesting (both unconditional and GARCH-conditional)
    results = evaluate_real(
        test_data=test_ds,
        diagnostics_list=test_diag_dicts,
        k_pred=k_pred,
        k_baseline=k_baseline,
        returns_lookup=returns_lookup,
        config=config,
        garch_test_data=garch_test_ds,
        garch_diagnostics_list=garch_test_diag_dicts,
        garch_k_pred=garch_k_pred,
        garch_k_baseline=garch_k_baseline,
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

    # Multi-level VaR coverage table
    multi_level = results.get("multi_level", {})
    if multi_level:
        logger.info("=" * 60)
        logger.info("Multi-Level VaR Coverage")
        logger.info("=" * 60)
        for method in sorted(multi_level.keys()):
            logger.info("  %s:", method)
            for pl in sorted(multi_level[method].keys()):
                info = multi_level[method][pl]
                vr = info.get("violation_rate", float('nan'))
                expected = info.get("expected", float('nan'))
                kup = info.get("kupiec", {})
                reject = kup.get("reject_5pct", "N/A")
                logger.info("    p=%.3f: VR=%.4f (expected=%.4f), Kupiec reject=%s",
                            pl, vr, expected, reject)
        logger.info("=" * 60)

    # Plot
    plot_real_results(results, f"{out_base}/figures/real",
                      history=history, garch_history=garch_history)

    # ── Sign-split experiments (loss / profit tails) ───────────────────────
    tail_modes = config["realdata"].get("tail_modes", ["abs"])
    for tail_mode in tail_modes:
        if tail_mode == "abs":
            continue  # already done above

        logger.info("=" * 60)
        logger.info("Sign-split experiment: tail_mode=%s", tail_mode)
        logger.info("=" * 60)

        # Prepare sign-split datasets
        ss_ds_path = f"outputs/data/real_datasets_{tail_mode}.pkl"
        if not args.fresh and os.path.exists(ss_ds_path):
            logger.info("[%s] Loading cached sign-split datasets from %s", tail_mode, ss_ds_path)
            with open(ss_ds_path, "rb") as f:
                ss_datasets = pickle.load(f)
        else:
            logger.info("[%s] Preparing sign-split datasets …", tail_mode)
            ss_datasets = prepare_real_datasets_signsplit(
                config, returns_lookup, datasets, tail_mode)
            with open(ss_ds_path, "wb") as f:
                pickle.dump(ss_datasets, f)
            logger.info("  → %d windows saved to %s", len(ss_datasets), ss_ds_path)

        if len(ss_datasets) == 0:
            logger.warning("[%s] No windows produced, skipping", tail_mode)
            continue

        # POT diagnostics on sign-split data
        ss_diag_path = f"outputs/data/real_diagnostics_{tail_mode}.pkl"
        if not args.fresh and os.path.exists(ss_diag_path):
            logger.info("[%s] Loading cached diagnostics from %s", tail_mode, ss_diag_path)
            with open(ss_diag_path, "rb") as f:
                ss_diagnostics = pickle.load(f)
        else:
            logger.info("[%s] Computing POT diagnostics (%d windows) …",
                         tail_mode, len(ss_datasets))
            pot_cfg = config["pot"]
            ss_diagnostics = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(process_one_dataset)(ds, pot_cfg) for ds in ss_datasets
            )
            with open(ss_diag_path, "wb") as f:
                pickle.dump(ss_diagnostics, f)

        # Build features and predict
        ssX, ssy, ss_meta = build_dataset_regression(ss_diagnostics, config)
        logger.info("[%s] Features: X %s", tail_mode, tuple(ssX.shape))

        # Time-ordered split
        ss_end_dates = [m.get("end_date", "") for m in ss_meta]
        ss_sorted = np.argsort(ss_end_dates)
        ss_n_train = int(len(ss_sorted) * train_frac)
        ss_test_idx = ss_sorted[ss_n_train:]

        ssX_test = ssX[ss_test_idx]
        ss_test_meta = [ss_meta[i] for i in ss_test_idx]
        ss_test_diags = [ss_diagnostics[i] for i in ss_test_idx]

        # Predict using the unconditional model (transfer)
        ss_pred_norm = predict(model, ssX_test, task="regression")
        ss_k_pred = np.array([
            int(np.clip(round(m["k_min"] + yp * (m["k_max"] - m["k_min"])),
                         m["k_min"], m["k_max"]))
            for yp, m in zip(ss_pred_norm, ss_test_meta)
        ])
        ss_k_baseline = np.array([diag["k_star"] for _, diag in ss_test_diags])
        ss_test_ds = [ds for ds, _ in ss_test_diags]
        ss_test_diag_dicts = [diag for _, diag in ss_test_diags]

        # Evaluate
        ss_results = evaluate_real_signsplit(
            test_data=ss_test_ds,
            diagnostics_list=ss_test_diag_dicts,
            k_pred=ss_k_pred,
            k_baseline=ss_k_baseline,
            returns_lookup=returns_lookup,
            config=config,
            tail_mode=tail_mode,
        )

        # Save
        ss_results_path = f"{out_base}/real_results_{tail_mode}.pkl"
        with open(ss_results_path, "wb") as f:
            pickle.dump(ss_results, f)
        logger.info("[%s] Results saved to %s", tail_mode, ss_results_path)

        # ── GARCH + sign-split ─────────────────────────────────────────────
        logger.info("[%s] GARCH + sign-split …", tail_mode)
        gss_ds_path = f"outputs/data/real_garch_datasets_{tail_mode}.pkl"
        if not args.fresh and os.path.exists(gss_ds_path):
            logger.info("[%s] Loading cached GARCH sign-split datasets from %s", tail_mode, gss_ds_path)
            with open(gss_ds_path, "rb") as f:
                gss_datasets = pickle.load(f)
        else:
            gss_datasets = prepare_real_datasets_garch_signsplit(
                config, returns_lookup, datasets, tail_mode)
            with open(gss_ds_path, "wb") as f:
                pickle.dump(gss_datasets, f)

        if len(gss_datasets) == 0:
            logger.warning("[%s] No GARCH sign-split windows, skipping", tail_mode)
            continue

        gss_diag_path = f"outputs/data/real_garch_diagnostics_{tail_mode}.pkl"
        if not args.fresh and os.path.exists(gss_diag_path):
            logger.info("[%s] Loading cached GARCH sign-split diagnostics", tail_mode)
            with open(gss_diag_path, "rb") as f:
                gss_diagnostics = pickle.load(f)
        else:
            logger.info("[%s] Computing GARCH sign-split POT diagnostics (%d windows) …",
                         tail_mode, len(gss_datasets))
            pot_cfg = config["pot"]
            gss_diagnostics = Parallel(n_jobs=args.n_jobs, verbose=10)(
                delayed(process_one_dataset)(ds, pot_cfg) for ds in gss_datasets
            )
            with open(gss_diag_path, "wb") as f:
                pickle.dump(gss_diagnostics, f)

        gssX, gssy, gss_meta = build_dataset_regression(gss_diagnostics, config)
        logger.info("[%s] GARCH sign-split features: X %s", tail_mode, tuple(gssX.shape))

        gss_end_dates = [m.get("end_date", "") for m in gss_meta]
        gss_sorted = np.argsort(gss_end_dates)
        gss_n_train = int(len(gss_sorted) * train_frac)
        gss_test_idx = gss_sorted[gss_n_train:]

        gssX_test = gssX[gss_test_idx]
        gss_test_meta = [gss_meta[i] for i in gss_test_idx]
        gss_test_diags = [gss_diagnostics[i] for i in gss_test_idx]

        gss_pred_norm = predict(garch_model, gssX_test, task="regression")
        gss_k_pred = np.array([
            int(np.clip(round(m["k_min"] + yp * (m["k_max"] - m["k_min"])),
                         m["k_min"], m["k_max"]))
            for yp, m in zip(gss_pred_norm, gss_test_meta)
        ])
        gss_k_baseline = np.array([diag["k_star"] for _, diag in gss_test_diags])
        gss_test_ds = [ds for ds, _ in gss_test_diags]
        gss_test_diag_dicts = [diag for _, diag in gss_test_diags]

        gss_results = evaluate_real_signsplit(
            test_data=gss_test_ds,
            diagnostics_list=gss_test_diag_dicts,
            k_pred=gss_k_pred,
            k_baseline=gss_k_baseline,
            returns_lookup=returns_lookup,
            config=config,
            tail_mode=tail_mode,
        )

        gss_results_path = f"{out_base}/real_results_garch_{tail_mode}.pkl"
        with open(gss_results_path, "wb") as f:
            pickle.dump(gss_results, f)
        logger.info("[%s] GARCH sign-split results saved to %s", tail_mode, gss_results_path)

    logger.info("Real-data pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger("real_pipeline").warning(
            "Interrupted by user. Previously completed steps are saved — "
            "re-run to resume from last checkpoint."
        )
