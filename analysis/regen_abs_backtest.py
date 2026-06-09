#!/usr/bin/env python
"""Regenerate and archive the absolute-tail unconditional backtest.

The main pipeline (run_real_pipeline.py) computes the abs-tail unconditional
backtest via evaluate_real() but, unlike the loss/profit sign-split tracks,
never pickles it. This reproduces that exact computation (same cached datasets,
diagnostics, time-ordered split, and transfer-learned CNN) and saves the result
to outputs/real_results_abs.pkl so the thesis's absolute-tail McNeil-Frey
numbers have a backing artifact.

Faithful to run_real_pipeline.py lines 144-160 (split) and 336-347 (evaluate).
"""
import os
import pickle

import numpy as np
import torch
import yaml

from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.train import predict
from src.evaluate_real import evaluate_real

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    with open(os.path.join(ROOT, "config/default.yaml")) as f:
        config = yaml.safe_load(f)

    feat_cfg = config.get("features", {})
    in_channels = len(feat_cfg.get("columns", [0, 1, 2, 3, 4, 5, 6]))
    model_cfg = config["model"]
    train_frac = config["realdata"]["train_fraction"]

    with open(os.path.join(ROOT, "outputs/data/real_datasets.pkl"), "rb") as f:
        saved = pickle.load(f)
    datasets = saved["datasets"]
    returns_lookup = saved["returns_lookup"]

    with open(os.path.join(ROOT, "outputs/data/real_diagnostics.pkl"), "rb") as f:
        all_diagnostics = pickle.load(f)

    # Build regression features + time-ordered split (pipeline lines 144-160)
    X, y, meta = build_dataset_regression(all_diagnostics, config)
    end_dates = [m.get("end_date", "") for m in meta]
    sorted_indices = np.argsort(end_dates)
    n_train = int(len(sorted_indices) * train_frac)
    test_idx = sorted_indices[n_train:]
    X_test = X[test_idx]
    test_meta = [meta[i] for i in test_idx]
    test_diags = [all_diagnostics[i] for i in test_idx]

    # Transfer-learned model (pipeline uses model_real_transfer.pt when tl enabled)
    tl_enabled = config.get("transfer_learning", {}).get("enabled", False)
    ckpt = "model_real_transfer.pt" if tl_enabled else "model_real.pt"
    model = ThresholdCNN(
        in_channels=in_channels,
        channels=model_cfg["channels"],
        kernel_size=model_cfg["kernel_size"],
        dropout=model_cfg["dropout"],
        pool_sizes=model_cfg.get("pool_sizes"),
        task="regression",
    )
    model.load_state_dict(torch.load(
        os.path.join(ROOT, "outputs/checkpoints", ckpt), weights_only=True))
    model.eval()

    y_pred_norm = predict(model, X_test, task="regression")
    k_pred = np.array([
        int(np.clip(round(m["k_min"] + yp * (m["k_max"] - m["k_min"])),
                    m["k_min"], m["k_max"]))
        for yp, m in zip(y_pred_norm, test_meta)
    ])
    k_baseline = np.array([diag["k_star"] for _, diag in test_diags])
    test_ds = [ds for ds, _ in test_diags]
    test_diag_dicts = [diag for _, diag in test_diags]

    # Unconditional evaluation only (garch args omitted -> unconditional summary
    # for cnn/baseline/fixed/historical_sim is identical to the pipeline's)
    results = evaluate_real(
        test_data=test_ds,
        diagnostics_list=test_diag_dicts,
        k_pred=k_pred,
        k_baseline=k_baseline,
        returns_lookup=returns_lookup,
        config=config,
    )
    results["tail_mode"] = "abs"

    out_path = os.path.join(ROOT, "outputs/real_results_abs.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print("Saved", out_path)
    print("=" * 64)
    for method, s in results["summary"].items():
        mf = s.get("mcneil_frey", {})
        kup = s.get("kupiec", {})
        print(f"{method:18s} VR={s.get('overall_violation_rate'):.4f} "
              f"n_win={s.get('n_windows')} "
              f"Kupiec p={kup.get('p_value'):.4f} rej={kup.get('reject_5pct')} | "
              f"MF t={mf.get('t_stat'):.3f} p={mf.get('p_value'):.4f} "
              f"rej={mf.get('reject_5pct')} nviol={mf.get('n_violations')}")


if __name__ == "__main__":
    main()
