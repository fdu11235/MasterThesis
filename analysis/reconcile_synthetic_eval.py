"""One-shot reconciliation of all synthetic evaluation numbers in the thesis.

Loads the deployed CNN (model_regression.pt, the run_pipeline.py model with
augmentation) and evaluates it on the original held-out test split (seed=42
permutation, no train leakage). Produces:

  outputs/data/synthetic_test_eval.pkl   — full evaluate_all result on test set
  outputs/tables/synthetic_per_family_relrmse.tex   — regenerated table
  outputs/figures/n1000/pred_vs_true.png            — regenerated scatter
  outputs/figures/n1000/agreement_rates.png         — regenerated bars
  outputs/figures/n1000/residuals.png               — regenerated histograms
  outputs/composite_test_decomp.txt                 — printable decomposition
                                                       numbers for the appendix

All §5.1 figures, the per-family table, and the appendix decomposition
sentences derive from this single source after this script runs.
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluate import evaluate_all, pot_es, true_es
from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.train import predict


CONFIG_PATH = os.path.join(ROOT, "config", "default.yaml")
DIAG_PATH = os.path.join(ROOT, "outputs", "data", "diagnostics.pkl")
CKPT_PATH = os.path.join(ROOT, "outputs", "checkpoints", "model_regression.pt")
OUT_PKL = os.path.join(ROOT, "outputs", "data", "synthetic_test_eval.pkl")
TABLE_PATH = os.path.join(ROOT, "outputs", "tables",
                          "synthetic_per_family_relrmse.tex")
FIG_DIR = os.path.join(ROOT, "outputs", "figures", "n1000")
DECOMP_TXT = os.path.join(ROOT, "outputs", "composite_test_decomp.txt")

os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def _xi_beta_at_k(diag, k):
    kg = np.asarray(diag["k_grid"])
    i = min(int(np.searchsorted(kg, k)), len(diag["params"]) - 1)
    return diag["params"][i]


def es_at_k(ds, diag, k, p):
    xi, beta = _xi_beta_at_k(diag, k)
    if np.isnan(xi) or np.isnan(beta):
        return np.nan
    xi = float(np.clip(xi, -0.5, 0.95))
    sd = np.sort(ds["samples"])[::-1]
    k = int(np.clip(k, diag["k_grid"][0], diag["k_grid"][-1]))
    return pot_es(sd, k, xi, beta, len(sd), p)


def oracle_k(ds, diag, p, es_true):
    kg = np.asarray(diag["k_grid"])
    sd = np.sort(ds["samples"])[::-1]
    n = len(sd)
    best_k, best = None, np.inf
    for i, k in enumerate(kg):
        xi, beta = diag["params"][i]
        if np.isnan(xi) or np.isnan(beta):
            continue
        xi_c = float(np.clip(xi, -0.5, 0.95))
        es = pot_es(sd, int(k), xi_c, beta, n, p)
        if np.isnan(es) or es <= 0:
            continue
        v = abs((es - es_true) / es_true)
        if v < best:
            best, best_k = v, int(k)
    return best_k


def main():
    print(f"Loading config from {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    test_frac = config["evaluate"]["test_fraction"]
    p = config["evaluate"]["quantile_p"]

    print(f"Loading cached diagnostics from {DIAG_PATH}")
    with open(DIAG_PATH, "rb") as f:
        all_diag = pickle.load(f)
    print(f"  {len(all_diag)} diagnostics loaded")

    print("Building regression dataset")
    X, y, meta = build_dataset_regression(all_diag, config)
    print(f"  X {tuple(X.shape)}, y {tuple(y.shape)}")

    N = len(X)
    torch.manual_seed(42)
    perm = torch.randperm(N)
    test_size = int(N * test_frac)
    test_idx = perm[:test_size]

    X_test = X[test_idx]
    y_test = y[test_idx]
    test_meta = [meta[i] for i in test_idx.tolist()]
    test_diags = [all_diag[i] for i in test_idx.tolist()]
    print(f"Test set: {len(X_test)} samples")

    # Reproduce the train/test split that run_pipeline.py uses.
    mc = config["model"]
    in_channels = len(config.get("features", {}).get("columns", [0, 1, 2, 3, 4, 5, 6]))
    print(f"Loading checkpoint from {CKPT_PATH}")
    model = ThresholdCNN(
        in_channels=in_channels,
        channels=mc["channels"],
        kernel_size=mc["kernel_size"],
        dropout=mc["dropout"],
        pool_sizes=mc.get("pool_sizes"),
        task="regression",
    )
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    model.eval()

    y_pred_norm = predict(model, X_test, task="regression")
    k_pred = np.array([
        int(np.clip(round(m["k_min"] + yp * (m["k_max"] - m["k_min"])),
                    m["k_min"], m["k_max"]))
        for yp, m in zip(y_pred_norm, test_meta)
    ])
    k_true = np.array([
        int(np.clip(round(m["k_min"] + yt * (m["k_max"] - m["k_min"])),
                    m["k_min"], m["k_max"]))
        for yt, m in zip(y_test.numpy(), test_meta)
    ])

    print("Running evaluate_all on test set")
    results = evaluate_all(test_diags, k_pred, k_true, config["evaluate"])

    print(f"  k R^2: {results.get('k_r2', float('nan')):.4f}")
    print(f"  k MAE: {results.get('k_mae', float('nan')):.2f}")
    print(f"  agreement: {results['agreement']}")
    print(f"  rel RMSE (VaR): {results['relative_rmse'] * 100:.2f}%")
    print(f"  rel RMSE (ES):  {results['es_relative_rmse'] * 100:.2f}%")

    # ── Persist the full result dict ──────────────────────────────────────
    with open(OUT_PKL, "wb") as f:
        pickle.dump(results, f)
    print(f"Wrote {OUT_PKL}")

    # ── Regenerate the per-family table ───────────────────────────────────
    rmse_by = results["rmse_by_dist"]
    # Sort ascending by ES rel RMSE
    rows = sorted(
        ((dist, m["relative_rmse"] * 100, m["es_relative_rmse"] * 100)
         for dist, m in rmse_by.items()),
        key=lambda r: r[2],
    )
    lines = [r"\begin{tabular}{@{}l r r@{}}", r"\hline",
             r"Distribution & VaR rel.\ RMSE & ES rel.\ RMSE \\",
             r"\hline"]
    for dist, vpct, epct in rows:
        dist_label = dist.replace("_", r"\_")
        lines.append(f"{dist_label} & ${vpct:.2f}\\%$ & ${epct:.1f}\\%$ \\\\")
    var_agg = results["relative_rmse"] * 100
    es_agg = results["es_relative_rmse"] * 100
    lines.append(r"\hline")
    lines.append(f"Aggregate & ${var_agg:.2f}\\%$ & ${es_agg:.1f}\\%$ \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    with open(TABLE_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {TABLE_PATH}")

    # ── Regenerate the three §5.1 figures from the same source ────────────
    dist_types = list(results["_dist_types"])
    rel_err_pct = np.asarray(results["_rel_errors"]) * 100.0
    k_err = k_pred.astype(float) - k_true.astype(float)
    k_r2 = float(results["k_r2"])

    # Scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    unique_dists = sorted(set(dist_types))
    cmap = plt.get_cmap("tab20")
    colors = {dist: cmap(i % cmap.N) for i, dist in enumerate(unique_dists)}
    for dist in unique_dists:
        mask = np.array([d == dist for d in dist_types])
        ax.scatter(k_true[mask], k_pred[mask], s=10, alpha=0.55,
                   color=colors[dist], label=dist, edgecolors="none")
    lo = min(k_true.min(), k_pred.min())
    hi = max(k_true.max(), k_pred.max())
    ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=0.8)
    ax.set_xlabel("$k^*$  (baseline scorer)")
    ax.set_ylabel("$k_{\\mathrm{pred}}$  (CNN)")
    ax.set_title(f"Predicted vs Baseline Threshold  "
                 f"($R^{{2}} = {k_r2:.3f}$, n=1{chr(8239)}000, "
                 f"{len(k_pred):,} samples)")
    ax.legend(fontsize=7, markerscale=1.5, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.grid(linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pred_vs_true.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # Agreement rates
    radii = [1, 3, 5, 10, 20]
    rates = [float(np.mean(np.abs(k_err) <= r)) for r in radii]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar([str(r) for r in radii], rates,
                  color="tab:blue", edgecolor="black", linewidth=0.6)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{rate * 100:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Tolerance radius $r$ (positions on the candidate grid)")
    ax.set_ylabel("Agreement rate  "
                  "$\\mathrm{Pr}(|k_{\\mathrm{pred}} - k^*| \\leq r)$")
    ax.set_title(f"Agreement Rate by Tolerance Radius  "
                 f"(n=1{chr(8239)}000, {len(k_err):,} samples)")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "agreement_rates.png"), dpi=150)
    plt.close(fig)

    # Residuals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    med_k = float(np.median(k_err))
    iqr_lo_k, iqr_hi_k = np.percentile(k_err, [25, 75])
    mae_k = float(np.mean(np.abs(k_err)))
    ax1.hist(k_err, bins=50, range=(-60, 60), color="tab:blue",
             edgecolor="black", linewidth=0.5, alpha=0.85)
    ax1.axvline(0, color="black", ls="-", lw=0.8)
    ax1.axvline(med_k, color="red", ls="--", lw=1.2,
                label=f"median = {med_k:+.1f}")
    ax1.set_xlabel("$k_{\\mathrm{pred}} - k^*$  (positions)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Threshold Prediction Residual  "
                  f"(MAE = {mae_k:.1f}, "
                  f"IQR = [{iqr_lo_k:+.0f}, {iqr_hi_k:+.0f}])")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.set_axisbelow(True)
    med_q = float(np.median(rel_err_pct))
    iqr_lo_q, iqr_hi_q = np.percentile(rel_err_pct, [25, 75])
    rrmse_q = float(np.sqrt(np.mean(rel_err_pct ** 2)))
    ax2.hist(rel_err_pct, bins=50, range=(-60, 80), color="tab:orange",
             edgecolor="black", linewidth=0.5, alpha=0.85)
    ax2.axvline(0, color="black", ls="-", lw=0.8)
    ax2.axvline(med_q, color="red", ls="--", lw=1.2,
                label=f"median = {med_q:+.1f}%")
    ax2.set_xlabel("Relative VaR error (%)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Downstream VaR Relative Error  "
                  f"(RRMSE = {rrmse_q:.1f}%, "
                  f"IQR = [{iqr_lo_q:+.0f}%, {iqr_hi_q:+.0f}%])")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "residuals.png"), dpi=150)
    plt.close(fig)

    # ── Composite-tail decomposition on the test set ──────────────────────
    decomp_lines = []
    decomp_lines.append("Composite-tail decomposition on the held-out test "
                        "split (same data as the table / figures above)\n")
    for fam in ("two_pareto", "lognormal_pareto_mix"):
        idx = [i for i, dt in enumerate(dist_types) if dt == fam]
        if not idx:
            decomp_lines.append(f"{fam}: no test samples found\n")
            continue
        fam_diags = [test_diags[i] for i in idx]
        fam_k_pred = [k_pred[i] for i in idx]
        es_pred_arr, es_oracle_arr, k_or_arr, true_es_arr = [], [], [], []
        for (ds, diag), kp in zip(fam_diags, fam_k_pred):
            et = true_es(fam, ds["params"], p)
            if et is None or et <= 0:
                continue
            ko = oracle_k(ds, diag, p, et)
            if ko is None:
                continue
            ep = es_at_k(ds, diag, kp, p)
            eo = es_at_k(ds, diag, ko, p)
            if np.isnan(ep) or np.isnan(eo):
                continue
            es_pred_arr.append((ep - et) / et)
            es_oracle_arr.append((eo - et) / et)
            k_or_arr.append(ko)
            true_es_arr.append(et)
        ep = np.asarray(es_pred_arr)
        eo = np.asarray(es_oracle_arr)
        rrmse_pred = np.sqrt(np.mean(ep ** 2)) * 100
        rrmse_oracle = np.sqrt(np.mean(eo ** 2)) * 100
        sel_share = ((rrmse_pred - rrmse_oracle) / rrmse_pred) * 100
        decomp_lines.append(
            f"{fam}: n={len(ep)} test samples\n"
            f"  RRMSE at CNN threshold:    {rrmse_pred:.2f}%\n"
            f"  RRMSE at oracle threshold: {rrmse_oracle:.2f}%\n"
            f"  Selection share of RRMSE:  ~{sel_share:.0f}%\n"
            f"  Median error at CNN:    {np.median(ep) * 100:+.2f}%\n"
            f"  Median error at oracle: {np.median(eo) * 100:+.2f}%\n"
            f"  Median oracle k:           {int(np.median(k_or_arr))}\n"
        )
    text = "\n".join(decomp_lines)
    print()
    print(text)
    with open(DECOMP_TXT, "w") as f:
        f.write(text)
    print(f"Wrote {DECOMP_TXT}")


if __name__ == "__main__":
    main()
