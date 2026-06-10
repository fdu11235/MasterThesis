"""Synthetic ES sensitivity to the high-xi historical-simulation trigger.

Mirrors analysis/reconcile_synthetic_eval.py exactly (same seed=42 split, same
deployed CNN, same k_pred), but recomputes the deployed Expected Shortfall with
the fallback trigger set to several values instead of the hard-coded 0.7. This
quantifies the claim in the Outlook that lowering the trigger to 0.5 or 0.3
worsens the synthetic ES accuracy.

Writes:
  outputs/trigger_sensitivity.txt              — per-family and aggregate table
  outputs/figures/results_chapter/trigger_sensitivity.png  — figure
"""

import os
import pickle
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

plt.style.use("ggplot")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluate import pot_quantile, true_es
from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.train import predict

CONFIG_PATH = os.path.join(ROOT, "config", "default.yaml")
DIAG_PATH = os.path.join(ROOT, "outputs", "data", "diagnostics.pkl")
CKPT_PATH = os.path.join(ROOT, "outputs", "checkpoints", "model_regression.pt")
OUT_TXT = os.path.join(ROOT, "outputs", "trigger_sensitivity.txt")
FIG_DIR = os.path.join(ROOT, "outputs", "figures", "results_chapter")
os.makedirs(FIG_DIR, exist_ok=True)

TRIGGERS = [0.7, 0.5, 0.3]


def pot_es_trigger(sorted_desc, k, xi, beta, n, p, trigger):
    """Deployed pot_es but with a configurable high-xi trigger (default 0.7)."""
    if xi <= trigger:
        var_est = pot_quantile(sorted_desc, k, xi, beta, n, p)
        u = sorted_desc[k]
        if abs(xi) < 1e-8:
            return var_est + beta
        one_minus_xi = max(1 - xi, 0.05)
        return (var_est + beta - xi * u) / one_minus_xi
    n_eff = len(sorted_desc)
    m = min(max(int(np.ceil(n_eff * (1.0 - p))), 1), n_eff)
    return float(np.mean(sorted_desc[:m]))


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    test_frac = config["evaluate"]["test_fraction"]
    p = config["evaluate"]["quantile_p"]

    with open(DIAG_PATH, "rb") as f:
        all_diag = pickle.load(f)
    print(f"{len(all_diag)} diagnostics loaded")

    X, y, meta = build_dataset_regression(all_diag, config)
    N = len(X)
    torch.manual_seed(42)
    perm = torch.randperm(N)
    test_size = int(N * test_frac)
    test_idx = perm[:test_size]
    test_meta = [meta[i] for i in test_idx.tolist()]
    test_diags = [all_diag[i] for i in test_idx.tolist()]
    X_test = X[test_idx]
    print(f"Test set: {len(X_test)} samples")

    mc = config["model"]
    in_channels = len(config.get("features", {}).get("columns", [0, 1, 2, 3, 4, 5, 6]))
    model = ThresholdCNN(
        in_channels=in_channels, channels=mc["channels"],
        kernel_size=mc["kernel_size"], dropout=mc["dropout"],
        pool_sizes=mc.get("pool_sizes"), task="regression",
    )
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    model.eval()

    y_pred_norm = predict(model, X_test, task="regression")
    k_pred = np.array([
        int(np.clip(round(m["k_min"] + yp * (m["k_max"] - m["k_min"])),
                    m["k_min"], m["k_max"]))
        for yp, m in zip(y_pred_norm, test_meta)
    ])

    # Per-family relative errors at each trigger; also count fallback routing.
    rel_by_fam = {t: defaultdict(list) for t in TRIGGERS}
    rel_all = {t: [] for t in TRIGGERS}
    fallback_frac = {t: 0 for t in TRIGGERS}
    n_used = 0

    for i, (ds, diag) in enumerate(test_diags):
        fam = ds["dist_type"]
        sorted_desc = np.sort(ds["samples"])[::-1]
        k = int(k_pred[i])
        k_idx = min(int(np.searchsorted(diag["k_grid"], k)), len(diag["params"]) - 1)
        xi, beta = diag["params"][k_idx]
        if np.isnan(xi) or np.isnan(beta):
            continue
        xi = float(np.clip(xi, -0.5, 0.95))  # same clamp as evaluate_all
        n = ds["n"]
        es_true = true_es(fam, ds["params"], p)
        if es_true is None or es_true <= 0 or np.isnan(es_true):
            continue
        n_used += 1
        for t in TRIGGERS:
            es_est = pot_es_trigger(sorted_desc, k, xi, beta, n, p, t)
            rel = (es_est - es_true) / es_true
            rel_by_fam[t][fam].append(rel)
            rel_all[t].append(rel)
            if xi > t:
                fallback_frac[t] += 1

    def rrmse(a):
        a = np.asarray(a)
        return float(np.sqrt(np.mean(a ** 2)) * 100)

    fams = sorted(rel_by_fam[TRIGGERS[0]].keys(),
                  key=lambda f: rrmse(rel_by_fam[0.7][f]))

    # Build text table
    lines = []
    lines.append(f"Synthetic ES RelRMSE vs fallback trigger "
                 f"(held-out test split, n_used={n_used} samples, p={p})\n")
    header = f"{'family':<22} " + "  ".join(f"xi>{t:<5}" for t in TRIGGERS)
    lines.append(header)
    lines.append("-" * len(header))
    for fam in fams:
        cells = "  ".join(f"{rrmse(rel_by_fam[t][fam]):>6.1f}" for t in TRIGGERS)
        lines.append(f"{fam:<22} {cells}")
    lines.append("-" * len(header))
    agg = "  ".join(f"{rrmse(rel_all[t]):>6.1f}" for t in TRIGGERS)
    lines.append(f"{'AGGREGATE':<22} {agg}")
    lines.append("")
    lines.append("Fraction of test windows routed to historical-sim fallback:")
    for t in TRIGGERS:
        lines.append(f"  trigger xi>{t}: {fallback_frac[t] / n_used * 100:5.1f}%  "
                     f"({fallback_frac[t]}/{n_used})")
    text = "\n".join(lines)
    print()
    print(text)
    with open(OUT_TXT, "w") as f:
        f.write(text + "\n")
    print(f"\nWrote {OUT_TXT}")

    # Figure: two panels — small synthetic cost vs large routing change
    xs = list(range(len(TRIGGERS)))
    labels = [f"$\\hat{{\\xi}} > {t}$" for t in TRIGGERS]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: synthetic aggregate + the three genuine heavy tails that worsen.
    heavy = ["two_pareto", "frechet", "log_gamma", "pareto"]
    axL.plot(xs, [rrmse(rel_all[t]) for t in TRIGGERS], "o-", lw=2.6,
             color="black", label="Aggregate", zorder=5)
    for t, x in zip(TRIGGERS, xs):
        axL.annotate(f"{rrmse(rel_all[t]):.1f}", (x, rrmse(rel_all[t])),
                     textcoords="offset points", xytext=(0, 8), ha="center",
                     fontsize=9, fontweight="bold")
    cmap = plt.get_cmap("tab10")
    for j, fam in enumerate(heavy):
        axL.plot(xs, [rrmse(rel_by_fam[t][fam]) for t in TRIGGERS], "o--",
                 lw=1.5, color=cmap(j + 1), alpha=0.85, label=fam)
    axL.set_xticks(xs)
    axL.set_xticklabels(labels)
    axL.set_xlabel("Fallback trigger")
    axL.set_ylabel("Synthetic ES RelRMSE (%)")
    axL.set_title("Synthetic cost is small (aggregate +0.7 pp)")
    axL.legend(fontsize=8, ncol=2)
    axL.invert_xaxis()  # 0.7 (deployed) on the left

    # Right: fraction of windows rerouted to the empirical fallback.
    fracs = [fallback_frac[t] / n_used * 100 for t in TRIGGERS]
    bars = axR.bar(xs, fracs, color="#C44E52", alpha=0.85, zorder=3)
    for b, fr in zip(bars, fracs):
        axR.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                 f"{fr:.0f}%", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")
    axR.set_xticks(xs)
    axR.set_xticklabels(labels)
    axR.set_xlabel("Fallback trigger")
    axR.set_ylabel("Test windows on empirical estimator (%)")
    axR.set_title("Routing change is large (24% to 61%)")
    axR.set_ylim(0, 75)
    axR.invert_xaxis()

    fig.suptitle("Lowering the high-$\\hat{\\xi}$ trigger reroutes most windows "
                 "for little synthetic gain", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(FIG_DIR, "trigger_sensitivity.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
