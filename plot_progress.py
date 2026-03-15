#!/usr/bin/env python
"""Plot autoresearch experiment progress from results.tsv.

Usage:
    python plot_progress.py                        # default: results.tsv → outputs/exp/figures/progress.png
    python plot_progress.py --tsv results.tsv --out progress.png
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_results(tsv_path: str) -> list[dict]:
    """Load results.tsv, skipping real-validation rows."""
    rows = []
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Skip real-validation rows
            if row.get("description", "").startswith("real-validation"):
                continue
            # Skip rows with missing primary metric
            try:
                float(row["mean_rel_rmse"])
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot autoresearch progress.")
    parser.add_argument("--tsv", default="results.tsv", help="Path to results.tsv")
    parser.add_argument("--out", default="outputs/exp/figures/progress.png",
                        help="Output PNG path")
    args = parser.parse_args()

    if not os.path.exists(args.tsv):
        print(f"No results file found: {args.tsv}", file=sys.stderr)
        sys.exit(1)

    rows = load_results(args.tsv)
    if len(rows) < 2:
        print("Need at least 2 experiments to plot.", file=sys.stderr)
        sys.exit(1)

    # Extract data
    exp_ids = list(range(len(rows)))
    mean_rmse = [float(r["mean_rel_rmse"]) for r in rows]
    n1000 = [float(r["n1000"]) if r.get("n1000") else None for r in rows]
    n2000 = [float(r["n2000"]) if r.get("n2000") else None for r in rows]
    n5000 = [float(r["n5000"]) if r.get("n5000") else None for r in rows]
    decisions = [r.get("decision", "") for r in rows]
    descriptions = [r.get("description", "") for r in rows]

    # Compute running best
    running_best = []
    best_so_far = float("inf")
    for val in mean_rmse:
        best_so_far = min(best_so_far, val)
        running_best.append(best_so_far)

    # Colors by decision
    colors = []
    for d in decisions:
        if d == "keep":
            colors.append("#2ecc71")     # green
        elif d == "discard":
            colors.append("#e74c3c")     # red
        elif d == "baseline":
            colors.append("#3498db")     # blue
        else:
            colors.append("#95a5a6")     # gray

    # ── Figure: 3 subplots ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 2, 1]})
    fig.suptitle("Autoresearch Progress", fontsize=16, fontweight="bold", y=0.98)

    # ── Panel 1: Mean RelRMSE with running best ───────────────────────
    ax1 = axes[0]
    ax1.scatter(exp_ids, mean_rmse, c=colors, s=40, zorder=3, edgecolors="white", linewidths=0.5)
    ax1.plot(exp_ids, running_best, color="#2c3e50", linewidth=2, label="Best so far", zorder=2)
    ax1.axhline(y=mean_rmse[0], color="#3498db", linestyle="--", alpha=0.5, label=f"Baseline ({mean_rmse[0]:.2f}%)")

    # Annotate best point
    best_idx = np.argmin(mean_rmse)
    ax1.annotate(
        f"{mean_rmse[best_idx]:.2f}%\n{descriptions[best_idx]}",
        xy=(best_idx, mean_rmse[best_idx]),
        xytext=(10, -25),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#2c3e50"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2ecc71", alpha=0.3),
    )

    ax1.set_ylabel("Mean Relative RMSE (%)", fontsize=11)
    ax1.set_title("Primary Metric: Mean RelRMSE (lower is better)", fontsize=12)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Custom legend for decisions
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Baseline'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='Keep'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Discard'),
    ]
    ax1.legend(handles=legend_elements + [
        Line2D([0], [0], color='#2c3e50', linewidth=2, label='Best so far'),
        Line2D([0], [0], color='#3498db', linestyle='--', alpha=0.5, label=f'Baseline ({mean_rmse[0]:.2f}%)'),
    ], loc="upper right", fontsize=8, ncol=2)

    # ── Panel 2: Per-n breakdown ──────────────────────────────────────
    ax2 = axes[1]
    valid_n1000 = [(i, v) for i, v in zip(exp_ids, n1000) if v is not None]
    valid_n2000 = [(i, v) for i, v in zip(exp_ids, n2000) if v is not None]
    valid_n5000 = [(i, v) for i, v in zip(exp_ids, n5000) if v is not None]

    if valid_n1000:
        ax2.plot(*zip(*valid_n1000), marker=".", markersize=4, label="n=1000", alpha=0.7, color="#e74c3c")
    if valid_n2000:
        ax2.plot(*zip(*valid_n2000), marker=".", markersize=4, label="n=2000", alpha=0.7, color="#f39c12")
    if valid_n5000:
        ax2.plot(*zip(*valid_n5000), marker=".", markersize=4, label="n=5000", alpha=0.7, color="#2ecc71")

    ax2.set_ylabel("Relative RMSE (%)", fontsize=11)
    ax2.set_title("Per Sample-Size Breakdown", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Keep/discard timeline ────────────────────────────────
    ax3 = axes[2]
    bar_colors = colors
    ax3.bar(exp_ids, [1] * len(exp_ids), color=bar_colors, edgecolor="none", width=1.0)
    ax3.set_yticks([])
    ax3.set_xlabel("Experiment #", fontsize=11)
    ax3.set_title("Decision Timeline", fontsize=12)

    # Add keep/discard counts
    n_keep = sum(1 for d in decisions if d == "keep")
    n_discard = sum(1 for d in decisions if d == "discard")
    total = len(decisions) - 1  # exclude baseline
    if total > 0:
        ax3.text(0.02, 0.5,
                 f"Keep: {n_keep}/{total} ({100*n_keep/total:.0f}%)  |  "
                 f"Discard: {n_discard}/{total} ({100*n_discard/total:.0f}%)  |  "
                 f"Improvement: {mean_rmse[0]:.2f}% → {min(mean_rmse):.2f}% "
                 f"({mean_rmse[0] - min(mean_rmse):+.2f}pp)",
                 transform=ax3.transAxes, fontsize=9, va="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Shared x-axis formatting
    for ax in axes:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Progress plot saved to {args.out}")


if __name__ == "__main__":
    main()
