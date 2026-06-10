"""Standalone regeneration of the §5.1 prediction-accuracy figures.

NOTE: analysis/reconcile_synthetic_eval.py is the canonical script that
produces synthetic_test_eval.pkl AND the three figures in one pass. This
helper script exists only as a thin wrapper for the figures, reading the
same canonical pickle. Outputs:
- outputs/figures/n1000/pred_vs_true.png
- outputs/figures/n1000/agreement_rates.png
- outputs/figures/n1000/residuals.png

Styling matches the other Results-chapter figures (analysis/make_results_figures.py):
the "ggplot" style with the shared rcParams (gray panel, white gridlines).
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Shared house style for the Results-chapter figures (see make_results_figures.py).
plt.style.use("ggplot")
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

# Palette consistent with the other Results-chapter figures.
BLUE = "#4C72B0"
ORANGE = "#DD8452"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "outputs", "data", "synthetic_test_eval.pkl")
OUT = os.path.join(ROOT, "outputs", "figures", "n1000")
os.makedirs(OUT, exist_ok=True)

with open(SRC, "rb") as f:
    er = pickle.load(f)
k_pred = np.asarray(er["k_pred"]).astype(float)
k_true = np.asarray(er["k_true"]).astype(float)
rel_err_pct = np.asarray(er["_rel_errors"]) * 100.0
dist_types = list(er["_dist_types"])
k_r2 = float(er["k_r2"])

k_err = k_pred - k_true


# --- Figure 5.1: predicted vs baseline threshold, coloured by family ---------
fig, ax = plt.subplots(figsize=(7, 6))
unique_dists = sorted(set(dist_types))
cmap = plt.get_cmap("tab20")
colors = {dist: cmap(i % cmap.N) for i, dist in enumerate(unique_dists)}
for dist in unique_dists:
    mask = np.array([d == dist for d in dist_types])
    ax.scatter(k_true[mask], k_pred[mask], s=10, alpha=0.55,
               color=colors[dist], label=dist, edgecolors="none", zorder=3)
lo = min(k_true.min(), k_pred.min())
hi = max(k_true.max(), k_pred.max())
ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=0.9, zorder=4)
ax.set_xlabel("$k^*$  (baseline scorer)")
ax.set_ylabel("$k_{\\mathrm{pred}}$  (CNN)")
ax.set_title(f"Predicted vs Baseline Threshold  ($R^{{2}} = {k_r2:.3f}$, "
             f"n=1{chr(8239)}000, {len(k_pred):,} samples)")
ax.legend(fontsize=7, markerscale=1.5, loc="upper left",
          bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pred_vs_true.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)


# --- Figure 5.2: agreement rate by tolerance radius --------------------------
radii = [1, 3, 5, 10, 20]
rates = [float(np.mean(np.abs(k_err) <= r)) for r in radii]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ax.bar([str(r) for r in radii], rates, color=BLUE, zorder=3)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{rate * 100:.1f}%", ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Tolerance radius $r$ (positions on the candidate grid)")
ax.set_ylabel("Agreement rate  $\\mathrm{Pr}(|k_{\\mathrm{pred}} - k^*| \\leq r)$")
ax.set_title(f"Agreement Rate by Tolerance Radius  (n=1{chr(8239)}000, "
             f"{len(k_err):,} samples)")
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "agreement_rates.png"), dpi=150)
plt.close(fig)


# --- Figure 5.3: residual and downstream relative-VaR-error histograms -------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

med_k = float(np.median(k_err))
iqr_lo_k, iqr_hi_k = np.percentile(k_err, [25, 75])
mae_k = float(np.mean(np.abs(k_err)))

ax1.hist(k_err, bins=50, range=(-60, 60),
         color=BLUE, edgecolor="black", linewidth=0.5, alpha=0.9, zorder=3)
ax1.axvline(0, color="black", ls="-", lw=0.8, zorder=4)
ax1.axvline(med_k, color="red", ls="--", lw=1.4,
            label=f"median = {med_k:+.1f}", zorder=4)
ax1.set_xlabel("$k_{\\mathrm{pred}} - k^*$  (positions)")
ax1.set_ylabel("Count")
ax1.set_title(f"Threshold Prediction Residual  (MAE = {mae_k:.1f}, "
              f"IQR = [{iqr_lo_k:+.0f}, {iqr_hi_k:+.0f}])")
ax1.legend(loc="upper right", fontsize=10)
ax1.set_axisbelow(True)

med_q = float(np.median(rel_err_pct))
iqr_lo_q, iqr_hi_q = np.percentile(rel_err_pct, [25, 75])
rrmse_q = float(np.sqrt(np.mean(rel_err_pct ** 2)))

ax2.hist(rel_err_pct, bins=50, range=(-60, 80),
         color=ORANGE, edgecolor="black", linewidth=0.5, alpha=0.9, zorder=3)
ax2.axvline(0, color="black", ls="-", lw=0.8, zorder=4)
ax2.axvline(med_q, color="red", ls="--", lw=1.4,
            label=f"median = {med_q:+.1f}%", zorder=4)
ax2.set_xlabel("Relative VaR error (%)")
ax2.set_ylabel("Count")
ax2.set_title(f"Downstream VaR Relative Error  (RRMSE = {rrmse_q:.1f}%, "
              f"IQR = [{iqr_lo_q:+.0f}%, {iqr_hi_q:+.0f}%])")
ax2.legend(loc="upper right", fontsize=10)
ax2.set_axisbelow(True)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "residuals.png"), dpi=150)
plt.close(fig)

print(f"Regenerated {os.path.join(OUT, 'pred_vs_true.png')}")
print(f"Regenerated {os.path.join(OUT, 'agreement_rates.png')}")
print(f"Regenerated {os.path.join(OUT, 'residuals.png')}")
print()
print("Summary statistics:")
print(f"  k_pred vs k*: R^2 = {k_r2:.3f}")
print(f"  Agreement rates: " + ", ".join(
    f"r={r} -> {rate * 100:.1f}%" for r, rate in zip(radii, rates)))
print(f"  k residual:  median={med_k:+.2f}, MAE={mae_k:.2f}, "
      f"IQR=[{iqr_lo_k:+.1f}, {iqr_hi_k:+.1f}]")
print(f"  Rel VaR err: median={med_q:+.2f}%, RRMSE={rrmse_q:.2f}%, "
      f"IQR=[{iqr_lo_q:+.1f}%, {iqr_hi_q:+.1f}%]")
