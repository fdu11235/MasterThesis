"""Generate illustrative figures for the Methodology chapter.

Three standalone, read-only figures:
A. Empirical log-survival of representative synthetic distributions.
B. POT diagnostic channels along the k-grid for a representative sample.
C. Original versus perturbed (random deletion, bootstrap) samples.

Reads only from existing pickles in `outputs/data/`. Writes PNGs to
`outputs/figures/methodology_chapter/`.

Run: python scripts/make_methodology_figures.py
"""

from __future__ import annotations

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "figures", "methodology_chapter")
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use("ggplot")
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


def _empirical_log_survival(x):
    """Return (log x_sorted, log P(X >= x)) on log-log axes, dropping x <= 0."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    x_sorted = np.sort(x)
    n = len(x_sorted)
    ranks = np.arange(n, 0, -1)
    surv = ranks / (n + 1)
    return x_sorted, surv


def _pick_first(synthetic, dist_type, params=None, n=1000):
    for ds in synthetic:
        if ds["n"] != n:
            continue
        if ds["dist_type"] != dist_type:
            continue
        if params is None or all(ds["params"].get(k) == v for k, v in params.items()):
            return ds
    return None


def fig_a_synthetic_distributions():
    with open(os.path.join(ROOT, "outputs/data/synthetic.pkl"), "rb") as f:
        syn = pickle.load(f)

    light = [
        ("lognormal", {"sigma": 1.0}, "lognormal $\\sigma = 1$", "#4C72B0"),
        ("weibull_stretched", {"c": 0.6}, "stretched Weibull $c = 0.6$", "#55A868"),
    ]
    heavy = [
        ("pareto", {"alpha": 2.0}, "Pareto $\\alpha = 2$", "#C44E52"),
        ("frechet", {"c": 3.0}, "Fréchet $c = 3$", "#CCB974"),
        ("log_gamma", {"b": 2.0}, "log-gamma $b = 2$", "#937860"),
        ("burr12", {"c": 2, "d": 1}, "Burr XII $c=2, d=1$", "#64B5CD"),
        ("student_t", {"df": 4}, "Student-$t$ $\\nu = 4$", "#8172B2"),
    ]

    fig, (ax_l, ax_h) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    for dist, params, label, color in light:
        ds = _pick_first(syn, dist, params)
        if ds is None:
            continue
        x_sorted, surv = _empirical_log_survival(ds["samples"])
        ax_l.loglog(x_sorted, surv, label=label, color=color, linewidth=1.5)

    for dist, params, label, color in heavy:
        ds = _pick_first(syn, dist, params)
        if ds is None:
            continue
        x_sorted, surv = _empirical_log_survival(ds["samples"])
        ax_h.loglog(x_sorted, surv, label=label, color=color, linewidth=1.5)

    for ax, title in [(ax_l, "Gumbel MDA  ($\\xi = 0$, rapidly varying)"),
                      (ax_h, "Fréchet MDA  ($\\xi > 0$, power-law)")]:
        ax.set_xlabel("$x$  (log scale)")
        ax.set_ylabel("$\\hat{S}(x) = P(X \\geq x)$")
        ax.set_title(title)
        ax.legend(loc="lower left", frameon=True)
        ax.grid(True, which="both", alpha=0.35)

    fig.suptitle("Empirical survival function of representative synthetic distributions, "
                 "$n = 1{,}000$", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "synthetic_log_survival.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_b_pot_diagnostics():
    with open(os.path.join(ROOT, "outputs/data/diagnostics.pkl"), "rb") as f:
        diag_pairs = pickle.load(f)

    # Find one Pareto alpha=2 sample at n=1000.
    chosen = None
    for ds, diag in diag_pairs:
        if (ds["n"] == 1000 and ds["dist_type"] == "pareto"
                and ds["params"].get("alpha") == 2.0):
            chosen = (ds, diag)
            break
    if chosen is None:
        raise RuntimeError("No Pareto alpha=2.0 sample at n=1000 in diagnostics.pkl")
    ds, diag = chosen

    k_grid = np.asarray(diag["k_grid"])
    params = np.asarray(diag["params"])  # (L, 2): xi, beta
    xi_mle = params[:, 0]
    hill = np.asarray(diag["hill_series"])
    ad = np.asarray(diag["score_gof"])
    qq = np.asarray(diag["qq_residual_series"])
    me_vals = np.asarray(diag["mean_excess_values"])
    me_score = np.asarray(diag["score_mean_excess"])
    s_stab = np.asarray(diag["score_stability"])
    s_pen = np.asarray(diag["score_penalty"])
    s_heavy = np.asarray(diag["score_heavy_tail"])
    total = np.asarray(diag["total_score"])
    k_star = diag["k_star"]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))

    # Panel 1: shape estimates
    ax = axes[0, 0]
    ax.plot(k_grid, xi_mle, label=r"$\hat{\xi}_{\mathrm{MLE}}(k)$", color="#4C72B0")
    ax.plot(k_grid, hill, label=r"$\hat{\xi}_{\mathrm{Hill}}(k)$", color="#C44E52")
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axvline(k_star, color="#444", linestyle="--", linewidth=1.0,
               label=rf"$k^* = {k_star}$")
    ax.set_xlabel("threshold count $k$")
    ax.set_ylabel(r"tail-index estimate $\hat{\xi}$")
    ax.set_title("Panel 1.  Shape estimates (MLE and Hill)")
    ax.legend(loc="upper right", frameon=True)

    # Panel 2: goodness of fit
    ax = axes[0, 1]
    color1, color2 = "#55A868", "#937860"
    line1 = ax.plot(k_grid, ad, color=color1, label=r"$A^2(k)$ Anderson--Darling")[0]
    ax.set_ylabel(r"$A^2$  (lower is better)", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_xlabel("threshold count $k$")
    ax.set_title("Panel 2.  Goodness-of-fit and QQ residual")
    ax.axvline(k_star, color="#444", linestyle="--", linewidth=1.0)
    ax2 = ax.twinx()
    line2 = ax2.plot(k_grid, qq, color=color2, label="QQ-residual RMSE")[0]
    ax2.set_ylabel("QQ-residual RMSE", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)
    ax.legend(handles=[line1, line2], loc="upper right", frameon=True)

    # Panel 3: mean excess
    ax = axes[1, 0]
    color1, color2 = "#8172B2", "#CCB974"
    line1 = ax.plot(k_grid, me_vals, color=color1, label="raw mean excess $e(k)$")[0]
    ax.set_ylabel("mean excess $e(k)$", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_xlabel("threshold count $k$")
    ax.set_title("Panel 3.  Mean excess function and linearity score")
    ax.axvline(k_star, color="#444", linestyle="--", linewidth=1.0)
    ax2 = ax.twinx()
    line2 = ax2.plot(k_grid, me_score, color=color2,
                     label=r"non-linearity $1 - R^2(k)$")[0]
    ax2.set_ylabel(r"$1 - R^2(k)$", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)
    ax.legend(handles=[line1, line2], loc="upper right", frameon=True)

    # Panel 4: composite score
    ax = axes[1, 1]

    def _mm(s):
        s = np.asarray(s)
        rng = s.max() - s.min()
        if rng <= 0:
            return np.zeros_like(s)
        return (s - s.min()) / (rng + 1e-12)

    components = {
        "stability": _mm(s_stab),
        "GoF (AD)": _mm(ad),
        "penalty $1/\\sqrt{k}$": _mm(s_pen),
        "mean-excess  (w=2)": _mm(me_score),
        "heavy-tail  (w=2)": _mm(s_heavy),
    }
    cmap = ["#4C72B0", "#55A868", "#937860", "#CCB974", "#C44E52"]
    for (name, vals), c in zip(components.items(), cmap):
        ax.plot(k_grid, vals, label=name, color=c, linewidth=1.1, alpha=0.85)
    total_n = (total - total.min()) / (total.max() - total.min() + 1e-12)
    ax.plot(k_grid, total_n, label="total Score(k)", color="#222", linewidth=2.0)
    ax.axvline(k_star, color="#444", linestyle="--", linewidth=1.0,
               label=rf"$k^* = {k_star}$")
    ax.set_xlabel("threshold count $k$")
    ax.set_ylabel("normalised score (0 = best)")
    ax.set_title("Panel 4.  Composite score components")
    ax.legend(loc="upper right", frameon=True, fontsize=8.5, ncol=1)

    fig.suptitle("POT diagnostic channels on a Pareto $\\alpha = 2$ sample, "
                 f"$n = {ds['n']}$",
                 y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "diagnostic_channels.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_c_perturbations():
    # Load originals
    with open(os.path.join(ROOT, "outputs/data/diagnostics.pkl"), "rb") as f:
        orig_pairs = pickle.load(f)
    with open(os.path.join(ROOT, "outputs/data/perturbed_diags_delete_10pct_rep0.pkl"),
              "rb") as f:
        del_pairs = pickle.load(f)
    with open(os.path.join(ROOT, "outputs/data/perturbed_diags_bootstrap_rep0.pkl"),
              "rb") as f:
        boot_pairs = pickle.load(f)

    # Find a moderate Pareto sample present across all three sets by matching index
    # The perturbation pickles include 'perturbation' and corresponding params.
    # The augmentation iterates over train_datasets in order; the first Pareto alpha=2
    # sample in the original list should correspond to indices in the same order in
    # the perturbed pickles (one perturb copy per training dataset).
    def _idx_for(pairs, dist, params):
        for i, item in enumerate(pairs):
            ds = item[0] if isinstance(item, tuple) else item
            if (ds.get("dist_type") == dist
                    and ds.get("n") == 1000
                    and all(ds.get("params", {}).get(k) == v for k, v in params.items())):
                return i, ds
        return None, None

    i_o, ds_o = _idx_for(orig_pairs, "pareto", {"alpha": 2.0})
    if ds_o is None:
        raise RuntimeError("No Pareto alpha=2 sample in original diagnostics")
    diag_o = orig_pairs[i_o][1]

    def _find_matching_pair(pairs, dist, params):
        for ds, diag in pairs:
            if (ds.get("dist_type") == dist
                    and all(ds.get("params", {}).get(k) == v for k, v in params.items())):
                return ds, diag
        return None, None

    ds_del, diag_del = _find_matching_pair(del_pairs, "pareto", {"alpha": 2.0})
    ds_boot, diag_boot = _find_matching_pair(boot_pairs, "pareto", {"alpha": 2.0})
    if ds_del is None or ds_boot is None:
        raise RuntimeError("Could not find Pareto alpha=2 in perturbed pickles")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # Panel 1: empirical log-survival
    for label, samples, color, lw in [
        ("original ($n = 1{,}000$)", ds_o["samples"], "#222", 1.8),
        ("random deletion 10%", ds_del["samples"], "#4C72B0", 1.3),
        ("bootstrap resample", ds_boot["samples"], "#C44E52", 1.3),
    ]:
        xs, surv = _empirical_log_survival(samples)
        ax_l.loglog(xs, surv, label=label, color=color, linewidth=lw, alpha=0.9)
    ax_l.set_xlabel("$x$  (log scale)")
    ax_l.set_ylabel("$\\hat{S}(x)$  (log scale)")
    ax_l.set_title("Panel 1.  Empirical survival, original vs perturbed")
    ax_l.legend(loc="lower left", frameon=True)
    ax_l.grid(True, which="both", alpha=0.35)

    # Panel 2: xi_hat(k) for each
    diag_pairs_for_xi = [("original", diag_o, "#222", 1.8)]
    if diag_del is not None:
        diag_pairs_for_xi.append(("deletion 10%", diag_del, "#4C72B0", 1.3))
    if diag_boot is not None:
        diag_pairs_for_xi.append(("bootstrap", diag_boot, "#C44E52", 1.3))

    for label, diag, color, lw in diag_pairs_for_xi:
        k_grid = np.asarray(diag["k_grid"])
        xi_mle = np.asarray(diag["params"])[:, 0]
        ax_r.plot(k_grid, xi_mle, label=label, color=color, linewidth=lw, alpha=0.9)
    ax_r.axhline(0.5, color="#888", linestyle=":", linewidth=1.0, alpha=0.7,
                 label=r"true $\xi = 0.5$ for $\alpha = 2$")
    ax_r.set_xlabel("threshold count $k$")
    ax_r.set_ylabel(r"GPD-MLE $\hat{\xi}(k)$")
    ax_r.set_title("Panel 2.  Shape estimate across perturbations")
    ax_r.legend(loc="upper right", frameon=True)

    fig.suptitle("Perturbation augmentation on a Pareto $\\alpha = 2$ training sample",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "perturbation_augmentation.png"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Writing figures to {OUT_DIR}")
    fig_a_synthetic_distributions()
    fig_b_pot_diagnostics()
    fig_c_perturbations()
    print("Done. Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, f)
        size_kb = os.path.getsize(path) / 1024.0
        print(f"  {f}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
