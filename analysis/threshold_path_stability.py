#!/usr/bin/env python
"""Temporal stability of the threshold path: CNN k_hat vs baseline scorer k*.

Tests the claim that the CNN, being a smooth (approximately Lipschitz) function
of the diagnostic grid, produces a threshold path that varies more smoothly
across heavily-overlapping rolling windows than the composite scorer's argmin,
which can hop between local minima of a flat/multimodal score surface even when
adjacent windows share 99.5% of their data (stride 5 on 1,000-obs windows).

Faithful to run_real_pipeline.py: same cached datasets/diagnostics, the
time-ordered split (train 80% / test 20% by end_date), and the transfer-learned
CNN (model_real_transfer.pt). Threshold paths are reconstructed on the
OUT-OF-SAMPLE test windows only, grouped by ticker and ordered by window_idx.
Jumps are measured only between truly adjacent rolling windows (window_idx
difference of 1).

Primary metric is the grid-invariant normalised threshold
    yhat = (k - k_min) / (k_max - k_min) in [0, 1]
(the space the CNN predicts in), with raw grid-position jumps as a secondary
view (the abs-pipeline grid is near-constant, k_min=30, k_max in [137,140]).

Run: PYTHONPATH=. python analysis/threshold_path_stability.py
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import yaml
from scipy.stats import wilcoxon

from src.features import build_dataset_regression
from src.model import ThresholdCNN
from src.train import predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PKL = os.path.join(ROOT, "outputs", "threshold_path_stability.pkl")
OUT_FIG = os.path.join(ROOT, "outputs", "figures", "results_chapter",
                       "threshold_path_stability.png")
OUT_TABLE = os.path.join(ROOT, "outputs", "tables",
                         "threshold_stability_per_ticker.tex")


def reconstruct_test_paths():
    """Return a list of per-window dicts on the out-of-sample test slice."""
    with open(os.path.join(ROOT, "config/default.yaml")) as f:
        config = yaml.safe_load(f)

    feat_cfg = config.get("features", {})
    in_channels = len(feat_cfg.get("columns", [0, 1, 2, 3, 4, 5, 6]))
    model_cfg = config["model"]
    train_frac = config["realdata"]["train_fraction"]

    with open(os.path.join(ROOT, "outputs/data/real_diagnostics.pkl"), "rb") as f:
        all_diagnostics = pickle.load(f)

    X, y, meta = build_dataset_regression(all_diagnostics, config)

    # Time-ordered split (pipeline lines 147-160)
    end_dates = [m.get("end_date", "") for m in meta]
    sorted_indices = np.argsort(end_dates)
    n_train = int(len(sorted_indices) * train_frac)
    test_idx = sorted_indices[n_train:]
    X_test = X[test_idx]
    test_meta = [meta[i] for i in test_idx]
    test_diags = [all_diagnostics[i] for i in test_idx]

    # Transfer-learned model
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

    rows = []
    for yp, m, (ds, diag) in zip(y_pred_norm, test_meta, test_diags):
        k_min, k_max = m["k_min"], m["k_max"]
        width = max(k_max - k_min, 1)
        k_pred = int(np.clip(round(k_min + yp * (k_max - k_min)), k_min, k_max))
        k_star = int(diag["k_star"])
        rows.append({
            "ticker": ds["ticker"],
            "window_idx": int(ds["window_idx"]),
            "end_date": ds["end_date"],
            "k_min": k_min, "k_max": k_max, "width": width,
            "k_pred": k_pred, "k_star": k_star,
            "yhat_pred": (k_pred - k_min) / width,
            "yhat_star": (k_star - k_min) / width,
            "at_bound_pred": int(k_pred <= k_min or k_pred >= k_max),
            "at_bound_star": int(k_star <= k_min or k_star >= k_max),
        })
    return rows


def adjacent_jumps(rows):
    """Per-ticker, ordered by window_idx, collect jumps between windows whose
    window_idx differ by exactly 1 (truly adjacent rolling windows)."""
    by_ticker = defaultdict(list)
    for r in rows:
        by_ticker[r["ticker"]].append(r)

    jumps = {"raw_pred": [], "raw_star": [],
             "norm_pred": [], "norm_star": [], "ticker": []}
    per_ticker_counts = {}
    for tk, lst in by_ticker.items():
        lst.sort(key=lambda r: r["window_idx"])
        n_adj = 0
        for a, b in zip(lst[:-1], lst[1:]):
            if b["window_idx"] - a["window_idx"] != 1:
                continue  # not adjacent (split boundary / gap)
            n_adj += 1
            jumps["raw_pred"].append(abs(b["k_pred"] - a["k_pred"]))
            jumps["raw_star"].append(abs(b["k_star"] - a["k_star"]))
            jumps["norm_pred"].append(abs(b["yhat_pred"] - a["yhat_pred"]))
            jumps["norm_star"].append(abs(b["yhat_star"] - a["yhat_star"]))
            jumps["ticker"].append(tk)
        per_ticker_counts[tk] = (len(lst), n_adj)
    for k in jumps:
        jumps[k] = np.asarray(jumps[k]) if k != "ticker" else np.asarray(jumps[k], dtype=object)
    return jumps, per_ticker_counts


def summarize(name, dp, ds):
    """dp = CNN jumps, ds = scorer jumps (paired, same transitions)."""
    out = {}
    out["n"] = len(dp)
    out["cnn_mean"] = float(np.mean(dp))
    out["star_mean"] = float(np.mean(ds))
    out["cnn_median"] = float(np.median(dp))
    out["star_median"] = float(np.median(ds))
    out["cnn_p90"] = float(np.percentile(dp, 90))
    out["star_p90"] = float(np.percentile(ds, 90))
    out["cnn_frac_unchanged"] = float(np.mean(dp == 0))
    out["star_frac_unchanged"] = float(np.mean(ds == 0))
    out["ratio_mean"] = out["cnn_mean"] / out["star_mean"] if out["star_mean"] else float("nan")
    # paired test on |Δ| differences
    diff = dp - ds
    nz = diff[diff != 0]
    if len(nz) > 0:
        stat, p = wilcoxon(dp, ds, zero_method="wilcox", alternative="less")
        out["wilcoxon_p_cnn_smaller"] = float(p)
    else:
        out["wilcoxon_p_cnn_smaller"] = float("nan")
    out["frac_cnn_smaller"] = float(np.mean(dp < ds))
    out["frac_cnn_larger"] = float(np.mean(dp > ds))
    out["frac_equal"] = float(np.mean(dp == ds))
    return out


def main():
    print("Reconstructing out-of-sample threshold paths (transfer CNN) ...")
    rows = reconstruct_test_paths()
    print(f"  test windows: {len(rows)}")
    bp = np.mean([r["at_bound_pred"] for r in rows])
    bs = np.mean([r["at_bound_star"] for r in rows])
    print(f"  windows at grid boundary  CNN={bp:.3%}  scorer={bs:.3%}  "
          f"(boundary saturation confound check)")
    kp = np.array([r["k_pred"] for r in rows]); ks = np.array([r["k_star"] for r in rows])
    print(f"  k level   CNN  median={np.median(kp):.0f}  IQR=[{np.percentile(kp,25):.0f},{np.percentile(kp,75):.0f}]")
    print(f"  k level scorer median={np.median(ks):.0f}  IQR=[{np.percentile(ks,25):.0f},{np.percentile(ks,75):.0f}]")
    print(f"  k level std   CNN={kp.std():.1f}  scorer={ks.std():.1f}  (level dispersion, not smoothness)")

    jumps, counts = adjacent_jumps(rows)
    print("\nadjacent transitions per ticker (n_windows_test, n_adjacent):")
    for tk, (nw, na) in sorted(counts.items()):
        print(f"  {tk:10s} windows={nw:4d}  adjacent_transitions={na:4d}")

    n_adj = len(jumps["raw_pred"])
    print(f"\nTotal adjacent transitions used: {n_adj}")

    raw = summarize("raw", jumps["raw_pred"], jumps["raw_star"])
    nrm = summarize("norm", jumps["norm_pred"], jumps["norm_star"])

    def show(title, s, unit):
        print(f"\n=== {title} (n={s['n']}) ===")
        print(f"  mean |Δ|        CNN={s['cnn_mean']:.4f}  scorer={s['star_mean']:.4f}  "
              f"ratio={s['ratio_mean']:.3f}  ({unit})")
        print(f"  median |Δ|      CNN={s['cnn_median']:.4f}  scorer={s['star_median']:.4f}")
        print(f"  p90 |Δ|        CNN={s['cnn_p90']:.4f}  scorer={s['star_p90']:.4f}")
        print(f"  frac unchanged CNN={s['cnn_frac_unchanged']:.3%}  scorer={s['star_frac_unchanged']:.3%}")
        print(f"  per-transition: CNN smaller={s['frac_cnn_smaller']:.3%}  "
              f"larger={s['frac_cnn_larger']:.3%}  equal={s['frac_equal']:.3%}")
        print(f"  Wilcoxon (H1: CNN jump < scorer jump) p={s['wilcoxon_p_cnn_smaller']:.2e}")

    show("RAW grid-position jumps |Δk|", raw, "grid positions")
    show("NORMALISED jumps |Δ yhat|", nrm, "fraction of grid width")

    # Reference: fixed sqrt(n) rule is a constant => zero jumps (trivially smooth
    # but data-blind). Historical-sim uses no threshold. The meaningful contrast
    # is CNN (data-adaptive AND smooth) vs scorer (data-adaptive but jumpy).
    print("\n[reference] fixed sqrt(n) rule: constant k => |Δk|=0 everywhere "
          "(maximally smooth but ignores the window).")

    # per-ticker mean |Δk|
    print("\nPer-ticker mean raw |Δk| (CNN vs scorer):")
    tkarr = jumps["ticker"]
    for tk in sorted(set(tkarr)):
        msk = tkarr == tk
        mp = jumps["raw_pred"][msk].mean(); ms = jumps["raw_star"][msk].mean()
        print(f"  {tk:10s} CNN={mp:.3f}  scorer={ms:.3f}  ratio={mp/ms if ms else float('nan'):.3f}  (n={msk.sum()})")

    result = {"raw": raw, "norm": nrm, "per_ticker_counts": counts,
              "n_adjacent": n_adj, "bound_frac": {"cnn": float(bp), "scorer": float(bs)},
              "rows": rows, "jumps": {k: (v.tolist() if k != "ticker" else list(v))
                                       for k, v in jumps.items()}}
    with open(OUT_PKL, "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved -> {OUT_PKL}")

    write_table(jumps)
    make_figure(rows, jumps)


def write_table(jumps):
    """Per-ticker mean |Delta k| (scorer vs CNN) plus a pooled row, as a LaTeX
    tabular matching the repo's generated-table convention."""
    tkarr = jumps["ticker"]
    rp, rs = jumps["raw_pred"], jumps["raw_star"]
    lines = [r"\begin{tabular}{@{}l r r r r@{}}", r"\hline",
             r"Ticker & Transitions & Scorer mean $|\Delta k|$ & "
             r"CNN mean $|\Delta k|$ & Ratio \\", r"\hline"]
    for tk in sorted(set(tkarr)):
        m = tkarr == tk
        ms, mp = rs[m].mean(), rp[m].mean()
        disp = tk.replace("^", r"\^{}").replace("_", r"\_")
        lines.append(f"{disp} & ${int(m.sum())}$ & ${ms:.2f}$ & ${mp:.2f}$ & "
                     f"${mp/ms:.3f}$ \\\\")
    lines.append(r"\hline")
    lines.append(f"Pooled & ${len(rp)}$ & ${rs.mean():.2f}$ & ${rp.mean():.2f}$ & "
                 f"${rp.mean()/rs.mean():.3f}$ \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    with open(OUT_TABLE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved table -> {OUT_TABLE}")


def make_figure(rows, jumps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # House style, matching analysis/make_results_figures.py
    plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    })
    c_scorer, c_cnn = "#C44E52", "#4C72B0"

    by_ticker = defaultdict(list)
    for r in rows:
        by_ticker[r["ticker"]].append(r)
    for lst in by_ticker.values():
        lst.sort(key=lambda r: r["window_idx"])

    # Path panel: the ticker whose scorer path is jumpiest, so the damping shows.
    tkarr = jumps["ticker"]
    rs = jumps["raw_star"]
    tk = max(set(tkarr), key=lambda t: rs[tkarr == t].mean())
    lst = by_ticker[tk]
    idx = [r["window_idx"] for r in lst]
    kp = [r["k_pred"] for r in lst]
    ks = [r["k_star"] for r in lst]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))

    ax[0].plot(idx, ks, color=c_scorer, lw=1.0, alpha=0.9, label=r"scorer $k^*$")
    ax[0].plot(idx, kp, color=c_cnn, lw=1.3, label=r"CNN $\hat{k}$")
    ax[0].set_xlabel("rolling-window index")
    ax[0].set_ylabel("selected threshold $k$")
    ax[0].set_title(f"Threshold path ({tk}, out-of-sample)")
    ax[0].legend(frameon=True, framealpha=0.9)

    # Complementary CDF (survival) of the adjacent-window jump, log-y. The two
    # rules overlap at small jumps and separate in the large-jump tail.
    rp = jumps["raw_pred"]
    tmax = int(max(rp.max(), rs.max()))
    grid = np.arange(0, tmax + 1)
    ccdf_p = np.array([np.mean(rp >= t) for t in grid])
    ccdf_s = np.array([np.mean(rs >= t) for t in grid])
    ax[1].step(grid, ccdf_s, where="post", color=c_scorer, lw=1.4,
               label=fr"scorer (mean {rs.mean():.2f})")
    ax[1].step(grid, ccdf_p, where="post", color=c_cnn, lw=1.4,
               label=fr"CNN (mean {rp.mean():.2f})")
    ax[1].set_yscale("log")
    ax[1].set_xlim(0, min(tmax, 40))
    ax[1].set_xlabel(r"adjacent-window jump $t$ (grid positions)")
    ax[1].set_ylabel(r"$P(|\Delta k| \geq t)$")
    ax[1].set_title("Jump-size tail (all eight tickers)")
    ax[1].legend(frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_FIG)
    print(f"Saved figure -> {OUT_FIG}")


if __name__ == "__main__":
    main()
