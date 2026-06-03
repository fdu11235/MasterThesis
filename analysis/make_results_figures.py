"""Generate the figures for the Results chapter of the thesis.

Reads existing pickles produced by `run_pipeline.py`,
`run_high_xi_experiment.py`, and `run_real_pipeline.py`. Writes six
PNGs to `outputs/figures/results_chapter/`. Does not modify any other
file.

Run: python scripts/make_results_figures.py
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "figures", "results_chapter")
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

TAIL_GROUPS = {
    "weibull_stretched": ("light", "#4C72B0"),
    "student_t": ("light", "#4C72B0"),
    "lognormal": ("light", "#4C72B0"),
    "burr12": ("moderate", "#55A868"),
    "dagum": ("moderate", "#55A868"),
    "inverse_gamma": ("moderate", "#55A868"),
    "gamma_pareto_splice": ("composite", "#C44E52"),
    "garch_student_t": ("clustered", "#8172B2"),
    "pareto": ("heavy", "#CCB974"),
    "frechet": ("heavy", "#CCB974"),
    "log_gamma": ("heavy", "#CCB974"),
    "lognormal_pareto_mix": ("composite", "#C44E52"),
    "two_pareto": ("composite", "#C44E52"),
}


def fig1_synthetic_es_relrmse():
    data = [
        ("weibull_stretched", 16.5),
        ("student_t", 15.6),
        ("burr12", 21.0),
        ("dagum", 23.3),
        ("inverse_gamma", 23.2),
        ("lognormal", 25.9),
        ("gamma_pareto_splice", 34.4),
        ("garch_student_t", 34.4),
        ("pareto", 54.3),
        ("frechet", 60.6),
        ("log_gamma", 60.8),
        ("lognormal_pareto_mix", 75.6),
        ("two_pareto", 82.5),
    ]
    data.sort(key=lambda x: x[1])
    fams = [d[0] for d in data]
    vals = [d[1] for d in data]
    colors = [TAIL_GROUPS.get(f, ("?", "#888"))[1] for f in fams]
    groups = [TAIL_GROUPS.get(f, ("?", "#888"))[0] for f in fams]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(fams))
    bars = ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("_", " ") for f in fams])
    ax.set_xlabel("ES relative RMSE (%)")
    ax.set_title("CNN Expected Shortfall relative RMSE by distribution, n = 1000")
    for i, (b, v, g) in enumerate(zip(bars, vals, groups)):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", ha="left", fontsize=8.5, color="#333")
    # Legend
    seen = []
    handles = []
    for g, c in [(TAIL_GROUPS[f][0], TAIL_GROUPS[f][1]) for f in fams]:
        if g not in seen:
            seen.append(g)
            handles.append(plt.Rectangle((0, 0), 1, 1, color=c))
    ax.legend(handles, seen, title="tail type", loc="lower right", frameon=True)
    ax.axvline(49.69, color="#444", linestyle="--", linewidth=1.0)
    ax.text(49.69 + 1.5, -0.6, "aggregate 49.7%", color="#444", fontsize=8.5)
    ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "synthetic_es_relrmse_by_family.png"))
    plt.close(fig)


def fig2_real_xi_hat_distribution():
    path = os.path.join(ROOT, "outputs", "data", "real_diagnostics_loss.pkl")
    with open(path, "rb") as f:
        pairs = pickle.load(f)
    xis = []
    for ds, diag in pairs:
        k_grid = np.asarray(diag["k_grid"])
        k_star = diag["k_star"]
        idx = int(np.searchsorted(k_grid, k_star))
        idx = min(idx, len(diag["params"]) - 1)
        xi = diag["params"][idx, 0]
        if np.isfinite(xi):
            xis.append(xi)
    xis = np.asarray(xis)
    frac_05 = (xis > 0.5).mean() * 100
    frac_07 = (xis > 0.7).mean() * 100

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.hist(xis, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="#CCB974", linestyle="--", linewidth=1.4,
               label=f"proposed trigger 0.5  ({frac_05:.1f}% above)")
    ax.axvline(0.7, color="#C44E52", linestyle="-", linewidth=1.4,
               label=f"current trigger 0.7  ({frac_07:.1f}% above)")
    ax.set_xlabel(r"GPD-MLE $\hat{\xi}$ at the baseline threshold $k^*$")
    ax.set_ylabel("number of loss-tail windows")
    ax.set_title("Distribution of GPD shape on real loss-tail rolling windows\n"
                 f"median {np.median(xis):.3f}, 95th percentile {np.quantile(xis, 0.95):.3f}, "
                 f"n = {len(xis)}")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "real_loss_tail_xi_hat_distribution.png"))
    plt.close(fig)


def fig3_mcneil_frey_by_ticker():
    path = os.path.join(ROOT, "outputs", "real_results_loss.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    meth = d["methods"]["cnn"]
    tk_list = meth["tickers"]
    n_future = meth["n_future_list"]
    tk_per_obs = []
    for tk, nf in zip(tk_list, n_future):
        tk_per_obs.extend([tk] * nf)
    tk_per_obs = np.array(tk_per_obs)
    viol = np.array(meth["violations_binary"])
    rets = np.array(meth["future_returns_all"])
    es = np.array(meth["es_all"])

    rows = []
    for tk in sorted(set(tk_list)):
        mask = (tk_per_obs == tk)
        m = viol[mask].astype(bool)
        n_v = int(m.sum())
        if n_v < 5:
            rows.append((tk, n_v, np.nan, np.nan, np.nan, np.nan))
            continue
        resid = (rets[mask][m] - es[mask][m]) / es[mask][m]
        t_stat, p_val = ttest_1samp(resid, 0)
        sd = resid.std(ddof=1)
        se = sd / np.sqrt(len(resid)) if len(resid) > 1 else np.nan
        t_crit = student_t.ppf(0.975, df=max(len(resid) - 1, 1))
        ci_lo = resid.mean() - t_crit * se
        ci_hi = resid.mean() + t_crit * se
        rows.append((tk, n_v, resid.mean(), ci_lo, ci_hi, p_val))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    y_pos = np.arange(len(rows))
    for i, (tk, n_v, mean_r, lo, hi, p) in enumerate(rows):
        if np.isnan(mean_r):
            ax.text(0, i, f"  {tk}  too few exceedances ({n_v})",
                    va="center", ha="left", color="#888", fontsize=9)
            continue
        color = "#C44E52" if p < 0.05 else "#4C72B0"
        ax.plot([lo, hi], [i, i], color=color, linewidth=2.2)
        ax.plot(mean_r, i, "o", color=color, markersize=6)
        ax.text(hi + 0.02, i, f"p={p:.4f}, n={n_v}", va="center",
                fontsize=8.5, color="#333")
    ax.axvline(0, color="#444", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(r"mean McNeil-Frey residual  $(r_t - ES_t)/ES_t$  at violations  (95% CI)")
    ax.set_title("Per-ticker McNeil-Frey on the real loss tail, CNN method\n"
                 "negative mean residual indicates ES too large (conservative)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "real_loss_tail_mcneil_frey_by_ticker.png"))
    plt.close(fig)


def fig4_es_vs_realised():
    path = os.path.join(ROOT, "outputs", "real_results_loss.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    meth = d["methods"]["cnn"]
    tk_list = meth["tickers"]
    n_future = meth["n_future_list"]
    tk_per_obs = []
    for tk, nf in zip(tk_list, n_future):
        tk_per_obs.extend([tk] * nf)
    tk_per_obs = np.array(tk_per_obs)
    viol = np.array(meth["violations_binary"]).astype(bool)
    rets = np.array(meth["future_returns_all"])
    es = np.array(meth["es_all"])

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    palette = {"AAPL": "#4C72B0", "AMZN": "#C44E52", "BTC-USD": "#8172B2",
               "ETH-USD": "#CCB974", "META": "#937860", "MSFT": "#64B5CD",
               "NVDA": "#55A868", "^NYFANG": "#000000"}
    tickers = sorted(set(tk_list))
    for tk in tickers:
        mask = viol & (tk_per_obs == tk)
        if mask.sum() == 0:
            continue
        ax.scatter(rets[mask], es[mask], s=42, alpha=0.85,
                   color=palette.get(tk, "#888"),
                   edgecolor="white", linewidth=0.6, label=tk)
    lim_max = max(es[viol].max() if viol.sum() else 0,
                  rets[viol].max() if viol.sum() else 0) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "--", color="#444",
            linewidth=1.0, label=r"$y = x$ (perfect calibration)")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("realised loss at violation")
    ax.set_ylabel("predicted Expected Shortfall")
    ax.set_title("Predicted ES versus realised loss at violations, CNN method\n"
                 "loss-tail real data, points above the diagonal indicate ES too large")
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "real_es_vs_realised_at_violations.png"))
    plt.close(fig)


def fig5_method_comparison():
    # Hard-code the McNeil-Frey p-values across the unconditional backtests
    # (sourced from the latest pipeline run; can be recomputed from pickles).
    methods = ["cnn", "baseline_k_star", "fixed_sqrt_n", "historical_sim"]
    tail_modes = ["abs", "loss", "profit"]
    # (method, tail_mode) -> McNeil-Frey p-value, unconditional
    p_vals = {
        ("cnn", "abs"): 0.0000,
        ("cnn", "loss"): 0.0011,
        ("cnn", "profit"): 0.1393,
        ("baseline_k_star", "abs"): 0.0000,
        ("baseline_k_star", "loss"): 0.0028,
        ("baseline_k_star", "profit"): 0.0072,
        ("fixed_sqrt_n", "abs"): 0.0177,
        ("fixed_sqrt_n", "loss"): 0.0045,
        ("fixed_sqrt_n", "profit"): 0.2529,
        ("historical_sim", "abs"): 0.7727,
        ("historical_sim", "loss"): 0.7261,
        ("historical_sim", "profit"): 0.1160,
    }
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(methods))
    width = 0.26
    colors = {"abs": "#4C72B0", "loss": "#C44E52", "profit": "#55A868"}
    for j, tm in enumerate(tail_modes):
        vals = [p_vals[(m, tm)] for m in methods]
        # Replace zeros with a small floor so they are visible
        vals = [max(v, 1e-4) for v in vals]
        ax.bar(x + (j - 1) * width, vals, width=width, color=colors[tm],
               edgecolor="white", linewidth=0.6, label=tm)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in methods])
    ax.set_ylabel("McNeil-Frey p-value, log scale")
    ax.axhline(0.05, color="#444", linestyle="--", linewidth=1.0,
               label="p = 0.05 cutoff")
    ax.set_title("McNeil-Frey p-value by method and tail mode, unconditional setting")
    ax.legend(loc="lower right", title="tail", frameon=True)
    ax.set_ylim(1e-4, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "method_comparison_mcneil_frey.png"))
    plt.close(fig)


def fig6_kstar_before_after():
    # Numbers come from the prior memory (pre-fix) and the current pickle (post-fix).
    alphas = [1.2, 1.3, 1.4]
    k_old = [118, 128, 131]    # baseline k* before the heavy-tail penalty
    k_new = [37, 35, 36]       # baseline k* after the heavy-tail penalty
    k_oracle = [33, 33, 36]    # oracle k

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(alphas))
    width = 0.32
    ax.bar(x - width / 2, k_old, width=width, color="#C44E52",
           edgecolor="white", linewidth=0.6, label="prior scorer  (4 weights)")
    ax.bar(x + width / 2, k_new, width=width, color="#4C72B0",
           edgecolor="white", linewidth=0.6, label="new scorer  (5 weights, heavy-tail penalty)")
    ax.plot(x, k_oracle, "o-", color="#222", linewidth=1.5,
            markersize=7, label="oracle  $k_{\\mathrm{oracle}}$")
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$\alpha = {a}$" for a in alphas])
    ax.set_ylabel(r"median $k^*$  over 300 replications")
    ax.set_title("Baseline threshold under the prior and new scorer on Pareto, n = 1000")
    ax.legend(loc="upper right", frameon=True)
    for xi_, ko, kn in zip(x, k_old, k_new):
        ax.text(xi_ - width / 2, ko + 3, str(ko), ha="center", fontsize=9, color="#333")
        ax.text(xi_ + width / 2, kn + 3, str(kn), ha="center", fontsize=9, color="#333")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "high_xi_kstar_before_after.png"))
    plt.close(fig)


TABLE_DIR = os.path.join(ROOT, "outputs", "tables")
os.makedirs(TABLE_DIR, exist_ok=True)

# Per-family ES relRMSE that the Results chapter currently quotes. These
# match the numbers in fig1 above and are the post-heavy-tail-penalty values
# from the run referenced by the thesis prose. The VaR counterpart is read
# (Per-family ES RelRMSE constants previously hardcoded here have been
# removed. The table_synthetic_per_family function below now reads
# outputs/data/synthetic_test_eval.pkl, produced by
# scripts/reconcile_synthetic_eval.py, so the table stays in sync with the
# §5.1 figures and the appendix decomposition numbers.)


def table_synthetic_per_family():
    """Emit a LaTeX table with VaR and ES relRMSE per distribution.

    Numbers come from outputs/data/synthetic_test_eval.pkl, the canonical
    test-set evaluation of the deployed model produced by
    scripts/reconcile_synthetic_eval.py. Re-run that script if the model
    or diagnostics change.
    """
    path = os.path.join(ROOT, "outputs", "data", "synthetic_test_eval.pkl")
    with open(path, "rb") as f:
        results = pickle.load(f)
    var_agg = results["relative_rmse"] * 100.0
    es_agg = results["es_relative_rmse"] * 100.0
    rmse_by = results["rmse_by_dist"]

    rows = sorted(
        ((dist, m["relative_rmse"] * 100, m["es_relative_rmse"] * 100)
         for dist, m in rmse_by.items()),
        key=lambda r: r[2],
    )

    lines = []
    lines.append(r"\begin{tabular}{@{}l r r@{}}")
    lines.append(r"\hline")
    lines.append(r"Distribution & VaR rel.\ RMSE & ES rel.\ RMSE \\")
    lines.append(r"\hline")
    for dist, vpct, epct in rows:
        dist_label = dist.replace("_", r"\_")
        lines.append(f"{dist_label} & ${vpct:.2f}\\%$ & ${epct:.1f}\\%$ \\\\")
    lines.append(r"\hline")
    lines.append(f"Aggregate & ${var_agg:.2f}\\%$ & ${es_agg:.1f}\\%$ \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    out_path = os.path.join(TABLE_DIR, "synthetic_per_family_relrmse.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_path}")


def table_real_aggregate_loss():
    """Aggregate VaR and ES backtest table on the real loss tail."""
    path = os.path.join(ROOT, "outputs", "real_results_loss.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)

    method_order = ["cnn", "baseline_k_star", "fixed_sqrt_n", "historical_sim"]
    lines = []
    lines.append(r"\begin{tabular}{@{}l r r r r r r@{}}")
    lines.append(r"\hline")
    lines.append(r"Method & VR & $n_{\mathrm{viol}}$ & mean ES & Kupiec $p$ & Chr.\ $p$ & MF $p$ \\")
    lines.append(r"\hline")
    for meth in method_order:
        s = d["summary"][meth]
        vr = s["overall_violation_rate"]
        nv = s["mcneil_frey"]["n_violations"]
        es = s["mean_es_estimate"]
        kp = s["kupiec"]["p_value"]
        cp = s["christoffersen"]["p_value_ind"]
        mp = s["mcneil_frey"]["p_value"]
        meth_label = "\\textsc{" + meth.replace("_", r"\_") + "}"
        lines.append(
            f"{meth_label} & ${vr:.4f}$ & ${nv}$ & ${es:.4f}$ & "
            f"${kp:.3f}$ & ${cp:.4f}$ & ${mp:.4f}$ \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    out_path = os.path.join(TABLE_DIR, "real_aggregate_loss.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_path}")


def table_real_aggregate_profit():
    """Aggregate VaR and ES backtest table on the real profit tail."""
    path = os.path.join(ROOT, "outputs", "real_results_profit.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)

    method_order = ["cnn", "baseline_k_star", "fixed_sqrt_n", "historical_sim"]
    lines = []
    lines.append(r"\begin{tabular}{@{}l r r r r r r@{}}")
    lines.append(r"\hline")
    lines.append(r"Method & VR & $n_{\mathrm{viol}}$ & mean ES & Kupiec $p$ & Chr.\ $p$ & MF $p$ \\")
    lines.append(r"\hline")
    for meth in method_order:
        s = d["summary"][meth]
        vr = s["overall_violation_rate"]
        nv = s["mcneil_frey"]["n_violations"]
        es = s["mean_es_estimate"]
        kp = s["kupiec"]["p_value"]
        cp = s["christoffersen"]["p_value_ind"]
        mp = s["mcneil_frey"]["p_value"]
        meth_label = "\\textsc{" + meth.replace("_", r"\_") + "}"
        lines.append(
            f"{meth_label} & ${vr:.4f}$ & ${nv}$ & ${es:.4f}$ & "
            f"${kp:.3f}$ & ${cp:.4f}$ & ${mp:.4f}$ \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    out_path = os.path.join(TABLE_DIR, "real_aggregate_profit.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_path}")


def _per_ticker_var_rows(pickle_path):
    from scipy.stats import chi2 as _chi2
    with open(pickle_path, "rb") as f:
        d = pickle.load(f)
    p_exp = 0.01
    method_order = ["cnn", "baseline_k_star", "fixed_sqrt_n", "historical_sim"]
    out = {}
    for meth in method_order:
        m = d["methods"][meth]
        tk_list = m["tickers"]
        n_future = m["n_future_list"]
        tk_per_obs = []
        for tk, nf in zip(tk_list, n_future):
            tk_per_obs.extend([tk] * nf)
        tk_per_obs = np.array(tk_per_obs)
        viol = np.array(m["violations_binary"])
        per_tk = {}
        for tk in sorted(set(tk_list)):
            mask = (tk_per_obs == tk)
            n_obs = int(mask.sum())
            n_v = int(viol[mask].sum())
            vr = n_v / n_obs if n_obs else float("nan")
            if 0 < n_v < n_obs:
                obs = n_v / n_obs
                lr = -2.0 * (
                    n_v * np.log(p_exp / obs)
                    + (n_obs - n_v) * np.log((1 - p_exp) / (1 - obs))
                )
                kup_p = float(1 - _chi2.cdf(lr, 1))
            else:
                kup_p = float("nan")
            per_tk[tk] = (n_obs, n_v, vr, kup_p)
        out[meth] = per_tk
    return out, sorted(set(tk_list))


def table_real_per_ticker_var(tail="loss"):
    """Per-ticker VaR coverage table for the real-data backtest.

    Eight tickers times four methods. Method label appears only on the first
    row of each ticker block; horizontal rules separate ticker blocks.
    """
    path = os.path.join(
        ROOT, "outputs", f"real_results_{tail}.pkl"
    )
    per_meth, tickers = _per_ticker_var_rows(path)
    method_order = ["cnn", "baseline_k_star", "fixed_sqrt_n", "historical_sim"]
    method_labels = {
        "cnn": r"\textsc{cnn}",
        "baseline_k_star": r"\textsc{baseline\_k\_star}",
        "fixed_sqrt_n": r"\textsc{fixed\_sqrt\_n}",
        "historical_sim": r"\textsc{historical\_sim}",
    }

    lines = []
    lines.append(r"\begin{tabular}{@{}l l r r r r@{}}")
    lines.append(r"\hline")
    lines.append(r"Ticker & Method & $n_{\mathrm{obs}}$ & $n_{\mathrm{viol}}$ & VR & Kupiec $p$ \\")
    lines.append(r"\hline")
    for ti, tk in enumerate(tickers):
        for mi, meth in enumerate(method_order):
            n_obs, n_v, vr, kup_p = per_meth[meth][tk]
            kup_s = f"{kup_p:.3f}" if not np.isnan(kup_p) else r"---"
            tk_label = (tk.replace("^", r"\^{}") if mi == 0 else "")
            lines.append(
                f"{tk_label} & {method_labels[meth]} & ${n_obs}$ & ${n_v}$ & "
                f"${vr:.4f}$ & ${kup_s}$ \\\\"
            )
        if ti < len(tickers) - 1:
            lines.append(r"\hline")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    out_path = os.path.join(TABLE_DIR, f"real_per_ticker_var_{tail}.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_path}")


def table_profit_per_ticker_mf():
    """Per-ticker McNeil-Frey table on the profit tail, CNN method."""
    from scipy.stats import ttest_1samp as _ttest1
    path = os.path.join(ROOT, "outputs", "real_results_profit.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    m = d["methods"]["cnn"]
    tk_list = m["tickers"]
    n_future = m["n_future_list"]
    tk_per_obs = []
    for tk, nf in zip(tk_list, n_future):
        tk_per_obs.extend([tk] * nf)
    tk_per_obs = np.array(tk_per_obs)
    viol = np.array(m["violations_binary"]).astype(bool)
    rets = np.array(m["future_returns_all"])
    es = np.array(m["es_all"])

    lines = []
    lines.append(r"\begin{tabular}{@{}l r r r r r@{}}")
    lines.append(r"\hline")
    lines.append(r"Ticker & $n_{\mathrm{viol}}$ & $\overline{r}$ & $\overline{\mathrm{ES}}$ & $t$ & $p$ \\")
    lines.append(r"\hline")
    for tk in sorted(set(tk_list)):
        mask = (tk_per_obs == tk)
        m_v = viol[mask]
        n_v = int(m_v.sum())
        tk_label = tk.replace("^", r"\^{}")
        if n_v < 5:
            lines.append(f"{tk_label} & ${n_v}$ & --- & --- & --- & --- \\\\")
            continue
        r = rets[mask][m_v]
        e = es[mask][m_v]
        resid = (r - e) / e
        t, p = _ttest1(resid, 0)
        lines.append(
            f"{tk_label} & ${n_v}$ & ${r.mean():.3f}$ & ${e.mean():.3f}$ "
            f"& ${t:+.2f}$ & ${p:.3f}$ \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    out_path = os.path.join(TABLE_DIR, "real_profit_per_ticker_mf.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out_path}")


def main():
    print(f"Writing figures to {OUT_DIR}")
    fig1_synthetic_es_relrmse()
    fig2_real_xi_hat_distribution()
    fig3_mcneil_frey_by_ticker()
    fig4_es_vs_realised()
    fig5_method_comparison()
    fig6_kstar_before_after()
    print(f"Writing tables to {TABLE_DIR}")
    table_synthetic_per_family()
    table_real_aggregate_loss()
    table_real_aggregate_profit()
    table_real_per_ticker_var("loss")
    table_profit_per_ticker_mf()
    print("Done. Figure files:")
    for f in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, f)
        size_kb = os.path.getsize(path) / 1024.0
        print(f"  {f}  ({size_kb:.0f} KB)")
    print("Table files:")
    for f in sorted(os.listdir(TABLE_DIR)):
        path = os.path.join(TABLE_DIR, f)
        size_kb = os.path.getsize(path) / 1024.0
        print(f"  {f}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
