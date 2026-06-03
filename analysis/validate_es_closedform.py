"""Validation appendix for ground-truth Expected Shortfall.

Addresses three items in Pasquale's feedback:

    (2) Quantify the Monte Carlo numerical error of the tail mean.
    (3) Compare closed-form ES against high-precision MC for a few cells.
    (4) Cross-check closed-form against scipy.integrate.quad.

Outputs:
    outputs/data/es_validation.csv         - full per-cell numerics
    outputs/figures/es_validation.png      - log-log plot of MC std vs N
    docs/appendix_es_validation.md         - summary table for the thesis appendix

Usage (from repo root):
    python scripts/validate_es_closedform.py
"""
from __future__ import annotations

import csv
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from scipy.integrate import quad

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import _analytical_es, _mc_es, true_quantile  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
CSV_PATH = REPO_ROOT / "outputs" / "data" / "es_validation.csv"
FIG_PATH = REPO_ROOT / "outputs" / "figures" / "es_validation.png"
MD_PATH = REPO_ROOT / "docs" / "appendix_es_validation.md"

P_LIST = [0.95, 0.975, 0.99]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def expand_params(dist_type: str, cfg: dict) -> list[dict]:
    """Mirror src.synthetic._param_combos for the 12 distribution families."""
    if dist_type == "student_t":
        return [{"df": d} for d in cfg["df"]]
    if dist_type == "pareto":
        return [{"alpha": a} for a in cfg["alpha"]]
    if dist_type == "burr12":
        return [{"c": c, "d": d}
                for c, d in itertools.product(cfg["c"], cfg["d"])]
    if dist_type == "frechet":
        return [{"c": c} for c in cfg["c"]]
    if dist_type == "dagum":
        return [{"c": c, "d": d}
                for c, d in itertools.product(cfg["c"], cfg["d"])]
    if dist_type == "inverse_gamma":
        return [{"a": a} for a in cfg["a"]]
    if dist_type == "lognormal":
        return [{"sigma": s} for s in cfg["sigma"]]
    if dist_type == "weibull_stretched":
        return [{"c": c} for c in cfg["c"]]
    if dist_type == "two_pareto":
        cp = cfg["changepoint_frac"]
        return [{"alpha1": a1, "alpha2": a2, "changepoint_frac": cp}
                for a1, a2 in itertools.product(cfg["alpha1"], cfg["alpha2"])]
    if dist_type == "gamma_pareto_splice":
        sq = cfg["splice_quantile"]
        return [{"gamma_shape": g, "pareto_alpha": a, "splice_quantile": sq}
                for g, a in itertools.product(cfg["gamma_shape"],
                                              cfg["pareto_alpha"])]
    if dist_type == "log_gamma":
        pp = cfg["p"]
        return [{"b": b, "p": pp} for b in cfg["b"]]
    if dist_type == "lognormal_pareto_mix":
        return [{"lognormal_mu": cfg["lognormal_mu"],
                 "lognormal_sigma": cfg["lognormal_sigma"],
                 "pareto_alpha": pa, "mix_frac": cfg["mix_frac"]}
                for pa in cfg["pareto_alpha"]]
    raise ValueError(dist_type)


def quad_es(dist_type: str, params: dict, p: float) -> float | None:
    """High-precision quadrature against the population density.

    Returns ``None`` for distribution families that are skipped here.
    """
    if dist_type == "log_gamma":
        # u = (b-1)*log(x) substitution avoids exp() overflow.
        from scipy.special import gammaincinv, gamma as gamma_fn
        b, p_param = params["b"], params["p"]
        if b <= 1:
            return None
        log_v = gammaincinv(p_param, p) / b
        a_lim = (b - 1.0) * log_v
        integrand = lambda u: u ** (p_param - 1) * np.exp(-u)
        val, _ = quad(integrand, a_lim, np.inf, limit=200)
        val *= (b / (b - 1.0)) ** p_param / gamma_fn(p_param)
        return float(val / (1.0 - p))

    if dist_type == "lognormal_pareto_mix":
        mu, sig = params["lognormal_mu"], params["lognormal_sigma"]
        a, w = params["pareto_alpha"], params["mix_frac"]
        if a <= 1:
            return None
        var = true_quantile(dist_type, params, p)
        density = lambda x: ((1 - w) * stats.lognorm.pdf(x, s=sig, scale=np.exp(mu))
                             + w * stats.pareto.pdf(x, b=a))
        val, _ = quad(lambda x: x * density(x), var, np.inf, limit=300)
        return float(val / (1.0 - p))

    if dist_type == "two_pareto":
        a1, a2 = params["alpha1"], params["alpha2"]
        cp = params["changepoint_frac"]
        if a2 <= 1 or p < 1 - cp:
            return None
        u = (1.0 / cp) ** (1.0 / a1)
        var = true_quantile(dist_type, params, p)
        density = lambda x: cp * (1.0 / u) * stats.pareto.pdf(x / u, b=a2)
        val, _ = quad(lambda x: x * density(x), var, np.inf, limit=300)
        return float(val / (1.0 - p))

    if dist_type == "gamma_pareto_splice":
        a = params["pareto_alpha"]
        sq = params["splice_quantile"]
        if a <= 1 or p < sq:
            return None
        u = stats.gamma.ppf(sq, a=params["gamma_shape"])
        var = true_quantile(dist_type, params, p)
        density = lambda x: (1 - sq) * (1.0 / u) * stats.pareto.pdf(x / u, b=a)
        val, _ = quad(lambda x: x * density(x), var, np.inf, limit=300)
        return float(val / (1.0 - p))

    # Single-family distributions: integrate x * pdf(x) on [VaR, inf).
    dist_obj_map = {
        "pareto": (stats.pareto, {"b": params.get("alpha")}),
        "burr12": (stats.burr12, params),
        "frechet": (stats.invweibull, params),
        "dagum": (stats.burr, params),
        "inverse_gamma": (stats.invgamma, {"a": params.get("a")}),
        "lognormal": (stats.lognorm, {"s": params.get("sigma")}),
        "weibull_stretched": (stats.weibull_min, params),
    }
    if dist_type in dist_obj_map:
        dist_cls, kw = dist_obj_map[dist_type]
        var = dist_cls.ppf(p, **kw)
        density = lambda x: dist_cls.pdf(x, **kw)
        val, _ = quad(lambda x: x * density(x), var, np.inf, limit=300)
        return float(val / (1.0 - p))
    if dist_type == "student_t":
        df = params["df"]
        # Synthetic samples are |T|; E[|T| | |T|>v] = E[T | T>v_upper] by symmetry.
        upper_p = (p + 1.0) / 2.0
        v = stats.t.ppf(upper_p, df=df)
        val, _ = quad(lambda t: t * stats.t.pdf(t, df=df), v, np.inf, limit=300)
        return float(val / ((1.0 - p) / 2.0))
    return None


# --------------------------------------------------------------------------
# Main computation
# --------------------------------------------------------------------------

def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    dists = cfg["synthetic"]["distributions"]

    rows = []
    t0 = time.time()
    print(f"{'dist':<24}{'params':<48}{'p':>6}{'closed':>14}"
          f"{'mc_N=1e7':>14}{'quad':>14}{'|cf-mc|/cf':>13}"
          f"{'|cf-q|/cf':>13}")
    print("-" * 146)

    # Skip GARCH wrappers in this validation (delegate to base distribution).
    skip = {"garch_student_t", "garch_pareto"}
    for dt in [d for d in dists if d not in skip]:
        for params in expand_params(dt, dists[dt]):
            for p in P_LIST:
                cf = _analytical_es(dt, params, p)
                m = _mc_es(dt, params, p, n_mc=10_000_000, seed=99999)
                q = quad_es(dt, params, p)
                err_mc = (abs(cf - m) / cf) if cf is not None else float("nan")
                err_q = (abs(cf - q) / cf) if (cf is not None and q is not None) else float("nan")
                rows.append({
                    "dist": dt,
                    "params": str(params),
                    "p": p,
                    "closed_form": cf if cf is not None else "",
                    "mc_N1e7": m,
                    "quad": q if q is not None else "",
                    "rel_err_mc": err_mc if not np.isnan(err_mc) else "",
                    "rel_err_quad": err_q if not np.isnan(err_q) else "",
                })
                cf_s = f"{cf:>14.6f}" if cf is not None else f"{'(MC)':>14}"
                q_s = f"{q:>14.6f}" if q is not None else f"{'-':>14}"
                e_mc = f"{err_mc:>13.2e}" if not np.isnan(err_mc) else f"{'-':>13}"
                e_q = f"{err_q:>13.2e}" if not np.isnan(err_q) else f"{'-':>13}"
                pstr = ", ".join(f"{k}={v}" for k, v in params.items())[:46]
                print(f"{dt:<24}{pstr:<48}{p:>6.3f}{cf_s}{m:>14.6f}{q_s}{e_mc}{e_q}")

    print(f"\nMain table: {len(rows)} rows; elapsed {time.time() - t0:.1f}s")

    # ----------------------------------------------------------------------
    # Stability check: 2 cells, N in {1e6, 1e7, 1e8}, 5 seeds each.
    # ----------------------------------------------------------------------
    stability_cells = [
        ("pareto", {"alpha": 1.5}, 0.99),       # heavy-tailed
        ("lognormal", {"sigma": 1.0}, 0.99),    # moderate
    ]
    n_grid = [1_000_000, 10_000_000, 100_000_000]
    seeds = [11, 22, 33, 44, 55]
    stability = {}  # cell -> {N: list of MC ES values}
    print("\nStability check (N x 5 seeds for two illustrative cells)...")
    for dt, params, p in stability_cells:
        cf = _analytical_es(dt, params, p)
        cell_key = (dt, str(params), p)
        stability[cell_key] = {"closed": cf, "by_n": {}}
        for n in n_grid:
            vals = []
            for s in seeds:
                vals.append(_mc_es(dt, params, p, n_mc=n, seed=s))
            stability[cell_key]["by_n"][n] = vals
            print(f"  {dt} {params} p={p} N={n:>10d}  "
                  f"mean={np.mean(vals):.4f}  std={np.std(vals, ddof=1):.4f}  "
                  f"closed={cf:.4f}")

    # ----------------------------------------------------------------------
    # Write CSV
    # ----------------------------------------------------------------------
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {CSV_PATH}")

    # ----------------------------------------------------------------------
    # Plot: MC std vs N on log-log; expect 1/sqrt(N) slope.
    # ----------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for cell_key, data in stability.items():
        ns = sorted(data["by_n"].keys())
        stds = [np.std(data["by_n"][n], ddof=1) for n in ns]
        ax.loglog(ns, stds, marker="o", lw=1.5,
                  label=f"{cell_key[0]} {cell_key[1]} p={cell_key[2]}")
    # Reference 1/sqrt(N) slope using the heavy-tail cell as anchor
    anchor_cell = list(stability.values())[0]
    ns_a = sorted(anchor_cell["by_n"].keys())
    std_anchor = np.std(anchor_cell["by_n"][ns_a[0]], ddof=1)
    ref = [std_anchor * np.sqrt(ns_a[0] / n) for n in ns_a]
    ax.loglog(ns_a, ref, "k--", lw=0.8, alpha=0.6, label=r"$\propto 1/\sqrt{N}$")
    ax.set_xlabel("N (Monte Carlo sample size)")
    ax.set_ylabel(r"std of MC ES estimate (across 5 seeds)")
    ax.set_title("Monte Carlo stability for tail-mean estimation")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {FIG_PATH}")

    # ----------------------------------------------------------------------
    # Markdown appendix
    # ----------------------------------------------------------------------
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MD_PATH, "w") as f:
        f.write("# Appendix: Validation of Ground-Truth Expected Shortfall\n")
        f.write("\\label{appendix:es-validation}\n\n")
        f.write("This appendix documents the numerical agreement between the\n")
        f.write("closed-form ES formulas used as ground truth and two\n")
        f.write("independent benchmarks: high-precision Monte Carlo (N = 10^7)\n")
        f.write("and `scipy.integrate.quad` adaptive quadrature on the population\n")
        f.write("density. Generated by `scripts/validate_es_closedform.py`.\n\n")

        f.write("## 1. Closed-form vs MC vs quadrature\n\n")
        f.write("| Distribution | Parameters | p | Closed | MC (N=1e7) | Quad | "
                "\\|cf-mc\\|/cf | \\|cf-quad\\|/cf |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            cf_str = f"{r['closed_form']:.4f}" if isinstance(r['closed_form'], float) else "(none)"
            q_str = f"{r['quad']:.4f}" if isinstance(r['quad'], float) else "—"
            err_mc = f"{r['rel_err_mc']:.2e}" if isinstance(r['rel_err_mc'], float) else "—"
            err_q = f"{r['rel_err_quad']:.2e}" if isinstance(r['rel_err_quad'], float) else "—"
            f.write(f"| {r['dist']} | {r['params']} | {r['p']} | {cf_str} | "
                    f"{r['mc_N1e7']:.4f} | {q_str} | {err_mc} | {err_q} |\n")

        f.write("\n## 2. MC stability across N\n\n")
        f.write("Five seeds at each N. Reported value is the empirical std\n")
        f.write("of the MC ES estimate; if MC is unbiased and tail variance\n")
        f.write("is finite, this should decay as 1/sqrt(N).\n\n")
        f.write("| Cell | N | Mean | Std | Closed-form |\n|---|---|---|---|---|\n")
        for cell_key, data in stability.items():
            for n, vals in sorted(data["by_n"].items()):
                f.write(f"| {cell_key[0]} {cell_key[1]} p={cell_key[2]} | "
                        f"{n:.0e} | {np.mean(vals):.4f} | "
                        f"{np.std(vals, ddof=1):.4f} | {data['closed']:.4f} |\n")
        f.write(f"\nFigure: `{FIG_PATH.relative_to(REPO_ROOT)}`\n")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
