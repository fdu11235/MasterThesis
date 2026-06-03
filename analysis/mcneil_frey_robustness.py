"""McNeil-Frey robustness check for the real loss-tail Expected Shortfall test.

Reproduces Table 5.x (tab:mf_robustness) in the thesis. For each estimator it
computes the loss-tail McNeil-Frey p-value under three residual definitions:

  * Relative : r_i = (X_i - ES_i) / ES_i        (the definition used in the thesis)
  * Absolute : r_i = X_i - ES_i                 (removes the per-violation weighting)
  * HAC      : relative residual, Newey-West HAC standard error (clustering-robust)

The "Relative" column reproduces the headline McNeil-Frey p-values reported in
the real-data backtest tables, confirming the computation is apples-to-apples.

Input: outputs/real_results_loss.pkl and outputs/real_results_profit.pkl, produced
by the real-data backtest.
Run:   python scripts/mcneil_frey_robustness.py            # both tails
       python scripts/mcneil_frey_robustness.py loss       # one tail
"""
import os
import sys
import pickle

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHOD_ORDER = ["cnn", "baseline_k_star", "fixed_sqrt_n", "historical_sim"]


def hac_se(x, max_lag):
    """Newey-West HAC standard error of the sample mean of x."""
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    s = np.dot(e, e) / n
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        s += 2.0 * w * np.dot(e[lag:], e[:-lag]) / n
    return np.sqrt(s / n)


def report(tail):
    path = os.path.join(ROOT, "outputs", f"real_results_{tail}.pkl")
    with open(path, "rb") as fh:
        methods = pickle.load(fh)["methods"]

    print(f"\n=== {tail} tail ===")
    header = f"{'method':18} {'nviol':>5} {'relative':>9} {'absolute':>9} {'HAC':>9} {'mean_res':>9}"
    print(header)
    print("-" * len(header))
    for name in METHOD_ORDER:
        m = methods[name]
        fr = np.asarray(m["future_returns_all"], float)
        var = np.asarray(m["var_all"], float)
        es = np.asarray(m["es_all"], float)
        mask = fr > var
        nv = int(mask.sum())
        if nv < 2:
            print(f"{name:18} {nv:>5}   untestable")
            continue
        rel = (fr[mask] - es[mask]) / es[mask]
        absr = fr[mask] - es[mask]
        _, p_rel = stats.ttest_1samp(rel, 0.0)
        _, p_abs = stats.ttest_1samp(absr, 0.0)
        max_lag = int(np.floor(4 * (nv / 100.0) ** (2.0 / 9.0))) + 1
        t_hac = rel.mean() / hac_se(rel, max_lag)
        p_hac = 2.0 * stats.norm.sf(abs(t_hac))
        print(f"{name:18} {nv:>5} {p_rel:9.4f} {p_abs:9.4f} {p_hac:9.4f} {rel.mean():+9.4f}")


def main():
    tails = sys.argv[1:] or ["loss", "profit"]
    for tail in tails:
        report(tail)


if __name__ == "__main__":
    main()
