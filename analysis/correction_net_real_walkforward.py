"""Walk-forward expanding-window training of the ES correction net on
real-data loss-tail backtests, using realized exceedance returns as
supervision.

Two correction models are fitted side by side:
  - scalar: a single multiplicative factor c = mean(realized) / mean(ES_pred)
            over all violations observed before the current window.
  - mlp:    the existing 9-feature ESCorrectionNet from src/es_correction.py.

Walk-forward parameters: warmup=200 windows (no correction applied before
this many windows of history), refit_every=50.

Reads the existing pickles produced by run_real_pipeline.py. Writes a
result pickle and one figure summarising the running correction factor
and McNeil-Frey p-value.

Run: python scripts/correction_net_real_walkforward.py
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import ttest_1samp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.es_correction import (  # noqa: E402
    ESCorrectionNet,
    extract_features,
    train_correction_net,
)

OUT_PKL = os.path.join(ROOT, "outputs", "correction_walkforward_results.pkl")
OUT_FIG = os.path.join(ROOT, "outputs", "figures", "results_chapter",
                       "correction_walkforward.png")

WARMUP = 200
REFIT_EVERY = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

plt.style.use("ggplot")


def load_returns_lookup(cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(ROOT, "outputs", "data")
    lookup = {}
    for fname in os.listdir(cache_dir):
        if not fname.startswith("returns_") or not fname.endswith(".csv"):
            continue
        # Skip the _old variants and the GSPC/IXIC/N225/FTSE/GDAXI legacy ones.
        if "_old" in fname:
            continue
        ticker_short = fname[len("returns_"):-len(".csv")]
        df = pd.read_csv(os.path.join(cache_dir, fname))
        if "signed_return" not in df.columns:
            continue
        lookup[ticker_short] = {
            "signed_returns": df["signed_return"].values,
            "abs_returns": df["abs_return"].values,
        }
    return lookup


def build_test_data(config, returns_lookup):
    """Rebuild the test slice for the loss tail and align with the result file."""
    diag_path = os.path.join(ROOT, "outputs", "data", "real_diagnostics_loss.pkl")
    res_path = os.path.join(ROOT, "outputs", "real_results_loss.pkl")
    with open(diag_path, "rb") as f:
        diag_list = pickle.load(f)
    with open(res_path, "rb") as f:
        results = pickle.load(f)

    end_dates = [ds["end_date"] for ds, _ in diag_list]
    sorted_idx = np.argsort(end_dates)
    train_frac = config["realdata"]["train_fraction"]
    n_train = int(len(sorted_idx) * train_frac)
    test_idx_full = sorted_idx[n_train:]

    backtest_horizon = config["realdata"]["backtest_horizon"]

    matched = []
    cnn = results["methods"]["cnn"]
    expected_tickers = cnn["tickers"]

    pos = 0
    for j in test_idx_full:
        ds, diag = diag_list[j]
        ticker = ds["ticker"]
        # Yahoo cache uses safe name; map ticker -> safe key
        safe = ticker.replace("^", "").replace("/", "_")
        sr = returns_lookup.get(safe, {}).get("signed_returns")
        if sr is None:
            continue
        series_end_idx = ds.get("series_end_idx", 0)
        future_end = series_end_idx + backtest_horizon
        if future_end > len(sr):
            continue

        if pos >= len(expected_tickers):
            break
        if expected_tickers[pos] != ticker:
            continue
        matched.append({
            "ds": ds,
            "diag": diag,
            "ticker": ticker,
            "end_date": ds["end_date"],
            "var_pred": float(cnn["var_estimates"][pos]),
            "es_pred": float(cnn["es_estimates"][pos]),
            "n_future": int(cnn["n_future_list"][pos]),
            "future_returns": sr[series_end_idx:future_end],
            "tail_mode": "loss",
        })
        pos += 1

    if pos != len(expected_tickers):
        log.warning("Matched %d of %d expected test entries", pos, len(expected_tickers))
    return matched


def mcneil_frey(residuals):
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) < 5:
        return float("nan"), float("nan"), len(residuals)
    t, p = ttest_1samp(residuals, 0)
    return float(t), float(p), len(residuals)


def kupiec(n_violations, n_obs, p):
    if n_violations == 0 or n_obs == 0:
        return float("nan"), float("nan")
    expected = p * n_obs
    if n_violations >= n_obs:
        return float("nan"), float("nan")
    rate = n_violations / n_obs
    # standard Kupiec POF likelihood ratio
    ll1 = (n_violations * np.log(rate)
           + (n_obs - n_violations) * np.log(1 - rate))
    ll0 = (n_violations * np.log(p)
           + (n_obs - n_violations) * np.log(1 - p))
    lr = -2 * (ll0 - ll1)
    # chi^2 with 1 dof
    from scipy.stats import chi2
    p_val = 1 - chi2.cdf(lr, df=1)
    return float(lr), float(p_val)


def main():
    cfg_path = os.path.join(ROOT, "config", "default.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    p = config["evaluate"]["quantile_p"]

    log.info("Loading return data and reconstructing test slice ...")
    returns_lookup = load_returns_lookup()
    log.info("  loaded returns for tickers: %s", sorted(returns_lookup.keys()))
    rows = build_test_data(config, returns_lookup)
    log.info("Reconstructed %d test windows on the loss tail", len(rows))

    # Convenience arrays for the BASELINE (no correction).
    # We treat each future-day observation independently.
    obs_records = []  # one entry per (window, future_day)
    for w_idx, r in enumerate(rows):
        var_pred = r["var_pred"]
        es_pred = r["es_pred"]
        fut = r["future_returns"]
        # tail_mode == "loss": violations = future loss exceeds VaR.
        # signed_returns < 0 means loss; magnitude is abs(signed) for violations
        # against VaR computed on loss-tail magnitudes.
        loss_mags = np.where(fut < 0, -fut, 0.0)
        for d_idx, mag in enumerate(loss_mags):
            obs_records.append({
                "w_idx": w_idx,
                "ticker": r["ticker"],
                "end_date": r["end_date"],
                "real_loss": float(mag),
                "var_pred": var_pred,
                "es_pred": es_pred,
                "violated": bool(mag > var_pred),
            })

    log.info("Total per-day observations: %d", len(obs_records))
    n_viol_total = sum(o["violated"] for o in obs_records)
    log.info("Total loss-tail violations (uncorrected VaR): %d", n_viol_total)

    # Walk-forward
    # State: list of "training pairs" accumulated from observed violations.
    # A training pair has (features, target = realized_loss / es_pred).
    train_X = []
    train_y = []
    train_meta = []

    scalar_history = []   # (w_idx, c_scalar)
    mlp_state = {"model": None, "X_mean": None, "X_std": None}
    refit_log = []

    cnn_features_cache = {}  # cache per w_idx

    def _features_for(w_idx):
        if w_idx in cnn_features_cache:
            return cnn_features_cache[w_idx]
        r = rows[w_idx]
        ds, diag = r["ds"], r["diag"]
        # use baseline k* as a proxy for the CNN k_pred (they coincide tightly
        # after the heavy-tail-penalty scorer fix).
        k = int(diag["k_star"])
        feats = extract_features(ds, diag, k, p=p, config=config)
        cnn_features_cache[w_idx] = feats
        return feats

    # Per-method predicted-ES arrays (corrected). Index: per future-day obs.
    n_obs = len(obs_records)
    es_scalar = np.array([o["es_pred"] for o in obs_records], dtype=float)
    es_mlp = np.array([o["es_pred"] for o in obs_records], dtype=float)
    correction_applied_from = None  # first w_idx with a non-trivial correction

    last_refit_w = None
    c_scalar_current = 1.0

    # Iterate windows in order.
    for w_idx, r in enumerate(rows):
        # Refit if needed.
        if w_idx >= WARMUP and (last_refit_w is None
                                 or w_idx - last_refit_w >= REFIT_EVERY):
            last_refit_w = w_idx
            # Accumulate training pairs from all violations in windows < w_idx.
            X_list, y_list = [], []
            for o in obs_records:
                if o["w_idx"] >= w_idx:
                    break
                if not o["violated"]:
                    continue
                feats = _features_for(o["w_idx"])
                if feats is None:
                    continue
                target = o["real_loss"] / max(o["es_pred"], 1e-10)
                X_list.append(feats)
                y_list.append(target)
            n_train = len(y_list)
            if n_train >= 5:
                X_arr = np.array(X_list, dtype=np.float32)
                y_arr = np.array(y_list, dtype=np.float32)
                # Scalar: mean of targets (least-squares-optimal under MSE
                # when the predictor is a constant). Equivalent to
                # mean(realized)/mean(es) for the multiplicative form when
                # using simple per-violation ratios.
                c_scalar_current = float(y_arr.mean())
                # MLP: train with the existing trainer. Adapt batch size to
                # the very small training-set size and lower max_epochs.
                config_mlp = dict(config)
                ec = dict(config.get("es_correction", {}))
                ec["batch_size"] = max(4, min(16, n_train // 2 or 1))
                ec["max_epochs"] = 200
                ec["patience"] = 20
                ec["val_fraction"] = 0.2 if n_train >= 20 else 0.0
                config_mlp["es_correction"] = ec

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if ec["val_fraction"] == 0.0:
                            # Train without validation split when too few points.
                            # Hack: pass a non-zero val_fraction and accept noisy.
                            ec["val_fraction"] = 0.25
                        model, _hist = train_correction_net(X_arr, y_arr, config_mlp)
                    mlp_state["model"] = model
                    mlp_state["X_mean"] = model.X_mean
                    mlp_state["X_std"] = model.X_std
                except Exception as e:
                    log.warning("MLP fit failed at w_idx=%d (n_train=%d): %s",
                                 w_idx, n_train, e)
                refit_log.append({"w_idx": w_idx, "n_train": n_train,
                                  "c_scalar": c_scalar_current})
                if correction_applied_from is None:
                    correction_applied_from = w_idx

        if w_idx < WARMUP or correction_applied_from is None:
            continue

        # Apply both corrections to this window's ES estimate
        for o_idx, o in enumerate(obs_records):
            if o["w_idx"] != w_idx:
                continue
            es_scalar[o_idx] = c_scalar_current * o["es_pred"]
            if mlp_state["model"] is not None:
                feats = _features_for(w_idx)
                if feats is None:
                    es_mlp[o_idx] = c_scalar_current * o["es_pred"]
                else:
                    feats_norm = (feats - mlp_state["X_mean"]) / mlp_state["X_std"]
                    with torch.no_grad():
                        c = float(mlp_state["model"](
                            torch.tensor(feats_norm, dtype=torch.float32).unsqueeze(0)
                        ).item())
                    es_mlp[o_idx] = c * o["es_pred"]
            else:
                es_mlp[o_idx] = c_scalar_current * o["es_pred"]

        scalar_history.append((w_idx, c_scalar_current))

    # Final aggregate over the eval slice (w_idx >= warmup and correction available).
    eval_idx_start = correction_applied_from or WARMUP
    eval_mask = np.array([o["w_idx"] >= eval_idx_start for o in obs_records])
    viol_mask = np.array([o["violated"] for o in obs_records])
    use = eval_mask & viol_mask

    es_uncorr = np.array([o["es_pred"] for o in obs_records])
    realised = np.array([o["real_loss"] for o in obs_records])

    summary = {}
    for name, es_arr in [("uncorrected", es_uncorr),
                         ("scalar", es_scalar),
                         ("mlp", es_mlp)]:
        if use.sum() < 5:
            summary[name] = {"n_viol": int(use.sum()), "t": float("nan"),
                             "p": float("nan"), "mean_real": float("nan"),
                             "mean_es": float("nan")}
            continue
        resid = (realised[use] - es_arr[use]) / es_arr[use]
        t, p_val, n = mcneil_frey(resid)
        summary[name] = {
            "n_viol": n, "t": t, "p": p_val,
            "mean_real": float(realised[use].mean()),
            "mean_es": float(es_arr[use].mean()),
        }

    # Per-ticker McNeil-Frey on eval slice
    per_ticker = {}
    tickers_per_obs = np.array([o["ticker"] for o in obs_records])
    for tk in sorted(set(tickers_per_obs)):
        per_ticker[tk] = {}
        sel = use & (tickers_per_obs == tk)
        if sel.sum() < 5:
            per_ticker[tk] = {"n": int(sel.sum())}
            continue
        for name, es_arr in [("uncorrected", es_uncorr),
                             ("scalar", es_scalar),
                             ("mlp", es_mlp)]:
            resid = (realised[sel] - es_arr[sel]) / es_arr[sel]
            t, p_val, n = mcneil_frey(resid)
            per_ticker[tk][name] = {"n": n, "t": t, "p": p_val}

    # Save
    out = {
        "summary": summary,
        "per_ticker": per_ticker,
        "scalar_history": scalar_history,
        "refit_log": refit_log,
        "warmup": WARMUP,
        "refit_every": REFIT_EVERY,
        "eval_idx_start": eval_idx_start,
        "n_test_windows": len(rows),
        "n_obs": int(len(obs_records)),
        "n_viol_total": int(viol_mask.sum()),
        "n_viol_eval": int(use.sum()),
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)
    log.info("Wrote %s", OUT_PKL)

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    if scalar_history:
        xs, cs = zip(*scalar_history)
        ax1.plot(xs, cs, color="#4C72B0", linewidth=1.5)
    ax1.axhline(1.0, color="#444", linestyle="--", linewidth=1.0,
                label="no correction")
    ax1.set_xlabel("test window index")
    ax1.set_ylabel("scalar correction $c(t)$")
    ax1.set_title("Expanding-window scalar correction factor")
    ax1.legend(loc="best", frameon=True)

    bar_methods = ["uncorrected", "scalar", "mlp"]
    bar_p = [summary[m]["p"] for m in bar_methods]
    bar_colors = ["#888", "#4C72B0", "#C44E52"]
    ax2.bar(bar_methods, bar_p, color=bar_colors,
            edgecolor="white", linewidth=0.6)
    ax2.axhline(0.05, color="#444", linestyle="--", linewidth=1.0,
                label="$p = 0.05$ cutoff")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 1.0)
    for i, m in enumerate(bar_methods):
        ax2.text(i, max(bar_p[i], 1e-4) * 1.15,
                 f"$p = {bar_p[i]:.3f}$" if not np.isnan(bar_p[i]) else "n/a",
                 ha="center", fontsize=9)
    ax2.set_ylabel("McNeil-Frey $p$-value, log scale")
    ax2.set_title("McNeil-Frey on the eval slice")
    ax2.legend(loc="best", frameon=True)

    fig.suptitle(
        f"Walk-forward ES correction on the real loss tail "
        f"(eval window indices {eval_idx_start} to {len(rows)-1}, "
        f"{out['n_viol_eval']} violations)",
        y=1.04, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", OUT_FIG)

    # Print summary
    log.info("=" * 60)
    log.info("Eval slice: %d windows >= idx %d, %d violations",
             len(rows) - eval_idx_start, eval_idx_start, out["n_viol_eval"])
    for name in bar_methods:
        s = summary[name]
        log.info("  %-12s  n=%d  t=%+.2f  p=%.4f   mean_real=%.4f  mean_es=%.4f",
                 name, s["n_viol"], s["t"], s["p"], s["mean_real"], s["mean_es"])


if __name__ == "__main__":
    main()
