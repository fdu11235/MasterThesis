"""Walk-forward expanding-window training of the ES correction net on
real-data loss-tail backtests, using realized exceedance returns as
supervision.

Two correction models are fitted side by side:
  - scalar: a single multiplicative factor c = mean(realized) / mean(ES_pred)
            over all violations observed before the current window.
  - mlp:    the existing 9-feature ESCorrectionNet from src/es_correction.py.

Walk-forward parameters: warmup=200 windows (no correction applied before
this many windows of history), refit_every=50.

The MLP features are extracted at the CNN-predicted threshold k_hat (recomputed
here from the transfer-learned model, exactly as run_real_pipeline.py does),
not the baseline scorer threshold k*. The MLP weight initialisation is
stochastic, so the MLP correction is run over several seeds and reported as the
mean with the across-seed range. The scalar correction and the uncorrected
baseline are deterministic (the scalar uses no features).

Reads the existing pickles produced by run_real_pipeline.py. Writes a
result pickle and one figure summarising the running correction factor
and McNeil-Frey p-value.

Run: PYTHONPATH=. python analysis/correction_net_real_walkforward.py
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
from src.features import build_dataset_regression  # noqa: E402
from src.model import ThresholdCNN  # noqa: E402
from src.train import predict  # noqa: E402

OUT_PKL = os.path.join(ROOT, "outputs", "correction_walkforward_results.pkl")
OUT_FIG = os.path.join(ROOT, "outputs", "figures", "results_chapter",
                       "correction_walkforward.png")

WARMUP = 200
REFIT_EVERY = 50
MLP_SEEDS = list(range(10))  # MLP weight init is stochastic; average over seeds

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


def compute_khat_lookup(config, p):
    """CNN-predicted threshold k_hat per loss-tail window, keyed by
    (ticker, series_end_idx). Replicates run_real_pipeline.py's loss-tail
    prediction: build regression features from the cached loss diagnostics,
    predict with the transfer model, denormalise to an integer k.
    """
    with open(os.path.join(ROOT, "outputs/data/real_diagnostics_loss.pkl"), "rb") as f:
        diag_list = pickle.load(f)
    X, _y, meta = build_dataset_regression(diag_list, config)

    mcfg = config["model"]
    feat_cfg = config.get("features", {})
    in_ch = len(feat_cfg.get("columns", [0, 1, 2, 3, 4, 5, 6]))
    tl = config.get("transfer_learning", {}).get("enabled", False)
    ckpt = "model_real_transfer.pt" if tl else "model_real.pt"
    model = ThresholdCNN(in_channels=in_ch, channels=mcfg["channels"],
                         kernel_size=mcfg["kernel_size"], dropout=mcfg["dropout"],
                         pool_sizes=mcfg.get("pool_sizes"), task="regression")
    model.load_state_dict(torch.load(os.path.join(ROOT, "outputs/checkpoints", ckpt),
                                     weights_only=True))
    model.eval()
    pred = predict(model, X, task="regression")
    lookup = {}
    for i, (ds, _diag) in enumerate(diag_list):
        m = meta[i]
        khat = int(np.clip(round(m["k_min"] + pred[i] * (m["k_max"] - m["k_min"])),
                           m["k_min"], m["k_max"]))
        lookup[(ds["ticker"], ds.get("series_end_idx", 0))] = khat
    return lookup


# Features are extracted at k_hat and do not depend on the MLP seed, so cache
# them once and reuse across seeds.
_FEAT_CACHE = {}


def _features_at_khat(w_idx, rows, khat_lookup, config, p):
    if w_idx in _FEAT_CACHE:
        return _FEAT_CACHE[w_idx]
    r = rows[w_idx]
    ds, diag = r["ds"], r["diag"]
    kstar = int(diag["k_star"])
    k = khat_lookup.get((ds["ticker"], ds.get("series_end_idx", 0)), kstar)
    feats = extract_features(ds, diag, k, p=p, config=config)
    _FEAT_CACHE[w_idx] = feats
    return feats


def walk_forward(seed, rows, obs_records, config, p, khat_lookup):
    """One expanding-window pass. The scalar correction and refit schedule are
    deterministic; only the MLP weight init depends on `seed`."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    es_scalar = np.array([o["es_pred"] for o in obs_records], dtype=float)
    es_mlp = np.array([o["es_pred"] for o in obs_records], dtype=float)
    scalar_history = []
    refit_log = []
    mlp_state = {"model": None, "X_mean": None, "X_std": None}
    correction_applied_from = None
    last_refit_w = None
    c_scalar_current = 1.0

    for w_idx, r in enumerate(rows):
        if w_idx >= WARMUP and (last_refit_w is None
                                or w_idx - last_refit_w >= REFIT_EVERY):
            last_refit_w = w_idx
            X_list, y_list = [], []
            for o in obs_records:
                if o["w_idx"] >= w_idx:
                    break
                if not o["violated"]:
                    continue
                feats = _features_at_khat(o["w_idx"], rows, khat_lookup, config, p)
                if feats is None:
                    continue
                y_list.append(o["real_loss"] / max(o["es_pred"], 1e-10))
                X_list.append(feats)
            n_train = len(y_list)
            if n_train >= 5:
                X_arr = np.array(X_list, dtype=np.float32)
                y_arr = np.array(y_list, dtype=np.float32)
                c_scalar_current = float(y_arr.mean())
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
        for o_idx, o in enumerate(obs_records):
            if o["w_idx"] != w_idx:
                continue
            es_scalar[o_idx] = c_scalar_current * o["es_pred"]
            if mlp_state["model"] is not None:
                feats = _features_at_khat(w_idx, rows, khat_lookup, config, p)
                fn = (feats - mlp_state["X_mean"]) / mlp_state["X_std"]
                with torch.no_grad():
                    c = float(mlp_state["model"](
                        torch.tensor(fn, dtype=torch.float32).unsqueeze(0)).item())
                es_mlp[o_idx] = c * o["es_pred"]
            else:
                es_mlp[o_idx] = c_scalar_current * o["es_pred"]
        scalar_history.append((w_idx, c_scalar_current))

    eval_start = correction_applied_from or WARMUP
    return {"es_scalar": es_scalar, "es_mlp": es_mlp,
            "scalar_history": scalar_history, "refit_log": refit_log,
            "eval_start": eval_start}


def _mf_stats(realised, es_arr, mask):
    resid = (realised[mask] - es_arr[mask]) / es_arr[mask]
    t, p_val, n = mcneil_frey(resid)
    return {"n_viol": n, "t": t, "p": p_val,
            "mean_real": float(realised[mask].mean()),
            "mean_es": float(es_arr[mask].mean())}


def main():
    cfg_path = os.path.join(ROOT, "config", "default.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    p = config["evaluate"]["quantile_p"]

    log.info("Loading return data and reconstructing test slice ...")
    returns_lookup = load_returns_lookup()
    rows = build_test_data(config, returns_lookup)
    log.info("Reconstructed %d test windows on the loss tail", len(rows))

    obs_records = []
    for w_idx, r in enumerate(rows):
        loss_mags = np.where(r["future_returns"] < 0, -r["future_returns"], 0.0)
        for mag in loss_mags:
            obs_records.append({
                "w_idx": w_idx, "ticker": r["ticker"], "end_date": r["end_date"],
                "real_loss": float(mag), "var_pred": r["var_pred"],
                "es_pred": r["es_pred"], "violated": bool(mag > r["var_pred"]),
            })
    log.info("Total per-day observations: %d; violations: %d",
             len(obs_records), sum(o["violated"] for o in obs_records))

    log.info("Computing CNN k_hat per window ...")
    khat_lookup = compute_khat_lookup(config, p)

    log.info("Running walk-forward over %d MLP seeds ...", len(MLP_SEEDS))
    runs = [walk_forward(s, rows, obs_records, config, p, khat_lookup) for s in MLP_SEEDS]

    realised = np.array([o["real_loss"] for o in obs_records])
    es_uncorr = np.array([o["es_pred"] for o in obs_records])
    viol_mask = np.array([o["violated"] for o in obs_records])
    eval_idx_start = runs[0]["eval_start"]
    use = np.array([o["w_idx"] >= eval_idx_start for o in obs_records]) & viol_mask

    # Deterministic methods (identical across seeds).
    summary = {
        "uncorrected": _mf_stats(realised, es_uncorr, use),
        "scalar": _mf_stats(realised, runs[0]["es_scalar"], use),
    }
    # MLP: one McNeil-Frey per seed, reported as mean + across-seed range.
    mlp_per_seed = [_mf_stats(realised, rn["es_mlp"], use) for rn in runs]
    mlp_p = np.array([s["p"] for s in mlp_per_seed])
    mlp_t = np.array([s["t"] for s in mlp_per_seed])
    mlp_es = np.array([s["mean_es"] for s in mlp_per_seed])
    summary["mlp"] = {
        "n_viol": mlp_per_seed[0]["n_viol"],
        "p": float(mlp_p.mean()), "p_median": float(np.median(mlp_p)),
        "p_min": float(mlp_p.min()), "p_max": float(mlp_p.max()),
        "n_pass": int((mlp_p > 0.05).sum()), "n_seeds": len(MLP_SEEDS),
        "t": float(mlp_t.mean()),
        "mean_real": mlp_per_seed[0]["mean_real"], "mean_es": float(mlp_es.mean()),
    }

    # Per-ticker (eval slice): scalar/uncorrected deterministic; MLP seed-mean.
    per_ticker = {}
    tickers_per_obs = np.array([o["ticker"] for o in obs_records])
    for tk in sorted(set(tickers_per_obs)):
        sel = use & (tickers_per_obs == tk)
        if sel.sum() < 5:
            per_ticker[tk] = {"n": int(sel.sum())}
            continue
        entry = {
            "uncorrected": _mf_stats(realised, es_uncorr, sel),
            "scalar": _mf_stats(realised, runs[0]["es_scalar"], sel),
        }
        tk_p = np.array([_mf_stats(realised, rn["es_mlp"], sel)["p"] for rn in runs])
        tk_t = np.array([_mf_stats(realised, rn["es_mlp"], sel)["t"] for rn in runs])
        entry["mlp"] = {"n": int(sel.sum()), "p": float(tk_p.mean()),
                        "p_min": float(tk_p.min()), "p_max": float(tk_p.max()),
                        "t": float(tk_t.mean())}
        per_ticker[tk] = entry

    out = {
        "summary": summary, "per_ticker": per_ticker,
        "scalar_history": runs[0]["scalar_history"], "refit_log": runs[0]["refit_log"],
        "warmup": WARMUP, "refit_every": REFIT_EVERY, "mlp_seeds": MLP_SEEDS,
        "feature_threshold": "k_hat", "mlp_p_per_seed": mlp_p.tolist(),
        "eval_idx_start": eval_idx_start, "n_test_windows": len(rows),
        "n_obs": int(len(obs_records)), "n_viol_total": int(viol_mask.sum()),
        "n_viol_eval": int(use.sum()),
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)
    log.info("Wrote %s", OUT_PKL)

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    if out["scalar_history"]:
        xs, cs = zip(*out["scalar_history"])
        ax1.plot(xs, cs, color="#4C72B0", linewidth=1.5)
    ax1.axhline(1.0, color="#444", linestyle="--", linewidth=1.0, label="no correction")
    ax1.set_xlabel("test window index")
    ax1.set_ylabel("scalar correction $c(t)$")
    ax1.set_title("Expanding-window scalar correction factor")
    ax1.legend(loc="best", frameon=True)

    bar_methods = ["uncorrected", "scalar", "mlp"]
    bar_p = [summary[m]["p"] for m in bar_methods]
    ax2.bar(bar_methods, bar_p, color=["#888", "#4C72B0", "#C44E52"],
            edgecolor="white", linewidth=0.6)
    # show the across-seed range for the MLP bar
    ax2.errorbar([2], [summary["mlp"]["p"]],
                 yerr=[[summary["mlp"]["p"] - summary["mlp"]["p_min"]],
                       [summary["mlp"]["p_max"] - summary["mlp"]["p"]]],
                 fmt="none", ecolor="#333", capsize=4, linewidth=1.0)
    ax2.axhline(0.05, color="#444", linestyle="--", linewidth=1.0, label="$p = 0.05$ cutoff")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 1.0)
    for i, m in enumerate(bar_methods):
        lbl = (f"$p = {bar_p[i]:.3f}$" if m != "mlp"
               else f"mean $p = {bar_p[i]:.3f}$")
        ax2.text(i, max(bar_p[i], 1e-4) * 1.25, lbl, ha="center", fontsize=9)
    ax2.set_ylabel("McNeil-Frey $p$-value, log scale")
    ax2.set_title("McNeil-Frey on the eval slice")
    ax2.legend(loc="best", frameon=True)

    fig.suptitle(
        f"Walk-forward ES correction on the real loss tail "
        f"(eval window indices {eval_idx_start} to {len(rows)-1}, "
        f"{out['n_viol_eval']} violations; MLP averaged over {len(MLP_SEEDS)} seeds)",
        y=1.04, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", OUT_FIG)

    log.info("=" * 60)
    log.info("Eval slice: %d windows >= idx %d, %d violations",
             len(rows) - eval_idx_start, eval_idx_start, out["n_viol_eval"])
    s = summary["uncorrected"]
    log.info("  uncorrected  n=%d  t=%+.2f  p=%.4f  mean_es=%.4f",
             s["n_viol"], s["t"], s["p"], s["mean_es"])
    s = summary["scalar"]
    log.info("  scalar       n=%d  t=%+.2f  p=%.4f  mean_es=%.4f",
             s["n_viol"], s["t"], s["p"], s["mean_es"])
    s = summary["mlp"]
    log.info("  mlp(k_hat)   n=%d  t=%+.2f  p=%.4f [%.4f,%.4f] pass %d/%d  mean_es=%.4f",
             s["n_viol"], s["t"], s["p"], s["p_min"], s["p_max"],
             s["n_pass"], s["n_seeds"], s["mean_es"])


if __name__ == "__main__":
    main()
