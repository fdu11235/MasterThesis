#!/usr/bin/env python
"""Walk-forward ES correction: compare features extracted at the baseline k*
versus the CNN's predicted threshold k_hat.

The deployed walk-forward script (correction_net_real_walkforward.py) extracts
the nine MLP features at the baseline scorer threshold k* as a proxy for the
CNN k_hat. The thesis text (Sec. ES correction) says the features are extracted
at k_hat. This driver reruns the experiment with the *actual* k_hat so the text
becomes literally true, and reports the difference.

Only the MLP variant can change: var/es being corrected come from the CNN method
results (already at k_hat), and the scalar correction uses no features at all.
torch is seeded identically for both modes so the only difference is the feature
matrix fed to the MLP.

Run: PYTHONPATH=. python analysis/walkforward_khat_compare.py
"""
import os
import sys
import warnings

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.correction_net_real_walkforward import (  # noqa: E402
    load_returns_lookup, build_test_data, mcneil_frey, WARMUP, REFIT_EVERY,
)
from src.es_correction import extract_features, train_correction_net  # noqa: E402
from src.features import build_dataset_regression  # noqa: E402
from src.model import ThresholdCNN  # noqa: E402
from src.train import predict  # noqa: E402
import pickle  # noqa: E402


def compute_khat_lookup(config):
    """k_hat per loss-tail window, keyed by (ticker, series_end_idx).

    Replicates run_real_pipeline.py's loss-tail prediction: build regression
    features from the cached loss diagnostics, predict with the transfer model,
    denormalise to an integer k in [k_min, k_max].
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


def run(rows, obs_records, config, p, use_khat, khat_lookup, seed=42):
    """Faithful copy of the walk-forward loop, parameterised on the feature k."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    feat_cache = {}
    kmismatch = []  # (khat, kstar) for windows whose features are actually used

    def _features_for(w_idx):
        if w_idx in feat_cache:
            return feat_cache[w_idx]
        r = rows[w_idx]
        ds, diag = r["ds"], r["diag"]
        kstar = int(diag["k_star"])
        if use_khat:
            k = khat_lookup.get((ds["ticker"], ds.get("series_end_idx", 0)), kstar)
        else:
            k = kstar
        kmismatch.append((k, kstar))
        feats = extract_features(ds, diag, k, p=p, config=config)
        feat_cache[w_idx] = feats
        return feats

    es_scalar = np.array([o["es_pred"] for o in obs_records], dtype=float)
    es_mlp = np.array([o["es_pred"] for o in obs_records], dtype=float)
    correction_applied_from = None
    last_refit_w = None
    c_scalar_current = 1.0
    mlp_state = {"model": None, "X_mean": None, "X_std": None}

    for w_idx, r in enumerate(rows):
        if w_idx >= WARMUP and (last_refit_w is None or w_idx - last_refit_w >= REFIT_EVERY):
            last_refit_w = w_idx
            X_list, y_list = [], []
            for o in obs_records:
                if o["w_idx"] >= w_idx:
                    break
                if not o["violated"]:
                    continue
                feats = _features_for(o["w_idx"])
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
                        model, _h = train_correction_net(X_arr, y_arr, config_mlp)
                    mlp_state["model"] = model
                    mlp_state["X_mean"] = model.X_mean
                    mlp_state["X_std"] = model.X_std
                except Exception as e:  # noqa
                    print("  MLP fit failed", w_idx, n_train, e)
                if correction_applied_from is None:
                    correction_applied_from = w_idx

        if w_idx < WARMUP or correction_applied_from is None:
            continue
        for o_idx, o in enumerate(obs_records):
            if o["w_idx"] != w_idx:
                continue
            es_scalar[o_idx] = c_scalar_current * o["es_pred"]
            if mlp_state["model"] is not None:
                feats = _features_for(w_idx)
                fn = (feats - mlp_state["X_mean"]) / mlp_state["X_std"]
                with torch.no_grad():
                    c = float(mlp_state["model"](
                        torch.tensor(fn, dtype=torch.float32).unsqueeze(0)).item())
                es_mlp[o_idx] = c * o["es_pred"]
            else:
                es_mlp[o_idx] = c_scalar_current * o["es_pred"]

    eval_start = correction_applied_from or WARMUP
    eval_mask = np.array([o["w_idx"] >= eval_start for o in obs_records])
    viol_mask = np.array([o["violated"] for o in obs_records])
    use = eval_mask & viol_mask
    realised = np.array([o["real_loss"] for o in obs_records])
    es_uncorr = np.array([o["es_pred"] for o in obs_records])

    out = {}
    for name, es_arr in [("uncorrected", es_uncorr), ("scalar", es_scalar), ("mlp", es_mlp)]:
        resid = (realised[use] - es_arr[use]) / es_arr[use]
        t, pv, n = mcneil_frey(resid)
        out[name] = {"n": n, "t": t, "p": pv,
                     "mean_real": float(realised[use].mean()),
                     "mean_es": float(es_arr[use].mean())}
    out["_kmismatch"] = kmismatch
    out["_eval_start"] = eval_start
    out["_n_eval_viol"] = int(use.sum())
    return out


def main():
    with open(os.path.join(ROOT, "config/default.yaml")) as f:
        config = yaml.safe_load(f)
    p = config["evaluate"]["quantile_p"]

    returns_lookup = load_returns_lookup()
    rows = build_test_data(config, returns_lookup)

    obs_records = []
    for w_idx, r in enumerate(rows):
        fut = r["future_returns"]
        loss_mags = np.where(fut < 0, -fut, 0.0)
        for mag in loss_mags:
            obs_records.append({"w_idx": w_idx, "ticker": r["ticker"],
                                "real_loss": float(mag), "var_pred": r["var_pred"],
                                "es_pred": r["es_pred"], "violated": bool(mag > r["var_pred"])})

    khat_lookup = compute_khat_lookup(config)

    seeds = list(range(10))
    base = run(rows, obs_records, config, p, use_khat=False, khat_lookup=khat_lookup, seed=0)
    print("=" * 72)
    print(f"Test windows: {len(rows)}  |  eval slice from idx {base['_eval_start']}  "
          f"|  eval-slice violations: {base['_n_eval_viol']}")
    print("=" * 72)
    # Seed-independent rows
    print(f"uncorrected : t={base['uncorrected']['t']:+.3f} p={base['uncorrected']['p']:.4f}"
          f"   (thesis reports p=0.040)")
    print(f"scalar      : t={base['scalar']['t']:+.3f} p={base['scalar']['p']:.4f}"
          f"   (thesis reports p=0.106; identical for k* and k_hat by construction)")
    print("-" * 72)
    print("MLP across seeds 0..9 (mean es ~ p-value distribution):")
    for label, uk in [("k* features ", False), ("k_hat feats ", True)]:
        ps, ts = [], []
        for s in seeds:
            r = run(rows, obs_records, config, p, use_khat=uk, khat_lookup=khat_lookup, seed=s)
            ps.append(r["mlp"]["p"]); ts.append(r["mlp"]["t"])
        ps = np.array(ps); ts = np.array(ts)
        npass = int((ps > 0.05).sum())
        print(f"  MLP {label}: p mean={ps.mean():.3f} median={np.median(ps):.3f} "
              f"min={ps.min():.3f} max={ps.max():.3f} | passes(>0.05) {npass}/10 | "
              f"t mean={ts.mean():+.2f}")
    print("-" * 72)
    khat_run = run(rows, obs_records, config, p, use_khat=True, khat_lookup=khat_lookup, seed=0)
    km = khat_run["_kmismatch"]
    diffs = np.array([kh - ks for kh, ks in km])
    same = int((diffs == 0).sum())
    print(f"k_hat vs k* on feature windows: exact {same}/{len(diffs)} "
          f"({100*same/len(diffs):.1f}%) | |k_hat-k*| mean={np.abs(diffs).mean():.2f} "
          f"median={np.median(np.abs(diffs)):.1f} max={np.abs(diffs).max()}")


if __name__ == "__main__":
    main()
