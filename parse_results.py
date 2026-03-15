#!/usr/bin/env python
"""Parse run.log output into a standardized metrics block.

Usage:
    python parse_results.py run.log              # synthetic mode (default)
    python parse_results.py run_real.log --real   # real-data mode
"""

import re
import sys


def parse_synthetic(log: str) -> dict:
    """Extract synthetic pipeline metrics from log text."""
    metrics = {}

    # ── Overall Relative RMSE per sample size ──────────────────────────
    # Format: "n=1000 (... samples):" followed by "Relative RMSE:   22.42%"
    rel_rmse_pattern = re.compile(
        r"n=(\d+)\s+\(\d+ samples?\):\s*\n"
        r"(?:.*\n)*?"  # non-greedy intermediate lines
        r".*?Relative RMSE:\s+([\d.]+)%",
        re.MULTILINE,
    )
    rel_rmse_by_n = {}
    for m in rel_rmse_pattern.finditer(log):
        n = int(m.group(1))
        val = float(m.group(2))
        rel_rmse_by_n[n] = val

    if rel_rmse_by_n:
        for n in sorted(rel_rmse_by_n):
            metrics[f"n{n}_rel_rmse"] = rel_rmse_by_n[n]
        metrics["mean_rel_rmse"] = sum(rel_rmse_by_n.values()) / len(rel_rmse_by_n)

    # ── Per-distribution RelRMSE at n=5000 ─────────────────────────────
    # Format: "student_t: RMSE=..., RelRMSE=25.61%, ..."
    # We want lines that appear after the n=5000 block
    n5000_block = re.search(
        r"n=5000\s+\(\d+ samples?\):\s*\n((?:.*\n)*)",
        log,
    )
    if n5000_block:
        block = n5000_block.group(1)
        dist_pattern = re.compile(
            r"(\w[\w_]*?):\s+RMSE=[\d.]+,\s+RelRMSE=([\d.]+)%"
        )
        for dm in dist_pattern.finditer(block):
            dist_name = dm.group(1).strip()
            val = float(dm.group(2))
            metrics[f"{dist_name}_n5000"] = val

    # ── ES Relative RMSE (mean across all n) ───────────────────────────
    es_pattern = re.compile(
        r"n=(\d+)\s+\(\d+ samples?\):\s*\n"
        r"(?:.*\n)*?"
        r".*?ES Rel\. RMSE:\s+([\d.]+)%",
        re.MULTILINE,
    )
    es_by_n = {}
    for m in es_pattern.finditer(log):
        n = int(m.group(1))
        val = float(m.group(2))
        es_by_n[n] = val

    if es_by_n:
        metrics["mean_es_rel_rmse"] = sum(es_by_n.values()) / len(es_by_n)

    # ── Training epochs ────────────────────────────────────────────────
    epoch_match = re.search(r"Early stopping at epoch (\d+)", log)
    if epoch_match:
        metrics["training_epochs"] = int(epoch_match.group(1))
    else:
        # Count epoch lines to estimate
        epoch_lines = re.findall(r"Epoch\s+(\d+):", log)
        if epoch_lines:
            metrics["training_epochs"] = int(epoch_lines[-1])

    # ── Status ─────────────────────────────────────────────────────────
    if "Pipeline complete" in log:
        metrics["status"] = "ok"
    elif "Error" in log or "Traceback" in log:
        metrics["status"] = "crash"
    else:
        metrics["status"] = "unknown"

    return metrics


def parse_real(log: str) -> dict:
    """Extract real-data pipeline metrics from log text."""
    metrics = {}

    # ── VaR Backtest Results section ───────────────────────────────────
    # Format: "  cnn_baseline       : VR=0.0271 (expected=0.0100), n=..."
    # and     "  cnn                : VR=0.0277 (expected=0.0100), n=..."
    vr_pattern = re.compile(
        r"([\w_]+)\s*:\s+VR=([\d.]+)\s+\(expected=([\d.]+)\)",
    )
    for m in vr_pattern.finditer(log):
        method = m.group(1).strip()
        vr = float(m.group(2))
        metrics[f"{method}_violation_rate"] = vr

    # ── McNeil-Frey p-values ───────────────────────────────────────────
    # Parse by finding each method's VR line, then scanning ahead for McNeil-Frey
    lines = log.splitlines()
    current_method = None
    for line in lines:
        vr_match = re.match(r"^\s*.*?\s+([\w_]+)\s*:\s+VR=[\d.]+", line)
        if vr_match:
            current_method = vr_match.group(1).strip()
        mf_match = re.search(r"McNeil-Frey:\s+t=[\d.eE+-]+,\s+p=([\d.eE+-]+)", line)
        if mf_match and current_method:
            pval = float(mf_match.group(1))
            metrics[f"{current_method}_mf_pvalue"] = pval

    # ── Status ─────────────────────────────────────────────────────────
    if "Real-data pipeline complete" in log:
        metrics["status"] = "ok"
    elif "Error" in log or "Traceback" in log:
        metrics["status"] = "crash"
    else:
        metrics["status"] = "unknown"

    return metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_results.py <logfile> [--real]", file=sys.stderr)
        sys.exit(1)

    log_path = sys.argv[1]
    real_mode = "--real" in sys.argv

    try:
        with open(log_path, "r") as f:
            log = f.read()
    except FileNotFoundError:
        print("status: crash", flush=True)
        print(f"error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    if not log.strip():
        print("status: crash", flush=True)
        print("error: log file is empty", file=sys.stderr)
        sys.exit(1)

    if real_mode:
        metrics = parse_real(log)
    else:
        metrics = parse_synthetic(log)

    if not metrics or (len(metrics) == 1 and "status" in metrics):
        print("status: crash", flush=True)
        sys.exit(1)

    # Print standardized block
    print("---")

    # Print mean_rel_rmse first if present (primary metric)
    key_order = []
    if not real_mode:
        key_order = ["mean_rel_rmse"]

    # Print in order: priority keys first, then alphabetical
    printed = set()
    for key in key_order:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                print(f"{key + ':':30s} {val:.2f}")
            else:
                print(f"{key + ':':30s} {val}")
            printed.add(key)

    for key in sorted(metrics.keys()):
        if key in printed:
            continue
        val = metrics[key]
        if isinstance(val, float):
            print(f"{key + ':':30s} {val:.2f}")
        elif isinstance(val, int):
            print(f"{key + ':':30s} {val}")
        else:
            print(f"{key + ':':30s} {val}")

    # Exit code
    if metrics.get("status") == "crash":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
