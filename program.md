# Autonomous Research Loop — ThresholdCNN Hyperparameter Search

You are an autonomous research agent running experiments on a CNN-based POT threshold selection model.
Your goal: **minimize mean Relative RMSE** across sample sizes n=1000, 2000, 5000.

## NEVER STOP

You must run experiments continuously. After each experiment, immediately start the next one.
Do not ask for permission. Do not wait for input. Do not summarize and stop.
If something fails, diagnose it, fix it, and continue experimenting.

**YOU MUST NEVER STOP RUNNING EXPERIMENTS.**

---

## Setup (do this once at the start)

1. Create and switch to a new branch:
   ```bash
   git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)
   ```

2. Read and understand the codebase:
   - `config/experiment.yaml` — your experiment config (tag: "exp")
   - `src/model.py` — ThresholdCNN architecture
   - `src/train.py` — training loop, optimizer, scheduler, loss
   - `run_pipeline.py` — the main runner (read-only for you)
   - `parse_results.py` — result parser

3. Run baseline experiment:
   ```bash
   python run_pipeline.py --config config/experiment.yaml > run.log 2>&1
   ```
   Parse results:
   ```bash
   python parse_results.py run.log
   ```
   Record baseline metrics in `results.tsv`.

4. Initialize `results.tsv` with a header:
   ```
   exp_id	description	mean_rel_rmse	n1000	n2000	n5000	es_rel_rmse	epochs	status	decision	timestamp
   ```

5. Set `keep_streak = 0`.

---

## Experiment Loop

For each experiment:

### a. Choose what to modify

Pick ONE thing to change per experiment (or a small coherent group). Ideas:

**Config changes** (edit `config/experiment.yaml`):
- `model.channels`: e.g. [16,32], [32,64,128], [64,128], [48,96]
- `model.kernel_size`: 3, 5, 7, 9, 11
- `model.dropout`: 0.1, 0.15, 0.2, 0.3, 0.4
- `model.lr`: 0.0005, 0.001, 0.002, 0.005
- `model.batch_size`: 32, 64, 128, 256
- `model.patience`: 20, 30, 50
- `model.max_epochs`: 300, 500, 800
- `pot.weights`: [1,1,1], [2,1,1], [1,2,1], [0.5,1,1.5]
- `features.columns`: subsets like [0,1,2,3,4,5,6], [0,1,2,3], [0,1,4,5,6]

**Model architecture changes** (edit `src/model.py`):
- Add residual connections
- Try different activations (GELU, LeakyReLU, SiLU)
- Add attention mechanism after conv layers
- Change pooling (AdaptiveMaxPool1d, combined avg+max)
- Add a second linear layer in the head
- Experiment with BatchNorm vs LayerNorm vs no norm

**Training changes** (edit `src/train.py`):
- Optimizer: AdamW, SGD with momentum, RAdam
- Scheduler: CosineAnnealingLR, OneCycleLR, StepLR
- Loss function: MSELoss, HuberLoss (different delta), L1Loss
- Gradient clipping

### b. Commit the change

```bash
git add -A && git commit -m "try: <short description of what changed>"
```

### c. Run the experiment

```bash
timeout 900 python run_pipeline.py --config config/experiment.yaml > run.log 2>&1
```

The 900s (15 min) timeout kills runs that hang. Normal runs take ~7-10 min.

### d. Parse results

```bash
python parse_results.py run.log
```

Extract `mean_rel_rmse` as the primary metric.

### e. Decide: keep or discard

**KEEP** if:
- `mean_rel_rmse` < previous best `mean_rel_rmse`
- AND no single per-distribution RelRMSE at n=5000 regressed by more than 2 percentage points vs baseline

**DISCARD** if:
- `mean_rel_rmse` >= previous best
- OR any distribution regressed >2pp
- OR status is crash/unknown

On **KEEP**:
- Log in `results.tsv`: metrics + "keep"
- Increment `keep_streak`
- The current state becomes the new baseline to beat

On **DISCARD**:
- `git reset --hard HEAD~1`
- Log in `results.tsv`: metrics + "discard"
- keep_streak stays the same

### f. Real-data validation (periodic)

After every 5th consecutive **keep** (`keep_streak` reaches 5):

```bash
timeout 1800 python run_real_pipeline.py --config config/experiment.yaml > run_real.log 2>&1
python parse_results.py run_real.log --real
```

- Log real metrics in `results.tsv` as a separate row with description "real-validation"
- A regression on real data does NOT trigger a discard — just log a WARNING
- Reset `keep_streak = 0`

### g. Update progress plot

After every experiment (keep or discard), regenerate the progress chart:

```bash
python plot_progress.py
```

This writes `outputs/exp/figures/progress.png` with three panels:
1. **Mean RelRMSE** over time with running-best line and baseline reference
2. **Per sample-size breakdown** (n=1000, 2000, 5000)
3. **Decision timeline** — color-coded keep/discard bar with summary stats

### h. Repeat

Go back to step (a). **NEVER STOP.**

---

## Scope Rules

### CAN modify
- `config/experiment.yaml` — any parameter
- `src/model.py` — ThresholdCNN class (architecture changes)
- `src/train.py` — optimizer, scheduler, loss, training loop

### CANNOT modify
- `src/pot.py` — POT scoring logic
- `src/evaluate.py` — evaluation metrics
- `src/synthetic.py` — data generation
- `src/features.py` — feature extraction
- `run_pipeline.py` — pipeline orchestration
- `run_real_pipeline.py` — real-data pipeline
- `parse_results.py` — result parser
- `plot_progress.py` — progress chart generator

### CRITICAL CONSTRAINTS
- **NEVER use `--fresh` flag.** Diagnostics computation takes ~30 minutes. The cached `outputs/data/diagnostics.pkl` must always be reused. Only training + evaluation runs each iteration.
- **NEVER install new packages.** Only use what's already available.
- **NEVER modify files outside the CAN-modify list.**
- **ALWAYS delete `outputs/exp/checkpoints/` before each run** so the model retrains from scratch:
  ```bash
  rm -rf outputs/exp/checkpoints/
  ```

---

## Metrics Reference

**Primary metric:** `mean_rel_rmse` — mean of Relative RMSE across n=1000, 2000, 5000. Lower is better.

**Secondary constraint:** No single distribution's RelRMSE at n=5000 should regress more than 2 percentage points from baseline.

**Tertiary (informational):** `mean_es_rel_rmse` — Expected Shortfall relative RMSE. Track but don't optimize directly.

---

## Strategy Tips

1. **Start with config-only changes** — they're safe and fast to iterate
2. **Change one thing at a time** — isolate what works
3. **If a direction shows promise, explore further** — e.g. if wider channels help, try even wider
4. **If 3 experiments in a row are discarded in the same direction, try something different**
5. **Architecture changes are higher-risk, higher-reward** — try them after exhausting easy config wins
6. **Keep notes** in your commit messages about what you tried and why
7. **Watch for overfitting signals** — if training epochs drop dramatically, the model might be memorizing

---

## results.tsv Format

Tab-separated, one row per experiment:

```
exp_id	description	mean_rel_rmse	n1000	n2000	n5000	es_rel_rmse	epochs	status	decision	timestamp
0	baseline	20.63	22.42	22.89	16.55	86.72	146	ok	baseline	2026-03-16T10:00:00
1	channels=[64,128]	19.85	21.10	21.50	17.00	84.50	152	ok	keep	2026-03-16T10:08:00
2	lr=0.002	21.10	23.00	22.50	17.80	88.00	98	ok	discard	2026-03-16T10:16:00
```

For real-validation rows, use a different format:
```
1r	real-validation	cnn_vr=0.027	baseline_vr=0.028	cnn_mf_p=0.588	baseline_mf_p=0.012	-	-	ok	real-check	2026-03-16T10:20:00
```
