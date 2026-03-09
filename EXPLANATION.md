# Codebase Explanation

This document explains how the codebase implements the plan described in `For_Francis.pdf` ("Operational guide: ML-assisted threshold selection for POT/GPD"). Each section maps a PDF step to the corresponding code.

---

## Goal (PDF: "Goal" & "Key design choice")

The PDF proposes building an automated method to choose a threshold for Peaks-over-Threshold (POT) fitting with a Generalized Pareto Distribution (GPD). The key design choice is to **predict k (number of exceedances)** rather than the threshold u directly, because k is comparable across datasets of different sizes.

The codebase implements this end-to-end in a 7-step pipeline orchestrated by `run_pipeline.py`. The threshold u is never predicted directly -- instead the model outputs a class index into a discrete k-grid, and u is derived as the (n-k)-th order statistic.

---

## Step 1 -- Generate synthetic datasets

**PDF:** Create i.i.d. samples from at least two heavy-tailed families: (1) pure heavy tail (Student-t/Pareto), (2) contaminated tail / mixture. Fix sample sizes (n = 1000, 2000, 5000) and generate many replications.

**Code:** `src/synthetic.py` + `config/default.yaml` (synthetic section)

Four distribution families are implemented, exceeding the PDF's minimum of two:

| Family | Generator function | PDF category | Parameters (from config) |
|---|---|---|---|
| Student-t (abs. values) | `_generate_student_t()` | Pure heavy tail | df: [3, 4, 5] |
| Pareto | `_generate_pareto()` | Pure heavy tail | alpha: [1.5, 2.0, 3.0] |
| Lognormal-Pareto mix | `_generate_lognormal_pareto_mix()` | Contaminated/mixture | pareto_alpha: [1.5, 2.0], mix_frac: 0.1 |
| Two-Pareto | `_generate_two_pareto()` | Contaminated/mixture | alpha1: [2.0, 3.0] x alpha2: [1.0, 1.5] |

The function `generate_all()` takes the Cartesian product of (distribution family x parameter combos x sample sizes x replications) and produces one dataset dict per combination. Each dict stores `{"samples", "params", "dist_type", "n"}`.

**Dataset count:** With the default config, there are 14 parameter combos (3 + 3 + 2 + 4) x 3 sample sizes x 200 replications = **7'200 datasets**.

Each dataset gets a deterministic seed (`base_seed + counter`) for reproducibility.

**Pipeline (`run_pipeline.py`):** Step 1 calls `generate_all(config["synthetic"])` and saves the result to `outputs/data/synthetic.pkl`. With checkpointing enabled, if this file already exists the step is skipped on re-run.

---

## Step 2 -- Define a grid of candidate k values

**PDF:** Choose k_min = 30, k_max = floor(0.15 * n).

**Code:** `src/pot.py` -> `candidate_k_grid(n, k_min, k_max_frac)`

```python
np.arange(k_min, int(np.floor(k_max_frac * n)) + 1)
```

This produces an integer array from k_min to floor(k_max_frac * n), inclusive. For example, with n=1000: k ranges from 30 to 150 (121 candidate values). With n=5000: k ranges from 30 to 750 (721 values).

Config values (`config/default.yaml`):
- `k_min: 30`
- `k_max_frac: 0.15`

---

## Step 3 -- For each candidate k, compute diagnostics

**PDF:** For each k in the grid: (3.1) build exceedances, (3.2) fit a GPD, (3.3) compute three scores: stability, goodness-of-fit, and penalty.

**Code:** `src/pot.py`

### 3.1 & 3.2 -- Exceedances and GPD fitting

`fit_gpd(sorted_desc, k)` implements both sub-steps:
- Computes exceedances: `sorted_desc[:k] - sorted_desc[k]` (the top k values minus the (k+1)-th value, which is the threshold u)
- Fits GPD via `scipy.stats.genpareto.fit(exceedances, floc=0)` to get (xi_hat, beta_hat)
- Returns (nan, nan) on convergence failure

`fit_all_k(sorted_desc, k_grid)` runs `fit_gpd` for every k, returning a (len(k_grid), 2) array of [xi, beta] pairs. **This is the computational bottleneck** -- for each of the 8,400 datasets, hundreds of GPD fits must run.

### 3.3 -- Three diagnostic scores

**1. Parameter stability** (PDF: S_stab(k) = Var(xi_hat(k-delta)...xi_hat(k+delta)))

`score_stability(xi_series, k_grid, delta)` computes the rolling variance of xi_hat in a window of size [i-delta, i+delta], truncating at boundaries. Default delta = 5.

**2. Goodness-of-fit** (PDF: S_gof(k) = KS statistic)

`score_gof(sorted_desc, k_grid, params)` computes the Kolmogorov-Smirnov statistic between the exceedances and the fitted GPD for each k. Returns 1.0 when parameters are NaN (worst score).

**3. Penalty** (PDF: S_pen(k) = 1/sqrt(k))

`score_penalty(k_grid)` returns `1.0 / sqrt(k)` for each k.

---

## Step 4 -- Baseline score and k* selection

**PDF:** Score(k) = w1 * S_stab(k) + w2 * S_gof(k) + w3 * S_pen(k), then k* = argmin Score(k).

**Code:** `src/pot.py` -> `compute_baseline_k_star()`

This function orchestrates the full diagnostics pipeline for one dataset:

1. Calls `fit_all_k()` to get all GPD parameter estimates
2. Computes the three scores (stability, gof, penalty)
3. **Min-max normalizes** each score to [0, 1] (the PDF does not specify this, but it ensures the three scores are on comparable scales before weighting)
4. Combines them: `total = w1 * s_stab_n + w2 * s_gof_n + w3 * s_pen_n`
5. Picks k* = k_grid[argmin(total)]

Default weights from config: `[1.0, 1.0, 1.0]` (equal, as the PDF suggests starting with).

Returns both k* and a diagnostics dict containing all intermediate arrays (k_grid, params, xi_series, individual scores, total_score, k_star) -- these are reused later as CNN features and labels.

**Pipeline (`run_pipeline.py`):** Steps 2-4 are executed together per dataset. The loop is parallelized with `joblib.Parallel` and saves results to `outputs/data/diagnostics.pkl`.

---

## Step 5 -- Turn diagnostics into ML inputs

**PDF:** Build a feature vector f(k) = [xi_hat(k), beta_hat(k), S_gof(k)] for each k. Stack into a matrix F over the candidate grid. Z-score normalize each feature channel.

**Code:** `src/features.py`

### Feature matrix construction

`build_feature_matrix(diagnostics)` stacks three columns:
- xi_hat(k) from `diagnostics["params"][:, 0]`
- beta_hat(k) from `diagnostics["params"][:, 1]`
- S_gof(k) from `diagnostics["score_gof"]`

This produces F of shape (L, 3), where L = len(k_grid). This matches the PDF's f(k) = [xi_hat(k), beta_hat(k), S_gof(k)].

### Normalization

`normalize_features(F)` applies z-score normalization per column: `(col - mean) / (std + 1e-10)`. As the PDF states: "each feature channel is z-scored using its mean and standard deviation computed within the current window."

### Dataset assembly

`build_dataset(all_diagnostics, config)` groups everything by sample size n (since different sample sizes produce different k_grid lengths, they can't be batched together). For each group:

1. Builds and normalizes the feature matrix for each dataset
2. Converts k* to a **class index** into k_grid via `np.searchsorted`
3. Stacks into tensors: X of shape **(N, 3, L)** (channels-first for Conv1d) and y of shape **(N,)**

The output is a dict keyed by sample size: `{1000: (X, y), 2000: (X, y), 5000: (X, y)}`.

---

## Step 6 -- Train a 1D CNN to predict k*

**PDF:** Train a simple 1D CNN: 2-3 conv layers, small kernels, dropout, early stopping. Labels are the baseline k* from Step 4. Predict k_hat = argmax p_theta(k | F).

**Code:** `src/model.py` (architecture) + `src/train.py` (training loop)

### Architecture: ThresholdCNN

```
Input: (batch, 3, L)    -- 3 feature channels, L = len(k_grid)
  |
  v
Conv1d(3 -> 16, kernel=5, padding=2) + ReLU + BatchNorm
Conv1d(16 -> 32, kernel=5, padding=2) + ReLU + BatchNorm
  |
  v
AdaptiveAvgPool1d(1)     -- reduces (batch, 32, L) -> (batch, 32, 1)
  |
  v
Dropout(0.2)
Linear(32, n_classes)     -- n_classes = len(k_grid)
  |
  v
Output: (batch, n_classes)  -- logits over candidate k values
```

This matches the PDF: 2 conv layers, kernel size 5, dropout, and the output is a probability distribution over k values (via softmax/cross-entropy).

### Training loop

`train_model(X, y, model, config)`:
- Splits data into train/val (80/20 by default)
- Uses Adam optimizer (lr=0.001) with CrossEntropyLoss
- Early stopping: monitors validation loss with patience=10
- Saves best model weights and restores them at the end

The classification framing (cross-entropy over k-grid indices) implements the PDF's "predict k_hat = argmax p_theta(k | F)".

### Inference

`predict(model, X)` returns `logits.argmax(dim=1)` -- the predicted class index for each input.

**Pipeline (`run_pipeline.py`):** A separate model is trained per sample size group (since k_grid lengths differ). Checkpoints are saved to `outputs/checkpoints/model_n{size}.pt`. On resume, existing checkpoints are loaded and training is skipped.

---

## Step 7 -- Evaluate on synthetic data

**PDF:** Two checks: (7.1) agreement with baseline within tolerance r in {5, 10}, and (7.2) downstream tail-risk via POT quantile error at p=0.99.

**Code:** `src/evaluate.py`

### 7.1 Agreement with baseline

`agreement_rate(k_pred, k_true, radius)` computes `mean(|k_pred - k_true| <= radius)`. This is exactly the PDF's P(|k_hat - k*| <= r).

Evaluated at r=5 and r=10 (from `config/default.yaml: agreement_radii: [5, 10]`).

### 7.2 Downstream quantile error

`pot_quantile(sorted_desc, k, xi, beta, n, p)` implements the POT quantile estimator:
- If xi != 0: Q(p) = u + (beta/xi) * ((n/k * (1-p))^(-xi) - 1)
- If xi == 0: Q(p) = u - beta * log(n/k * (1-p))

`true_quantile(dist_type, dist_params, p)` computes the analytical (or approximate) true quantile for each distribution family. For mixtures (lognormal-Pareto, two-Pareto), it uses tail-dominance approximations since closed-form quantiles don't exist.

`evaluate_all()` computes RMSE between estimated and true quantiles at p=0.99, as suggested by the PDF's Err(p) = |q_hat_p - q_true_p|.

### Plots

`plot_results()` generates:
1. **Agreement rate bar chart** -- bar plot of agreement rates at each radius
2. **xi(k) and Score(k) curves** -- for the first 4 test datasets, shows how xi_hat evolves over k and where the total score is minimized (k* marked with a red dashed line)

These correspond to the PDF's suggested report items: "examples of xi_hat(k) curves, GOF curves, Score curves, baseline vs CNN agreement."

Saved to `outputs/figures/n{size}/`.

---

## Pipeline orchestration

`run_pipeline.py` ties everything together:

```
Step 1: generate_all()           -> outputs/data/synthetic.pkl
Steps 2-4: _process_one_dataset() -> outputs/data/diagnostics.pkl  (parallelized with joblib)
Step 5: build_dataset()          -> in-memory grouped tensors
For each sample size:
  Step 6: train_model()          -> outputs/checkpoints/model_n{size}.pt
  Step 7: evaluate_all()         -> outputs/figures/n{size}/
```

**Checkpointing:** Each step checks if its output file exists before running. The `--fresh` flag forces a full re-run.

**Parallelism:** Steps 2-4 (the bottleneck) run in parallel via `joblib.Parallel(n_jobs=-1)`. Controlled with `--n-jobs`.

**Interrupt safety:** `main()` is wrapped in a `KeyboardInterrupt` handler that logs which steps are saved.

---

## What is NOT yet implemented (PDF Step 8)

The PDF describes a future **Step 8**: moving to pseudo-labels on real data (rolling windows of real returns, VaR/ES evaluation). This is not yet in the codebase -- the current version only works with synthetic data, as the PDF suggests doing first.

---

## Configuration reference

All hyperparameters live in `config/default.yaml`:

| Section | Key | Value | Maps to PDF |
|---|---|---|---|
| synthetic | sample_sizes | [1000, 2000, 5000] | "Fix a few sample sizes" |
| synthetic | n_replications | 200 | "generate many replications" |
| synthetic | distributions | 4 families, 14 combos | "at least two families" |
| pot | k_min | 30 | "k_min = 30" |
| pot | k_max_frac | 0.15 | "k_max = floor(0.15n)" |
| pot | delta | 5 | "delta = 5 or 10" |
| pot | weights | [1.0, 1.0, 1.0] | "equal weights (1,1,1)" |
| model | channels | [16, 32] | "2-3 conv layers" |
| model | kernel_size | 5 | "small kernel sizes" |
| model | dropout | 0.2 | "dropout" |
| model | patience | 10 | "early stopping" |
| evaluate | agreement_radii | [5, 10] | "r in {5, 10}" |
| evaluate | quantile_p | 0.99 | "p = 0.99" |

---

## File map

```
MasterThesis/
  config/default.yaml          -- all hyperparameters
  src/
    synthetic.py               -- Step 1: data generation (4 distributions)
    pot.py                     -- Steps 2-4: GPD fitting, diagnostics, baseline scoring
    features.py                -- Step 5: feature matrix + normalization + dataset assembly
    model.py                   -- Step 6: ThresholdCNN architecture
    train.py                   -- Step 6: training loop + early stopping + inference
    evaluate.py                -- Step 7: agreement rates, quantile RMSE, plots
  run_pipeline.py              -- orchestrates Steps 1-7 with checkpointing
  tests/
    test_synthetic.py          -- tests for data generation
    test_pot.py                -- tests for GPD fitting and scoring
    test_features.py           -- tests for feature construction
    test_model.py              -- tests for CNN forward pass and overfitting
    test_evaluate.py           -- tests for agreement rate and quantile estimation
  For_Francis.pdf              -- the operational guide this codebase implements
  outputs/                     -- generated at runtime (git-ignored)
    data/synthetic.pkl         -- cached synthetic datasets
    data/diagnostics.pkl       -- cached POT diagnostics
    checkpoints/model_n*.pt    -- trained model weights
    figures/n*/                -- evaluation plots
```
