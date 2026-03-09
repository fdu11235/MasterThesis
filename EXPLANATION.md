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

### 3.3 -- Diagnostic scores

**1. Parameter stability** (PDF: S_stab(k) = Var(xi_hat(k-delta)...xi_hat(k+delta)))

`score_stability(xi_series, k_grid, delta)` computes the rolling variance of xi_hat in a window of size [i-delta, i+delta], truncating at boundaries. Default delta = 5.

**2. Goodness-of-fit** (PDF: S_gof(k))

`score_gof(sorted_desc, k_grid, params)` computes the **Anderson-Darling** (AD) statistic between the exceedances and the fitted GPD for each k. AD was chosen over the original Kolmogorov-Smirnov test because it weights tail deviations more heavily — critical for GPD fit quality. Returns 100.0 when parameters are NaN (worst score). The AD statistic is computed manually via the formula: `AD = -n - (1/n) * sum_{i=1}^{n} (2i-1) * [ln(F(z_i)) + ln(1 - F(z_{n+1-i}))]`.

**3. Mean excess linearity** (new diagnostic, not in original PDF)

`score_mean_excess(sorted_desc, k_grid)` measures how well the exceedances satisfy the GPD's theoretical property that the mean excess function `e(u) = E[X-u | X>u]` is linear in u. For each k, it computes the empirical mean excess at ~10 sub-thresholds, fits a linear regression, and returns `1 - R²` (lower = more linear = better GPD fit). This adds a fundamentally different diagnostic signal compared to AD, which only tests distributional fit.

**4. Penalty** (PDF: S_pen(k) = 1/sqrt(k))

`score_penalty(k_grid)` returns `1.0 / sqrt(k)` for each k.

---

## Step 4 -- Baseline score and k* selection

**PDF:** Score(k) = w1 * S_stab(k) + w2 * S_gof(k) + w3 * S_pen(k), then k* = argmin Score(k).

**Code:** `src/pot.py` -> `compute_baseline_k_star()`

This function orchestrates the full diagnostics pipeline for one dataset:

1. Calls `fit_all_k()` to get all GPD parameter estimates
2. Computes the four scores (stability, gof, mean excess, penalty)
3. **Min-max normalizes** each score to [0, 1] (the PDF does not specify this, but it ensures the scores are on comparable scales before weighting)
4. Combines them: `total = w1 * s_stab_n + w2 * s_gof_n + w3 * s_pen_n` (mean excess is stored in diagnostics but not included in the weighted score — it serves as a CNN feature only)
5. Picks k* = k_grid[argmin(total)]

Default weights from config: `[1.0, 1.0, 1.0]` (equal, as the PDF suggests starting with).

Returns both k* and a diagnostics dict containing all intermediate arrays (k_grid, params, xi_series, individual scores including score_mean_excess, total_score, k_star) -- these are reused later as CNN features and labels.

**Pipeline (`run_pipeline.py`):** Steps 2-4 are executed together per dataset. The loop is parallelized with `joblib.Parallel` and saves results to `outputs/data/diagnostics.pkl`.

---

## Step 5 -- Turn diagnostics into ML inputs

**PDF:** Build a feature vector f(k) = [xi_hat(k), beta_hat(k), S_gof(k)] for each k. Stack into a matrix F over the candidate grid. Z-score normalize each feature channel.

**Code:** `src/features.py`

### Feature matrix construction

`build_feature_matrix(diagnostics)` stacks four columns:
- xi_hat(k) from `diagnostics["params"][:, 0]`
- beta_hat(k) from `diagnostics["params"][:, 1]`
- S_gof(k) from `diagnostics["score_gof"]` (Anderson-Darling statistic)
- S_me(k) from `diagnostics["score_mean_excess"]` (mean excess linearity)

This produces F of shape (L, 4), where L = len(k_grid). The first three channels correspond to the PDF's f(k) = [xi_hat(k), beta_hat(k), S_gof(k)]; the fourth (mean excess) was added as an improvement to provide the CNN with a complementary diagnostic signal.

### Normalization

`normalize_features(F)` applies z-score normalization per column: `(col - mean) / (std + 1e-10)`. As the PDF states: "each feature channel is z-scored using its mean and standard deviation computed within the current window."

### Dataset assembly

`build_dataset(all_diagnostics, config)` groups everything by sample size n (since different sample sizes produce different k_grid lengths, they can't be batched together). For each group:

1. Builds and normalizes the feature matrix for each dataset
2. Converts k* to a **class index** into k_grid via `np.searchsorted`
3. Stacks into tensors: X of shape **(N, 4, L)** (channels-first for Conv1d) and y of shape **(N,)**

The output is a dict keyed by sample size: `{1000: (X, y), 2000: (X, y), 5000: (X, y)}`.

There is also a `build_dataset_regression()` variant that produces a unified dataset across all sample sizes by zero-padding shorter k_grids to L_max and normalizing labels to [0, 1]. This is used by both the synthetic regression pipeline and the real-data pipeline. It also stores per-sample metadata (k_min, k_max, n, window_idx, end_date, dist_type) for time-ordered splitting and evaluation.

---

## Step 6 -- Train a 1D CNN to predict k*

**PDF:** Train a simple 1D CNN: 2-3 conv layers, small kernels, dropout, early stopping. Labels are the baseline k* from Step 4. Predict k_hat = argmax p_theta(k | F).

**Code:** `src/model.py` (architecture) + `src/train.py` (training loop)

### Architecture: ThresholdCNN

```
Input: (batch, 4, L)    -- 4 feature channels, L = len(k_grid)
  |
  v
Conv1d(4 -> 16, kernel=5, padding=2) + ReLU + BatchNorm
Conv1d(16 -> 32, kernel=5, padding=2) + ReLU + BatchNorm
  |
  v
AdaptiveAvgPool1d(1)     -- reduces (batch, 32, L) -> (batch, 32, 1)
  |
  v
Dropout(0.2)
Linear(32, n_classes)     -- n_classes = len(k_grid) [classification]
  OR
Linear(32, 1) + Sigmoid   -- output in [0, 1] [regression]
  |
  v
Output: (batch, n_classes) or (batch,)
```

This matches the PDF: 2 conv layers, kernel size 5, dropout. The model supports both classification (cross-entropy over k-grid indices) and regression (Sigmoid output, SmoothL1 loss). The regression mode is used for the unified dataset and real-data pipeline.

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
2. **Diagnostic curves** -- for the first 4 test datasets, shows xi_hat(k), Anderson-Darling GOF(k), and total Score(k) side by side (k* marked with red dashed line)
3. **Quantile RMSE by distribution** -- bar chart of relative RMSE per distribution family with overall average line
4. **Quantile error box plot** -- box plots of relative error grouped by distribution+parameter combination, sorted by true quantile magnitude

These correspond to the PDF's suggested report items: "examples of xi_hat(k) curves, GOF curves, Score curves, baseline vs CNN agreement, downstream quantile error."

Saved to `outputs/figures/n{size}/`.

---

## Pipeline orchestration

`run_pipeline.py` ties everything together:

```
Step 1: generate_all()            -> outputs/data/synthetic.pkl
Steps 2-4: process_one_dataset()  -> outputs/data/diagnostics.pkl  (parallelized with joblib)
Step 5: build_dataset_regression() -> in-memory (X, y, meta)
Step 6: train_model()             -> outputs/checkpoints/model_regression.pt
Step 7: evaluate_all()            -> outputs/figures/n{size}/
```

**Checkpointing:** Each step checks if its output file exists before running. The `--fresh` flag forces a full re-run.

**Parallelism:** Steps 2-4 (the bottleneck) run in parallel via `joblib.Parallel(n_jobs=-1)`. Controlled with `--n-jobs`.

**Interrupt safety:** `main()` is wrapped in a `KeyboardInterrupt` handler that logs which steps are saved.

---

## Step 8 -- Real financial data with pseudo-labels

**PDF:** Move to real data using pseudo-labels from the baseline score, then evaluate with out-of-sample VaR backtesting.

**Code:** `src/realdata.py` (data loading) + `src/evaluate_real.py` (VaR backtesting) + `run_real_pipeline.py` (orchestration)

### Data loading

`src/realdata.py` downloads daily prices via `yfinance`, computes absolute log-returns `abs(log(P_t / P_{t-1}))`, and builds rolling windows.

- `load_returns(tickers, start, end)` -- downloads and caches to CSV
- `rolling_windows(abs_returns, dates, window_size, step_size, ticker)` -- creates overlapping windows in the same dict format as synthetic data (`samples`, `n`, `dist_type="real"`, etc.)
- `prepare_real_datasets(config)` -- orchestrates both, returns datasets + a `returns_lookup` dict for future-returns extraction

Default config: 5 global indices (S&P 500, NASDAQ, FTSE 100, Nikkei 225, DAX), window_size=1000, step_size=50. This produces ~1,200 windows total.

### Time-ordered train/test split

Unlike the synthetic pipeline (random split), the real pipeline sorts all windows by `end_date` and splits chronologically: first 80% for training, last 20% for testing. This prevents look-ahead bias.

### VaR backtesting

`src/evaluate_real.py` evaluates each method by computing VaR(p) from the GPD fit, then checking the next `backtest_horizon` days (default 250 = ~1 year) for violations (abs_return > VaR).

Four methods are compared:
1. **CNN** -- predicted k from the trained model
2. **Baseline k*** -- the scoring function's optimal k (what the CNN was trained to imitate)
3. **Fixed sqrt(n)** -- naive rule of thumb k = sqrt(1000) = 31
4. **Historical simulation** -- empirical quantile (no GPD, purely nonparametric)

### Statistical tests

Two backtesting tests are applied:

**Kupiec (1995) POF test** (`kupiec_test`): Tests whether the overall violation rate equals the expected 1-p. Uses a log-likelihood ratio statistic, chi-squared with 1 df.

**Christoffersen (1998) independence test** (`christoffersen_test`): Tests whether violations are independent by examining transition probabilities between consecutive days (n00, n01, n10, n11 counts). Detects violation clustering from volatility regimes. Uses LR statistic, chi-squared with 1 df.

### Plots

`plot_real_results()` generates:
1. **Violation rate bar chart** -- overall violation rate per method with expected rate line
2. **VaR time series** -- CNN vs baseline VaR estimates over test windows
3. **Per-ticker violation box plots** -- violation rate distributions by ticker and method

### Pipeline orchestration

`run_real_pipeline.py` mirrors `run_pipeline.py`:

```
Step 1: prepare_real_datasets()    -> outputs/data/real_datasets.pkl
Steps 2-4: process_one_dataset()   -> outputs/data/real_diagnostics.pkl  (parallelized)
Step 5: build_dataset_regression() -> in-memory (X, y, meta)
Split: time-ordered 80/20
Step 6: train_model()              -> outputs/checkpoints/model_real.pt
Step 7: evaluate_real()            -> outputs/figures/real/
```

Same checkpointing and interrupt safety as the synthetic pipeline.

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
| model | task | regression | "predict k" |
| model | channels | [16, 32] | "2-3 conv layers" |
| model | kernel_size | 5 | "small kernel sizes" |
| model | dropout | 0.2 | "dropout" |
| model | patience | 15 | "early stopping" |
| evaluate | agreement_radii | [5, 10] | "r in {5, 10}" |
| evaluate | quantile_p | 0.99 | "p = 0.99" |
| realdata | tickers | [^GSPC, ^IXIC, ^FTSE, ^N225, ^GDAXI] | Step 8: real data |
| realdata | window_size | 1000 | "rolling windows" |
| realdata | step_size | 50 | overlap reduction |
| realdata | backtest_horizon | 250 | "out-of-sample VaR" |
| realdata | train_fraction | 0.8 | time-ordered split |

---

## File map

```
MasterThesis/
  config/default.yaml          -- all hyperparameters
  src/
    synthetic.py               -- Step 1: data generation (4 distributions)
    pot.py                     -- Steps 2-4: GPD fitting, diagnostics (AD GOF, mean excess),
                                  baseline scoring, process_one_dataset()
    features.py                -- Step 5: feature matrix (4 channels) + normalization + dataset assembly
    model.py                   -- Step 6: ThresholdCNN architecture (classification + regression)
    train.py                   -- Step 6: training loop + early stopping + inference
    evaluate.py                -- Step 7: agreement rates, quantile RMSE, diagnostic/error plots
    realdata.py                -- Step 8: real data loading (yfinance), rolling windows
    evaluate_real.py           -- Step 8: VaR backtesting, Kupiec test, Christoffersen test
  run_pipeline.py              -- orchestrates Steps 1-7 (synthetic) with checkpointing
  run_real_pipeline.py         -- orchestrates Step 8 (real data) with checkpointing
  tests/
    test_synthetic.py          -- tests for data generation
    test_pot.py                -- tests for GPD fitting and scoring
    test_features.py           -- tests for feature construction
    test_model.py              -- tests for CNN forward pass and overfitting
    test_evaluate.py           -- tests for agreement rate and quantile estimation
  For_Francis.pdf              -- the operational guide this codebase implements
  outputs/                     -- generated at runtime (git-ignored)
    data/synthetic.pkl         -- cached synthetic datasets
    data/diagnostics.pkl       -- cached POT diagnostics (synthetic)
    data/real_datasets.pkl     -- cached real-data rolling windows
    data/real_diagnostics.pkl  -- cached POT diagnostics (real)
    data/returns_*.csv         -- cached downloaded price data
    checkpoints/model_*.pt     -- trained model weights
    figures/n*/                -- synthetic evaluation plots
    figures/real/              -- real-data VaR backtest plots
```
