# MasterThesis

ML-Assisted Threshold Selection for Peaks-over-Threshold (POT) with Generalized Pareto Distribution (GPD).

## Overview

Automated method to choose a threshold for POT/GPD fitting using synthetic data with known tail behavior. A baseline scoring rule selects k* (number of exceedances), then a 1D CNN is trained to replicate and generalize that selection from diagnostic features.

## Current Results

### Synthetic Pipeline (Steps 1-7)

The CNN is trained on 7,200 synthetic datasets (4 distribution families, 3 sample sizes, 200 replications each) to predict the optimal number of exceedances k from diagnostic curves.

**Relative RMSE of quantile estimates (p=0.99):**

For each test sample, the CNN predicts k, the GPD is fitted at that k to obtain parameters (xi, beta), and the POT quantile formula gives an estimate of the 99th percentile. This is compared against the true quantile (known analytically for Student-t/Pareto, or via Monte Carlo for mixtures). The relative RMSE normalizes the error by the true quantile so that results are comparable across distributions with very different tail magnitudes.

| Sample size | Relative RMSE |
|-------------|---------------|
| n=1000 | 18.9% |
| n=2000 | 16.2% |
| n=5000 | 13.9% |

Performance improves with larger samples as expected. Student-t is the hardest family (~25% RMSE) because its tails are lighter than what GPD naturally fits, causing systematic overestimation. Pareto and mixtures are easiest (~5-11%) since their tails match GPD theory.

**Diagnostic curves** — xi(k), Anderson-Darling GOF(k), and composite Score(k) for example datasets (n=1000), with baseline k* marked:

![Diagnostic curves](docs/figures/diagnostic_curves.png)

**Quantile estimation error by distribution** — box plots of relative error grouped by distribution+parameters (n=1000):

![Quantile error boxplot](docs/figures/quantile_error_boxplot_n1000.png)

**Per-distribution RMSE breakdown** (n=5000):

![RMSE by distribution](docs/figures/quantile_rmse_by_dist_n5000.png)

### Real Data Pipeline (Step 8)

The pipeline downloads daily returns for 5 global equity indices (S&P 500, NASDAQ, FTSE 100, Nikkei 225, DAX), builds 1,216 rolling windows (size=1000, step=50), and evaluates with out-of-sample VaR backtesting.

**VaR violation rates** (expected = 1% at p=0.99):

| Method | Violation Rate |
|--------|---------------|
| CNN | 1.52% |
| Baseline k* | 1.51% |
| Fixed sqrt(n) | 1.46% |
| Historical simulation | 1.55% |

All methods produce reasonable violation rates (1.5-1.6%). The CNN closely tracks the baseline k*, confirming it learned the pseudo-labels. Kupiec tests reject at 5% for all methods (typical for real data with volatility clustering). Christoffersen independence tests also reject, confirming violations cluster during crisis periods.

![Violation rates](docs/figures/violation_rates.png)

**VaR estimates over time** — CNN vs baseline across test windows (2016-2025):

![VaR time series](docs/figures/var_time_series.png)

**Per-ticker violation rates** — box plots across 5 indices and 4 methods:

![Per-ticker violations](docs/figures/violation_rates_by_ticker.png)

---

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the synthetic pipeline (Steps 1-7):

```bash
python run_pipeline.py --config config/default.yaml
```

Run the real data pipeline (Step 8):

```bash
python run_real_pipeline.py --config config/default.yaml
```

Both pipelines cache intermediate results. Use `--fresh` to recompute from scratch.

## Tests

```bash
pytest tests/ -v
```

## Project Structure

- `config/default.yaml` — All hyperparameters
- `src/synthetic.py` — Synthetic data generation (Student-t, Pareto, mixtures)
- `src/pot.py` — GPD fitting, Anderson-Darling GOF, mean excess diagnostic, baseline scoring
- `src/features.py` — Feature matrix construction (4 channels) for CNN
- `src/model.py` — 1D CNN architecture (classification + regression)
- `src/train.py` — Training loop with early stopping
- `src/evaluate.py` — Agreement metrics, quantile evaluation, diagnostic plots
- `src/realdata.py` — Real data loading (yfinance), rolling windows
- `src/evaluate_real.py` — VaR backtesting, Kupiec test, Christoffersen independence test
- `run_pipeline.py` — Synthetic pipeline entry point (Steps 1-7)
- `run_real_pipeline.py` — Real data pipeline entry point (Step 8)
- `outputs/` — Generated at runtime (git-ignored)
