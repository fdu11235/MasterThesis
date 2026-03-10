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

**Baseline vs CNN agreement rate:**

Agreement rate measures P(|k_hat - k*| <= r), the fraction of CNN predictions within radius r of the baseline k*:

| Sample size | Agree@5 | Agree@10 |
|-------------|---------|----------|
| n=1000      | 53.2%   | 76.8%    |
| n=2000      | 26.1%   | 45.5%    |
| n=5000      | 5.8%    | 13.6%    |

Rates drop with sample size because the candidate k-grid grows proportionally (from 121 values at n=1000 to 721 at n=5000), so a fixed radius covers a shrinking fraction of the range. This does not indicate worse model quality: the quantile RMSE table above shows accuracy *improving* with n, because neighboring thresholds in stable xi(k) regions produce nearly identical GPD fits.

**Expected Shortfall (ES) relative RMSE (p=0.99):**

ES is computed from the GPD closed-form formula ES(p) = (VaR(p) + beta - xi * u) / (1 - xi) and compared against Monte Carlo ground truth (10M samples). ES errors are larger than VaR errors because ES depends on the entire tail shape beyond the quantile, amplifying estimation errors in xi and beta.

| Sample size | VaR Rel. RMSE | ES Rel. RMSE |
|-------------|---------------|--------------|
| n=1000 | 18.2% | 117.1% |
| n=2000 | 16.2% | 116.8% |
| n=5000 | 14.5% | 70.0% |

The high overall ES RMSE is driven almost entirely by the two-Pareto distribution (123-191% ES RMSE), whose regime change makes the tail especially hard to capture. The other families perform well:

| Distribution (n=5000) | VaR Rel. RMSE | ES Rel. RMSE |
|------------------------|---------------|--------------|
| Student-t | 25.2% | 7.4% |
| Lognormal-Pareto mix | 4.5% | 10.8% |
| Pareto | 5.6% | 16.1% |
| Two-Pareto | 8.6% | **122.8%** |

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

**Expected Shortfall backtesting (McNeil & Frey, 2000):**

| Method | Mean ES | McNeil-Frey t-stat | p-value | Reject 5% |
|--------|---------|-------------------|---------|-----------|
| CNN | 0.0543 | 12.06 | <0.0001 | Yes |
| Baseline k* | 0.0543 | 12.16 | <0.0001 | Yes |
| Fixed sqrt(n) | 0.0538 | 12.82 | <0.0001 | Yes |
| Historical simulation | 0.0535 | 11.59 | <0.0001 | Yes |

The McNeil-Frey test examines whether, conditional on a VaR breach, the actual loss exceeds the ES estimate. All methods reject strongly (t > 11), meaning that when VaR is breached, losses systematically exceed what ES predicts. This is consistent with the Kupiec and Christoffersen rejections and points to a structural limitation: the unconditional rolling-window POT approach does not account for **volatility clustering** in financial returns. During stress periods, not only do VaR breaches occur more frequently (Kupiec rejection), they cluster in time (Christoffersen rejection), and the magnitude of exceedances over ES is systematically positive (McNeil-Frey rejection). A conditional model (e.g., GARCH-filtered residuals followed by POT) would likely improve these results by adapting to time-varying volatility.

![Violation rates](docs/figures/violation_rates.png)

**VaR estimates over time** — CNN vs baseline across test windows (2016-2025):

![VaR time series](docs/figures/var_time_series.png)

**Per-ticker violation rates** — box plots across 5 indices and 4 methods:

![Per-ticker violations](docs/figures/violation_rates_by_ticker.png)

---

### Extensions Beyond the Original Plan

The PDF roadmap (Steps 1-8) prescribed a minimal pipeline: three diagnostics (xi stability, KS goodness-of-fit, 1/sqrt(k) penalty), a 1D CNN, and basic agreement/quantile evaluation. It noted that "a second iteration can add: mean excess diagnostics, alternative GOF (Anderson-Darling), and time-series effects (declustering, tail index, etc.)." All of those second-iteration items are now implemented, plus several further extensions:

**Second-iteration items (all completed):**
- Mean excess linearity score as a diagnostic channel and baseline scoring component
- Anderson-Darling GOF (replaces KS throughout)
- Runs declustering for real-data rolling windows
- Hill tail index estimator as a feature channel

**Additional extensions:**
- **7 feature channels** for the CNN (xi, beta, AD GOF, mean excess score, Hill estimator, QQ-plot residual RMSE, raw mean excess) vs the 3 suggested in the PDF
- **Expected Shortfall** — closed-form GPD ES, Monte Carlo ground truth (10M samples), and McNeil-Frey backtesting on real data
- **Statistical backtesting suite** — Kupiec proportion-of-failures, Christoffersen conditional coverage, and McNeil-Frey ES tests
- **Transfer learning** — pre-train on synthetic data, fine-tune on real-data pseudo-labels with discriminative learning rates (backbone at 0.1x)
- **GARCH(1,1) filtering** — McNeil & Frey (2000) approach: fit GARCH to signed returns, apply POT to standardized residuals, scale VaR/ES back by forecasted volatility
- **Differentiable POT pipeline (Pathway M)** — end-to-end differentiable threshold selection via sigmoid soft-masking, probability-weighted-moment GPD estimation, and direct VaR/ES optimization with pinball + Fissler-Ziegel joint loss (bypasses proxy-label misalignment)
- **Regression mode** — unified model across all sample sizes with normalized k targets (replaces per-size classification)
- **Bootstrap 95% confidence intervals** on relative RMSE and ES RMSE

**Currently in progress:**
- Pathway D (Temporal Transformer) — causal Transformer encoder that outputs time-varying GPD parameters, replacing the fixed rolling-window assumption

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
- `src/evaluate.py` — Agreement metrics, VaR/ES quantile evaluation, diagnostic plots
- `src/realdata.py` — Real data loading (yfinance), rolling windows
- `src/evaluate_real.py` — VaR/ES backtesting, Kupiec test, Christoffersen independence test, McNeil-Frey ES test
- `run_pipeline.py` — Synthetic pipeline entry point (Steps 1-7)
- `run_real_pipeline.py` — Real data pipeline entry point (Step 8)
- `outputs/` — Generated at runtime (git-ignored)
