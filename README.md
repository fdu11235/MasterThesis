# MasterThesis

ML-assisted threshold selection for Peaks-over-Threshold (POT) modelling with the Generalized Pareto Distribution (GPD).

## Overview

Choosing the threshold for a POT/GPD fit is the central problem in extreme value tail estimation. This project automates that choice.

The method works in three parts:

1. **Synthetic training.** Datasets are generated from distributions with known tail behaviour (Student-t, Pareto, and mixtures). A baseline scoring rule selects the optimal number of exceedances k* from a grid of candidate thresholds. A 1D CNN is then trained to replicate and generalize that selection from diagnostic curves (xi stability, Anderson-Darling goodness-of-fit, mean excess, Hill estimator, and others).
2. **Real-data evaluation.** The trained model is fine-tuned by transfer learning and applied to daily returns of five global equity indices. Tail risk is evaluated with out-of-sample VaR and Expected Shortfall (ES) backtesting, including Kupiec, Christoffersen, and McNeil-Frey tests. POT is fitted both on raw returns and on GARCH(1,1)-standardized residuals.
3. **ES correction.** A separate correction network addresses the systematic bias in GPD-based ES estimates, which is most severe in the high tail-index regime.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the synthetic pipeline (Steps 1-7):

```bash
python run_pipeline.py --config config/default.yaml
```

Run the real-data pipeline (Step 8):

```bash
python run_real_pipeline.py --config config/default.yaml
```

Both pipelines cache intermediate results. Pass `--fresh` to recompute from scratch.

Run all three core pipelines end to end (synthetic, perturbation, real):

```bash
bash run_all.sh
```

One-off experiment drivers live in `experiments/`. Launch them from the repo root so relative output paths resolve, for example:

```bash
python experiments/run_high_xi_experiment.py --config config/high_xi.yaml
```

## Tests

```bash
pytest tests/ -v
```

## Project structure

Core source and entry points:

- `config/` - YAML configuration (`default.yaml` plus ablation and window-size variants)
- `src/synthetic.py` - synthetic data generation (Student-t, Pareto, mixtures)
- `src/pot.py` - GPD fitting, Anderson-Darling GOF, mean excess diagnostic, baseline scoring
- `src/features.py` - 7-channel feature matrix for the CNN (xi, beta, AD GOF, mean excess score, Hill estimator, QQ residual, raw mean excess)
- `src/model.py` - 1D CNN architecture (classification and regression heads)
- `src/train.py` - training loop with early stopping and transfer learning
- `src/evaluate.py` - agreement metrics, VaR/ES quantile evaluation, diagnostic plots
- `src/evaluate_real.py` - VaR/ES backtesting (Kupiec, Christoffersen, McNeil-Frey)
- `src/realdata.py` - real data loading (yfinance), rolling windows, declustering
- `src/garch.py` - GARCH(1,1) volatility filtering
- `src/es_correction.py` - ES correction network
- `src/perturbation.py` - perturbation robustness utilities
- `run_pipeline.py` - synthetic pipeline entry point (Steps 1-7)
- `run_real_pipeline.py` - real-data pipeline entry point (Step 8)
- `run_perturbation_experiment.py` - perturbation experiment (also part of the overnight chain)
- `run_all.sh` - runs the three core pipelines end to end

Supporting code and material:

- `experiments/` - one-off experiment drivers (run from the repo root)
- `analysis/` - figure generation, ES closed-form validation, and robustness checks
- `docs/` - appendices, investigation notes, and figures. Reference PDFs are kept locally and git-ignored.
- `tests/` - pytest suite
- `outputs/` - generated at runtime (git-ignored)

## Results

The full results, figures, and discussion live in the LaTeX thesis under `latex/` (kept local, git-ignored). Generated figures are in `docs/figures/`, and the technical appendices are `docs/appendix_es_validation.md` and `docs/high_xi_es_investigation.md`.
