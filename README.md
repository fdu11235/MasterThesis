# MasterThesis

ML-Assisted Threshold Selection for Peaks-over-Threshold (POT) with Generalized Pareto Distribution (GPD).

## Overview

Automated method to choose a threshold for POT/GPD fitting using synthetic data with known tail behavior. A baseline scoring rule selects k* (number of exceedances), then a 1D CNN is trained to replicate and generalize that selection from diagnostic features.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline (Steps 1-7):

```bash
python run_pipeline.py --config config/default.yaml
```

## Tests

```bash
pytest tests/ -v
```

## Project Structure

- `config/default.yaml` — All hyperparameters
- `src/synthetic.py` — Synthetic data generation (Student-t, Pareto, mixtures)
- `src/pot.py` — GPD fitting, diagnostics, baseline scoring
- `src/features.py` — Feature matrix construction for CNN
- `src/model.py` — 1D CNN architecture (PyTorch)
- `src/train.py` — Training loop with early stopping
- `src/evaluate.py` — Agreement metrics, quantile evaluation, plotting
- `run_pipeline.py` — Main entry point
- `outputs/` — Generated at runtime (git-ignored)
