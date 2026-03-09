# Approach: End-to-End Differentiable POT for Tail Risk Estimation

This document describes the full approach taken in this thesis, including the
baseline pipeline (Pathway A), the differentiable extension (Pathway M), what
worked, what failed, and the lessons learned.

---

## 1. Problem Statement

The goal is to automate threshold selection for Peaks-over-Threshold (POT)
fitting with a Generalized Pareto Distribution (GPD). Given a sample of
heavy-tailed observations, the method must choose the number of exceedances
*k* (equivalently a threshold *u*), fit GPD parameters (xi, sigma), and
produce estimates of Value-at-Risk (VaR) and Expected Shortfall (ES) at the
99th percentile.

The key design choice is to **predict k rather than u**, because k is
scale-invariant and comparable across datasets of different sizes and
magnitudes.

---

## 2. Pathway A: CNN with Pseudo-Labels (Baseline)

### Architecture

```
Synthetic data (7,200 datasets, 4 distribution families)
    |
    v
For each candidate k in [30, floor(0.15n)]:
    fit GPD -> compute 4 diagnostic channels:
      1. xi_hat(k)         -- GPD shape parameter
      2. beta_hat(k)       -- GPD scale parameter
      3. S_gof(k)          -- Anderson-Darling goodness-of-fit
      4. S_me(k)           -- Mean excess linearity (1 - R^2)
    |
    v
Baseline scoring: k* = argmin [w1*S_stab + w2*S_gof + w3*S_pen]
    (equal weights, min-max normalized scores)
    |
    v
1D CNN (2 conv layers, kernel=5, adaptive pooling)
    trained via SmoothL1 regression on normalized k_tilde = (k* - k_min) / (k_max - k_min)
    |
    v
At inference: denormalize k -> fit GPD at predicted k -> POT VaR and ES formulas
```

### Results

**Synthetic data (VaR relative RMSE at p=0.99):**

| n     | VaR Rel. RMSE | ES Rel. RMSE |
|-------|---------------|--------------|
| 1,000 | 18.9%         | 117.1%       |
| 2,000 | 16.2%         | 116.8%       |
| 5,000 | 13.9%         | 70.0%        |

Performance improves with sample size as expected. Student-t is hardest (~25%)
because its tails are lighter than what GPD naturally fits. Pareto and
mixtures are easiest (~5-11%).

**Real financial data (VaR backtesting):** All methods (CNN, baseline k*,
fixed sqrt(n), historical simulation) produce similar violation rates around
1.5% and fail both Kupiec and Christoffersen tests, confirming that
unconditional POT cannot handle volatility clustering.

### Identified Limitations

1. **Proxy-loss misalignment** -- The CNN optimizes agreement with baseline k*
   (a proxy), not VaR accuracy (the actual target). The best k for the scoring
   rule is not necessarily the best k for VaR.
2. **ES sensitivity** -- ES depends on the entire tail shape beyond the
   quantile. Small errors in xi and beta get amplified, especially for the
   two-Pareto distribution (123-191% ES RMSE) where a regime change makes the
   tail especially hard to capture.

---

## 3. Pathway M: Differentiable POT (End-to-End)

### Motivation

Pathway A trains the CNN on a *surrogate objective* (match baseline k*) and
hopes that good k-agreement implies good VaR. Pathway M removes this
indirection: make the entire threshold-selection -> GPD fitting -> VaR/ES
computation chain differentiable, and train end-to-end on the actual target
(pinball loss on VaR, direct ES loss).

### Architecture

```
Same 1D CNN backbone as Pathway A
    |
    v
Sigmoid head -> k_tilde in (0, 1) -> denormalize to k_cont in [k_min, k_max]
    |
    v
Soft threshold mask: weights_i = sigmoid(temperature * (k_cont - i))
    Monotonically decreasing; high temperature -> hard cutoff.
    Fully differentiable.
    |
    v
Differentiable GPD estimation via Probability Weighted Moments (PWM):
    - Soft threshold u via Gaussian kernel around k_cont
    - Weighted exceedances: e_i = relu(x_i - u)
    - M_0 = sum(w_i * e_i) / sum(w_i)
    - M_1 = sum(w_i * e_i * F_i) / sum(w_i)  where F_i = empirical CDF rank
    - xi = (2 - M_0/(M_0 - 2*M_1)).clamp(-0.5, 0.95)
    - sigma = softplus(2 * M_0 * M_1 / (M_0 - 2*M_1))
    |
    v
Differentiable VaR: Q(p) = u + (sigma/xi) * ((n/k * (1-p))^(-xi) - 1)
Differentiable ES:  ES(p) = (VaR + sigma - xi*u) / (1 - xi)
    |
    v
Multi-component loss:
    - Pinball loss at multiple quantile levels (0.95, 0.975, 0.99)
    - Direct ES loss (smooth L1 on ratio es_hat/es_true)
    - Optional: auxiliary k-agreement loss (MSE on k_tilde vs baseline k*)
```

### Key Design Decisions

**PWM over MLE for GPD estimation.** Maximum likelihood estimation involves an
iterative optimizer (scipy.optimize), which is not naturally differentiable.
Probability Weighted Moments (Hosking & Wallis, 1987) give closed-form
estimators for xi and sigma, making the GPD parameter layer a simple
computation graph that PyTorch can differentiate through.

**Double precision internally.** The PWM formulas and the POT quantile formula
involve ratios of small numbers and power functions with potentially large
exponents. Float32 caused numerical instability (NaN gradients). All GPD
computations run in float64, with the result cast back to float32 for the CNN.

**Multi-quantile pinball loss.** A single pinball loss at tau=0.99 provides
very sparse gradient signal (only 1% of data contributes to the
under-prediction penalty). Evaluating the VaR formula at multiple quantile
levels (0.95, 0.975, 0.99) using the same GPD parameters gives denser
gradients.

**Sigmoid soft-threshold mask** instead of the originally planned `torchsort`
differentiable sorting. The data is already sorted descending (pre-processing
step), so we only need to differentiably select how many observations to
include. A sigmoid mask centered at k_cont achieves this simply and stably.

---

## 4. What Failed in Pathway M

### The Fundamental Gradient Bottleneck

The differentiable chain CNN -> k_tilde -> soft mask -> PWM -> GPD params ->
VaR/ES is **too long and too nonlinear** for effective gradient flow. Despite
the entire pipeline being technically differentiable, the CNN backbone never
learned to produce VaR/ES estimates meaningfully better than its random
initialization.

**Observed behavior:** Training loss would decrease slowly, but the CNN's
k_tilde output remained essentially flat (near 0.5 for all inputs). The
downstream GPD/VaR computation adapted through its other inputs (the actual
data), but the CNN's contribution was negligible.

**Why:** Consider the gradient path. The pinball loss gradient w.r.t. VaR is
simple. But VaR depends on xi, sigma, u, and k_eff, each of which depends on
the soft mask weights, which depend on k_cont, which depends on k_tilde from
the CNN. Each link in this chain introduces nonlinearity (sigmoid, softmax,
power functions, division), and the gradient must traverse all of them. By the
time it reaches the CNN parameters, it is either vanishingly small or
dominated by noise.

This is analogous to the vanishing gradient problem in early deep networks,
but worse: the intermediate layers are not learned parameters but fixed
mathematical formulas (GPD equations) with sharp nonlinearities.

### The k-Auxiliary Loss Experiment

**Hypothesis:** If the gradient path from VaR to the CNN is too noisy, maybe
we can provide a direct supervisory signal on k_tilde to "guide" the CNN,
then let the VaR loss fine-tune from there.

**Implementation:** Added an auxiliary MSE loss term on k_tilde vs the
Pathway A baseline k*, with configurable weight (`k_aux_weight`). Annealed
the auxiliary weight from full strength to zero over training, so the model
would start by learning baseline k* and gradually shift to VaR-optimal k.

**Result:** This actually made things *worse*. The auxiliary loss pulled k
toward the scoring-function-optimal k*, which conflicts with the VaR/ES loss
that wants a different k. The two objectives fought each other, and the model
converged to a compromise that was worse than either pure objective. The
auxiliary weight was set to 0 in the final configuration.

### Temperature Annealing Experiment

**Hypothesis:** Start with a low temperature (soft mask, smooth gradients) and
gradually increase it (sharper mask, closer to hard threshold) as the model
learns.

**Implementation:** Temperature annealed from 5.0 to 20.0 over the first
half of training.

**Result:** Created a **train/val mismatch**. Validation always used the
final temperature (20.0) for consistency, but training used varying
temperatures. The model learned to exploit the softer training mask in ways
that did not transfer to the harder validation mask. Disabling annealing
(using fixed temperature=20.0 throughout) gave more consistent results. The
final configuration uses `temperature_anneal: false`.

### ES Estimation Instability

**Problem:** ES estimates would occasionally blow up to extremely large values
(10^6 or more), caused by xi approaching 1.0 where the ES formula has a
singularity: ES = (VaR + sigma - xi*u) / (1 - xi).

**Fixes applied (these worked):**
1. **xi clamp to 0.95** -- Clamped the GPD shape parameter to [-0.5, 0.95]
   in both the differentiable PWM estimator and the scipy-based Pathway A
   evaluation. This keeps the denominator (1 - xi) >= 0.05.
2. **ES denominator guard** -- Added `max(1 - xi, 0.05)` in the ES formula
   as a safety net, both in the differentiable `diff_pot.py` and in the
   evaluation `evaluate.py`.

These two fixes dramatically improved ES estimation for **both pathways** --
a genuine positive finding from the Pathway M investigation.

---

## 5. What Worked

### From Pathway A

- **The CNN reliably learns baseline k*.** Agreement rates are ~90% within
  radius 10, and the model generalizes well across distribution families.
- **Anderson-Darling GOF** is a robust diagnostic. Tail-weighted testing
  catches GPD misfit that Kolmogorov-Smirnov would miss.
- **Mean excess linearity** as a fourth diagnostic channel provides a
  complementary signal to AD (tests the GPD's theoretical mean excess
  property rather than distributional fit).
- **Real-data backtesting** infrastructure correctly identifies the
  unconditional POT limitation (volatility clustering).

### From Pathway M

- **The differentiable POT architecture is sound.** Soft masking, PWM-based
  GPD estimation, and the differentiable VaR/ES formulas are all numerically
  stable (with the double-precision and xi-clamp fixes). The unit tests
  confirm that the differentiable VaR matches the analytical formula, ES > VaR,
  and gradients flow through the full computation graph.
- **ES stability fixes** (xi clamp + denominator guard) improved both
  pathways. Before these fixes, ES relative RMSE was much worse.
- **Multi-quantile pinball loss** is a sound idea for dense gradient signal,
  even though the overall training failed.
- **The Fissler-Ziegel FZ0 loss** (joint scoring function for VaR and ES) is
  theoretically principled and was implemented correctly, but could not
  overcome the gradient bottleneck.

### Key Insight

The Pathway M experiment demonstrates that **not all differentiable pipelines
benefit from end-to-end training**. When the computation graph between the
learned parameters and the loss function is long, highly nonlinear, and
involves sharp mathematical operations (power functions, division by small
numbers), gradient-based optimization struggles even though gradients
technically exist.

This is a useful negative result: it rules out the naive end-to-end approach
and points toward either (a) shorter gradient paths (e.g., predicting GPD
parameters directly rather than going through threshold selection), or (b)
non-gradient methods for threshold optimization (e.g., reinforcement learning,
evolutionary strategies).

---

## 6. Final Configuration

The code is preserved in its final state with all experiments documented.
The default configuration (`config/default.yaml`) reflects the best-known
settings:

```yaml
diff_pot:
  temperature_start: 5.0
  temperature_end: 20.0
  temperature_anneal: false     # annealing caused train/val mismatch
  temperature_u: 50.0
  quantiles: [0.95, 0.975, 0.99]
  quantile_weights: [0.2, 0.3, 0.5]
  pinball_weight: 0.7
  es_weight: 0.3
  k_aux_weight: 0.0            # auxiliary k loss hurts results; disabled
  lr: 0.0005
  batch_size: 32
  max_epochs: 300
  patience: 30
  xi_clamp: [-0.5, 0.95]       # prevents ES singularity at xi=1
```

### What is kept and why

| Component | Status | Rationale |
|-----------|--------|-----------|
| `src/diff_pot.py` | Kept | Documents the differentiable architecture (positive contribution) |
| `src/train_diff.py` | Kept | Documents loss functions and training loop (tells the story) |
| `run_diff_pipeline.py` | Kept | Runnable comparison of Pathway M vs A on the same test set |
| `tests/test_diff_pot.py` | Kept | 28 tests verifying numerical correctness of all components |
| `k_aux_weight` | Kept (disabled) | Documents the auxiliary loss experiment |
| `es_direct_loss`, `fz0_loss` | Kept | Both are theoretically sound; failure was in gradient flow, not loss design |
| Temperature annealing code | Kept (disabled) | Documents the experiment; easily re-enabled for future work |

---

## 7. File Map

```
Pathway A (baseline):
  src/synthetic.py         -- Data generation (4 heavy-tailed families)
  src/pot.py               -- GPD fitting, diagnostics, baseline k* scoring
  src/features.py          -- 4-channel feature matrix for CNN
  src/model.py             -- ThresholdCNN (conv -> pool -> linear)
  src/train.py             -- Training loop with early stopping
  src/evaluate.py          -- Agreement, VaR/ES RMSE, plots
  run_pipeline.py          -- Orchestration (Steps 1-7)

Pathway M (differentiable):
  src/diff_pot.py          -- Soft mask, PWM GPD, differentiable VaR/ES
  src/train_diff.py        -- Pinball, FZ0, ES direct, combined loss, training loop
  run_diff_pipeline.py     -- Orchestration + Pathway A comparison

Real data (Step 8):
  src/realdata.py          -- yfinance loading, rolling windows
  src/evaluate_real.py     -- VaR backtesting, Kupiec, Christoffersen, McNeil-Frey
  run_real_pipeline.py     -- Orchestration

Config & docs:
  config/default.yaml      -- All hyperparameters
  Pipeline.md              -- Step-by-step code-to-PDF mapping
  APPROACH.md              -- This document
```

---

## 8. Lessons and Future Directions

1. **Shorter gradient paths.** Instead of CNN -> k -> mask -> PWM -> GPD -> VaR,
   predict GPD parameters (xi, sigma) directly from the feature matrix. This
   skips the threshold-selection bottleneck entirely. The differentiable VaR/ES
   formulas from Pathway M can still be used for end-to-end training.

2. **Conditional models for real data.** The unconditional rolling-window
   approach fails backtesting because it ignores volatility dynamics. A
   GARCH-filtered residuals + POT combination would likely improve results.
   Pathway D (Temporal Transformer for non-stationary POT) was proposed but
   not implemented.

3. **Hybrid approaches.** Use Pathway A's reliable k-prediction as
   initialization, then fine-tune the GPD parameters directly with a VaR loss
   over a few gradient steps. This avoids the long gradient path while still
   benefiting from end-to-end alignment.

4. **Non-gradient optimization.** For the threshold selection sub-problem,
   reinforcement learning or evolutionary strategies could optimize VaR/ES
   directly without requiring differentiability of the GPD fitting step.
