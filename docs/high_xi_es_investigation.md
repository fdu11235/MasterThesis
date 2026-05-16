# High Tail-Index ES Investigation

Investigation into why Expected Shortfall (ES) estimation is unreliable when
the estimated GPD shape xi-hat exceeds 0.7. Conducted 2026-05-16.

This document is the technical record. The thesis version is Appendix B
("Expected Shortfall Estimation in the High Tail-Index Regime") in
`latex/main.tex`.

Reproduce with:

```bash
python run_high_xi_experiment.py --config config/high_xi.yaml   # stress experiment
python run_xi_es_analysis.py --config config/default.yaml        # full-set residual bias
python investigate_two_pareto.py --config config/default.yaml    # composite tails
python run_pipeline.py --config config/default.yaml              # full synthetic eval
```

## 1. Motivation

A thesis reviewer flagged that the network-determined xi is problematic for
xi > 0.5, badly so above 0.7. In this pipeline xi is not a direct network
output. The CNN predicts the exceedance count k, a GPD is fitted by MLE at
that k, and ES follows from `src/evaluate.py:pot_es`.

To get clean evidence, Pareto data was generated at alpha in {1.2, 1.3, 1.4}
so xi = 1/alpha in {0.833, 0.769, 0.714}, at n in {1000, 2000, 5000}, 300
replications. Pareto is used because the true ES is exactly
`VaR * alpha/(alpha-1)`.

## 2. Diagnosis

The ES relative error `E = (ES_est - ES_true) / ES_true` was decomposed at
three thresholds: the CNN threshold, the baseline threshold, and an oracle
threshold. With the original trimmed-mean fallback, median results at
n = 1000:

| alpha | xi | E_cnn | E_irr (oracle) | E_sel (CNN selection) |
|---|---|---|---|---|
| 1.2 | 0.833 | -72% | -49% | -21% |
| 1.3 | 0.769 | -62% | -29% | -30% |
| 1.4 | 0.714 | -55% | -15% | -37% |

Findings:

1. The deployed pipeline underestimated ES by 55 to 72 percent on genuine
   heavy Pareto tails.
2. Two errors of opposite sign that do not cancel. The raw GPD closed form
   overestimates by about +82 percent (the GPD-MLE shape is upward biased).
   The trimmed-mean fallback then overcorrected into a large underestimate.
3. The amplification convexity of `1/(1-xi)` is a minor term (at most 15
   percent at the oracle threshold). It is not the main problem.

## 3. Attempt 1: Hill/Weissman semi-parametric estimator

The fallback was replaced by a Hill/Weissman semi-parametric ES:
`xi_H` = Hill index at k, `VaR_W = u (k/(n(1-p)))^xi_H`,
`ES = VaR_W / (1 - xi_H)`.

On the clean Pareto stress data this worked. Median E_cnn fell to -5.6,
-5.5, -2.8 percent at n = 1000.

But the full synthetic re-run across all 13 families showed a severe
regression. Aggregate ES relative RMSE rose from 35.4 percent (trimmed mean)
to 117.5 percent. The regression was concentrated in light-tailed families:
lognormal 381.8 percent, stretched Weibull 134.0 percent, log-gamma 111.2
percent. Splitting by branch confirmed it is entirely in the xi-hat>0.7
cases. For lognormal the xi-hat<=0.7 cases had 17.0 percent RMSE while the
xi-hat>0.7 cases had 924.9 percent.

## 4. Root cause: identifiability, and why a larger sample does not help

When the GPD xi-hat exceeds 0.7 on a lognormal sample, the Hill estimate is
also large (median 0.96, above 0.7 in 97.5 percent of those cases). Both
estimators are fooled. This is the lognormal versus power-law
identifiability problem. At n approx 1000 a lognormal tail and a power-law
tail with index near one produce statistically similar samples but require
very different ES. The Hill/Weissman estimator extrapolates a power law and
overshoots when the tail is not genuinely power-law.

No discriminator and no machine-learning model resolves this. Any such model
is a function of features computed from the sample, and the features do not
separate the regimes. It is an information limit.

A larger sample does not solve it. The Hill bias on a lognormal tail decays
only about as 1/log(n), so the misfire becomes rarer but does not vanish.
And the application cannot use a larger sample. Financial tail risk is
estimated on rolling windows that must be short enough to be one stationary
regime. A 1000-day window already spans about four years. A 5000-day window
spans about twenty years and mixes market regimes, and the crypto assets do
not have that much history. The sample size near 1000 is imposed by the
application.

## 5. Attempt 2: plain empirical tail mean

The Hill/Weissman estimator was replaced by the plain empirical tail mean,
the mean of the observations exceeding the estimated VaR, keeping all
exceedances. This removes the catastrophic light-tail blow-ups (lognormal
fell from 381.8 to 44.8 percent, stretched Weibull from 134.0 to 23.4
percent).

But the aggregate ES relative RMSE was 85.4 percent, still worse than the
35.4 percent of the original trimmed mean. The cause is variance. The
xi-hat>0.7 tail has only about 8 to 10 observations above the VaR. The mean
of so few points is dominated by a single large draw. A frechet sample with
an unusually large maximum gave an ES estimate of 51.4 against a true 20.0,
a +157 percent error. RMSE squares such cases, which pushed `frechet` to
128.0 percent and `lognormal_pareto_mix` to 295.3 percent.

The original fallback dropped the single largest exceedance. That step was
not a bias mistake. It is variance reduction, removing the dominant outlier
from a 10-point mean.

## 6. Decision and the deployed estimator

Summary of the options on the same synthetic set:

| xi-hat>0.7 estimator | aggregate ES RelRMSE |
|---|---|
| Trimmed empirical mean (drop largest) | ~35% |
| Plain empirical mean | 85% |
| Hill/Weissman | 117% |

With only about 10 tail observations there is no estimator that is both
unbiased and low variance. The trimmed mean has the lowest RMSE but it is
biased low, which underestimates a risk measure. The plain empirical mean is
mean-unbiased but noisy.

Decision: the deployed `pot_es` uses the plain empirical tail mean for
xi-hat>0.7. The estimator is mean-unbiased, which is preferred for a risk
measure over the lower-RMSE but downward-biased trimmed mean. Its limitation
is a high variance for individual windows, because the tail has only about
ten observations. For xi-hat<=0.7 the GPD closed form is unchanged.

The ES correction network is left disabled. It cannot help here for the same
identifiability reason, and on synthetic data it previously made the
aggregate worse (35 to 40 percent).

## 7. Composite tails (two_pareto)

`investigate_two_pareto.py` examined the two-regime Pareto family, a
Pareto-alpha1 bulk with the top 5 percent spliced to a heavier Pareto-alpha2.
The genuine heavy tail is only the top 5 percent, about 50 observations at
n = 1000. The CNN selects a threshold of about 124 to 144 observations, so
the estimation window straddles the splice and the tail index is estimated
from a mixture of the two regimes. At the oracle threshold the error is near
zero. This is a threshold-selection problem rather than an estimator
problem.

| alpha2 | true xi | E @ CNN k | E @ changepoint | E @ oracle k |
|---|---|---|---|---|
| 1.05 | 0.952 | -89.8% | -44.0% | -7.9% |
| 1.10 | 0.909 | -82.0% | -34.2% | -4.1% |
| 1.50 | 0.667 | -38.5% | -17.8% | -0.6% |

## 8. Outcome

The deployed estimator is the plain empirical tail mean for xi-hat>0.7 and
the GPD closed form for xi-hat<=0.7. The investigation is documented in the
thesis as Appendix B. The broader finding is that ES estimation for genuine
xi>0.7 tails at n approx 1000 has an irreducible difficulty. The closed form,
the Hill/Weissman estimator, and the empirical tail mean each fail in a
different way, and no method reliably distinguishes the cases.
