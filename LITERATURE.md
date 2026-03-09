# Literature Review: ML-Based & Automated Threshold Selection for POT/GPD

## 1. Classical References & Reviews

### Coles (2001) — An Introduction to Statistical Modeling of Extreme Values
- **Authors:** Stuart Coles
- **Venue:** Springer Series in Statistics
- **Link:** [Springer](https://link.springer.com/book/10.1007/978-1-4471-3675-0)
- **Summary:** The foundational textbook on extreme value theory. Develops the theoretical framework (GEV, GPD, point process models) and inferential techniques for threshold-based modelling, including graphical diagnostics like the mean residual life plot and threshold stability plots.

### Danielsson & de Vries (2000) — Value-at-Risk and Extreme Returns
- **Authors:** Jon Danielsson, Casper G. de Vries
- **Venue:** Annales d'Economie et de Statistique, No. 60, pp. 239-270
- **Link:** [PDF (EUR)](https://personal.eur.nl/cdevries/Articles/value_at_risk_and_extrene_returns.pdf)
- **Summary:** Proposes a semi-parametric VaR method where the largest risks are modelled parametrically via EVT while smaller risks use the empirical distribution. Shows that at low probability levels, standard methods strongly underpredict VaR whereas the semi-parametric EVT approach is most accurate.

### Scarrott & MacDonald (2012) — A Review of Extreme Value Threshold Estimation and Uncertainty Quantification
- **Authors:** Carl Scarrott, Anna MacDonald
- **Venue:** REVSTAT — Statistical Journal, 10(1), 33-60
- **Link:** [REVSTAT (PDF)](https://www.ine.pt/revstat/pdf/rs120102.pdf)
- **Summary:** Comprehensive review of threshold estimation methods for the GPD, covering graphical diagnostics, mixture models, and Bayesian approaches. Highlights the fundamental bias-variance trade-off: too low a threshold introduces model misspecification bias, too high a threshold wastes data and increases estimation uncertainty.

---

## 2. Classical Automated Methods

### Langousis et al. (2016) — Threshold Detection for the GPD: Review and Application to NOAA Data
- **Authors:** Andreas Langousis, Antonios Mamalakis, Massimo Puliga, Roberto Deidda
- **Venue:** Water Resources Research, 52(4), 2659-2681
- **Link:** [Wiley (WRR)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2015WR018502)
- **Summary:** Reviews representative GPD threshold detection methods (nonparametric change-point, graphical, GoF-based) and applies them to 1714+ daily rainfall records from the NOAA NCDC database. Finds that nonparametric methods are generally unreliable, while GP-asymptotic methods tend to yield unrealistically high threshold and shape parameter estimates.

### Northrop, Attalides & Jonathan (2017) — Cross-Validatory Extreme Value Threshold Selection
- **Authors:** Paul J. Northrop, Nicolas Attalides, Philip Jonathan
- **Venue:** Journal of the Royal Statistical Society: Series C, 66(1), 93-120
- **Link:** [Oxford Academic (JRSS-C)](https://academic.oup.com/jrsssc/article/66/1/93/7068122)
- **Summary:** Uses Bayesian leave-one-out cross-validation to compare predictive performance across candidate thresholds. Importance sampling reduces computation to only two posterior samples per threshold. Applied to ocean storm severity data. Implemented in the R package `threshr`.

### Bader, Yan & Zhang (2018) — Automated Threshold Selection via Ordered GoF Tests with FDR Control
- **Authors:** Brian Bader, Jun Yan, Xuebin Zhang
- **Venue:** Annals of Applied Statistics, 12(1), 310-329
- **Link:** [Project Euclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-12/issue-1/Automated-threshold-selection-for-extreme-value-analysis-via-ordered-goodness/10.1214/17-AOAS1092.full)
- **Summary:** Develops an efficient Anderson-Darling GoF test for GPD exceedances and automates threshold selection using a stopping rule that controls the false discovery rate (FDR) in ordered hypothesis testing. Fully automated — no subjective graphical judgment required.

### Silva Lomba & Fraga Alves (2020) — L-Moments for Automatic Threshold Selection in EVA
- **Authors:** Juliana Silva Lomba, M. Isabel Fraga Alves
- **Venue:** Stochastic Environmental Research and Risk Assessment, 34, 465-491
- **Link:** [Springer](https://link.springer.com/article/10.1007/s00477-020-01789-x)
- **Summary:** Proposes a truly automated, data-driven threshold selection method based on L-moments theory. Eliminates subjective judgment, is computationally efficient, and handles batch processing of large collections of extremes data. Good performance demonstrated on both simulations and wave height data.

### Murphy, Tawn & Varty (2024) — Automated Threshold Selection and Associated Inference Uncertainty
- **Authors:** Conor Murphy, Jonathan A. Tawn, Zak Varty
- **Venue:** Technometrics, 67, 215-224
- **Link:** [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/00401706.2024.2421744) | [arXiv](https://arxiv.org/abs/2310.17999)
- **Summary:** Develops a novel methodology that directly tackles the bias-variance trade-off in threshold selection and, crucially, propagates threshold uncertainty through to high quantile inference. Addresses the gap where most automated methods select a single threshold without accounting for the uncertainty of that choice.

---

## 3. ML / Neural Network Approaches

### Gnecco, Terefe & Engelke (2024) — Extremal Random Forests
- **Authors:** Nicola Gnecco, Edossa Merga Terefe, Sebastian Engelke
- **Venue:** Journal of the American Statistical Association, 119(548)
- **Link:** [Taylor & Francis (JASA)](https://www.tandfonline.com/doi/full/10.1080/01621459.2023.2300522) | [arXiv](https://arxiv.org/abs/2201.12865)
- **Summary:** Combines random forests with GPD extrapolation for extreme conditional quantile estimation. A quantile random forest provides local likelihood weights for GPD parameter estimation, with a penalized shape parameter for regularization. Consistency proven under general domain-of-attraction conditions. R package `erf` available.

### Allouche, Girard & Gobet (2024) — Estimation of Extreme Quantiles from Heavy-Tailed Distributions with Neural Networks
- **Authors:** Michael Allouche, Stephane Girard, Emmanuel Gobet
- **Venue:** Statistics and Computing, 34, 12
- **Link:** [Springer](https://link.springer.com/article/10.1007/s11222-023-10331-2) | [GitHub](https://github.com/michael-allouche/nn-quantile-extrapolation)
- **Summary:** Proposes new neural network parametrizations for extreme quantile estimation in heavy-tailed settings (conditional and unconditional). Features bias correction based on higher-order regular variation. Outperforms classical bias-reduced EVT estimators in difficult heavy-tailed scenarios. Applied to extreme rainfall prediction in southern France.

### Rai et al. (2024) — Fast Parameter Estimation of GEV Distribution Using Neural Networks
- **Authors:** Sweta Rai et al.
- **Venue:** Environmetrics, 35(3), e2845
- **Link:** [Wiley (Environmetrics)](https://onlinelibrary.wiley.com/doi/10.1002/env.2845) | [arXiv](https://arxiv.org/abs/2305.04341) | [GitHub](https://github.com/Sweta-AMS/GEV_NN)
- **Summary:** Proposes a likelihood-free neural network method for GEV parameter estimation. Achieves comparable accuracy to MLE but with significant computational speedup. The NN is trained on simulated GEV samples and directly outputs parameter estimates — useful for large-scale or real-time applications.

### Pasche & Engelke (2024) — Neural Networks for Extreme Quantile Regression (EQRN)
- **Authors:** Olivier C. Pasche, Sebastian Engelke
- **Venue:** Annals of Applied Statistics, 18(4), 2818-2839
- **Link:** [Project Euclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-18/issue-4/Neural-networks-for-extreme-quantile-regression-with-an-application-to/10.1214/24-AOAS1907.short) | [R package](https://opasche.github.io/EQRN/) | [GitHub](https://github.com/opasche/EQRN_Results)
- **Summary:** EQRN combines neural networks with EVT to extrapolate conditional risk measures (quantiles, exceedance probabilities) beyond the range of training data. A recurrent variant captures temporal dependence. Applied to flood risk forecasting in the Swiss Aare catchment using spatial-temporal covariates.

### Iroko, Tukur & Adeyanju (2025) — Threshold Selection for POT Models Using Logistic Regression
- **Authors:** T. C. Iroko, I. Tukur, V. Adeyanju
- **Venue:** Earthline Journal of Mathematical Sciences, 15(6), 1021-1036
- **Link:** [Earthline Publishers](https://earthlinepublishers.com/index.php/ejms/article/view/1093)
- **Summary:** Proposes logistic regression as an ML-assisted threshold selection strategy for POT models. Unlike graphical methods (MRL plot) that require subjective visual interpretation, the logistic regression approach offers transparent, reproducible threshold choices with easily interpretable coefficients — relevant for actuarial applications.

---

## 4. EVT for Anomaly Detection

### Siffer et al. (2017) — SPOT/DSPOT: Anomaly Detection in Streams with EVT
- **Authors:** Alban Siffer, Pierre-Alain Fouque, Alexandre Termier, Christine Largouet
- **Venue:** KDD 2017 (ACM SIGKDD)
- **Link:** [ACM Digital Library](https://dl.acm.org/doi/10.1145/3097983.3098144) | [HAL](https://hal.science/hal-01640325)
- **Summary:** Introduces SPOT and DSPOT algorithms for automatic anomaly detection in streaming data using POT/GPD. Requires no hand-set thresholds and makes no distributional assumptions — only the risk level (false positive rate) is specified. DSPOT extends SPOT to handle concept drift. Widely cited as a key bridge between EVT and applied ML.

### Spilak & Hardle (2021) — Tail-Risk Protection: Machine Learning Meets Modern Econometrics
- **Authors:** Bruno Spilak, Wolfgang Karl Hardle
- **Venue:** Encyclopedia of Finance (Springer); also available as IRTG 1792 Discussion Paper
- **Link:** [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3714632) | [arXiv](https://arxiv.org/abs/2010.03315) | [Springer](https://link.springer.com/10.1007/978-3-030-91231-4_94)
- **Summary:** Proposes a dynamic tail risk protection strategy using VaR-based exceedance classification. Compares GARCH, MLP, and LSTM neural networks as weak classifiers for predicting extreme losses. An ensemble meta-strategy improves both generalization and trading performance. Demonstrates the practical value of combining EVT with modern ML in financial risk management.

---

## 5. Relevance to Thesis

| Category | Relevance |
|----------|-----------|
| **Classical references** | Coles (2001) and Danielsson & de Vries (2000) provide the theoretical foundation for POT/GPD modelling. Scarrott & MacDonald (2012) frames the threshold selection problem that the thesis aims to solve. |
| **Classical automated methods** | These papers (Bader et al., Northrop et al., Murphy et al., Silva Lomba & Fraga Alves, Langousis et al.) represent the statistical state-of-the-art in automated threshold selection — the benchmarks against which an ML-based approach should be compared. Murphy et al. (2024) is particularly relevant for uncertainty propagation. |
| **ML/NN approaches** | Directly relevant prior work. Rai et al. and Allouche et al. show that neural networks can estimate EVT parameters and extreme quantiles effectively. Pasche & Engelke (EQRN) and Gnecco et al. (ERF) demonstrate how ML models can be combined with GPD extrapolation in a principled way. Iroko et al. (2025) is the closest to the thesis topic — using ML specifically for POT threshold selection. |
| **EVT + anomaly detection** | SPOT/DSPOT (Siffer et al.) shows that POT can be made fully automatic in streaming settings, providing practical motivation. Spilak & Hardle demonstrates the combination of EVT with neural networks in finance, a key application domain for the thesis. |
