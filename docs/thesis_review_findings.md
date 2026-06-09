# Thesis Review — Findings and Status

Scope: full review of `latex/main.tex` (104 pages, 6 chapters + 5 appendices),
cross-checked against `src/`, `config/`, `outputs/`, `references.bib`, and the
generated figures. Severity: **CRITICAL** · **MAJOR** · **MINOR**.

Status after three review rounds: compiles clean (`latexmk` exit 0), **0 undefined
citations/references**, 104 pages. Almost everything actionable is resolved; the
remaining items are provenance/cosmetic/optional and listed at the bottom.

---

## ✅ Resolved

### Round 1 — EVT terminology
- `\xi` is the "shape parameter / extreme-value index" throughout (not "tail index");
  Table 2.1 header → "Shape `\xi`"; tail index reserved for `\alpha=1/\xi`.
- Gumbel sentence (318): rewritten as "tails that decay faster than any power law …
  the lognormal, which is rapidly varying yet subexponential" (subexponentiality as a
  cross-cutting property, not a Gumbel sub-level).
- "high tail-index regime" → "high-`\xi`" everywhere incl. Appendix B chapter/section titles.
- Hill / moment uses → "extreme-value index" (code returns `\xi=1/\alpha`).

### Round 2 — first correctness pass
- **ES definition** → coherent average-VaR integral, with the CTE form as the
  continuous-case reduction (`eq:es_definition` / `eq:es_cte`).
- **Hill estimator** restricted to regularly varying / Fréchet (`\xi>0`) tails.
- **Parameter count** "≈1.0 M" → "≈0.75 M (747 k)" + caption notes standard PyTorch count.
- **Anderson–Darling** formula re-symboled `n`→`k` (exceedance count).
- **Augmentation**: verified 36,720 effective (1 deletion + 1 bootstrap copy / sample);
  table fixed to "36,720 (12,240 originals, two copies each)", prose "triples" kept.
- **Profit caption** "1,572" → "1,593 profit-tail windows".
- **Gnedenko (1943)** and **Fissler & Ziegel (2016)** added to `references.bib` and cited;
  "asymmetric Laplace … member of the joint scoring functions" reworded.
- **`\sigma` / convolution**: split conv (`eq:cnn_convolution`) from activation
  (`eq:cnn_activation`); after discussion kept `\sigma` for the activation in §2.7/§2.8
  with a localized clarifying sentence.
- **`\xi<0` wording**, **directional-bias claim** ("inflates … biases" → "can distort"),
  **informal ES sentence** — all softened/aligned.

### Round 3 — reviewer feedback (this round, all verified against code/figure/table)
- **Point 1 (CNN contribution).** Reframed in §5.5 (`\subsection*` heading replacing the
  inline `\textbf`), the Conclusion, with the Management Summary left as-is. Verified the
  CNN input *is* the full `L×7` diagnostic grid, so the grid is computed regardless and the
  forward pass is no faster than the scorer's argmin. The "faster/scalable/automated" claim
  is kept but anchored to the **manual-expert** baseline (what makes an 8,147-window study
  feasible); the "beyond the scorer" gain is reframed as **determinism + differentiability /
  adaptability**, not speed. Dropped "substantially faster at inference / makes the backtest
  tractable" as a CNN-vs-scorer claim.
- **Point 2 ("mean-unbiased").** All 3 occurrences (1350, 1586, 1970) → "non-parametric and
  consistent under standard regularity conditions" / "bounded by the observed data".
- **Point 3 (Student-$t$).** Figure was **also wrong** — Student-$t$ ν=4 was plotted in the
  "Gumbel/light-tailed" panel. Regenerated `synthetic_log_survival.png` via
  `make_methodology_figures.py` (moved Student-$t$ to the Fréchet panel; retitled panels
  "Gumbel MDA ($\xi=0$, rapidly varying)" / "Fréchet MDA ($\xi>0$, power-law)"); text (754)
  regrouped to Fréchet-domain (incl. Student-$t$) vs Gumbel-domain.
- **Point 4 (notation).** Unified to a positive tail variable `X_t` (`=-r` loss, `r` profit,
  `|r|` abs; verified against `evaluate_real.py`). Walk-forward `c_t` eq, MLP target, prose
  (1272, 1614), and the TikZ violation node now use `X`. MF *residual* renamed `r_i → e_i`.
- **Point 5 (draw vs expectation).** All 3 (651, 1272, 1614) → "a single draw from the tail
  conditional distribution beyond VaR, whose mean is the conditional tail expectation …".
- **Point 6 (VaR "read off order statistics").** All 3 (1444, 1606, 1659) → "anchored at the
  threshold and extrapolated through the fitted GPD … without the `1/(1-\hat\xi)` amplification".
- **Point 7 (declustering).** Clarified (1005): 90th-pct identifies clusters; grid runs on
  the declustered sample; declustering is prior to and distinct from the POT threshold.
- **Point 8 (trading days).** First use (874) → "daily observations (trading days for equities
  and calendar days for cryptocurrencies)"; other window/horizon uses → "daily observations" / "days".
- **Point 9 (FANG+).** "the individual constituents" → "selected large technology constituents, namely …".
- **Point 11 (Gumbel heading).** §4.2 heading → "Gumbel MDA ($\xi=0$, rapidly varying tails)".
- **Point 12 (Student-$t$ exception).** Verified (table: VaR 26.24% > ES 15.6%). "tail mean
  bounded by finite kurtosis" → moderate shape `\hat\xi=1/\nu\le1/3` keeps ES amplification
  small while VaR is sensitive to the body-to-tail transition.

### Round 4 — alignment pass (RQ / results / conclusion / appendix), all verified against the output pkls

Reviewer raised eight points on the alignment between research question, results, conclusion,
and appendix wording. All eight verified against `outputs/real_results_{profit,loss}.pkl`,
`outputs/data/real_diagnostics_{profit,loss}.pkl`, and `outputs/tables/`. The profit-tail point
was the most serious and contained a **confirmed sign error the reviewer only half-caught**.

- **Point 5 (profit-tail sign error) — CONFIRMED, three linked errors fixed.** Verified from the
  pkls: profit CNN `overall_violation_rate = 0.01368` (1.37%), Kupiec p=0.0225, **rejects** →
  1.37% > 1% nominal is **under-coverage** (too many violations), not over-coverage. Loss CNN
  `overall_violation_rate = 0.01113` (1.11%), Kupiec p=0.487, **passes** → near-nominal, not
  over-covered. Shapes (8,147 windows, ξ̂ at baseline k*): profit median 0.5048 vs loss 0.5039;
  p95 0.831 vs 0.798; ξ̂>0.7 fires 13.77% vs 12.74% → profit tail is **marginally heavier**, not
  lighter. Three thesis locations corrected: (1672/now §Discussion "they over-cover" → "under-cover";
  appendix §2035 "rejection direction is over-coverage / over-covers the loss tail / lighter shape"
  → "under-coverage", loss near-nominal, shape marginally heavier, causal story (heavier shape ⇒
  larger VaR ⇒ fewer violations) noted as the opposite of observed; appendix §2072 "GPD shape
  distribution is lighter on the profit tail" → "marginally heavier", asymmetry reattributed to
  time dependence not shape.
- **Point 1 (RQ rewording).** Primary RQ (§Research Question) → "make threshold selection in the
  Peaks-Over-Threshold method more systematic, and under which conditions does this improve …".
  Conclusion restatement (§Answer to the RQ) mirrored. Sub-questions already consistent.
- **Point 2 (NYFANG constituents).** Results-chapter intro "eight tickers across NYFANG constituents
  and two cryptocurrencies" → "the NYSE FANG+ index, selected large technology stocks, and two
  cryptocurrencies". (Methodology §868 already fixed in Round 3.)
- **Point 3 (Student-t grouping).** "Light and moderate tails (Student-t, …)" → "Families with
  comparatively lower ES error, including …" (empirical, not theoretical — these are Fréchet/
  power-law). Verified table: the six named families are exactly the 15.6–25.9% ES-error band.
- **Point 3b (ξ̂ vs ξ).** "$\hat{\xi}=1/\nu\le1/3$" → "$\xi=1/\nu\le1/3$" (1/ν is the population
  value, not the estimate); "moderate shape" → "moderate theoretical shape".
- **Point 4 (appendix order-statistic formula).** §appendix:score_me_grid `e_i = X_{(i)} - X_{(k+1)}`
  → `e_i = X_{(n-k+i)} - X_{(n-k)}` to match the global ascending convention and §892/§888
  ($u_k = X_{(n-k)}$). Examiner-catchable; fixed.
- **Point 6 (absolute impossibility claims).** Two "No discriminator and no machine-learning
  correction can separate/resolves …" → "No discriminator based only on these finite-sample
  diagnostics … at the available sample sizes". "No threshold rule recovers a number that the data
  does not contain" → "cannot reliably recover tail information that is not represented in the window".
- **Point 7 ("a good threshold is enough").** → "without the additional $1/(1-\hat{\xi})$
  amplification, so accurate threshold selection appears sufficient for acceptable VaR coverage in
  this empirical setting".
- **Point 8 (absolute-tail p=0.77) — RESOLVED by regeneration, number is correct and now backed.**
  The abs-tail unconditional backtest was computed by `evaluate_real()` in `run_real_pipeline.py`
  (line 336) but never pickled (only the loss/profit sign-split tracks are saved). Added
  `analysis/regen_abs_backtest.py`, which faithfully reproduces the abs-tail path (same cached
  datasets/diagnostics, time-ordered split, transfer-learned CNN) and saves
  `outputs/real_results_abs.pkl`. Regenerated `historical_sim` McNeil–Frey **p = 0.7727 → 0.77**,
  confirming the hard-coded constant. The other abs-tail methods also back the Conclusion's
  "only method that passes on both tails": cnn/baseline reject at p≈0, fixed rejects at p=0.018,
  historical passes at p=0.77. **No text change needed; the provenance gap is closed.**

Post-edit: `latexmk` exit 0, 104 pages, 0 undefined references/citations.

### Round 5 — walk-forward correction now extracts features at the CNN k̂ (was baseline k\*)

`analysis/correction_net_real_walkforward.py` previously extracted the nine MLP
features at the baseline scorer threshold `k_star` with a comment claiming it
"coincides tightly" with the CNN `k_pred`. The thesis text (Sec. ES correction,
methodology) already said the features are extracted at $\hat{k}$, so prose and
code disagreed. Verified the "coincide tightly" claim is too strong: on the real
loss tail $\hat{k}$ and $k^*$ agree only 14.3% exactly (mean $|\hat{k}-k^*|\approx
3.8$ grid positions) — the same modest agreement as the synthetic set.

Fix (makes the prose literally true): patched `_features_for` to recompute
$\hat{k}$ from the transfer model (`compute_khat_lookup`, same denormalisation as
`run_real_pipeline.py`) and use it. Because the MLP weight init is stochastic
(`train_correction_net` seeds the data split but not the weights), the script now
averages the MLP over 10 seeds and stores the per-seed sweep; the scalar and
uncorrected methods are deterministic and use no features. Regenerated
`outputs/correction_walkforward_results.pkl` and the figure.

Impact (eval slice, 1,422 windows, 34 violations): uncorrected p=0.0397 and
scalar p=0.1057 reproduce the thesis exactly (deterministic, feature-independent).
MLP under $\hat{k}$: mean p=0.375 (range [0.074, 0.873], **passes 10/10**), t=+0.89,
mean ES=0.100 — replacing the old single-draw MLP row (p=0.170, t=+1.40, ES=0.098).
Conclusion unchanged: the scalar correction (the recommended method) is unaffected.
Thesis updated: `tab:walkforward_summary` MLP row, the p-value sentence (1618) with
a seed-averaging footnote, the figure caption (now shows the across-seed whisker),
and the per-ticker MLP values (AMZN 0.17→mean 0.32, ETH-USD 0.63→mean 0.57; scalar
per-ticker unchanged). Comparison harness kept at `analysis/walkforward_khat_compare.py`.

---

## Open items (optional / provenance / cosmetic)

- **[MINOR/clarity] ES-correction output mode (`sec:es_correction`).** Note that the deployed
  head is the softplus floor `[0.5,∞)` and the bounded sigmoid `[0.5,3.0]` alternative exists
  but is unused (so a reader hitting the class default isn't confused). Not yet applied.
- ~~**[MINOR/provenance] Absolute-tail McNeil–Frey `p=0.77`** is a hard-coded constant with no
  backing pickle.~~ **RESOLVED in Round 4:** regenerated via `analysis/regen_abs_backtest.py` →
  `outputs/real_results_abs.pkl`; `historical_sim` p = 0.7727 confirms the number.
- **[MINOR/provenance] Pre-fix four-term `k*` (118–131)** and the superseded semi-parametric
  block in Appendix B come from a non-archived run. Footnote that they are a logged earlier run.
- **[nice-to-have] Citations:** Davison & Smith (1990), Smith (1987), de Haan & Ferreira (2006).
- **[cosmetic] Overfull `\hbox`:** lognormal–Pareto list (~4 pt), composite-scoring TikZ (~86 pt),
  Appendix A file path (~124 pt), ES-validation longtable (~21 pt). And the `\sigma`/`tocloft`/
  `csquotes`/`todonotes`/`scrhack` package warnings (all harmless).
- **[cosmetic] Figure accent:** `synthetic_log_survival.png` legend/title shows "Fr\'echet"
  (matplotlib doesn't render the LaTeX accent). Pre-existing; fix by using a unicode "é" in
  `make_methodology_figures.py` if desired.

## Considered and intentionally NOT changed
- **CNN distills the scorer → "circularity / what does it add".** Already addressed in §5.5 and
  the Conclusion; not a gap. (Point 1 sharpens the wording but adds no new limitation.)
- No full Hill formula, ES derivation, or expanded FTG theorem — concise-correct exposition preferred.
