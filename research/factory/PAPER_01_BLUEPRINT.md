# Paper 01 blueprint — From Unit Labels to Causal Calculators

**Status:** research design / manuscript seed, **not** a registered protocol, submitted preprint, or new empirical result.
**Proposed title:** *When Does a Unit Label Become a Calculation? Recovering and Rewriting Quantity Semantics in Tiny Transformers*
**Repository foundation:** NRI docs 14–19 on `main`; unit-calculus v2 design and exploratory feasibility probe in [PR #11](https://github.com/Mnehmos/mnehmos.llm.playground/pull/11) (doc 20, not yet a registration).
**Scope:** one falsifiable paper, not one paper per interesting heatmap. The looped-transformer experiments remain a **separate** paper candidate; they are not required for Paper 01.

## Single-sentence claim we would be testing

*Can the operational meaning of a learned unit label be reconstructed from network activations alone and then causally changed, in a way that predicts behavior on novel combinations?*

This is more specific than "models learn units" and more interesting than reproducing known Fourier circuits. A quantity can be behaviorally equivalent across unit expressions *without* sharing a unit-invariant hidden state. We will distinguish (i) canonicalization, (ii) label-conditioned composition / algebraic routing, (iii) memorized lookup, and (iv) no robust identifiable mechanism.

## Existing evidence vs unknowns

- **Existing, exploratory:** doc 20 reports about 0.94 multiplicative-character concentration for the modular product and about 0.92 for the final answer; labels remain strongly linearly recoverable. These are measured on **12 previously trained undeclared networks** with only three units per dimension. The exploratory spectra **do not** prove a causal log-space circuit or successful scale-table reconstruction.
- **Existing, preregistered:** doc 19 found installed unit-aware edits of about 0.90 intended for mass and 0.87 for acceleration on editor-reserved questions. These questions were **not withheld from model training**, and the within-weights re-fit was large (35–59% relative norms across affected matrices). It does not establish minimal edits or novel-combination generalization.
- **Unknown:** whether labels implement measurable log-scale rotations; whether independently recovered scales compose; whether a minimal weight edit changes precisely that semantic factor; and whether this holds on unseen label pairings. Those unknowns are the paper.

## Three headline hypotheses (prospective; thresholds to freeze after pilot)

**P1 / MEANING:** Recover each label's scale/sign from hidden states **without exposing the true scale table**. Report circular error of the inferred discrete-log shift, exact recovery rate, equal-ratio consistency, and alignment to a permutation null. Compare natural, declared, meaningless-label and untrained arms.

**P2 / COMPOSITION:** Infer label meanings from permissible calibration data, then predict answers and hidden-state changes for **entirely withheld label-pair combinations**. Compare recovered-map prediction against memorization, raw token-classification, pooled table, and an oracle using the true scale table. Report generalization accuracy per mode and label family, alongside confidence intervals over independent network seeds.

**P3 / CAUSALITY:** Use recovered scale information to precommit to **label-only** counterfactual interventions: activation rotation and permanent embedding-row edit with bounded relative norm. Score intended / wrong-variable / unchanged / other outcomes and preservation of unrelated cases; compare exact real-activation patch ceiling, random/norm-matched edits, shuffled maps, and full layer re-fit (separately, because it changes much more of the network).

**No circular validation:** do not use the true unit table to pick a recovery direction or an edit hyperparameter, or use the held-out test split to search sites and then claim independent confirmation.

## Study population and design

**Task:** modular arithmetic mod 13 in a symbolic unit-calculus v2 setting (six labels per dimension, selected equal-ratio pairs, inverse labels, product and quotient modes). This is *not* an empirical demonstration that models understand real SI units: modular numerical relations and real conversion tests are separate follow-ups.

**Baseline architecture:** repository's 2-attention-layer + MLP, `d=64` (TinyTyped), with vocabulary changes proposed in PR #11.

**Arms:** (A) natural, (B) declared/interchange-trained, (C) meaningless labels (every scale = 1), (D) untrained weights. Main comparison unit is independently trained network, not input row. Target 12 fixed seeds/arm, unless pilot and preregistration select different powered sample sizes. **Failed or non-converged runs remain in all analyses.**

**Split policy:** three distinct and recorded boundaries:
1. **Training vs actual OOD:** disjoint (label1,label2) combinations never used in model training; ensure all individual labels and underlying values remain represented in training.
2. **Fit vs evaluation:** no question used to fit a probe, recovery angle, location search, or edit is in confirmatory evaluation.
3. **Pilot vs confirmatory:** pilot/checkpoint reuse permits method development; future seeds and reserved rows are untouched until frozen design.

**Design weakness to monitor:** with small modular groups, the finite task has symmetries and non-identifiable phase origins. Fix a reference convention *without looking at the hidden scale table*; score recovered relations invariant under permissible global phase gauge. Zero must be treated separately in discrete-log analyses; harmonic bases must be orthogonal on the actual support, with nondegenerate sign frequency. Negative controls must match probe capacity, sample size and site selection.

**Statistics:** exact paired seed-level tests where pairs exist; otherwise per-seed effect sizes with 95% CIs. One **small prespecified family** for P1–P3 (Holm correction); remaining analyses exploratory. A paper that finds a null or a negative result is still complete.

## The first 12 experiment cards: screen fast, then confirm

| ID | Experiment | Minimum readout | Tier | Status |
|---|---|---|---|---|
| P01-01 | Character-spectrum reanalysis with orthonormal, comparable bases | permutation-adjusted multiplicative-minus-additive power, per seed | 0 | existing checkpoints; exploratory |
| P01-02 | Leave-one-unit-out linear probe | chance-adjusted scale recoverability | 0 | exploratory |
| P01-03 | Equal-ratio pairs, matched value distributions | recovered angle difference vs true ratio group | 0/1 | may require v2 data |
| P01-04 | Raw-route vs canonical-quantity variance matched for category complexity | cross-validated partial R² | 0 | existing checkpoints |
| P01-05 | Label-only counterfactual patch, same value | intended/wrong/preserve fractions | 0 | existing checkpoints |
| P01-06 | Phase-angle table recovery blinded to true scale | circular error; exact scale proportion | 1 | v2 primary |
| P01-07 | Product vs quotient sign reversal | signed angle agreement | 1 | v2 primary |
| P01-08 | Inverse labels vs ordinary labels | inverse-angle error | 1 | v2 primary |
| P01-09 | Held-out label pair calculator | trained-vs-held accuracy, accuracy predicted from recovered map | 1 | v2 primary |
| P01-10 | Embedding-row redefinition of one unit, map-guided | intended/wrong/preserve, edit rank/norm | 1 | v2 primary |
| P01-11 | Shuffled-map and norm-matched random interventions | specificity difference vs causal map | 1 | v2 primary |
| P01-12 | Copy-ceiling and network-failure audit | edit-success / copy-ceiling ratio on all seeds | 0/1 | mandatory control |

**Tier 0:** inference/analysis on checkpoints already stored; do not claim these exploratory investigations were preregistered.
**Tier 1:** training or validation within v2 study; doc 20's approximate 8 CPU-minutes per network and 6.5 CPU-hours for 48 networks are *planning estimates*, not measured guarantees.

## Preprint figures (must be auto-generated from immutable data)

1. **Research diagram:** value + label → recovered scale → calculator, with the three competing representation mechanisms.
2. **Spectral sanity:** within-network orthogonal additive/multiplicative spectra vs permutation controls, all seeds shown.
3. **Where is a label's meaning?** layer × role heatmap of held-out recovery score, with causal localization overlay, not probe-only claims.
4. **Recovered unit map:** predicted vs true scale rotations (after fixed gauge alignment), errors by label and seed.
5. **Held-out composition:** seen vs unseen label-pair accuracy by operator mode; predictions from recovered table vs controls.
6. **Edit reliability and price:** intended and collateral vs relative weight-change norm; copy ceiling explicitly displayed.
7. **Failure modes:** nonconverged seeds, zero values, ambiguous phases, interventions that change the wrong variable.

## Evidence gates and pivot options

- **Gate A, validity:** positive-control declaration should yield detectable localized variables; permutation/meaningless controls should not produce spurious scale tables. If not, fix the measurement before hypothesis claims.
- **Gate B, identification:** if angles fail to recover true relations under the fixed gauge, reject P1; do **not** relabel correlations as "internal scale meanings."
- **Gate C, generalization:** if recovering a table predicts seen pairs only, reject P2's compositional claim. Report lookup as the discovered limitation.
- **Gate D, edits:** if interventions fail to predict intended outcomes or have large collateral effects, reject P3 even if P1/P2 succeeded.
- **Stop rule:** no manuscript claiming a mechanism before an independent confirmatory test and causal falsification. A bounded negative result can be reported as a workshop note with explicit limitations.

## Paper outline

1. Introduction: numerical equivalence does not imply representation identity.
2. Related work: modular Fourier/grokking, concept emergence, causal localization vs editing, recurrent loops as later extension.
3. Dataset, architecture, split construction and instruments.
4. Table-free semantic recovery (P1).
5. Compositional prediction on withheld unit pairs (P2).
6. Causal redefinition and specificity (P3).
7. Alternative mechanisms and failures.
8. Limitations (toy modular group, few seed populations, synthetic labels, identification gauge, no scaling claim).
9. Reproducibility (single-command regenerate, hashes, full seed table, environment, executor).
10. Conclusion conditional on results.

## Relationship to the multi-paper program

- **Paper 01:** causal recovery of unit meaning, grounded in doc 20 / PR #11.
- **Paper 02:** loops and latent iteration, training step × virtual depth, linked to [issues #12–18](https://github.com/Mnehmos/mnehmos.llm.playground/issues/12).
- **Paper 03:** smallest reliable within-weight edits and edit composition.
- **Paper 04:** ANLU type supervision, unit equivariance and generalization.
- **Paper 05:** machine-checkable NRI explanations and blinded interpreter benchmark.
- **Paper 06:** research automation itself as an auditable negative-result-friendly factory.

Papers 02–06 are **candidates**; publishing multiple preprints is conditional on distinct validated contributions, not splitting one study into tiny overlapping papers.

## Literature / novelty boundary

Do not claim Fourier circuits, grokking phases, recurrent computation, or automated research agents as inventions here. Compare against:
- Nanda et al., *Progress measures for grokking via mechanistic interpretability* (2023), https://arxiv.org/abs/2301.05217
- *Inside the Loop* (2026), https://arxiv.org/abs/2610.24135
- *Loop, Think, & Generalize* (2026), https://arxiv.org/abs/2604.07822
- Hase et al., *Does Localization Inform Editing?* (2023), https://research.google/pubs/does-localization-inform-editing-surprising-differences-in-causality-based-localization-vs-knowledge-editing-in-language-models/

The candidate increment is **blind recovery of unit-meaning operations + unseen algebraic composition + causal rewriting within the same controlled experiment**, if the results support all three. Verify an updated literature search before any novelty claim in a submission.
