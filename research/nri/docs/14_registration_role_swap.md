# 14. Registration: does declaring a variable make *that* variable easier to edit? (role swap)

*This document, together with `run_swap.py` in the same commit, is the registration. Both were
committed **before** the main run. Earlier registrations and results (docs 11–13, PRs #3–#4) are
preserved unchanged.*

## 14.1 Question and motivation

PR #4 found, post hoc, that declaring **mass** with interchange intervention training made the
mass inversion edit more variable-specific than the acceleration edit. That comparison confounds
two things: *which variable was declared*, and *which variable it was*. Mass might simply be
easier to edit in this setup. This study **reverses the roles**. If declaration deserves the
credit, the advantage must switch sides when acceleration is declared instead.

## 14.2 Groups

These use the same dimensional task as docs 11–12, with units and wording unchanged.

| group | interchange training |
|---|---|
| `mass_only` | declares canonical mass M in block dims 0–15 |
| `acc_only` | declares canonical acceleration A in the **same** block dims 0–15 |
| `neither` | none |

**Identical across groups:**
- architecture: 69.8K parameters, 2-layer attention + MLP, d = 64;
- block size and location (16 dims, 0–15);
- interchange weight (λ = 3.0, with the schedule in §14.4);
- training budget: 6,000 steps, batch 512, AdamW, lr 2e-3, wd 0.01, OneCycle;
- evaluation (untyped inference).

**Matched within each seed:** the initialization (same torch seed), and the base training batches
(a dedicated generator, `10_000 + seed`). The interchange source batches come from a separate
generator (`20_000 + seed`), so they never shift the base data stream.

## 14.3 Seeds and compute

| use | seeds | note |
|---|---|---|
| pilot (schedule choice only) | 200–204 | scratch zoo `runs/zoo_swap_pilot`; never analysed for edits |
| **main** | **300–311 (12 seeds per group, 36 networks)** | none used anywhere before, including PRs #3–#4 and all pilots |

**Compute:** about 8 CPU-minutes per declared network and 4 per `neither` network, so about
1.2 CPU-hours of training. Analysis is one pass.

**Restart rule:** **no restarts.** Every scheduled network counts in the headline analysis,
converged or not.

**Convergence threshold:** ID accuracy ≥ 0.95. It is used only for the secondary
converged-seeds table.

## 14.4 Interchange-weight schedule (chosen by pilot, then locked)

The pilot compared a **constant** λ = 3.0 against a **warm-up** (λ rising linearly from 0 to
3.0 over the first 2,000 steps), on pilot seeds 200–204, for `mass_only` and `acc_only`.

**Selection rule,** fixed before the pilot was inspected: choose the schedule with more
converged pilot networks (ID ≥ 0.95 **and** designed-block interchange accuracy ≥ 0.9, pooled
over both groups). Ties go to `const`, the setting used in PRs #3–#4.

**Pilot outcome:** warm-up converged 9/10 pilot networks and constant 6/10.
**Locked schedule: `warmup`** (λ rising linearly from 0 to 3.0 over steps 1–2,000, then constant). It
applies to both declared groups; `neither` has no interchange term.

**Noted before the main run:** under the constant schedule, declaring *acceleration* converged in only
1/5 pilots, against 5/5 for mass. Declared variables are not equally easy to
train, and the main run's convergence table will show whether that persists under warm-up.

## 14.5 Frozen edit and scoring procedure

- **Primary edit: inversion modulo 13**, V → V⁻¹ (mod 13). **Zero maps to zero**, so examples
  with V = 0 cannot be edited and are excluded by the coincidence rule below.
- **Edit construction (frozen in `fit_edit`):**
  - an affine map inside the subspace located for V;
  - fitted by least squares to move each class mean of V to the class mean of its edited value;
  - folded into `W_in` as (ΔW, Δb).
  - **Primary location:** the DAS subspace (k = 16), found *blind* (it does not know the block).
- **Data separation (frozen seeds, never used before):**
  - edit fitting: `FIT_S = 51515`;
  - DAS search: `PAIR_S = 61616`, `DAS_S = 81818`;
  - **final evaluation: `TEST_S = 71717`**, a separate 16,384-example sample.
- **Eligibility.** Let F₀ be the unchanged answer, F_M the answer with M edited, and F_A the
  answer with A edited. An example is eligible iff F₀, F_M and F_A are **pairwise distinct**.
  This is one set, **shared by both variables**: the M-edit's wrong-variable answer is the
  A-edit's intended answer, and vice versa. For inversion it requires M, A ∉ {0, ±1} and A ≠ ±M.
  Of the 16,384 evaluation examples, 7,385 are eligible.
- **Scoring.** Every eligible output is classified as **intended / wrong-variable / unchanged /
  other**. Specificity = P(intended) − P(wrong-variable). All four proportions and the eligible
  count are reported. Specificity alone can hide a high failure rate.
- **Unit of evidence: the trained network.** All tests are across seeds; examples within a network
  are not independent evidence.

## 14.6 Hypotheses and analysis

For group g and seed s, let spec_g(V) be the inversion specificity for an edit of V (DAS-located),
and gap_g(s) = spec_g(M) − spec_g(A).

**Primary** (single test, α = 0.05, two-sided):
- Δ(s) = gap_mass_only(s) − gap_acc_only(s), paired by seed.
- **Prediction: mean Δ > 0.**
- Test: an exact sign-flip permutation over the 12 seeds (4,096 patterns), with a 95% percentile
  bootstrap CI over seeds (10,000 resamples).

**Key secondary: both directions separately** (Holm across the two, α = 0.05). These guard
against a strong mass result hiding a flat acceleration result.
- direction M: spec_mass_only(M) − spec_acc_only(M) > 0;
- direction A: spec_acc_only(A) − spec_mass_only(A) > 0.

**Decomposition against `neither`** (reported with CIs, uncorrected), separating help to the
declared variable from damage to the other:
- `mass_only`: own M − neither, and other A − neither;
- `acc_only`: own A − neither, and other M − neither.

**Secondary edit: add-one** (V → V+1 mod 13; pre-specified, because it failed in PR #4). This uses
the same scoring, eligibility rule (pairwise-distinct answers), tests and tables.

**Diagnostic** (on the trained networks themselves). For each group and variable, compare the
intended-answer rate of:
1. **copying a real pattern.** Take a real source input with the same other quantity (value and
   unit), the same output unit and order, and the edited target value, with its unit drawn from
   in-distribution options. Copy its block activation into the base example's block.
2. the fitted edit applied in the **known block** (dims 0–15);
3. the fitted edit at the **DAS location**;
4. the source pattern copied into the DAS location.

Designed-block interchange accuracy is also reported. Interpretation:

| copy | known-block edit | DAS edit | diagnosis |
|---|---|---|---|
| works | fails | — | the edit's *construction* is the problem |
| — | works | fails | the *location search* is the problem |
| fails | — | — | check task learning and whether the block carries the variable usably |

**Secondary:** converged-seeds-only primary statistic, plus convergence counts, ID and OOD
accuracy. Converged-only results are not the headline.

## 14.7 Decision rules, fixed in advance

| outcome | conclusion |
|---|---|
| Primary Δ > 0 (significant) **and** both directions positive | **The advantage switches sides with declaration.** This supports *declaration itself helps* for the inversion edit. |
| Advantage follows mass even when A is declared (direction A ≤ 0) | Declaration is not the cause. Investigate units, inputs, or the editor. |
| Inversion replicates **and** add-one fails again | Declaration helps with **one tested transformation**, not with editing in general. |
| Neither edit replicates | Report directly; PR #4's inversion result does not replicate on fresh networks. |

The hypothesis "different units leave different internal patterns" remains a possible
explanation only. It is not a conclusion until it predicts a result on fresh networks.

## 14.8 Out of scope

The paraphrase experiment is deferred until an edit is shown to follow the declared variable
reliably.

## 14.9 Pilot table (schedule selection only)

| schedule | network | ID acc | designed-block IIA | converged |
|---|---|---|---|---|
| const | mass_only_s200 | 1.000 | 1.000 | yes |
| const | mass_only_s201 | 1.000 | 1.000 | yes |
| const | mass_only_s202 | 1.000 | 1.000 | yes |
| const | mass_only_s203 | 1.000 | 1.000 | yes |
| const | mass_only_s204 | 1.000 | 1.000 | yes |
| const | acc_only_s200 | 0.631 | 0.641 | no |
| const | acc_only_s201 | 0.242 | 0.248 | no |
| const | acc_only_s202 | 0.375 | 0.363 | no |
| const | acc_only_s203 | 1.000 | 1.000 | yes |
| const | acc_only_s204 | 0.598 | 0.591 | no |
| warmup | mass_only_s200 | 1.000 | 1.000 | yes |
| warmup | mass_only_s201 | 1.000 | 1.000 | yes |
| warmup | mass_only_s202 | 1.000 | 1.000 | yes |
| warmup | mass_only_s203 | 1.000 | 1.000 | yes |
| warmup | mass_only_s204 | 1.000 | 1.000 | yes |
| warmup | acc_only_s200 | 1.000 | 1.000 | yes |
| warmup | acc_only_s201 | 0.605 | 0.601 | no |
| warmup | acc_only_s202 | 1.000 | 1.000 | yes |
| warmup | acc_only_s203 | 1.000 | 1.000 | yes |
| warmup | acc_only_s204 | 1.000 | 1.000 | yes |

Converged: const = 6/10 (mass 5/5, acc 1/5); warmup = 9/10 (mass 5/5, acc 4/5).
