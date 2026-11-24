# 37 — Registration: Birth of a Concept, 4 × 2 factorial (96 networks)

Committed before any registered training. Builds on the Stage-1 re-analysis (doc 35) and the qualification pilot
(doc 36, preserved unchanged in `runs/concept_pilot/`). Code at registration: `sci/concept.py` (sha256 prefix
`5c557e86e781cf24`), `sci/concept_audit.py` (`a03d5d7530d3cc6c`), `sci/factorial.py` (`6c89bb66955bea54`); any
later code change is logged as a deviation in the results document.

## 37.1 Questions

1. Does canonical-quantity supervision causally promote compositional generalisation (C vs its matched control D)?
2. Does weight decay interact with semantic supervision?
3. Why does a decodable canonical representation exist thousands of steps before it is used compositionally?
4. What changes in weight space, gradient dynamics and causal circuitry accompany that transition?
5. What mechanism produces the temporary collapses and recoveries seen in the pilot?

Working hypothesis for 2 (falsifiable, not assumed): weight decay weakens route-specific memorisation / competing
pathways, so that the network comes to *use* the canonical quantity it already represents.

## 37.2 Design (frozen)

* **Conditions**: population ∈ {A narrow, B diverse, C canonical auxiliary, D route-scrambled auxiliary} × weight
  decay ∈ {0.1, 1.0}; definitions exactly as in `sci/concept.py` (doc 36 §36). **12 seeds: 2001-2012**, the same
  12 in every condition. The seed fixes the initial weights, which are therefore identical across all 8 conditions
  for a given seed (tested), so contrasts are **paired by seed**. 96 networks.
* Student: SemStudent, 129,280 inference parameters; AdamW lr 1e-3 (100-step linear warm-up, then constant),
  batch 128, 6,000 steps, auxiliary weight 1.0 (C, D), deterministic algorithms, one CPU thread per worker.
* Task and split: modular unit conversion (mod 13, fictional units — not real-world arithmetic). Split sha
  `c103174ee0bfc1fa`: 120 trained (value, route) combinations, 18 development combinations (used by audits and
  onset definitions), **18 reserved test combinations** used only once, at step 6,000:
  `acc:0→1 v8, v10; acc:0→2 v7, v9; acc:1→0 v8; acc:2→0 v9, v12; acc:2→1 v6; mass:0→1 v2; mass:0→2 v10;
  mass:1→0 v5, v12; mass:1→2 v1, v7, v8; mass:2→0 v5; mass:2→1 v2, v4` — 72 items (4 training frames),
  best-constant baseline 0.167; plus the same 18 in the unseen frame C03 (secondary).

## 37.3 No interventions

The 96 registered trajectories are never modified after initialisation: no curriculum changes, no edits, no
early stopping. Any intervention is performed on checkpoint copies, or in an explicitly registered separate
experiment, and is never written back.

## 37.4 Preservation

Every 100 steps (61 checkpoints per network including initialisation): weights, auxiliary head, full AdamW state
(both moments, step counts), torch and Python RNG, and the exact decomposition of the 100 AdamW updates into the
decoupled weight-decay part and the adaptive-gradient part (summed per parameter); per-step interval log (loss
parts, gradient norm, batch hash, lr, worst item loss). Training resumes bit-exactly after interruption (tested).
Tensors are stored **outside Git** as per-run `tar.xz` archives with a per-file SHA-256 manifest and an archive
checksum; Git holds manifests, audit summaries, interval logs (gzipped), cohort reviews, code and instructions.

## 37.5 Audit protocol (every 100-step cohort)

**Automated measurement, every network, every checkpoint** (`sci/concept_audit.py`, on saved checkpoints only):
behaviour (practiced, own-trained, paraphrase, development held-out, held-out in a new frame); per-site tracing of
task-loss and auxiliary-loss gradients through embeddings, every attention and MLP output, every residual stream,
the readout hidden layer and logits; task-aux gradient alignment per token and per parameter group; per-item
losses; discrete-log and additive geometry of raw values and of canonical quantities at x0/x1m/x2m;
decodability of canonical quantity, raw value and source unit; class-subspace bases (lineage) and subspace
ablations; causal `patch_semantics` at x0/x1/x2 (number token; number + unit) and the novel-combination patch;
weight norms; the AdamW/weight-decay decomposition.

**Supervisor review, every cohort**: matched cohorts (all 96 networks at the same step) are reviewed together
from an automated cohort table; every network is measured; automatically flagged anomalies (held-out change
≥ 0.15 within 100 steps, own-trained accuracy drop ≥ 0.05, gradient-norm spike > 5× the interval median,
canonical-semantics change ≥ 0.15, onset events) are investigated individually. Each review records what
changed, the evidence, competing explanations, a falsifiable prediction, whether a new instrument was needed, and
the outcome of the previous prediction. New deterministic instruments may be built at any time; they run only on
checkpoint copies, are tested on planted or known cases before use, and are flagged *retrospective* when applied
to earlier checkpoints.

## 37.6 Primary endpoint and confirmatory contrasts

Primary endpoint: **reserved-test accuracy at step 6,000** (72 items per network). Unit of analysis: the seed.
Each contrast is a per-seed paired difference d_s (12 values), tested with an **exact sign-flip permutation test**
(2¹² sign patterns) on the mean; effect size = mean d with a 95% seed-bootstrap CI (10,000 resamples, seed 0).
Holm correction across the five tests, family-wise α = 0.05.

| ID | Contrast | Direction |
|---|---|---|
| H1 | C − D at wd 1.0 | one-sided, > 0 |
| H2 | (C − D at wd 1.0) − (C − D at wd 0.1) | one-sided, > 0 (interaction) |
| H3 | C(wd 1.0) − C(wd 0.1) | one-sided, > 0 |
| H4 | C − B at wd 1.0 | one-sided, > 0 |
| H5 | B − A at wd 1.0 | two-sided |

Secondary (reported, not corrected): the same contrasts on the new-frame test items; all 8 condition means.

## 37.7 Onset definitions (frozen)

Evaluated on audits at 100-step resolution; an onset is the first audited step at which the criterion holds at that
step **and** the next audit.

* **Decodable**: canonical quantity at x1m (source-number token) decodable on development held-out combinations,
  kNN ≥ 0.50.
* **Causally utilised**: `semantics_x1_src` canonical − number ≥ 0.20.
* **Compositional**: development held-out accuracy ≥ best constant + 0.20.

## 37.8 Registered mechanistic predictions (secondary; each with its falsifier)

* **M1 — ordering.** In networks with both onsets, causal utilisation precedes compositional onset (lead ≥ 100
  steps) in more networks than it follows it (sign test over networks with both onsets, α = 0.05). Falsified if
  utilisation follows composition at least as often as it precedes it.
* **M2 — collapse coupling.** For C networks after their compositional onset, among 100-step changes in development
  held-out accuracy of magnitude ≥ 0.05, the change in x1 canonical semantics has the same sign in ≥ 70%
  (binomial test against 50%). Falsified if agreement ≤ 50%.
* **M3 — weight decay and canonical use.** At step 6,000, x1 canonical − number semantics in C is larger at
  wd 1.0 than at wd 0.1 (paired by seed, sign-flip test). If it is not, the hypothesis that weight decay acts by
  making the network *use* the x1 canonical code is not supported at that site (composition would then be gated
  downstream).
* **M4 — representation without use (Q3).** In C at wd 0.1, the canonical quantity is decodable (kNN ≥ 0.5) at
  step 6,000 in ≥ 9 of 12 seeds while compositional onset is not reached in ≥ 9 of 12. Falsified otherwise.

## 37.9 Exploratory (declared now, not confirmatory)

Weight-space and gradient precursors of compositional onset and of collapses (AdamW vs weight-decay update norms
and their cosine per parameter group; task-aux gradient alignment at x1m; weight-norm trajectories; subspace
lineage); collapse/recovery mechanism (Q5); A and B trajectories; any instrument built during the run. Precursor
candidates found in this experiment can only be *tested* in a later registered replication.

## 37.10 Stopping and deviations

All 96 networks train to 6,000 steps. Infrastructure failures are resumed from the last complete checkpoint
(bit-exact) and logged. A network whose training cannot be resumed is reported as missing; it is not replaced.
The reserved test is evaluated once (`python -m sci.factorial final` refuses a second run).

## 37.11 Reproduce

```
python -m pytest -q tests/test_concept.py tests/test_factorial.py
for i in 0 1 2; do python -m sci.factorial train $i 3 & done; python -m sci.factorial audit --loop &
python -m sci.factorial final                       # after all networks reach 6,000
python -m sci.factorial archive <run>               # per run: archive outside Git + manifest into Git
```
