# 46 — The mechanistic origins of generalizable concepts (role-swap and factorial), and candidate causal mechanisms

Continuation of PR #33, at the owner's request (INIT "The Mechanistic Origins of Generalizable Concepts").

**Status.** Everything in this document is **retrospective and exploratory**. It uses the exact reconstructions
(36/36 bitwise-identical endpoints, doc 41R), the registered 10-step window replay (doc 45; bitwise-qualified at
steps 1,000 and 2,000 for all five networks) and the archived factorial weights. No prospective training was run.

**Labels.**
- **[O] observed:** a measurement in these networks.
- **[I] inferred:** an interpretation of observations.
- **[E] experimentally demonstrated:** none yet in this document, since every intervention here is a single-pass
  measurement on a fixed network, not a change to training.

**Terms**, defined the first time they matter:
- *Generalisation to unseen inputs*, or *OOD accuracy*, is accuracy on (unit, value) renderings that never
  appeared in training: unit 2 with values 9–12.
- The *declared block* is the 16 coordinates (0–15) of the last-token residual where the interchange objective
  trained the canonical mass M = value × unit-scale (mod 13).
- A *rendering* is one way of writing a quantity, e.g. "5 g" vs "8 kg" for the same canonical mass.

---

## 46.1 Critical window: what separates the generalisers 307/311 from the failures 304/310

Sources:
- `runs/swap_recon/window_battery_10step.json`: 705 states, 10-step resolution, steps 600–2,000.
- `runs/swap_recon/window_factorization.json`.
- `runs/swap_recon/factorization_trajectories.json`: all 24 declared networks, 100-step resolution.
- `/home/user/swap_window/*/steps.jsonl`: per-step gradients.

### (a) The finding: factorization is kept or lost during the escape [O]

**Factorization** (R²_add) means the declared block's activation is a sum of a part that depends only on the
written value and a part that depends only on the written unit, plus context. It is measured as the R² of an
additive (value + unit) model of the block's (unit, value) class means. 1.00 means fully factorized, with no
value×unit interaction.

| | step 0–600 | during escape | after escape (frozen) | declared-variable OOD at 6,000 |
|---|---|---|---|---|
| 307 | 0.99–1.00 | dips to 0.93 (step 900) | **1.00** by 1,050 | 1.00 |
| 311 | 0.99–1.00 | dips to 0.91 (step 1,000) | **0.98** | 1.00 |
| 305 | 0.99–1.00 | dips to 0.96 (1,200) | 0.96–0.97 | 0.46 |
| 304 | 0.98–1.00 | falls to 0.75 (600 → 950) | **0.71** | 0.09 |
| 310 | 1.00 | falls to 0.74 (1,300 → 1,500) | **0.68** | 0.09 |

**Population check, all 24 declared networks** (100-step reconstructions):
- Final factorization correlates with declared-variable OOD accuracy at ρ = **0.82** among the 16 that fit their
  training data, and ρ = 0.67 across all 24.
- **Necessary:** every near-perfect generaliser (OOD ≥ 0.95; 4/4) has final factorization ≥ 0.98.
- **Not sufficient:** of the 6 fitted networks with factorization ≥ 0.98, two only partially generalise:
  `mass_only_s300` (0.57) and `acc_only_s305` (0.82).
- **Not an early predictor:** at step 600, before any escape, essentially every network is still fully factorized
  (most ≥ 0.97; the lowest are two networks that never fit their training data, at 0.84–0.86).

### (b) Why factorization matters: unseen renderings are predictable from their parts [O]

Endpoint test, seven fitted `mass_only` networks:

| network | OOD (mass) | additive R² | canonical R² | error predicting the *unseen* renderings' activations from the additive model | same, from the canonical model |
|---|---|---|---|---|---|
| 307 | 1.00 | 1.000 | 0.29 | **0.03** | 2.84 |
| 311 | 1.00 | 0.983 | 0.24 | 0.23 | 1.77 |
| 300 | 0.57 | 0.986 | 0.38 | 0.07 | 2.88 |
| 305 | 0.46 | 0.963 | 0.39 | 0.19 | 2.65 |
| 302 | 0.19 | **0.171** | 0.92 | 1.31 | 1.04 |
| 310 | 0.09 | 0.683 | 0.74 | 1.19 | 1.42 |
| 304 | 0.09 | 0.708 | 0.63 | 0.88 | 1.99 |

Errors are relative to the typical class-mean norm.

**Factorized networks.** In 307, the block activation for "unit 2 with value 9–12", never seen in training, is
exactly what the value part (learned from units 0 and 1) plus the unit part (learned from values 0–8) predicts.
The downstream head, the computation after the block, then decodes it.

**Failures.** The block holds a value×unit **interaction**, a code that depends on the specific pairing and
behaves like a lookup over trained pairs. Unseen pairings land where neither model predicts.

At layer 1, all seven networks are factorized at the unit token (0.87–1.00). **The interaction is introduced at
the declared block** (layer 2, last token), where the interchange objective acts.

### (c) Other measures, and what they do *not* show [O]

- **Route-specificity of the block freezes in failures.** It drops to 0.62 (304) or about 0.70 (310) during the
  escape, then stays constant while training accuracy rises from 0.50 to 1.00 (304, steps 850–1,200) and
  thereafter. In generalisers it keeps falling to 0.26–0.30 while training accuracy rises. This is the class-mean
  counterpart of (a).
- **Single-frequency multiplicative circle.** Failures' block class means lie on an almost perfect
  discrete-logarithm circle: 0.88 (304, frequency 1), 0.95 (310, frequency 5), 0.90 (302, frequency 2).
  Generalisers' do not (0.30–0.54). The "textbook" multiplicative code is the *non-generalising* solution here: it
  is canonical for trained renderings, but it is reached through the interaction, not through composable parts.
  At endpoints this measure correlates with OOD at ρ = −0.89 (7 fitted networks).
- **Attention routes are not consistent.** 304 binds mass to its unit through the unit token (unit→value attention
  0.46; last token→unit 0.29). 310 binds through the value token (value→unit 0.48). Generalisers differ again. So the
  failure is not a particular attention route.
- **Loss order is not consistent.**
  - Generalisers reduce the ordinary task loss before the interchange loss: 307 at step 900 has task 1.49 and IIT
    2.11; 311 at step 1,000 has task 1.06 and IIT 2.10.
  - But 304 shows the same lead (mean IIT − task during the escape is 0.34 for both 304 and 307).
  - Across the 16 fitted declared networks, the lead correlates only weakly with OOD (ρ = 0.43).
  - **Prediction "computation-first networks generalise": contradicted.**
- **No pre-window predictor.** Across the 24 declared networks, none of the following predicts the outcome
  (|ρ| < 0.4; leave-one-out threshold rules 0.33–0.75):
  - route-specificity at any step 0–1,000;
  - escape time;
  - early task loss;
  - early gradient norm.
- **Gradients.**
  - Once the escape starts, the interchange gradient is larger than the task gradient on the attention weights in
    every network: about 1.4–1.8× on attention 1 and 1.4–1.7× on attention 2 between steps 1,000 and 1,200.
  - The two gradients become aligned (cosine 0.5–0.8).
  - Before the escape they are nearly orthogonal or opposed (cosine −0.2 to +0.3).
  - None of these quantities separates the groups.

### (d) What the evidence supports [I]

The divergence is **not** a difference in whether the declared block forms. It forms in every fitted network, with
designed-block interchange of 1.00.

It is a difference in **how** the block computes the declared quantity:
- **Generalisers:** composable parts, value + unit, which the head composes.
- **Failures:** a value×unit interaction learned only for trained pairings.

That interaction appears during the escape in every network. It is **reversed** in generalisers within about 100–150
steps, and **retained and frozen** in failures once training loss saturates. With weight decay at 0.01 nothing
removes it afterwards: the failures' factorization is unchanged from step 2,000 to step 6,000.

**What decides between reversal and retention is not identified.** The candidates are in §46.4.

---

## 46.2 Mathematical organisation of the canonical subspace (factorial)

Source: `runs/instruments/subspace_math.json` (C1.0, C0.1, D1.0, B1.0 × 12 seeds × steps 1,000/3,000/6,000) and
doc 41R.7. Site: residual after block 1 at the source-number token.

- **Information present vs readable vs controlling** [O]. See §46.2 table (filled from the JSON in 46R).
- **Effective causal dimension is about 4, not 12** [O]. In C1.0 the top 4 directions of the canonical-mean subspace
  carry nearly the full intervention effect (seed 2001: 0.79 of 0.80).
- **Necessary as well as sufficient** [O]. Mean-ablating the subspace during *ordinary* prediction destroys it:
  trained accuracy 1.00 → 0.09, held-out → 0.00. A random subspace of equal rank leaves trained accuracy at about 1.00.
- **Geometry decides which edits are linear** [O].
  - In C1.0, a single linear map inside the subspace, fitted on 9 of 13 canonical classes, *doubles* and *inverts*
    canonical quantities of the 4 held-out classes. Add-one transfers to none.
  - This is what a multiplicative (discrete-log) geometry predicts. Multiplying and inverting are rotations and
    reflections of a discrete-log circle; adding one is not.
  - Exceptions are listed in 46R.
- **Alignment across independently trained networks:** Procrustes transplant results in 46R.

## 46.3 Generalisation and editability

### Population [O]

Across the 24 declared role-swap networks, declared-variable OOD accuracy correlates **negatively** with the
registered (doc 15) inversion edit's specificity (ρ = −0.55; −0.63 with overall OOD). The four near-perfect
generalisers have edit specificity ≈ 0.00. The most editable networks are middling or failing generalisers:
`acc_only_s306` 0.96, `acc_only_s300` 0.73, `acc_only_s310` 0.69.

### Ladder (focus networks; `runs/swap_recon/edit_ladder.json`) [O]

| rung | generalisers 307/311 | failures 304/310 |
|---|---|---|
| L1 localisation (designed-block interchange) | 1.00 | 1.00 |
| L2 replacement with a real donor activation | 1.00, including unseen renderings | 1.00 |
| L2′ replacement restricted to the class-mean subspace | 1.00 | 1.00 |
| L3 class-mean-based transformation (global, in-span) | 0.08–0.14 | 0.14 |
| L3 per-unit transformation | 0.08–0.39 | 0.16–0.32 |
| L4 permanent: folded into W_in only | 0.07–0.08 | 0.08 |
| L6 preservation of inversion fixed points under L3 | 0.09–0.21 | 0.35–0.41 |

### Interpretation [I]

1. **Localisation and replacement are not the limitation.** Every network carries the declared quantity in the
   designed block and in the subspace spanned by its class means.
2. **The class-mean transformation fails everywhere, worst in generalisers.**
   - In generalisers the block is a *sum of value and unit parts*: a given mass has as many internal patterns as it
     has renderings.
   - A class mean averages these patterns, and no single map sends every rendering's pattern for M to a valid
     pattern for f(M), even within the 12-dimensional span.
   - PR #34's alias measurements (block RMS between equal-canonical renderings 12–22, with 99.9% prediction
     agreement) are independent evidence for this "many patterns, same effect" organisation.
3. **What a generalising network computes with is the pair (value, unit), not the canonical value.** An edit must
   act on the composable parts. For inversion, which is multiplicative, the corresponding edit maps
   value → value⁻¹ and unit-scale → unit-scale⁻¹. It is therefore a per-part edit, not a class-mean map.
   Doc 19's unit-aware editor reached 87–90% where linear edits reached 27–34%; that is consistent with this
   account.
4. **Proposed, not run.** A per-part edit fitted on the value and unit components of the additive model; see
   §46.5 P4.

---

## 46.4 Candidate causal mechanisms

Each mechanism lists: explanation, supporting and contradicting evidence, alternatives, earliest observable point,
falsifiable prediction, and a causal experiment. All experiments are **proposals**, not run (§46.5).

### C1. Interaction lock-in (primary candidate)

**Explanation.** During the escape, gradient pressure, mostly from the interchange objective, installs a value×unit
interaction in the declared block. If the ordinary and interchange losses saturate while the interaction still
carries part of the solution, the remaining gradient is too small and weight decay (0.01) too weak to remove it. The
block stays a pairwise code and unseen pairings fail. If the factorized solution reduces the loss faster, the
interaction is reversed.

**Supports.**
- §46.1(a), (b): necessary-condition pattern across 24 networks; the freezing in 304 and 310; transient dips in
  307 and 311.
- In the factorial, strong decay continued to prune route-specific structure *after* the task was fit (route
  specificity 0.17 → 0.08), while weak decay plateaued (doc 41R.6).

**Contradicts or limits.**
- Factorization ≥ 0.98 is not sufficient (300: 0.57; 305: 0.46).
- 304's factorization fell *before* its training accuracy saturated (0.75 at step 950, with training accuracy 0.93).
- Not observable before the window.

**Alternatives.**
- (i) The decoder, the computation after the block, decides. Factorized blocks are necessary, and the head must
  also compose generically.
- (ii) Initialisation or data order fixes both. PR #34 will separate these.

**Earliest observable.** About 200 steps before each failure's escape: 304 at step 650 (0.97); 310 at step 1,400
(0.95).

**Prediction.** In any declared network, final factorization < 0.95 implies declared-variable OOD < 0.95.
Applies to fresh seeds and to PR #34's crossed cells.

**Causal test (reciprocal).**
- **Rescue:** resume 304 and 310 from step 2,000 with a factorization-restoring pressure. That is either strong
  weight decay (1.0), or an auxiliary penalty on the interaction component of the block (the residual of the
  additive model computed on the batch). Predicted: factorization rises above 0.95 and OOD rises.
- **Disrupt:** resume 307 and 311 from step 850, before their dip, with a penalty that *rewards* the interaction
  component, or with the learning rate raised so the losses saturate before reversal. Predicted: factorization
  stays below 0.9 and OOD falls.
- **Controls:** a matched penalty on a random direction; the same resumes with no change, which are exact
  replays.

### C2. Decoder generality (secondary; explains non-sufficiency)

**Explanation.** A factorized block is necessary. The head (residual + MLP) must also implement the composition for
unseen pairings, and in 300 and 305 it does so only partly.

**Supports.** 300 and 305 are factorized but partial. In them, prediction from the *additive model's* predicted
block is correct only partly (to be measured, 46R).

**Prediction.** Among factorized networks, OOD equals the head's accuracy when fed the additive-model block for unseen
pairings.

**Test.** Retrospective first: feed the predicted block into the head. Then prospectively, train only the head on
trained pairings from a factorized block and test unseen pairings.

### C3. Strong decay converts a canonical code into a reusable reader (factorial)

**Explanation.**
- A coherent semantic target builds a canonical code at the number token under any decay.
- Strong decay then prunes the leftover route-specific structure, and the reader becomes a reusable conversion
  over thousands of steps.

**Supports.** Doc 41R.6: route-specificity 0.08 vs 0.18; reader 0.14 → 0.47 vs flat.

**Contradicts.** It does not explain variation *between seeds* within C1.0: route-specificity ρ = −0.05 with held-out
accuracy.

**Causal test.** Switch decay mid-training:
- C0.1 → 1.0 at step 2,000 should start the reader's growth.
- C1.0 → 0.1 at step 2,000 should stop it.

This is reciprocal, and both arms resume from archived checkpoints.

### C4. Computation-first ordering: weakened

The task loss leading the interchange loss does not separate the groups (ρ = 0.43; 304 equals 307). It is kept only
as a component of C1: the order matters only through whether the interaction is in place when the losses saturate.

### C5. Attention binding route: contradicted

The failures use opposite routes, so the route alone is not the mechanism.

---

## 46.5 Proposed targeted intervention experiments

Not run. Each requires its own registration and owner approval.

| id | type | networks | manipulation | prediction | control |
|---|---|---|---|---|---|
| P1 | rescue (C1) | 304, 310 from step 2,000 | auxiliary penalty on the block's value×unit interaction component (additive-model residual), λ ∈ {0.1, 1} | factorization > 0.95 and declared OOD > 0.9 by step 6,000 | penalty on a random 16-dim direction; plain resume (exact replay) |
| P2 | disrupt (C1) | 307, 311 from step 850 | penalty that *rewards* the interaction component, or IIT weight ×3 (faster saturation) | factorization stays < 0.9; OOD < 0.5 | random-direction penalty; plain resume |
| P3 | decay switch (C1/C3) | 304, 310 at 2,000, wd 0.01 → 1.0; factorial C0.1 at 2,000, 0.1 → 1.0, and C1.0 at 2,000, 1.0 → 0.1 | decay only | route-specificity falls and OOD rises for the 0.01→1.0 and 0.1→1.0 arms; the reader stops growing for 1.0→0.1 | unchanged resumes |
| P4 | per-part edit (editability) | all fitted declared networks, retrospective | edit value and unit components separately (inversion: v → v⁻¹, s → s⁻¹) in the block's additive coordinates | in factorized networks the per-part edit succeeds where class-mean edits fail; in failures, the reverse | class-mean edit; random-part edit |
| P5 | window necessity | 307 from step 850 | freeze the block's input weights for 200 steps | the dip is not reversed; OOD falls | freeze an unrelated tensor |

P4 is a single-pass measurement on copies and can be run without new training. It is scheduled as retrospective
follow-up work.

---

## 46.6 Relation to PR #34 (independent study)

PR #34 crosses initialisation × ordinary data × interchange data for the 307/304 and 311/310 pairs. It also
re-scores all five focus networks over all 1,776 unseen prompts, with a discriminating interchange that excludes
coincidental answers.

**Agreement so far.** Its exhaustive scores (307: 1,772/1,776; 311: 1,774; 305: 1,286; 304: 478; 310: 493) and its
discriminating unseen-source swap (304: 0 of 2,692; 310: 3 of 2,692) agree with and strengthen doc 41R.

**What its crossing will decide for C1:**
- If one stream (for example initialisation) carries the outcome, C1 predicts that the cells with that stream lose
  factorization during their escape and the others do not.
- If the outcome depends on interactions between streams, C1 still predicts the factorization–outcome relation
  *within* each cell. That is a sharp test, because the factorization measure (`sci/factorization.py`) can be
  applied to its checkpoints without retraining.

Neither study modifies the other.
