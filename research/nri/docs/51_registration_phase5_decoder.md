# 51 — Registration: Phase V, the birth of compositional computation (the decoder)

Committed **before** any experiment below runs.

**Authorisation.** The owner's Phase V INIT approves:
- a registered investigation of decoder composition;
- the Phase IV documentation updates;
- durable preservation of the reconstruction and window archives.

The reliability experiment remains deferred. No original archive or historical result is modified. Neither PR #34's
branch nor its registration is touched.

## Definitions

- **Decoder.** The computation after the declared block. In these networks that is the head:
  logits = W_U(h + W_out · ReLU(W_in h + b)), where h is the last-token residual (64 dims).
  - h holds the declared mass block (dims 0–15) and the rest (dims 16–63), which carries acceleration, output unit
    and order.
  - The **encoder** is everything before h: the embeddings and two attention layers.
- **Controlled inputs.** Decoder inputs built from parts fitted on *familiar* data only (calibration seeds disjoint
  from all test panels). For network N, fit an additive model of N's full residual h over the latent factors:
  - mass value, mass unit, acceleration value, acceleration unit, output unit and order;
  - each as one-hot cell means, by least squares on familiar renderings.

  A **synthetic input** for any combination, including never-trained ones, is the sum of the fitted factor parts.
  This tests the decoder on representations that are valid by construction and independent of whether the encoder
  produced them for that combination.
- **Decoder classes** (fixed now), from accuracy on synthetic inputs, all answers scored against the task's exact
  arithmetic:

  | class | trained cells | unseen cells near a trained twin | unseen cells far from any twin |
  |---|---|---|---|
  | **algorithmic** | ≥ 0.95 | ≥ 0.95 | ≥ 0.95 |
  | **interpolative** | ≥ 0.95 | ≥ 0.9 | < 0.5 |
  | **lookup** | ≥ 0.95 | < 0.5 | < 0.5 |

  "Near" and "far" are by block distance to the trained rendering of the same canonical mass, using the thresholds
  of doc 49 §49.3 (< 0.3 and > 2). Other patterns are reported as mixed.

## D1 — Decoder isolation (single pass)

**Networks.** All 16 declared networks that fit their training data, plus the `neither` networks as reference.

- **Measures:**
  - synthetic-input accuracy per class of cell (trained / near / far);
  - the decoder's accuracy on canonical class-mean inputs (doc 49 D1);
  - **perturbation resistance:** accuracy when the synthetic block is perturbed by isotropic noise of relative size
    0.1, 0.25 and 0.5.
- **Prediction.**
  - 307 and 311 algorithmic;
  - 300 and 305 interpolative;
  - 302, 304 and 310 lookup on factorised synthetic inputs, but correct on canonical inputs (their decoder is
    canonical-reading).

## D2 — Birth of the decoder (single pass on archived checkpoints)

The D1 measures are applied:
- every 100 steps for all 12 `mass_only` networks;
- every 10 steps (steps 600–2,000) for the 5 focus networks.

Parts are refitted at each checkpoint from that checkpoint's own familiar activations.

**Additional measures:**
- **Invariance:** agreement of the decoder's prediction across equal-canonical synthetic renderings.
- **Operand reusability:** accuracy on synthetic inputs whose mass part is paired with acceleration and output-unit
  parts never seen with it.
- **Tolerance to representation change:** accuracy when the block's interaction is perturbed (the doc-48 E3 operator
  at κ = 0.5, in-subspace).

**Analysis.**
- **Onset:** the first state with far-cell synthetic accuracy ≥ 0.9 at two consecutive states.
- **Gradual or discrete:** discrete if the onset step-change is ≥ 0.5 within 50 steps (10-step data); otherwise
  gradual.
- **Prediction:** the composing decoder appears during the escape (steps 900–1,400 for 307/311), before or at the
  time OOD accuracy rises, and never in 304/310.

## D3 — Developmental scaffolding (single pass + existing interventions)

**Tolerance curve.** At every 100-step checkpoint, inject the in-subspace interaction (κ = 0.5) at run time and
measure the fraction of OOD accuracy retained.

**Hypothesis.** "Factorisation is temporary scaffolding". It predicts that tolerance rises from about 0 to about 1
after the decoder becomes algorithmic. It is tested in 307 and 311, against the P2c result: 311 tolerated
destruction of factorisation after step 2,000.

**Controls:** random-direction and magnitude-matched injections outside the causal subspace.

## D4 — Reverse-engineering the learned algorithm (single pass; 307 primary, 311 replication)

- **Domain.** Enumerate the complete finite domain: 9,126 written prompts.
- **Hypotheses**, tested on the decoder's hidden units (256) and logits:
  - **H-F, Fourier/discrete-log.** Hidden activations and logits are explained by cos/sin of k·(dlog M + dlog A
    + dlog S) for a few frequencies k. Equivalently, the answer F = M·A·S is computed as a sum of discrete-log phases.
  - **H-L, lookup.** Activations are explained only by cell identity, without low-frequency structure.
- **Measures:**
  - fraction of hidden-activation variance explained by the best 1–3 discrete-log frequencies of the product phase,
    vs of the individual phases, vs cell identity;
  - logit structure: whether logit(F) ≈ Σ_k c_k cos(k·2π(dlog F − dlog(MAS))/12).
- **Explicit program.** Fit a closed-form program (phases per factor and coefficients per frequency) on *familiar*
  inputs only. Validate it against the actual network over the complete 9,126-prompt domain: agreement of
  predictions, including on unseen renderings.

## D5 — Decoder transplantation (single pass)

From generalisers (307, 311) into partial (300, 305) and failed (302, 304, 310) networks:

| variant | what is transplanted |
|---|---|
| (a) direct | head parameters copied |
| (b) aligned | head + an affine map of the recipient's residual into the donor's, fitted on familiar inputs |
| (c) factor-aligned | map fitted on factor parts (cell means), not on individual examples |
| (d) layer-wise | W_in only, or W_out + W_U only |
| (e) minimal causal subspace | the recipient's block parts mapped into the donor's block subspace only |
| (f) low-rank correction | of the recipient's W_in, fitted on familiar inputs to reproduce the donor-composed outputs for familiar synthetic inputs |

- **Qualifying repair:** OOD ≥ 0.90 with familiar ≥ 0.95.
- **Constraints:** no held-out answer or held-out activation is used in constructing any repair.
- **Report:** the smallest qualifying intervention, by number of changed parameters.

## D6 — Constructing the decoder (no training)

If D4 identifies the algorithm, construct a head **by hand inside the existing architecture**, with the same W_in,
ReLU, W_out and W_U shapes. Hidden units implement:
- (i) detectors for the factor parts, fitted on familiar data;
- (ii) the arithmetic that composes them, using the task's mathematical rule, not held-out answers.

**Compared decoders:**
- naturally trained (307);
- surgically repaired (D5, if any);
- constructed (D6);
- randomly initialised;

each on 307, 300 and 304 representations.

**Measures:**
- accuracy, familiar and OOD;
- causal use (the discriminating swap);
- parameter count used;
- editability (per-part edits);
- compositionality (D1 classes);
- compatibility with existing representations.

## D7 — Programming internal operators (no hooks, no token remapping)

Distinguish:
- **input substitution:** token-row edits (Phase IV);
- **operator modification:** changes to the decoder's own parameters that change the computed function, e.g. making
  the head compute F = M²·A·S, or F = M·A·S + 1.

**Target.** For the constructed decoder (D6), and if D4 succeeds for the natural decoder, compile operator edits
exactly into W_in, W_out or W_U.

**Measures:**
- exactness over the full domain;
- reversibility;
- composition with token-level programs.

## Training (only if required)

- No new training is planned in D1–D7.
- If D3/D4 require a causal developmental test (e.g. transplanting a composing decoder into a network at step 850
  and continuing training, to test whether the critical-window dependency disappears), it will be added as a
  separately registered addendum before running. It will use the existing intervention driver with bitwise matched
  controls.

## Standards

- No seed selection.
- Negative results are kept.
- Familiar-only fitting for every qualifying intervention; oracle diagnostics are labelled.
- Exploratory except for the classification and qualifying criteria fixed above.
