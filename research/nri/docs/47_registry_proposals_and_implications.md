# 47 — Proposed updates to the Frontier Registry (Issue #31) and implications for the program

These are **proposals for owner review**. They are not posted to Issue #31: the registry's own rule is that status
changes need owner review and a pull request. Statuses use the registry's codes:
- **C-I**: closed in a specified setting by our registered evidence;
- **C-N**: not supported;
- **P**: partial or exploratory;
- **O**: open.

Retrospective analyses remain exploratory (P) even when their effects are large.

## 47.1 Updates to existing records

| ID | current | proposed | evidence | boundary |
|---|---|---|---|---|
| **Q03** (canonical auxiliary target × weight decay) | P | **C-I** for the registered estimand: the canonical target improves reserved compositional accuracy **only with strong weight decay**. **C-N** for "independently of weight decay". | Doc 38: H1 +0.55, interaction +0.48, Holm p = 0.0012, 12 paired seeds. Doc 40: independent re-derivation. | One task (mod-13 conversion), one architecture. Mechanism (doc 41R.6) exploratory. |
| **I14** (pilot) | P | superseded by the factorial (doc 38). Keep the record and its history. | — | — |
| **Q04** (decodability, use and reuse at distinguishable times) | O | **P** | Doc 46R.1: in C1.0 the canonical code is present and readable (0.87–0.88) by step 1,000, while control of unseen combinations is 0.14 and grows to 0.47 by 6,000. Validated held-out instrument (doc 41R.1). Canonical reading precedes composition in 11/12 at 100-step resolution. | Retrospective; one site; registered onsets only for M1. |
| **Q01** (identify a natural concept vs a probe- or patch-induced substitute) | O | **P** (internal method evidence) | Planted-mechanism validation P1–P5, including a correct code with a trained-only reader (doc 41R.1). Complement (I5) and random-subspace (I7) controls give ≤ 0.014 across 96 networks. **False-negative probe illusion:** probes fitted on familiar inputs read 0.00 on unseen inputs that a network demonstrably processes at 0.99 (doc 41R.2). | Toy networks; planted mechanisms are adapters, not trained networks. |
| **Q05** (early forecasting) | O | **P, negative so far** | Doc 46 §46.1(c): no measure at steps 0–1,000 predicts the role-swap outcome (24 networks; \|ρ\| < 0.4; leave-one-out 0.33–0.75). The outcome becomes measurable only during the escape: factorisation loss starts about 200 steps before a failure's escape. | Retrospective, small n; prospective test not run. |
| **Q06** (upstream binding supervision vs editability) | O | **O, sharpened** | Doc 46R.2: generalisation and editability are **not** in conflict once the edit acts on composable parts (307: per-part edit 1.00 on familiar and unseen renderings). The registered conflict (I06: declared arms OOD 0.53/0.51) is a *distribution* over seeds: some declared networks keep factorisation and generalise perfectly. | The upstream-supervision factorial is still untested. |
| **Q07** (does geometry predict which edits work?) | O | **P** | Doc 46R.1: in the factorial, a multiplicative (discrete-log) geometry makes inversion and doubling linear across held-out classes (0.78 / 0.42) and add-one not (0.06). This reproduces I09 across 12 networks. Doc 46R.2: in role-swap networks the block's *factorised* (value + unit) geometry predicts that per-part edits work and class-mean edits fail. | Predictions made before fitting only for P4. |
| **Q12** (transfer of learned circuits when the representation moves) | O | **P** | Doc 46R.1: an orthogonal map fitted on class means transfers the canonical code between independently trained networks (switch 0.41 vs own 0.47 vs random rotation 0.05; 12 pairs). | Single site; same architecture and data. Not a transplant of a circuit into a different host. |
| **I06** (declaring a variable improves its editability) | C-I | keep C-I; **add limitation** | The registered class-mean edit is worst in the best generalisers (ρ = −0.55 across 24). A per-part edit reverses this for factorised networks (doc 46R.2). | Per-part edit is retrospective and single-pass. |

## 47.2 Proposed new records

| proposed ID | status | question and answer | evidence | limits |
|---|---|---|---|---|
| **I24** | C-I (method) | Can historical small networks be replayed bit for bit years of code later? **Yes**: 36/36 historical role-swap networks and all 2,196 reconstructed checkpoints reproduce bitwise; instrumented, resumed and 10-step window replays are also bitwise. | Doc 41R.3, `runs/swap_recon/qualification.json`; PR #34 independently reproduces 307/311 including 61 checkpoint hashes. | Same CPU stack. Other BLAS builds or hardware not tested. |
| **I25** | P (v1: see 47.5) | What separates near-perfect from failed generalisation under identical hyperparameters? Whether the declared block **stays factorised** (a value part + a unit part) through the escape from a shared plateau, or **retains a value×unit interaction** that is then frozen. Factorisation ≥ 0.98 is necessary (4/4 near-perfect generalisers) but not sufficient. | Doc 46 §46.1; 24 declared networks; 10-step window for 5. | Retrospective; causal tests P1–P3 proposed, not run. PR #34's crossing will test whether one random stream decides it. |
| **I26** | P | Is a near-perfect generaliser editable? **Yes**, perfectly, when the edit acts on the composable parts: 307 inversion and doubling 1.00 on familiar and unseen renderings, with fixed points preserved. Class-mean edits fail (0.14). | Doc 46R.2. | Activation-level edit. A permanent weight form of the per-part edit is not yet constructed. |
| **I27** | P | Does the factorial's canonical code control computation as soon as it is decodable? **No**: in C1.0, present and readable (≥ 0.87) at step 1,000, controlling only 0.14; control grows to 0.47 by step 6,000; in C0.1 it never exceeds 0.14. | Doc 46R.1; doc 41R.6. | Exploratory. |
| **Q23** | O, P0 (v1: see 47.5) | Can a targeted training-time intervention restore factorisation in a locked-in failure, or destroy it in a generaliser (reciprocal), and does generalisation follow? | Proposals P1–P3 (doc 46 §46.5). | Requires registration and approval. |

## 47.3 Implications for the program

**[E]** marks experimentally established results (registered or bitwise). **[R]** marks retrospective findings.
**[F]** marks future research opportunities.

### NRI (interpretation and editing)

- **[E]** Planted-mechanism validation is a usable gate for causal instruments. Untrained networks and complement
  controls behave as required on 2,304 cells across 96 networks; 24 cells in the n = 33 stratum exceed the
  untrained-network bound, as reported in doc 41R.7.
- **[R]** For editing, the *coordinates* of the edit matter more than locating the concept. Localisation and
  replacement succeed everywhere; transformation succeeds only in the network's own composable coordinates.
- **[R]** Probes fitted on familiar inputs can silently fail on unseen inputs.
- **[F]**
  - Build permanent per-part editors.
  - Test Q07 prospectively: predict edit success from geometry before fitting.

### Paper Three (preservation of computational authority)

- **[R]** The role-swap failures show *authority without generality*: the declared block has full causal
  authority on trained inputs (IIA 1.00) but encodes a lookup that cannot extend.
- **[R]** In 305, partial generalisation erodes after fitting (0.75 → 0.46) because nothing anchors unseen pairings.
  This is a mechanism of *authority loss without any competing task*, worth comparing with Paper Three's
  displacement.
- **[F]** Does protecting factorisation preserve generalisation under continued training? A direct analogue of I20's
  protection objective.

### Paper Five (concept lifecycle)

- **[R]** A concrete lifecycle with measurable stages:
  - formation (present, readable);
  - authority (controlling);
  - reorganisation (the escape: factorisation dips and recovers, or locks);
  - consolidation (frozen);
  - drift (305).
- **[R]** The 100-step and 10-step records with checked predictions are a template for Paper Five's prospective
  component.
- **[F]**
  - Pre-register the factorisation measure as a forecast of generalisation on fresh seeds and on PR #34's cells.
  - Q05 is still negative before the window.

### Project Genesis (direct construction)

- **[R]** Canonical codes can be aligned and transplanted between independently trained networks by a rotation
  fitted on class means (0.41 vs 0.47 within-network). This is a minimal positive result for Q09/Q12 at the
  representation level.
- **[R]** A factorised value + unit block plus a composing head is a *constructible* target. The decomposition is
  explicit (additive model R² ≈ 1.00 in 307), so the parts could be written directly rather than learned.
- **[F]** Construct a factorised block by hand and test whether a trained head composes unseen pairings: a direct
  construction test of C2.

### Chess Vision Studio (capability per parameter)

- **[R]** Under identical hyperparameters, the same 69.8K-parameter network ends either generalising perfectly or
  memorising, depending on a few hundred steps of development. Capability per parameter is limited by *which
  solution development locks in*, not only by capacity.
- **[F]** Monitor factorisation-like composability during training and intervene inside the window, if P1–P3 confirm
  causation. Strong decay in the factorial suggests one lever: pruning route-specific structure after fit.

## 47.5 Revisions after Phase IV and Phase V (v1)

The owner approved preparing these registry revisions for PR #36. They remain **proposals**: no registry record is
changed until the owner merges them through the registry's own process.

| ID | v0 proposal | v1 proposal | evidence | boundary |
|---|---|---|---|---|
| **I25** | P | **C-I for necessity during development** (setting: TinyTyped role-swap, mass declared); **C-N for sufficiency** | Doc 49 §49.4: rewarding the interaction from step 850 breaks 307 (0.99 → 0.35) and 311 (1.00 → 0.33), familiar 1.00, bitwise controls. Prevention gives only 0.52–0.61. | One run per condition; two generalisers. |
| **Q23** | O, P0 | **P**: disruption yes (both); rescue partial and only before lock-in; after consolidation factorisation is dispensable (311: 0.92 at factorisation 0.15) | Doc 49 §49.4, 46 runs | Rescue criterion never met. |
| **I26** | P | **P, extended**: exact weight-level programs for token and output edits; per-part value edits remain activation-level | Doc 49 §49.6; doc 52 §52.6 | Programs act on inputs and outputs, not the core operator. |
| **Q06** | O, sharpened | **P** | Programmability on unseen inputs tracks generalisation (ρ = 0.96, 7 networks) | Exploratory. |
| **Q12** | P | **P, negative within role-swap**: affine maps and direct head transplants fail even on familiar inputs (≤ 0.43) and at chance on unseen | Doc 49 §49.2; doc 52 §52.5 | A familiar-only repair cannot select composition over interpolation (doc 52 §52.5). |

### Proposed new records (v1)

| proposed ID | status | question and answer | evidence | limits |
|---|---|---|---|---|
| **I28** | C-I (exact) | Can semantic token edits and output relabellings be compiled permanently into weights? **Yes**: embedding-row replacement and output-row permutation are identities of functions. They compose, and they invert bit for bit. | Doc 49 §49.6; doc 52 §52.6; full 9,126-prompt domain | A mass-only value edit has no exact one-step linear form. |
| **I29** | P | Can a memorising network be repaired without training or oracle information? **For one of three** (302: 0.19 → 0.93, familiar 1.00), using trained classes, bijectivity of unit rescaling and the network's own geometry. | Doc 49 §49.1 | Post-registration method; fails on 304. |
| **I30** | P | What does a composing decoder compute? In 307 and 311, hidden pre-activations are 0.90/0.92 additive in per-token discrete-log phases at one frequency coprime to 12; failures read a canonical code or use k = 2. | Doc 52 §52.3 | No compact exact program; the reading as phase-addition is proposed. |
| **I31** | P | When does the composing decoder form? Gradually, together with fitting (307: steps 900–1,400), not after it; never in failures. | Doc 52 §52.2 | Two generalisers; 311's composition is shared with its encoder. |
| **Q24** | O | Can a composing decoder be constructed from the task's mathematics, or transplanted with a structural prior, and does it give exact core-operator programming (e.g. F = M·A·S + 1 inside the head)? | Doc 52 §52.7 | Next registered work. |

### Implications (v1 additions)

- **NRI.** Programmability is bounded by generalisation. A network is programmable on unseen inputs exactly to the
  extent that it composes.
- **Paper Three.** Authority without generality has a decoder signature: canonical-reading decoders retain full
  authority on trained inputs and cannot be extended by input-side surgery. Removing their interaction lowers familiar
  accuracy to 0.33–0.40.
- **Paper Five.** The lifecycle now has a causal stage boundary. Before consolidation, the representation's format
  is causally necessary; after it, the decoder's training dynamics protect the composing solution (311).
- **Genesis.** Direct construction now has a concrete target and a negative baseline. Learned decoders are not
  transplantable, and an approximate program is not exact. A constructed decoder must come from the task's
  mathematics.
- **Chess Vision.** Monitoring is most useful before lock-in. Interventions after it fail, even when they restore the
  representation's measured structure.

## 47.6 Revisions after Phase VI (v2, proposals)

| ID | proposal | evidence | boundary |
|---|---|---|---|
| **Q24** | **C-I (constructive)** for construction and core-operator programming: an exact, training-free network for the task in the existing architecture; F = M²AS, MA²S, M±A, M/A compiled into the head; units, binding and decoding edited independently | Doc 54 §54.2–54.3, exhaustive, float32 bitwise-symbolic | Modular single-statistic expressions; one hidden layer. **C-N** for transplanting constructed parts into learned networks (§54.7). |
| **I32** (new) | P: the learned decoders' unexplained residual is essential readout structure made of pairwise token interactions (cross-operand in 307, unit conversion in 311, lookup in failures) | Doc 54 §54.4 | 7 networks. |
| **I33** (new) | P: learned generalisers share the construction's causal variables (factorised 2-D mass code, interchange-equivalent) but store them as a circle; exactness and robustness trade off | Doc 54 §54.5 | Exploratory. |
| **I34** (new) | C-I (method): a compiler from modular specifications to transformer weights, exact over primes 5–23, 2–3 declared variables, monomial, inverse, sum and difference expressions | Doc 54 §54.6 | Single-statistic expressions; larger domains sampled. |

## 47.7 Revisions after Phase VII (v3, proposals)

| ID | proposal | evidence | boundary |
|---|---|---|---|
| **I34** | extend: the compiler handles computational graphs with observable, programmable intermediates (C-I, exhaustive) | doc 58 §58.4 | single-statistic nodes; 13-valued variables |
| **I35** (new) | P: compiled and learned computations coexist in one network if learned writes into the compiled subspace are blocked; with scarce data an output-level preservation objective is also required | doc 58 §58.6 | one task, 3 seeds per condition |
| **Q25** (new) | O: can exact construction be made as robust as learning (distributed soft routing)? | doc 58 §58.3 | current gap about 10× |

## 47.8 Revisions after Phase VIII (v4, proposals)

| ID | proposal | evidence | boundary |
|---|---|---|---|
| **I35** | revise: the hybrid advantage is a **semantic-interface** effect (canonical unit-converted quantities at the readout), not a compiled-arithmetic effect; with matched parameters or compute, plain learning reaches competence in 1/3 seeds at 12–47× more FLOPs per success | doc 60 §60.1–60.2 | one task, 3 seeds; trainable count corrected to 152,860 |
| **I36** (new) | C-I (guarantee): structural read-only regions, write masks and deterministic output ownership make a compiled operation exact for **every** value of the trainable parameters; objective-based protection does not | doc 60 §60.3 [E, M] | output-gated networks generalised the learned operation less reliably (1/3 seeds) |
| **I37** (new) | P: read-only computational registers give fast, compositional learning of operations in their span (held-out combinations 1.00) with bitwise retention of all earlier modules; restricting learned modules to read only the semantic interface gives data efficiency (≤ 25% data) | doc 60 §60.5–60.6 | read-restriction reading is post hoc; 2–3 seeds |
| **Q26** (new) | O: how can a network learn a relationship **outside** the span of its compiled registers so that it generalises? (independent linear rule: memorised, held-out 0.01–0.20) | doc 60 §60.5 | — |
| **Q27** (new) | O: under what registered criterion can a rule be extracted from a module that is wrong on withheld inputs? (registered criterion failed; post-hoc recompilation generalised perfectly) | doc 60 §60.8 | — |
| **Q25** | update: routing redesign alone does not close the robustness gap (best 0.52 at 1% multiplicative noise); an independent unverified report locates the residual in the arithmetic decoder | doc 60 §60.7 | — |
| **R1** (new) | reproducibility: per-seed developmental outcomes of the historical networks are exactly reconstructible but not numerically stable; seed-level mechanistic claims describe observed trajectories, and their prevalence is unresolved | doc 61; ledger 71 | population-level reproducibility untested |

## 47.4 Change history

v0 (2026-11-25): first proposal set, for owner review. No registry record was changed.
v1 (2026-11-25): revisions after Phase IV and Phase V (§47.5), prepared for PR #36 as approved. Still proposals;
no registry record was changed.
v2 (2026-11-25): Phase VI proposals (§47.6). Still proposals; no registry record was changed.
v3 (2026-11-25): Phase VII proposals (§47.7). Still proposals; no registry record was changed.
