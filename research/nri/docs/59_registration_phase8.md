# 59 — Registration: Phase VIII, protected neural computation and modular architecture

Committed **before** any Phase VIII experiment runs.

**Authorisation.** The owner's Phase VIII INIT.

**Standing rules.**
- The large reliability study stays deferred.
- Original archives and earlier results are preserved.
- The failed sector instruments (doc 58 §58.2) are excluded from every mechanistic claim.
- Nothing is merged.

**Compute discipline.**
- The machine has 4 cores. At most 3 training workers run, each single-threaded, plus at most one single-threaded
  analysis process.
- Every training run checkpoints its full state (model, optimiser, RNG, history) every 500 steps and resumes from the
  last checkpoint.
- Completed runs are never rerun or replaced.
- Evaluation runs every 200 steps (Phase VII used 100) to reduce evaluation compute. Audit reproductions keep 100 steps
  so they can be compared with Phase VII.

**Labels.**
- Kinds of evidence: **[M]** mathematical guarantee, **[E]** exhaustive verification, **[S]** sampled verification,
  **[T]** observed in trained networks.
- Kinds of network: learned, constructed, hybrid-trained, recompiled.

## A. Audit of the Phase VII hybrid results

- **A1. Reproduction.** Re-run B, K2 and K5 (full data, seeds 0–2) and K2 and K5 (25% data, seed 0) with the unchanged
  `sci/hybrid.py`. Compare the complete metric histories with the stored JSONs. Report bitwise equality, or the maximum
  discrepancy.
- **A2. Parameter accounting.** For every Phase VII condition, report:
  - total parameters;
  - constructed (frozen) parameters;
  - structurally zeroed parameters (the K2/K5 write-block);
  - frozen-at-initialisation parameters (K3);
  - **effective trainable parameters** = total − constructed − zeroed − frozen-at-initialisation.

  Phase VII reported 191,516 trainable for K2/K5, which ignores the zeroed write-block. This is corrected here.
- **A3. Matched baselines.** Rerun B at the *effective* trainable count of K2/K5, with the same depth and heads and the
  width chosen to match within 2% (**B_p**). Also rerun B at the same training FLOPs as K5 (**B_c**): K5 spends extra
  forward/backward passes on its interchange loss, so B_c gets proportionally more steps.
- **A4. Cost accounting**, for every run in this phase:
  - analytic training FLOPs per step: forward, backward ≈ 2 × forward, and extra passes for auxiliary losses;
  - **dense FLOPs**, counted even for frozen or zeroed weights, because they are still multiplied;
  - wall-clock time (single thread);
  - peak resident memory;
  - steps, FLOPs and wall-clock time to competence.

  **Competence** = OP₂ familiar ≥ 0.99 **and** OP₂ unseen ≥ 0.95.
- **A5. Reporting rule.** Compiled sub-circuit correctness and end-to-end correctness are always reported separately.
  No compute reduction is claimed from optimiser steps alone; compute claims use FLOPs to competence.

## S. Source of the hybrid advantage

**Setup.** The same architecture and task as Phase VII H (d = 128, 8 heads, MLP 512, dual operation). Every condition
uses K2-style write protection for whatever is installed: learned components cannot write into installed dimensions.

| condition | what is installed and frozen | trainable |
|---|---|---|
| **S0** | nothing (= B) | everything |
| **S1 typed embeddings** | the Design-C embedding columns 0–63 (value one-hots, unit and force flags, OP flag key) and the positional code | every head (incl. 0–3), the MLP, the readout |
| **S2** | S1 + **binding**: the layer-1 copy head (each unit position receives its value's one-hot) | as S1, minus that head |
| **S3** | S2 + **canonical quantities and unit conversion**: correction heads and layer-2 gathers, so one-hot M, A, S at "=" | as S1, minus the installed heads |
| **S4** | S3 + **arithmetic**: the OP₁ ladder with gate (= K2) | as above |
| **S_rand** | the same frozen slices as S4, with the same write protection, but filled with seeded random values at Design-C's per-tensor RMS | everything outside the slices |

**Measures** (3 seeds, full data): steps and FLOPs to competence, final OP₁/OP₂ familiar and unseen accuracy, and
OP₁ sub-circuit exactness (S4 only).

**Prediction.**
- The acceleration is carried mainly by S3, the canonical semantic variables.
- S4 adds little for OP₂, but makes OP₁ exact.
- S_rand is no better than S0, so architectural isolation alone does not help.

## P. Read/write permissions and output ownership

**Architecture (ProtectedNet).** A TinyTyped-shaped network in which permissions are enforced *in the forward pass*: a
fixed mask multiplies every write path, so they are not restored after each step.

| permission | rule |
|---|---|
| **read-only region** | the compiled semantic dimensions (0–63). Any component may read them (unrestricted q/k/v/W_in columns). Only the compiled components may write them: every learned head's o rows and every learned MLP unit's W_out rows into 0–63 are masked to 0. |
| **independently trainable regions** | dimensions 64–127 are partitioned per learned module |
| **shared communication region** | dimensions 120–127, writable by every learned module |
| **output ownership** | logits = Σ_op g_op(x) · logits_op(x), where g is the op one-hot computed by a compiled op-flag head and logits_op is the owning module's readout. This is a deterministic multiplicative gate, an explicit architectural primitive. |
| **authorised operators** | only named compiled instructions (the Phase VI unit/op edits) may change compiled parameters; the trainer cannot address them |

**Guarantee [M].** For every input assigned to a compiled operation:
- the compiled module's inputs (read-only regions) are unchanged by any learned parameter values, because the write
  masks are structural;
- its output is the only non-gated term.

So **its answer is exact for every value of the trainable parameters.**

**Tests.**
- **P1 [E].** Verify the guarantee over the full domain for 20 random draws of all trainable parameters (default
  initialisation × 10 scale factors from 1 to 10⁴) and for adversarially optimised trainable parameters: 500 Adam steps
  *maximising* OP₁ loss. Report OP₁ exactness. The same tests are applied to K5 (objective-based protection) as the
  comparison.
- **P2 [T].** Train ProtectedNet on the dual task (full and 25% data, 3 seeds) against K5. Report the A4 measures,
  OP₁/OP₂ end-to-end accuracy and exactness.
- **P3.** Permission-violation audit: confirm that no gradient reaches any masked entry and that the masked writes are
  exactly 0 in every forward pass.

**Prediction.** P1: ProtectedNet stays exact under every trainable value; K5 fails under adversarial values. P2:
ProtectedNet matches K5 on OP₂ and is exact on OP₁ at both data levels.

## G2. Reusable compiled operators (extends doc 58 §58.4)

| feature | how it is implemented |
|---|---|
| **G2a common-subexpression elimination** | hash-consing of identical subtrees |
| **G2b lifetime analysis and register allocation** | a region is freed after its last use by a 13-unit clear (W_out = −1 on ReLU(x_d); exact for one-hot codes) and reused greedily (linear-scan allocation) |
| **G2c operator sharing / function call** | one weight-tied block applied repeatedly to an argument register (move-in, tied ladder, move-out); demonstrated by repeated squaring x^(2ⁿ) for n = 1–6 with **one** set of unique ladder parameters |
| **G2d conditional execution** | select(c, X, Y) = c·X + (1 − c)·Y on one-hot codes, exact with gated ReLUs, e.g. "if M = 0 then A else M + A" |
| **G2e multiple declared variables** | already exact in the 3-role compiler (doc 54); re-verified, with graph expressions over 3 variables |

**Measures:**
- dense parameters, unique parameters (with tying), nonzero parameters;
- FLOPs per prompt;
- residual width and depth;
- exactness [E], observability and interchange for every live intermediate.

**Expressions:** the 60 random trees of doc 58 (same seed), together with constructed cases that have shared
subexpressions.

## Q. Progressive acquisition of capabilities

**Task.** The operation token takes 6 values at position 5:

| operation | definition | how it enters the network |
|---|---|---|
| OP₁ | M·A·S | compiled |
| OP_add | M + A | compiled |
| OP₂ | M·A + S | learned in stage 1 |
| OP₃ | (M + A)·S | learned in stage 2; a different combination of compiled operators |
| OP₄ | (a·M + b·A + c·S + d) mod 13 | learned in stage 3; **independently generated**, with a, b, c, d drawn by a seeded RNG in a module the learner and extractor never import. Coefficients are revealed only to the verification oracle. |

**Architecture (ModularNet).** ProtectedNet plus:
- **compiled intermediate registers**, read-only: one-hot M·A and one-hot M + A, computed by compiled stages
  (graph compiler) before the learned modules;
- **one learned module per new operation**: a 2-layer MLP reading the read-only semantic and register regions, with its
  own private hidden width and readout, owning its operation's output through the gate.

After a stage, that module is **frozen and its output ownership locked**.

**Data.** Familiar renderings only. Additionally, for OP₂ to OP₄, **20% of canonical (M, A) pairs are held out** of
training entirely; they form the unfamiliar-composition test.

**Measures**, after every stage, for every previously established operation:
- accuracy on familiar prompts, unseen renderings and held-out compositions;
- exactness (compiled operations);
- bitwise identity of the outputs of earlier learned modules before and after later stages (retention).

**Conditions.**
- **Q-reg:** registers visible.
- **Q-base:** registers hidden; only M, A, S visible.

**Prediction.**
- Every earlier operation is retained bitwise [E].
- OP₂ and OP₃ are learned faster and generalise to held-out compositions only in Q-reg, by reusing the M·A and M + A
  registers.
- OP₄ (a linear relationship, not a register combination) is learned in both conditions, but does not benefit from
  the registers.

## E. Data and compute efficiency

**Learning curves.** Data fractions {25, 40, 55, 70, 85, 100}% of familiar (prompt, operation) pairs, with random
sampling, 2 seeds. Architectures:
- B;
- K5;
- ProtectedNet;
- ProtectedNet with M·A register (the Q-reg design restricted to OP₁/OP₂).

**Structured coverage.** At 25% and 40%, compare random sampling with *semantic-coverage sampling*: the same budget,
chosen so that every canonical (M, A, S) triple appears in OP₂ training as uniformly as possible.

**Diagnosis of the 25% failure.** For each run, report:
- coverage of semantic variable values;
- coverage of canonical (M, A, S) triples and (M, A) pairs in the OP₂ training set;
- **training-set accuracy** (high training accuracy with low familiar-held accuracy → overfitting; low training
  accuracy → optimisation);
- OP₁ output interference (end-to-end against sub-circuit);
- register use: ablate the register region at evaluation and measure the drop.

**Minimum data to qualify:** the smallest fraction at which a condition reaches competence in all of its seeds.

**Prediction.** The 25% failure is a *coverage of canonical operand combinations* failure for modules that must learn
the M·A interaction. With the M·A register, competence is reached at a lower fraction, and structured coverage
lowers it further.

## R2. Robust distributed routing (constructed; no optimisation)

**Base.** Design C in a d = 128, 8-head TinyTyped, so there is room for redundancy. Constructions:

| construction | description |
|---|---|
| **R2a spread scoring** | each routing head's query/key signal is replicated over all 16 head dimensions (8 rotary pairs), each with gain g/8; the score is unchanged and noise averages |
| **R2b redundant heads with consensus** | every routing head duplicated (2 copies), each OV scaled by ½; outputs sum |
| **R2c moderate magnitudes** | minimal argmax-exact gains (as R1), applied *after* R2a/R2b |
| **R2d alternative routing** | binding without content-dependent attention: positions are fixed, so binding reads position 1 → 0 and 4 → 3 through a fixed positional pattern (keys from positional codes only); unit-specific conversion is applied by the per-unit gate in the value path ((P_s − I) gated by unit flags through a second head) rather than by query selection |

**Controls separating architecture from rescaling:**
- the same parameter rescalings applied to the unmodified Design-C layout;
- R2a/R2b at the original gains.

**Measures:** noise curves (a), (b), (c) as in Phase VII; exactness at σ = 0 [E]; the instruction-system checks; dense
and nonzero parameters.

**Target:** ≥ 0.95 accuracy at noise (b) σ = 10⁻² while still exact at σ = 0.

**Prediction.** R2a + R2b + R2c reach the target under (b). Under (a), the target requires splitting the gain-carrying
q/k from the value weights into separate tensors (an architectural change), which is reported separately.

## MR. Mechanistic recompilation (learn → interpret → formalise → compile → verify → protect)

- **Learn.** The OP₄ module from Q (and, if available, OP₂ from Q-base).
- **Interpret.** Query the frozen learned module on all 13 × 13 × 3 canonical (M, A, S) register states, by direct
  register assignment through the read-only interface. This yields its I/O table. The extractor imports only the
  network and the compiler, never the rule generator.
- **Formalise.** Search the compiler's hypothesis space, g(φ_M(M) + φ_A(A) + φ_S(S)) with φ ∈ {integer value, discrete
  log} × integer coefficients in [−12, 12] and g ∈ {mod 13, exp mod 13}, plus zero gates. Choose the candidate with the
  highest agreement with the learned table.
  - **Acceptance:** agreement ≥ 0.99, and a unique best candidate.
  - **Negative control:** the same procedure on a module trained on a **random lookup table** (outside the hypothesis
    space). It must return "no candidate".
- **Compile** the accepted candidate into a protected module with the graph compiler.
- **Verify.** Exhaustively [E]:
  - behavioural agreement with the learned module, on all canonical states and all prompts;
  - causal equivalence: interchange on each register gives identical outputs for the compiled and the learned module;
  - agreement with the independent oracle (the generator's rule, used only here).
- **Protect and extend.** Replace the learned module with the compiled one, then learn OP₅ = OP₄(M, A, S)·M (a new
  module reading the OP₄ register). Compare against learning OP₅ with the learned OP₄ module left in place but no
  register exposed.
- **Claim rule.** "Recovered without access to the answer" is claimed only if extraction succeeds on the hidden rule
  *and* the negative control is correctly rejected.

## Order of execution and budget

1. **First:** A1–A4, S, P1, R2, G2. The constructions are cheap; the training uses about 30 runs.
2. **Next:** P2, E, Q, MR (about 60 runs).

All training uses at most 3 workers. Every experiment is evaluated only after this document is committed. Deviations are
logged in the ledger.
