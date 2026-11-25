# 62 — Registration: Phase IX, read access, pathway selection and reuse

**Status.** Draft registration for owner approval, committed **before** any Phase IX experiment runs. No Phase IX
training or evaluation starts until the owner approves this document. Phase VIII is closed (owner approval, 25 Nov
2026). PR #36 stays unmerged.

**Standing rules carried over:**
- reliability study deferred;
- archives preserved;
- registration before evaluation;
- every negative result and deviation logged (doc 43);
- at most 3 single-threaded training workers;
- resumable checkpoints;
- failed sector instruments excluded from mechanistic claims;
- nothing merged.

## 62.0 Context: updated findings from the independent tracks

These are recorded as reported. **Nothing below is verified by this program unless marked so.** The tracks stay
independent: Phase IX imports no code from them, and their results cannot change Phase IX's design after this
document is committed.

| track | latest report (owner-relayed, 25 Nov 2026) | status here |
|---|---|---|
| **Track C** (polynomial attention → decoder continuation) | Two decoder constructions, **restored ternary arithmetic** and **factored binary arithmetic**, report 100% accuracy at 1% tensor-relative additive and 5% multiplicative noise, over 20 fresh perturbation seeds per condition. The factored decoder uses 8,125 parameters (original decoder 33,920), retains exact intermediate product semantics and preserves the instruction system in clean evaluation. | **Unverified.** The package is not yet available to this session. Verification is §62.V; the manuscript's robustness conclusions do not change until it passes. |
| **Track A** (polynomial packing, through MM-02AC) | Restricted shared-weight packing gives about 4.39% nominal arithmetic savings on a 1.44M-weight synthetic workload, but is slower than direct GEMM. | Unverified. GEMM stays the default. Liaison question in §62.T. |
| **Track B** (CPU-01) | Frozen activation reuse and several data-path optimisations qualified; no reliable end-to-end training acceleration established. Numerical-reproducibility findings remain central (doc 61). | Doc 61: this program's part is verified and committed. |
| **Longitudinal reuse experiment** (separate, about 5M parameters) | At its latest checkpoint, an acquired primitive was preserved bitwise and supported perfect downstream transfer on familiar states, but failed on new states. | Independent; continues separately. Cited only as context. |

**Numerical-reproducibility qualification (doc 61; ledger 71).**
- Exact historical reconstruction holds.
- Numerical stability of per-seed outcomes fails for the historical generalisers.
- Population-level reproducibility is untested.

Therefore Phase IX:
- runs every cell in **one fixed numerical environment**: the original trainer path (no numeric-tier speedups),
  PyTorch 2.15.0, 1 thread, deterministic algorithms;
- compares conditions **only as distributions over seeds**;
- replays 3 randomly chosen completed runs to confirm bitwise reproducibility (§62.6).

## 62.1 Question and hypothesis

**Primary objective:** understand how networks *select* reusable computational pathways, and what is needed to
discover genuinely unfamiliar algorithms.

**Hypothesis H-read** (from doc 60 §60.6, post hoc there; tested prospectively here):
- providing semantic variables is not enough when the network can bypass them through rendering-specific features;
- restricting read access matters as much as protecting write access.

**Four properties of a variable or register, kept distinct:**

| property | definition | measured by |
|---|---|---|
| **availability** | the variable's value is present in the residual stream | by construction (installed), or an exact linear readout (learned) |
| **accessibility** | the learned component's read mask permits reading it | structural fact |
| **causal sensitivity** | ablating it changes the output | fraction of OP₂ predictions changed by mean-ablation of the region |
| **meaningful reuse** | the output follows the variable's *semantics* under intervention, including on withheld combinations | interchange accuracy: set the region to the value from a source input; score against the answer implied by the swapped value |

None is assumed to imply the next.

## 62.2 Design: a matched 2 × 2 × 2 × 2 factorial

**Task.** The dual-operation task (doc 58 addendum A):
- OP₁ = M·A·S;
- OP₂ = M·A + S;
- domain 18,252 prompts;
- unseen renderings = unit 2 with values 9–12.

**Additionally:** 20% of canonical (M, A) pairs are withheld from OP₂ training, using the same seed (59) and
procedure as Q (doc 60 §60.5). This gives the **withheld-combination** test.

**Network.** The fixed shell is d = 128, 8 heads, MLP 512: 269,952 dense parameters in every cell. It has two parts:
- **The encoder** is attention heads 0–3, layers 1–2. It forms the semantic variables and may read any position and
  dimension.
- **The operation module** is heads 4–7 (both layers), the MLP and the readout. It computes the operations.

Region layout:

| region | dimensions |
|---|---|
| Design-C semantic code | 0–63 |
| registers | 64–89 (M·A one-hot 64–76, M + A one-hot 77–89) |
| op flag / key | 51–52 (as in Phase VII) |
| learned working space | 90–127 |

| factor | level + | level − |
|---|---|---|
| **CV: canonical variables** | encoder = Design C, installed and frozen: exact one-hot M, A, S at "=" (dims Q, A, FU) | encoder heads 0–3 have the same shapes but are trainable from random initialisation, and are trained end to end. The canonical slots exist, but contain whatever the encoder learns. |
| **R: rendering-specific read access** | permitted: the operation module reads every dimension at every position (as in Phase VIII P/K5s) | **prohibited:** the operation module's heads may attend only to the "=" position (attention mask). Its q/k/v and the MLP W_in read only the canonical slots (Q, A, FU at "="), the registers and the op flag. Raw values at other positions, unit flags (MU, AU), NULL and POS are unreadable. |
| **REG: compiled registers** | M·A and M + A compiled from the canonical slots (graph-compiler ladders, frozen) into 64–89 | dims 64–89 are present but never written (zero) |
| **PROT: protection of installed parts** | **structural**: write masks block the operation module from writing into 0–63 and 64–89, and OP₁'s answer is owned by its compiled ladder through the deterministic output gate (Phase VIII P) | **objective**: no write masks and no gate. The K5 semantic-preservation (interchange) loss on the installed parts, as Phase VIII K5s. |

**Held constant.**
- The shell (dense parameters and FLOPs per forward pass are identical except the PROT− interchange passes).
- The data distribution and sampler: the random sampler for all fractions; no structured sampler (doc 60 item 71).
- The optimiser: AdamW, lr 2e-3, wd 0.01, batch as Phase VIII.
- The training budget: 6,000 steps.
- The evaluation cadence: every 200 steps.

The OP₁ ladder is installed in every cell and reads the canonical slots, so it is exact only when CV+ or when the
encoder learns an exact canonical code.

**Unavoidable differences, reported rather than hidden:**
- *Effective trainable parameters* differ by cell. CV− trains the encoder; masks zero different entries. They are
  reported per cell with the A2 accounting.
- PROT− costs about 1.95× FLOPs per step.

Cost comparisons therefore use FLOPs, never steps.

**Cells and runs.**
- 16 cells × 3 data fractions {25%, 55%, 100%} × 3 seeds = **144 runs**.
- About 5 h at 3 workers, more for PROT− cells. Checkpoints every 500 steps; resumed after harness limits.

**Construction validation (allowed before outcome evaluation; it touches no OP₂ outcome):**
- installed parts exact at initialisation, for every cell with CV+ and REG+;
- read-mask audit: zero gradient to masked input columns, and attention weight exactly 0 off the "=" position for R−;
- write-mask audit as P3.

## 62.3 Measures

For every run, at every evaluation:
- OP₂ familiar accuracy; OP₂ unseen renderings; OP₂ **withheld canonical combinations**;
- OP₁ end to end and sub-circuit;
- training accuracy.

**Derived measures:**
- **Competence:** OP₂ familiar ≥ 0.99 and unseen ≥ 0.95.
- **Compositional competence:** competence and withheld ≥ 0.95.
- **Data requirement:** the minimum fraction at which every seed is competent (and compositionally competent).
- **Cost to competence:** analytic training FLOPs, and FLOPs spent per competent network including unsuccessful
  seeds. Wall-clock and RSS are recorded but are machine-specific.

**Pathway measures** (every 400 steps on fixed evaluation subsets; final values exhaustive):

| measure | intervention | reads as |
|---|---|---|
| canonical causal sensitivity | mean-ablate the canonical slots at "=" | fraction of OP₂ predictions changed |
| register causal sensitivity | mean-ablate the register region (REG+) | fraction changed |
| register **reuse** (IIA) | write the register value of a source input with a different M·A; target = (M·A)_source + S | interchange accuracy, familiar and withheld |
| canonical reuse (IIA) | swap the M slot from a source input | as above |
| **rendering invariance** (bypass test) | replace the prompt by another *rendering* of the same canonical (M, A, S) | fraction of predictions unchanged; a network that computes only through canonical variables is invariant |
| bypass share | 1 − rendering invariance on familiar prompts whose canonical triple was trained | — |

**Pathway selection** is the time course of these measures. The questions:
- when does register dependence appear, relative to competence;
- does it appear at all when bypass is possible (R+)?

## 62.4 Predictions (registered)

1. **H-read.** With CV+, R− qualifies at **≤ 25%** data in all PROT/REG cells, while R+ needs ≥ 55% or fails
   (replicating doc 60 §60.6 prospectively).
2. **Availability ≠ use.** In CV+ R+ REG+ cells, register causal sensitivity and register reuse are **low** (< 0.5),
   and the network shows bypass. In CV+ R− REG+ cells, register reuse is ≥ 0.9 on familiar **and withheld**
   combinations.
3. **Compositional generalisation needs REG+ and R−.** Withheld-combination accuracy ≥ 0.95 only in CV+ R− REG+ cells.
   It stays ≤ 0.3 elsewhere, including CV+ R− REG− (canonical one-hots alone do not compose M·A over unseen pairs).
4. **Learned encoders under a read bottleneck.**
   - CV− R− learns more slowly than CV+ R−. When it reaches competence, it is rendering-invariant (≥ 0.95) and its
     canonical slots become exactly decodable.
   - CV− R+ behaves like the Phase VIII baseline (fails).
   - CV− REG+ registers are not reused unless the encoder's slot code becomes one-hot-canonical.
5. **Protection.**
   - Structural protection keeps OP₁ exact end to end in every CV+ cell, at every fraction.
   - Objective protection does so at full data but not reliably at 25% (Phase VII/VIII).
   - On OP₂, PROT has a smaller effect than R.

## 62.U Unfamiliar rules

Sub-experiment on the same shell, 3 seeds, 100% data, 20% withheld pairs. Target: an independently generated rule.
- It is generated by a seeded generator module, which the trainer imports only for labels and which no extractor ever
  imports.
- Its form is outside the span of the registers: (a·M + b·A + c·S + d) mod 13, with new coefficients from seed 990001.
- Cells:
  - (i) CV+ R− REG+ PROT+;
  - (ii) CV+ R+ REG− PROT−;
  - (iii) CV− R+ REG− PROT−;
  - (iv) CV+ R− REG+ PROT+ with an **additional compiled representation register**: M and A as integer-valued
    linear (non-one-hot) codes, i.e. a structured, ordered input representation.

**Measure:** withheld-pair accuracy.

**Prediction:** (i)–(iii) memorise (withheld ≤ 0.3). (iv) is the registered test of whether an ordered representation
lets a rule outside the register span generalise. No prediction of success is made, and a success is reported as such.

**Claim rule:** "discovers an unfamiliar algorithm" only if withheld-pair accuracy ≥ 0.95 in all seeds of a cell.

## 62.V Verification of Track C's decoder results (independent; precedes any manuscript change)

**On receipt of the complete package:**
1. **Source pin.** Record the package's source revision and hashes. Confirm that its NRI base revision exists in this
   repository and that the files it depends on are unchanged.
2. **Re-run.** Run their implementation unmodified, at the pinned revision, in this environment.
3. **Independent checks**, using *this program's* harness (`sci/robust.py` curves) rather than theirs:
   - clean exactness on all 9,126 prompts (argmax, and logits exactly one-hot where they claim it);
   - unseen renderings;
   - parameter count (dense and nonzero; expected 8,125 for the factored decoder and 33,920 for the original);
   - intermediate product semantics: exact readout and interchange of the intermediate;
   - the instruction-system checks of doc 54 / doc 58 on the clean network;
   - noise: **20 fresh perturbation seeds per condition, chosen here** (seeds 620000–620019), at 1% tensor-relative
     additive and 5% multiplicative noise. Confirm that their noise definitions match model (a)/(b), or report the
     difference.
4. **Independent reimplementation**, where their written construction suffices: rebuild both decoders from the
   description without their code, and repeat step 3.

**Acceptance (all required):**
- step 3 reproduces ≥ 0.95 accuracy (their claim: 1.00) at both noise conditions, in all 20 seeds, for each decoder;
- clean exactness 9,126/9,126;
- the parameter counts match;
- the instruction checks pass.

Reported as **verified**, **partially verified** (with what failed), or **not verified**. Until verified, the
manuscript keeps "Robustness remains unresolved", and the Track C findings stay labelled unverified.

## 62.T Track A liaison question (after §62.V)

Does the verified factored decoder offer a better *structured-execution* workload for Track A's packing than
synthetic GEMM? Measure:
- the decoder's matrix shapes, sparsity and repeated-structure counts;
- the share of training and inference FLOPs it represents;
- whether Track A's restricted shared-weight pattern applies.

This is an analysis handed to Track A. GEMM remains the production default, and no packing code enters this program's
training path.

## 62.6 Analysis rules

- Every cell is reported, with all seeds and fractions.
- Factor effects are estimated as cell-level differences in the rate of competent seeds and in median FLOPs to
  competence. There are 3 seeds per cell, so these are descriptions, not population estimates. An effect is called
  "clear" only if it holds in every seed pair.
- Post-hoc analyses are labelled.
- Bitwise replay of 3 randomly drawn completed runs (seed 62 draw) confirms the fixed-environment tier.
- Deviations are logged as ledger items 76 and later.
