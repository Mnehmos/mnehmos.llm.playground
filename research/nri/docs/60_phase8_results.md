# 60 — Results: Phase VIII, protected neural computation and modular architecture

**Registration.** Doc 59 (commit ae9ec28), committed before any Phase VIII experiment ran.

**Data and code.** Data: `runs/phase8/`. Code:
- `sci/p8_audit_repro.py`: A1;
- `sci/p8_cost.py`: A2, A4;
- `sci/p8.py`: S, A3, P2, E;
- `sci/p8_p1.py`: P1, P3;
- `sci/graph2.py`: G2;
- `sci/routing2.py`: R2;
- `sci/modular.py`, `sci/p8_q.py`, `sci/p8_rules.py`: Q, E-modular;
- `sci/p8_extract.py`, `sci/p8_mr_verify.py`: MR.

**Labels.**
- Kinds of evidence: **[M]** mathematical guarantee, **[E]** exhaustive verification, **[S]** sampled verification,
  **[T]** observed in trained networks.
- Kinds of network: learned, constructed, hybrid-trained, recompiled.

**Compute.**
- At most 3 single-threaded training workers ran on a 4-core machine.
- Wall-clock times are specific to this machine and its load. FLOPs are analytic (training ≈ 3 × forward, plus extra
  passes for interchange losses) and are the primary cost measure.

## Plain-language summary

**The central question: can networks accumulate capabilities by learning, compiling, protected storage and semantic
interfaces, without relearning or compromising earlier computations?** Within this task family: **yes for retention,
partly for acquisition.**

- **Nothing previously established was ever lost.**
  - Every compiled operation stayed exact, and every earlier learned module stayed bitwise identical, through every
    later stage.
  - The structural guarantee held against random and adversarial parameter values.
- **Reuse is what helps.**
  - New operations that are combinations of compiled quantities were learned in 200–400 steps and generalised to
    operand combinations never seen in training.
  - An operation that was not such a combination was learned on familiar data but did **not** generalise.
- **The Phase VII hybrid advantage is real but was partly mis-described.**
  - It reproduces bitwise.
  - Its trainable-parameter count was overstated by 25%.
  - It comes almost entirely from the **canonical semantic variables** (unit-converted M, A, S at "="), not from the
    compiled arithmetic.
- **Several registered targets were not met.** All are reported as failures:
  - distributed routing robustness;
  - output-gated protection matching K5 on the learned operation;
  - the registered extraction criterion;
  - learning from 25% data.

## 60.1 Audit of the Phase VII hybrid results (A1–A4)

### A1. Reproduction — bitwise [E]

Re-running the unchanged `sci/hybrid.py` reproduced **all 11 audited runs with bitwise-identical metric histories**
(maximum difference 0.0). The audited runs were B, K2 and K5 on full data with seeds 0–2, and K2 and K5 on 25% data
with seed 0. Competence steps match Phase VII:

| condition | competence step | 25% data |
|---|---|---|
| B | never | — |
| K2 | 600 / 800 / 700 | never |
| K5 | 600 / 600 / 600 | never |

### A2. Parameter accounting — correction to Phase VII

| condition | total | constructed (frozen) | write-blocked (zeroed) | frozen at init | **effective trainable** | Phase VII reported |
|---|---|---|---|---|---|---|
| B | 269,952 | 0 | 0 | 0 | 269,952 | 269,952 |
| C | 269,952 | 78,436 | 0 | 0 | 191,516 | 191,516 |
| K2 | 269,952 | 78,436 | 38,656 | 0 | **152,860** | 191,516 |
| K3 | 269,952 | 78,436 | 0 | 38,656 | **152,860** | 191,516 |
| K5 | 269,952 | 78,436 | 38,656 | 0 | **152,860** | 191,516 |

Phase VII counted the structurally zeroed write-block (38,656 entries) as trainable. The correct effective count for
K2, K3 and K5 is **152,860**. This is a reporting error in doc 58; doc 58 is not altered, and the correction is entered
in the ledger (doc 43, item 64).

**Dense FLOPs.**
- Every condition multiplies the same dense tensors: 2.43 M forward FLOPs per prompt.
- B, K2 and K3 cost 1.86 × 10⁹ training FLOPs per step.
- K5's interchange loss adds forward/backward passes: **3.63 × 10⁹ per step, 1.95× B.**

### A3/A4. Matched baselines and cost to competence

The fixed task is OP₂ (M·A + S) in the dual-operation network. Competence = OP₂ familiar ≥ 0.99 and OP₂ unseen ≥ 0.95.

| condition | trainable | FLOPs / step | seeds competent | steps to competence | **FLOPs to competence** |
|---|---|---|---|---|---|
| B (Phase VII) | 269,952 | 1.86e9 | 0/3 in 6,000 steps | — | > 1.1e13 |
| **B_p** (parameter-matched, d = 96) | 153,312 | 1.06e9 | 1/3 | 4,200 | 4.5e12 |
| **B_c** (compute-matched to K5: 12,000 steps) | 269,952 | 1.86e9 | 1/3 | 8,600 | 1.6e13 |
| K2 | 152,860 | 1.86e9 | 3/3 | 600–800 | 1.1e12–1.5e12 |
| K5 | 152,860 | 3.63e9 | 3/3 | 600 | 2.2e12 |
| S3 (§60.2) | 160,640 | 1.86e9 | 3/3 | 600–800 | 1.1e12–1.5e12 |

**Reading.**
- With matched parameters, plain learning reaches competence in 1 of 3 seeds, at 4,200 steps.
- With K5's compute budget (12,000 steps), it reaches competence in 1 of 3 seeds, at 8,600 steps.
- The hybrids reach competence in every seed, with **about 3–15× fewer training FLOPs**: K2 1.1–1.5e12 and K5 2.2e12,
  against 4.5e12 and 1.6e13 for the single successful matched-baseline seeds.

This is a FLOPs comparison, not a step count, as the INIT requires.

**Sub-circuit versus end-to-end.**
- OP₁'s compiled sub-circuit is exact at every evaluation in K2/K5/S4/P.
- End-to-end OP₁ accuracy is a separate measure; it is 1.000 in all K5 and P runs.
- In S1–S3, OP₁ has no compiled sub-circuit (sub-circuit exactness 0) and is learned end to end.

**Wall-clock and memory.** On this machine, under load, the single-thread wall-clock to competence was:
- 33–63 s for S3/S4/K2;
- 57–59 s for K5;
- 280 s (B_p) and 834 s (B_c) for the one successful seed of each.

Peak resident memory was 635–698 MB in every run.

## 60.2 Source of the hybrid advantage (S ablation)

Same architecture and task. Each level installs and freezes more of Design C, with K2-style write protection.

| level | installed | seeds competent | steps to competence | final OP₂ familiar / unseen (3 seeds) |
|---|---|---|---|---|
| S1 typed embeddings | embedding columns 0–63 + positional code | 0/3 | — | 0.97–1.00 / 0.83–0.88 |
| S2 + binding | + layer-1 copy head | 1/3 | 3,800 | 1.00/0.98; **0.43/0.32**; 0.98/0.86 |
| **S3 + canonical quantities and conversion** | + correction heads and layer-2 gathers | **3/3** | **600 / 800 / 800** | 1.00 / 0.97–1.00 |
| S4 + arithmetic (= K2) | + OP₁ ladder with gate | 3/3 | 600 / 1,200 / 800 | 1.00 / 0.97–1.00 |
| S_rand | S4's slices, frozen, with **random** values at the same RMS | 0/3 | — | 0.11–0.15 / 0.09–0.10 |

**Predictions** (doc 59): all three confirmed [T].
- **The acceleration is carried by S3**, the canonical semantic variables.
- **S4 adds nothing for OP₂** (its steps-to-competence range overlaps S3's), but it makes OP₁ exact. OP₁'s sub-circuit
  is exact only in S4; in S3, OP₁ is learned end to end (familiar and unseen 0.999–1.000).
- **S_rand is worse than no installation.** It does not even fit its training set (training accuracy 0.11–0.15, an
  optimisation failure). Frozen, write-protected random features obstruct learning, so architectural isolation alone
  does not help.

So **the hybrid advantage is a semantic-interface effect.** Supplying the unit-converted operands at the readout
position makes the remaining function (M·A + S over canonical one-hots) easy to learn. Typed embeddings alone, or with
binding, are not enough.

## 60.3 Read/write permissions and output ownership (P1–P3)

### P1. Guarantee regardless of trainable parameters [E over the full domain; M by construction]

| network | OP₁ end-to-end, 20 random parameter draws (scales 1 to 10⁴) | after 500 adversarial Adam steps maximising OP₁ loss | OP₁ sub-circuit |
|---|---|---|---|
| **P** (structural read-only + write masks + output-ownership gate) | **1.000 (minimum)** | **1.000** | exact in all |
| S4 (= K2/K5 architecture, no output gate) | 0.067 (minimum) | 0.070 | exact in all |

**Answer to the INIT question: correctness of a compiled operation can be guaranteed regardless of the trainable
parameters,** but only with deterministic output ownership.
- Write protection keeps the compiled sub-circuit exact in both networks.
- Without the gate, learned components can still overwrite the answer at the logits.
- K5's protection is a training objective and therefore cannot constrain arbitrary parameter values. This is shown
  directly by S4, which has K5's architecture.

### P3. Permission audit [E]

- **Zero** nonzero gradients reach any blocked entry.
- **Zero** nonzero effective writes into write-blocked entries.
- Both hold in P and in S4.

### P2. Training (3 seeds; full and 25% data)

| condition | FLOPs / step | full data: competent | full: OP₂ unseen | 25%: OP₂ familiar / unseen | OP₁ end-to-end (all runs) |
|---|---|---|---|---|---|
| **P** | 1.86e9 | 1/3 (800 steps) | 1.00 / 0.88 / 0.84 | 0.43–0.52 / 0.15–0.20 | **1.000** |
| K5s (K5 objective on the same masks) | 3.63e9 | 3/3 (600 steps each) | 0.98 / 1.00 / 1.00 | 0.51–0.57 / 0.21–0.30 | 1.000 |

**Prediction "P matches K5 on OP₂": not supported.**
- P is exact on OP₁ at both data levels, with half the per-step compute.
- On full data, however, P generalises OP₂ to unseen renderings in only 1 of 3 seeds.
- A post-hoc hypothesis, not tested: K5's interchange loss acts as a regulariser on the learned pathway in addition to
  protecting OP₁.

At 25% data both conditions fail, with **training accuracy 1.00 against familiar 0.43–0.57**. This is overfitting, not
an optimisation failure (see §60.6).

## 60.4 Reusable compiled operators (G2)

All 63 expression trees were compiled in 4 variants: {CSE off/on} × {register allocation off/on}. The trees were the 60
random trees of doc 58 plus 3 constructed shared-subexpression cases. Every variant is **exact and observable** [E],
63/63 each.

| variant | mean nodes | mean residual width D | max D | mean dense params | mean nonzero | mean decoder FLOPs/prompt |
|---|---|---|---|---|---|---|
| baseline | 5.59 | 136.6 | 324 | 118,011 | 4,075 | 95,002 |
| CSE | 5.41 | 134.4 | 298 | 115,513 | 3,998 | 90,031 |
| register allocation | 5.59 | 122.6 | 259 | 129,580 | 4,195 | 118,170 |
| CSE + register allocation | 5.41 | **120.3** | **233** | 125,953 | 4,113 | 110,949 |

- **CSE (hash-consing)** removed nodes in 8 of 63 trees. It saves width, parameters and FLOPs, but only where subtrees
  repeat.
- **Lifetime analysis and register allocation** (exact 13-unit clears, linear scan):
  - reduced D in 30 of 63 trees, by 14 on average and by up to 65;
  - **costs** about 10% more dense parameters and about 23% more FLOPs, because each clear is an extra ReLU stage.

  It trades width for compute. This matters only when width is the binding constraint.
- **Conditional execution** (select): 3/3 cases exact, e.g. `if M = 0 then A else M + A`. Select operands must be
  13-valued (an S-operand test case was replaced, ledger item 66).
- **Operator sharing / function call:** a weight-tied loop block computes x^(2ⁿ) exactly for n = 1–6 with **7,510
  unique decoder parameters**, against 29,595 when unrolled at n = 6. Cost grows with n in FLOPs, not in unique
  parameters.
- **Multiple variables:** re-verified (all trees use M, A, S).
- Observability and interchange hold for every live intermediate within its live interval. After a clear, a region
  belongs to its next occupant.

## 60.5 Progressive acquisition with retention (Q)

**Network.** ModularNet: a protected encoder, a compiled stage (OUT1 = M·A·S, OUT_add = M + A), and read-only registers
X = M·A and Y = M + A. One learned module per new operation reads only read-only dimensions, owns its operation's
output through the exact op gate, and is frozen after its stage.

**Data.** Familiar prompts only. 20% of canonical (M, A) pairs are withheld from every learned operation's training.

| stage | operation | Q-reg (registers visible): steps / held-out-pair accuracy | Q-base (only M, A, S): steps / held-out |
|---|---|---|---|
| 1 | OP₂ = M·A + S | 200–400 / **1.00** | 1,200–1,400 / 0.12–0.15 |
| 2 | OP₃ = (M + A)·S | 200 / **1.00** | 1,600–1,800 / 0.00–0.01 |
| 3 | OP₄ = (aM + bA + cS + d) mod 13, independently generated | 600–800 / **0.01–0.03** | 800–1,000 / 0.17–0.20 |
| control | OP_R (random table) | familiar 1.00 / held-out 0.06–0.14 (chance ≈ 0.08) | — |

Familiar and unseen-rendering accuracy are 1.00 for every operation and condition.

- **Retention [E]:** after every stage, OP₁ and OP_add are exact and every earlier learned module's outputs are
  **bitwise identical** (6/6 runs, all stages). Prediction confirmed.
- **Reuse [T]:** combinations of compiled quantities are learned **3–9× faster** and **generalise to unseen operand
  combinations** only when the registers are visible. Prediction confirmed.
- **Learning an unknown relationship versus recombining operators [T]:** OP₄ is learned on familiar data in both
  conditions. In both it fails to generalise to the withheld pairs: it **memorises** the trained pairs. That
  generalisation is no better than chance with registers and only slightly better (0.17–0.20) without them.

  Prediction "OP₄ does not benefit from registers": confirmed for speed. The registers in fact slightly *hurt* held-out
  generalisation. The architecture accumulates capabilities that are compositions of what it already has. It does not,
  from this data, induce a new rule outside that span.

## 60.6 Data and compute efficiency (E)

### E-modular: protected network with the M·A register (Q-reg restricted to OP₂)

2 seeds per cell. Each cell gives the number of seeds competent and the steps to competence.

| data | with M·A register: random | register: structured | no register: random | no register: structured |
|---|---|---|---|---|
| 25% | 2/2, 400 | 2/2, 200 | **0/2** (familiar 0.97–0.99, unseen 0.95–0.96) | 2/2, 1,600–1,800 |
| 40% | 2/2, 200–400 | 2/2, 200 | 2/2, 2,200–3,200 | 2/2, 1,400–1,600 |
| 55%–100% | 2/2, 200 | — | 2/2, 1,600–2,400 | — |

**Minimum data to qualify** (all seeds competent):
- with register: **≤ 25%** (random or structured);
- without register: **40%** random, **≤ 25%** structured.

**Diagnosis.**
- (M, A) pair coverage is 1.00 at every fraction.
- Canonical (M, A, S) triple coverage under random 25% sampling is 0.96–0.98. Structured sampling raises it to 1.00,
  which is what turns the no-register 25% runs from near-misses into competent runs.
- Training accuracy is 1.00.

So the residual 25% failure *without* the register is a **coverage of operand combinations** effect, as registered. With
the register it disappears.

### E: B, K5s and P on the dual-operation transformer

**PENDING.** The group is running (48 runs: 3 conditions × 6 fractions, plus structured 25%/40%, × 2 seeds). This
section is filled in when the group completes.

## 60.7 Robust distributed routing (R2, constructed)

All variants are exact at σ = 0 [E]. Accuracy is under noise model (b), multiplicative weight noise, with 3 seeds per σ.

| variant | dense / nonzero params | σ = 10⁻³ | 3·10⁻³ | **10⁻²** |
|---|---|---|---|---|
| Design C in d = 128 (base) | 204,935 / 1,291 | 0.43 | 0.28 | 0.20 |
| R2a spread scoring (original gains) | 204,935 / 1,634 | 0.54 | 0.35 | 0.23 |
| R2b redundant heads, consensus (original gains) | 204,935 / 1,571 | 0.52 | 0.36 | 0.30 |
| R2a + R2b | 204,935 / 2,257 | 0.53 | 0.36 | 0.23 |
| R2a + minimal gain (R2c) | 204,935 / 1,634 | 1.00 | 0.98 | 0.51 |
| R2b + minimal gain | 204,935 / 1,571 | 1.00 | 0.98 | 0.52 |
| R2a + R2b + minimal gain | 204,935 / 2,257 | 1.00 | 0.98 | **0.52** |
| R2d alternative routing (positional binding + AND-unit conversion) | 91,453 / 2,413 | 0.27 | 0.20 | 0.18 |
| *control: Phase VII R1 (minimal gain on unmodified Design C)* | — | — | — | *0.46* |

**Registered target (≥ 0.95 at (b) σ = 10⁻²): not met.**
- The best variant reaches 0.52, against 0.46 for the rescaling-only control. So the architectural additions
  (spreading, redundancy) contribute about +0.06 on top of rescaling. Most of the Phase VII→VIII gain is rescaling.
- Under (a), RMS-relative additive noise, the minimal-gain variants reach 0.93–0.99 at 10⁻³ and 0.42–0.63 at 3·10⁻³.
- Noise model (c), additive on h, is unchanged by routing changes (identical curves). It acts on the residual stream
  after routing.
- The alternative routing R2d is *less* robust: AND-unit conversion is itself a sharp, saturated construction.
- **Independent corroboration (external, unverified package).** Track C's polynomial-attention report (owner-supplied,
  25 Nov 2026; source pin `f5ab660` on this branch) reached 0.24 at 1% additive and 0.24 at 5% multiplicative noise. It
  attributes the residual error to the **arithmetic decoder** (W_in-only perturbation leaves 28%, W_out-only 55%), not
  the routing. This is consistent with R2: routing redesign alone cannot reach the target.

## 60.8 Mechanistic recompilation (MR)

**Learn.** The OP₄ module from Q-reg, 3 seeds. **Interpret.** The extractor (`sci/p8_extract.py`, which imports neither
the rule generator nor the trainer, asserted at run time) queried each frozen module through the network on all 507
canonical (M, A, S) states. **Formalise.** It searched the linear (28,561 candidates) and discrete-log multiplicative
(1,728 candidates) hypothesis spaces.

| module | registered criterion (all 507 states): best / second / accepted | trained pairs only: best / second |
|---|---|---|
| OP₄ s0–s2 | (8, 5, 9, 6) at **0.81** / 0.29–0.29 / **no** | (8, 5, 9, 6) at 1.00 / 0.33 |
| OP_R s0–s2 (negative control) | best 0.13 / **no** | best 0.14 / no |

- **The registered acceptance failed in all 3 seeds** (agreement 0.81 < 0.99). The reason is §60.5: the learned module
  is wrong on the 20% of withheld pairs, so no rule can agree with it there.
- The top candidate is unique by a wide margin (0.81 against 0.29) and **equals the hidden rule**.
- The negative control is correctly rejected.
- **Supplementary, post hoc, not registered:** restricting the query to the pairs the module was trained on gives the
  unique candidate (8, 5, 9, 6) at agreement 1.00. It is labelled post-hoc. Per the claim rule, **no claim of
  recovering a rule without access to the answer is made.** The extractor never saw the answer, but the registered
  criterion was not met.

**Compile, verify, protect** (applied to the post-hoc candidate).
- Compiled R4 is exact against the oracle on all states [E].
- It agrees with the learned module on 1.00 of trained pairs, 0.01–0.03 of held-out pairs.
- Interchange on M gives identical outputs for the compiled and learned modules in distribution (1.00).
- **On held-out pairs, the recompiled module is correct (1.00) where the learned one is not (0.01–0.03).** Recompilation
  turned a memorised relationship into a general one.
- **Extend:** OP₅ = OP₄·M.

  | | competence | held-out accuracy |
  |---|---|---|
  | with R4 as a protected register | 400–600 steps | 0.21–0.28 |
  | without | 600 steps | 0.07–0.10 |

  OP₅ still does not generalise fully: the final multiplication by M is again learned.

## 60.9 Answer to the central question

| capability | status |
|---|---|
| Retaining compiled and earlier learned computations while learning new ones | **demonstrated [E]** (bitwise, all stages) |
| Guaranteeing compiled correctness regardless of trainable parameters | **demonstrated [E, M]** with output ownership; **not** with objective-based protection |
| Learning compositions of compiled operators fast and with compositional generalisation | **demonstrated [T]** (registers visible) |
| Learning an unknown relationship that generalises beyond its training combinations | **not demonstrated** (memorisation) |
| Recovering a learned rule by extraction under the registered criterion | **not met**; post-hoc success labelled as such |
| Recompiling an extracted rule to obtain generalisation | **demonstrated [E]**, post-hoc candidate |
| Robust distributed routing at 1% multiplicative noise | **not met** (0.52) |

## Deviations (ledger items 64–70)

- **64.** The Phase VII trainable count was overstated (191,516 → 152,860). Reported in §60.1; doc 58 unchanged.
- **65.** `spec_terms('lin')` added the constant d for one force unit only (compiled R4 agreed with the oracle on 0.33).
  It was found during verification, fixed and re-run; the earlier output was discarded before reporting.
- **66.** G2 select with an S operand failed (3-dimensional leaf). An assert was added and the test case changed to use
  A; the full G2 suite was re-run.
- **67.** The E-modular coverage key collided (`M*100 + A*10 + fu`). Fixed to `M*1000 + …` and E-modular re-run. The
  sampler itself already used the correct key.
- **68.** Not supported or not met:
  - P matches K5 on OP₂;
  - the R2 target;
  - the MR registered criterion;
  - all 25% conditions on the dual-operation transformer (pending E).
- **69.** P2 at 25% used the Phase VII sampler (`random_phase7`) for comparability, as registered.
- **70.** The A3 B_c baseline was run at 12,000 steps (the compute match to K5's 1.95× per-step cost); B_p at d = 96
  gives 153,312 trainable (+0.3%).
