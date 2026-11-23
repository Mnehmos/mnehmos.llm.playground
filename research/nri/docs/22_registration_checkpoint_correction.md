# 22. Registration: checkpoint-guided causal correction during training (exploratory pilot)

*Registration for Issue #20, amended after the owner's review. This document, `run_correction.py`
and `tests/test_correction.py` are committed **before any study network is trained**. The pilot
is **exploratory** (n = 3 networks). It makes no statistical claims. Earlier registrations,
results and checkpoints are untouched.*

## 22.1 Question and hypothesis

**Question:** can we observe a developing representation during training, install a validated
persistent weight change, and get a better final network than uninterrupted training?

**Hypothesis (amended):** a small, validated, persistent **downstream readout correction** at 50%
of training can make the declared mass easier to edit at the final checkpoint, compared with the
same network trained on uninterrupted, **while preserving task accuracy and held-out
generalization**.

**What is *not* assumed.**
- Doc 17 found that the same mass leaves **route-specific** patterns, depending on unit and
  position. This study does **not** assume that variation is a defect: it may be how the network
  converts units.
- The study therefore measures whether removing it destroys information the calculation needs
  (§22.5, D1–D2).

**What the correction is and is not.**
- The correction is folded into `W_in`. It changes **how the MLP reads** the mass block, not the
  block's activations.
- It is labeled a **readout correction** throughout. Whether continued training then changes how
  the **upstream** representation forms is measured (C5), not assumed.
- Correcting upstream, representation-producing weights directly is left to a later experiment.

## 22.2 Design (fixed)

| item | value |
|---|---|
| task, arm | dimensional task (docs 11–15); `mass_only`: canonical mass M declared at block dims 0–15 |
| recipe | the doc-14 recipe, unchanged (`run_swap.train_one`): 6,000 steps, batch 512, AdamW lr 2e-3, wd 0.01, OneCycle, λ = 3 with a warm-up over steps 1–2,000 |
| networks | **seeds 600, 601, 602**, fresh, from scratch (69.8K parameters) |
| checkpoints | trunk: 0 (initialization), 600, 1,500, 3,000; then **both** branches at 4,500 and 6,000; plus the correction branch right after installation ("3,000-edited") |
| branch point | step 3,000. Control: resume. Correction: apply the validated fold, then resume |
| matching | identical generator states (identical batches, verified by per-step hashes), OneCycle position, λ schedule and remaining steps. AdamW moments are carried over unchanged, so the edited parameters start with the control's stale moments (declared) |
| no restarts | every trajectory and branch is kept, including stuck seeds and rejected corrections |

**Gate before any result counts.** For seed 600, the control branch must be **bitwise identical**
to an uninterrupted run of the original `run_swap.train_one`, which runs in parallel. Tests
already check this on short runs:
- the wrapper reproduces `train_one` bitwise;
- a run split at its midpoint is bitwise identical to an unsplit run.

## 22.3 Data separation (frozen)

| use | questions |
|---|---|
| calibration (fold fitting, validation, editor fitting, class means) | **non-reserved** questions from `run_perm`'s fixed split: fit seed 70701, validation seed 70702 |
| final evaluation | **every reserved question once** (2,205; 1,011 eligible for the edit scorer); interchange pairs drawn within reserved (seed 70703) |
| OOD | held-out unit–value combinations: the 8,192-question set `run_dimensional.acc` uses |
| DAS search | non-reserved pairs (`run_perm.das_basis`) |

Calibration and evaluation never share a question.

## 22.4 The correction (fixed): a block canonicalization readout fold

1. **Fit.** On calibration data at step 3,000, fit an affine map by least squares. It sends each
   example's block vector (h, dims 0–15) to the **pooled** mean of its M class.
2. **Install.** Fold it with strength α into the existing `W_in` columns 0–15 and `b_in`.
   - At most **4,352** parameters change.
   - The activations at h are unchanged.
3. **Choose α.** Try α ∈ {1.0, 0.5, 0.25}, largest first. The first α that passes **all** of
   V1–V5 is applied. If none passes, no edit is applied, the correction branch is **not trained**,
   and that is the result.

**Validation** (calibration only: fit on seed 70701, measured on seed 70702):

| check | criterion |
|---|---|
| V1 causal target | doc-15 pooled inversion edit at the block, on top of the fold: intended rate ≥ unedited + 0.05 |
| V2 localization | designed-block IIA(M) ≥ unedited − 0.01 |
| V3 task | accuracy ≥ unedited − 0.005 |
| V4 behaviour | ≥ 0.99 agreement with the unedited predictions |
| V5 scope | finite; only `W_in[:, 0:16]` and `b_in` change |

**Supervisor:** an LLM may propose interpretations and candidates in
`results/correction/supervisor_notes.md`. Only this candidate is executed, and only V1–V5 decide.

## 22.5 Measurements (frozen; `run_correction.audit`)

**At every checkpoint, on reserved questions:**
- ID and OOD accuracy;
- designed-block IIA(M), plus the legacy `run_dimensional.designed_iia`;
- blind DAS IIA(M) (k = 16) and the DAS–block overlap;
- the probe for M at the block, which is **reported, never counted as evidence**;
- the pooled inversion edit at the block and at DAS (intended / wrong / unchanged / other);
- copy-into-block;
- non-declared A: DAS IIA(A) and the A edit at DAS;
- the model hash;
- `acts.pt`, with class means per M and per (M, route) and 512 sample activations.

**Organization at two levels,** as the share of within-M variance explained by route:
- **upstream:** the block activations themselves;
- **readout:** the block's contribution to the MLP input, `W_in[:, 0:16] · z`.

**Diagnostics:**
- **D1 information necessity** (activation-level, not persistent), at every checkpoint: replace
  each example's block by its M-class mean, then measure ID and OOD accuracy. A drop means the
  route variance carries information the calculation uses.
- **D2 fold validation by checkpoint** (calibration only, never applied): V1–V5 for every α at
  600, 1,500, 3,000 and the control's 6,000. This shows whether canonicalizing the readout becomes
  destructive as training proceeds.
- **Persistence:** ‖W_in(corr) − W_in(ctrl)‖ / ‖W_in(ctrl)‖ at 3,000-edited, 4,500 and 6,000.
- **Ledger:** CPU-seconds per phase and bytes stored.

## 22.6 Claims, scored separately per network

| claim | criterion |
|---|---|
| **C1** representation identified | first checkpoint with designed-block IIA(M) ≥ 0.9 **and** ID ≥ 0.95 |
| **C2** behaviour changed by a causal intervention | copy-into-block intended ≥ 0.8 at step 3,000 |
| **C3** persistent readout correction installed | V1–V5 passed; on frozen evaluation at 3,000-edited, edit intended ≥ step-3,000 value + 0.05 **and** ID ≥ step-3,000 value − 0.005 |
| **C4** correction improved subsequent training | see the outcome definitions below |
| **C5** upstream organization changed during continued training | descriptive: upstream route share, correction vs control at 4,500 and 6,000 |

**C4 outcomes at step 6,000** (correction minus control):
- **Improved:** Δ edit intended ≥ +0.05, **and** ΔID ≥ −0.005, **and** ΔOOD ≥ −0.02.
- **Traded off:** Δ edit ≥ +0.05 but ID or OOD is worse beyond those margins.
- **No effect:** |Δ edit| < 0.05.
- **Harmed:** anything else.

**Not applicable:** if no correction passes validation.

## 22.7 Disclosed observation before registration (a non-study network)

The audit and validation code was smoke-tested on an **existing, converged** network,
`runs/zoo_swap/mass_only_s300` from doc 15. It was read only; outputs went to scratch.

- **The fold failed validation at every α.** Calibration accuracy went from 1.00 to 0.27, 0.40
  and 0.62 for α = 1.0, 0.5 and 0.25. Designed-block IIA went from 0.95 to 0.20, 0.33 and 0.56.
  V1 passed only at α = 1.0 (+0.08).
- **D1 showed the same dependence:** without route variance, ID accuracy fell to 0.77.
- **Probe vs interchange:** mass was not linearly readable from the block (probe 0.27), although
  the block is causally sufficient for it (interchange 0.96).

**Prediction:** the readout canonicalization may also fail validation at step 3,000 in the study
networks, because route variance carries information the calculation needs. If so, the
pre-registered result is:
- **"no validated correction"**;
- the D1/D2 trajectories showing **when** that dependence forms.

**No candidate, threshold or α grid was changed after this observation.** The only addition was
the zero-training diagnostic D2.

## 22.8 Failure conditions and non-claims

**Failure conditions,** each reported as a result:
- the resume-fidelity gate fails (results invalid until fixed);
- no validated correction exists;
- the correction washes out;
- generalization trades off;
- canonicalization destroys necessary information.

**Non-claims:**
- **Exploratory, n = 3:** no p-values, no population claims.
- **Narrow scope:** one task, one arm, one correction family, one branch point.
- **A readout correction is not a representation correction.**
- **One verified intervention is not continuous LLM-supervised training.**

## 22.9 Procedure and compute

- **Procedure:** `python run_correction.py run` (3 seeds and the fidelity reference in parallel)
  → `python run_correction.py report` → docs/23.
- **Estimate:** about 1.6 CPU-hours and about 15 MB.
