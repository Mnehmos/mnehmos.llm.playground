# 11. Pre-registration: specification quality vs programmability on dimensional reasoning

*Written and committed **before** the main run. The commit that adds this file is the
registration timestamp. Any later deviation is listed in doc 12 under "Deviations".*

## 11.1 Motivation

An external review of PR #2 raised four challenges:

1. IIT costs extra compute per step, and the learning-speed comparison counted steps, not compute.
2. Five seeds, many uncorrected tests.
3. The causal variables and target subspaces were supplied exactly. The open question is
   **incomplete, noisy or partially wrong specifications**.
4. The LLM-interpreter study was too small. That is deferred to a later study and is not part
   of this registration.

The review also proposed moving from modular arithmetic to **dimensional reasoning**, which has a
real invariance under unit conversion. This study addresses challenges 1–3 on such a task.

## 11.2 Task (`nri/dimtask.py`)

**Input:** `x1 u1 × x2 u2 → f =`. One pair is a mass and the other an acceleration, in random
order. Which is which is identifiable only from the unit token's dimension.

**Output:** F = (m·s_mu)·(a·s_au)·s_f⁻¹ mod 13. The unit scales are mass (1, 5, 8),
acceleration (1, 2, 7) and force (1, 6, 11).

**Canonical, unit-invariant causal variables:** M = m·s_mu, A = a·s_au, S = s_f⁻¹; F = M·A·S.
The task is exactly invariant under unit rescaling, which was checked in code.

**OOD split, never trained:** mass unit 2 with m ∈ {9–12}, and acceleration unit 2 with
a ∈ {9–12}. Every canonical value these produce occurs in training through other units, which was
also checked.

## 11.3 Arms

All arms share one architecture: 2-layer attention-only transformer + MLP, d = 64, about
69.8K parameters, identical across arms. All are trained deterministically, use **untyped
inference**, and run **10 seeds each** (seeds 0–9).

| arm | objective | role in the design |
|---|---|---|
| `none` | task, batch 512 | baseline |
| `none_2x` | task, batch 1024 | **compute-matched** to IIT: the same number of encoder forward passes per step |
| `aux_types` | task + auxiliary contextual-role loss (7 roles) | ANLU functional supervision |
| `iit_full` | task + IIT on canonical M, A, S (fixed blocks: M dims 0–15, A 16–31, S 32–35) | full, correct specification |
| `iit_partial` | task + IIT on M only | **partial** specification |
| `iit_noisy` | full spec, with 25% of counterfactual targets replaced by uniform random labels | **noisy** specification |
| `iit_raw` | IIT on *raw* values m, a (and S) | a valid but **non-invariant** (wrong-variable) specification |

**Step budget:** 6,000 steps for every arm, with λ_IIT = 3.0 (all IIT arms). The OneCycle learning-rate schedule spans 6,000 steps. The budget is chosen in
pilot runs (seed 0 of `iit_full` only, run in a scratch directory and **not** part of the
results). The criterion: designed-block interchange accuracy for M and A ≥ 0.9, so the IIT
treatment is actually applied. Pilot outcomes are recorded in §11.7.

## 11.4 Measurements

All measurements use untyped inference and are **blind to the training specification**, except
where a column is labelled "designed block".

- **Task accuracy**, in-distribution and **OOD** (held-out unit–value combinations).
- **Learning cost:** steps to 95% accuracy, and *samples encoded* to 95% (steps × encoder
  forwards per step × batch). The latter is the compute-matched comparison.
- **Localization:** DAS IIA (k = 16) and class-mean IIA for M and A. Designed-block IIA.
- **Held-out "new unit system" edits:** M → 2M (every mass unit rescaled by 2) and A → 2A. The
  edit is an affine map inside the located subspace (DAS-located, class-mean-located, or designed
  block), fitted by least squares on activations and folded into `W_in`. No arm is trained on any
  edit. Success means the edited network outputs 2M·A·S (respectively M·2A·S). The no-op rate is
  reported.
- **Additive R²** of the site on one-hot (M, A, S).

## 11.5 Hypotheses

**Primary family** (Holm-corrected at α = 0.05; two-sided permutation tests, 10 vs 10 seeds):

- **P1, programmability with a full specification:** `iit_full` > `none` on M→2M edit
  success (DAS-located, blind).
- **P2, invariance generalization:** `iit_full` > `none` on OOD accuracy.

**Secondary**, exploratory and reported uncorrected, with predictions stated in advance:

- **S1, partial specification:** `iit_partial` > `none` on the M→2M edit (the declared variable).
  *Open question:* does it help the A→2A edit (the undeclared variable)?
- **S2, noisy specification:** `iit_noisy` > `none` on M→2M. It retains at least half of
  `iit_full`'s gain.
- **S3, wrong variables:** `iit_raw` ≤ `iit_full` on OOD accuracy. A non-invariant specification
  should not help unit generalization.
- **S4, ANLU functional supervision:** `aux_types` > `none` on DAS localization of M.
- **S5, compute:** `iit_full` vs `none_2x` on samples encoded to 95%. *No direction predicted.*
  The R6 claim of "no significant slowdown" is retracted unless this comparison supports it.
- **S6, compute-matched baseline:** `none_2x` does **not** differ from `none` on the M→2M edit.
  If it does, extra data or compute, not structure, explains part of the IIT effect.

## 11.6 Decision rules

- If **P1 fails**, the R6 programmability result does not transfer to dimensional reasoning, and
  docs 09/12 will say so.
- If **P2 fails**, IIT gives no measurable invariance-generalization benefit on this task.
- If S2 fails (noisy spec retains < 50% of the gain), the method is fragile to specification
  errors. That would be a key limitation for real-world use.

## 11.7 Pilot log (budget selection; seed 0 of `iit_full`; scratch only)

| steps | λ | designed IIA M / A / S | acc ID / OOD | decision |
|---|---|---|---|---|
| 1,500 | 1.0 | not measured | 0.59 / 0.47 | not converged |
| 4,000 | 1.0 | 0.26 / 0.26 / 0.99 | 1.00 / 0.97 | **treatment not applied** (M and A blocks unused) |
| 6,000 | 3.0 | **0.999 / 0.999 / 1.00** | 1.00 / 0.997 | **criterion met → registered budget** |
| 10,000 | 1.0 | (stopped at step 5,250: acc 0.72) | — | too slow; stopped to free CPU |
| 16,000 | 1.0 | (stopped at step 5,250: acc 0.25) | — | too slow; stopped to free CPU |

The main run's seed 0 of `iit_full` is bit-identical to the 6,000-step pilot (deterministic
training), so the pilot was the first replicate. This is disclosed rather than discarded. All
other arm × seed combinations were never run before registration.
