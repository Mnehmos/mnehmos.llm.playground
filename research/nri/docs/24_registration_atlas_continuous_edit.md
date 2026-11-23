# 24. Registration: two-trajectory concept-formation atlas with a continuous upstream edit loop (exploratory pilot)

*Registration for Issue #22. This document, `nri/atlas.py`, `run_atlas.py` and `tests/test_atlas.py`
are committed **before either study network is trained**. The pilot is **exploratory**: one matched
pair of networks, no population or significance claims. Earlier registrations, code, checkpoints and
results (including PR #21's `run_correction.py`, docs 22–23 and `runs/zoo_correction/`) are not
modified; they are imported or read only.*

## 24.1 Questions

**Primary A — observation (equal priority with B).** In one network trained normally under the
existing interchange objective, *when, where and how* do mass, acceleration, raw values, units,
equivalent quantities and the multiplication form, become decodable, and become causally used? Do
equivalent quantities (the same canonical mass written in different units) share one representation
or travel separate unit-dependent routes, and does that change during training?

**Primary B — supervised organization.** Does a continuous, validated, **upstream** edit of the
weights that *produce* the declared mass block — attempted every 100 steps, installed only when
deterministic checks pass — give a network whose mass is more causally interpretable and more
persistently editable than its exact twin, **without** losing accuracy or held-out compositional
generalization?

**Falsifiable claim** (Issue #22): continuous, targeted, validated upstream supervision produces a
more causally interpretable and persistently editable model than the same model trained normally,
without sacrificing compositional generalization. With n = 1 pair this can be *contradicted* or
*illustrated*, not established.

## 24.2 Design (fixed)

| item | value |
|---|---|
| task, arm | dimensional task (`nri/dimtask.py`); `mass_only`: canonical mass M declared in block dims 0–15 by IIT |
| recipe | doc-14 recipe unchanged, via `run_correction.new_state` / `train` (bitwise `run_swap.train_one`): 6,000 steps, batch 512, AdamW lr 2e-3 wd 0.01, OneCycle (10% warm-up), IIT λ = 3 with linear warm-up over steps 1–2,000 |
| networks | **exactly two**: `control` and `supervised`, both **seed 700** (fresh), 69,831 parameters |
| matching | same initialization (torch seed 700), same base-batch generator (10,700) and IIT-source generator (20,700), same optimizer and schedule; per-interval batch digests are recorded and compared |
| only difference | `supervised` installs accepted edits; `control` runs the identical policy in **shadow mode** (proposals and validation computed, never applied), so audit compute is matched |
| checkpoints | step 0 and every 100 steps to 6,000: **61 per trajectory**. A checkpoint is the state after that step's update and **before** any edit at that step; an accepted edit's result is saved as `model_edited.pt` beside it |
| no branches, no restarts | one uninterrupted trajectory each. No third model, no sham trajectory, no extra seeds |

## 24.3 Data separation (frozen; `nri/atlas.partition`)

| use | questions |
|---|---|
| **FINAL** (sealed, used once at step 6,000) | the 2,205 `run_perm` reserved ID questions (30%) |
| **FIT** (policy fitting, probe fitting, editor fitting) | 50% of the remaining 5,145 ID questions (2,572), split with seed 24,100 |
| **VAL** (policy validation only) | 25% (1,286) |
| **OBS** (the atlas panel: every-100-step observation) | 25% (1,287) |
| **OOD_DEV** (development composition check, policy) | the 864 held-out-combination questions whose held-out values are only in {9, 10} |
| **OOD_FINAL** (sealed) | the 864 whose held-out values are only in {11, 12}; the 48 mixing both are unused |

The policy reads only FIT, VAL and OOD_DEV. The atlas reads FIT (probe fitting) and OBS. Nothing
before the sealed evaluation reads FINAL or OOD_FINAL. A test checks the six sets are disjoint and
that FIT ∪ VAL ∪ OBS ∪ FINAL is the full ID question set. DAS (offline only) uses `run_perm`'s
non-reserved pairs.

## 24.4 The upstream edit (fixed): shrink the mass × route interaction in the block's producer

**Why this tensor.** In `nri/model.py`, the declared block is `h[0:16]`, `h = x1[last] + attn2.o(heads)`.
Rows 0–15 of `attn2.o.weight` and `attn2.o.bias[0:16]` (1,040 parameters) are the only weights that
write the layer-2 attention contribution into the block, and the last position is the only one read
downstream. Editing them changes the **activations of the block itself** (upstream), unlike
PR #21's fold into `W_in` (a downstream readout correction).

**What the edit targets, and why it keeps unit information.** Decompose the block over cells
(mass class M, route = unit × position): `μ_{M,r} = μ + a_M + b_r + I_{M,r}`, least squares over
present cells. Doc 17 showed the doc-15 pooled editor fails because one mass leaves different
patterns per route; in this decomposition that failure is the **interaction** `I`, not the route main
effect `b_r`. If `I = 0`, translating a mass class mean moves every route's cluster correctly while
`b_r` (the unit/position information) is untouched. The edit therefore removes part of `I` and
**preserves** `a_M`, `b_r` and the within-cell variation. It does not collapse the block to class
means (PR #21's seed 602 showed that destroys the calculation).

**Fit.** On FIT, ridge least squares (λ = 1e-3·n) of `[dW | db]` such that
`dW · heads_last + db ≈ −I_{M,r}` (the full, α = 1 direction). **Candidates**: that direction scaled
so `‖[dW, db]‖ / ‖[W_o, b_o]_{rows 0–15}‖` = **0.25, 0.125, 0.0625**, tried largest first; the first
that passes every check is installed. **Budgets**: per edit ≤ 0.25 (relative); cumulative sum of
accepted relative norms ≤ **1.5**; only those 1,040 parameters. The AdamW moments are **not** reset
(stale moments, declared).

**How this differs from "more IIT".** IIT rewards the block for being interchange-*sufficient* for M;
it is indifferent to whether M is coded additively across routes. A sufficient block can still have
a large interaction (doc 17; PR #21: 0.43–0.61 of within-mass variance). The policy records, at
every proposal, an evaluation-only **IIT-gradient comparator** (a step of equal norm on the same
rows along −∇ IIT loss) and **E9 random-direction shams**, so the targeted direction's effect is
compared with "more IIT" and with "any perturbation" at matched norm.

## 24.5 Policy and validation (fixed; `nri/atlas.decide`)

At every checkpoint s = 100 … 6,000, on a **deep copy** of the model:

1. **Gates.** Skip if VAL block IIA(M) < 0.5 ("no representation yet"), if the VAL block interaction
   share < 0.05 ("nothing to organize"), or if the cumulative budget is spent.
2. **Checks** for each candidate (before vs after, VAL unless stated):

| check | criterion |
|---|---|
| E1 editability (target) | doc-15 pooled inversion edit at the block (fit FIT, score VAL), intended rate gain ≥ **+0.01** |
| E2 localization | block IIA(M) ≥ before − 0.01 |
| E3 task | accuracy ≥ before − 0.005 |
| E4 behaviour | ≥ 0.99 prediction agreement with the unedited copy |
| E5 composition (dev) | OOD_DEV accuracy ≥ before − 0.01 |
| E6 unit information kept | route (6-class) probe from the block and mass-unit probe from h each ≥ before − 0.02 |
| E7 specificity | pooled-edit wrong-variable rate ≤ before + 0.02 |
| E8 scope and budget | finite; relative norm ≤ 0.25; cumulative ≤ 1.5 (only the 1,040 parameters can change, enforced by construction and a test) |
| E9 not a generic perturbation | E1 gain − mean gain of 3 fixed random directions of equal norm (seed 7,700) ≥ **+0.005** (evaluated only if E1–E8 pass) |

3. **Install** (supervised only) the first passing candidate in place; record it. Otherwise record
   `rejected` with every check value, or `skipped` with the gate reason. Rejection or skipping never
   touches the training model or optimizer (proposals live on a copy; tested). The same candidate
   family is used throughout; no new candidate is searched mid-run.

**Development disclosure (before registration, non-study networks only).** The mechanism was
exercised read-only on PR #21's existing networks (seeds 601 and 602 at steps 700–1,500, replayed
bitwise from their saved step-600 states; seeds 600–602 at 1,500 and 3,000). Observations that
shaped the fixed values: (i) removing the whole interaction needs a change 2.2–2.8× the block rows'
norm, so a ≤ 10% budget did nothing measurable and the per-edit budget was set to 25%; (ii) at that
size, pooled-edit gains were +0.005–0.020 in seed 601 and ≤ 0 in 602, where accuracy and OOD_DEV
fell (E3/E5 reject); (iii) **a random direction of equal norm sometimes produced comparable gains**
(seed 601 at 1,400–1,500: +0.012/+0.018 vs +0.020/+0.013), which is why E9 was added. With these
values the policy would have accepted 2 of 9 checkpoints in seed 601 (before E9) and 0 of 9 in seed
602. **Expectation stated now:** accepted edits will be few and individually small; the edit may be
rejected throughout if the study network resembles seed 602 or 600. That is a valid outcome.

## 24.6 Observation: the every-100-step atlas (fixed; `nri/atlas.observe`, `weight_dynamics`, `attribution`)

Both trajectories, every checkpoint, on OBS (probes fitted on FIT):

- **Behaviour:** OBS accuracy; OOD_DEV accuracy.
- **Decodability (descriptive only):** chance-adjusted ridge-probe accuracy κ for 11 factors
  (M, m, mass unit, A, a, acc unit, output unit, order, F, M·A, mass route) at 14 sites (layer-1
  residual at the mass-value, mass-unit, acc-value, acc-unit and last positions; attn2 output; each
  attn2 head; h; block; h dims 16–63; MLP neurons).
- **Equivalent quantities:** cross-unit transfer probes (train on two units, test on the third) for M
  and A at five sites, with within-unit accuracy as reference.
- **Geometry:** M × route decomposition (route main, interaction, residual shares) at the block, h and
  MLP; per-unit CKA of mass-class configurations; discrete-log (multiplicative) vs additive character
  concentration of M and F class means; class-mean subspaces of M, A, F, route and output unit at h
  (rank, share inside the block, overlap with the previous checkpoint); CKA of h to the previous
  checkpoint.
- **Causal (activation-level interchange on OBS pairs; never persistent):** block / dims 16–63 for M
  and A (and output factor S); each attn2 head's output for M and A; layer-1 positional swaps with
  matched counterfactuals (value+unit → M or A; value only → raw m or a; unit only → unit change);
  pooled persistent inversion edit at the block (fit FIT) and copy-into-block; MLP neuron
  mean-ablation importance.
- **Features:** per-neuron families (argmax η² over 11 factors if ≥ 0.25, else `mixed`; `dead` if
  active < 1%) for MLP neurons, attn2 head dims and h dims; 50-dim response signatures; Hungarian
  matching to the previous checkpoint (stable / relocated / uncertain / new at cosine 0.8 / 0.5);
  family transitions.
- **Weights:** per tensor norm, relative change from init and from the previous checkpoint, update
  sparsity, update direction cosine with the previous interval, top singular value, effective rank,
  top-1 share of the update spectrum, top-3 singular-subspace overlap with previous; gradient norms
  of the task and IIT terms on a fixed FIT batch; **trapezoid path attribution** of the change in each
  term over each interval to tensors, attention heads and MLP neurons.
- **Stored per checkpoint:** `model.pt` (both trajectories, all 61); full training state (model,
  optimizer, scheduler, generators, global RNG) at 0, 1,000, …, 6,000; `acts.pt` (fp16 sample of 128
  OBS examples at h, attn2 heads, MLP, x1[last], last-row attention; signatures; η²; neuron
  importance); a JSON card with every number above, state hash and batch digest. Every other
  checkpoint's full state is reproducible by deterministic replay (the accepted deltas are saved).

**Offline milestones** (`run_atlas.py analyse`, never read by the policy): 0, 500, 600, …, 1,500
(every 100), 2,000, 3,000, 4,500, 6,000, plus any checkpoint where OBS accuracy or block IIA(M)
moves ≥ 0.15 from the previous one (and its predecessor), plus post-edit states at those steps:
DAS (k = 16) for M and A, DAS IIA and block overlap, DAS-located persistent edit, D1 (block replaced
by mass class means: accuracy on OBS and OOD_DEV), family-level MLP ablations, held-out edit kind
(add-one) at the block.

## 24.7 Registered atlas criteria

- **Decodable** at a site: κ ≥ 0.5 (weak: 0.2–0.5). **Causal**: the concept's matched interchange
  test ≥ 0.5; **causally verified**: ≥ 0.9. No concept is called represented on decodability alone,
  and no site is said to "mean" a concept on a probe, a cluster or an LLM label.
- **Sustained**: holds at a checkpoint and the next two.
- **Mass emergence**: OBS accuracy ≥ 0.5 and block IIA(M) ≥ 0.5, sustained.
- **Mass identified (C1 of doc 22)**: block IIA(M) ≥ 0.9 and accuracy ≥ 0.95, sustained.
- **Coalescence** of equivalent quantities: mass unit-transfer ratio (transfer / within) ≥ 0.8 at
  the block (and separately at h) while block IIA(M) ≥ 0.9, for 5 consecutive checkpoints. Regimes
  per checkpoint: shared (≥ 0.8), partially shared (0.5–0.8), unit-dependent (< 0.5).
- If a measure has no clean threshold crossing, the trajectory is reported without a transition.

## 24.8 Sealed final evaluation (once, both networks at step 6,000; `run_atlas.py final`)

On FINAL and OOD_FINAL: ID accuracy; held-out-combination accuracy (OOD_FINAL; legacy
`run_dimensional.acc(…, "ood")` also reported); block IIA(M), IIA(M) outside the block, IIA(A)
at the block; DAS IIA(M) and DAS–block overlap; copy-into-block; block interaction share;
**persistent** pooled edits folded into `W_in` (fit FIT) for inversion (the policy's target) and two
**held-out** edits never used by the policy, add-one and doubling, at the block and at DAS: intended /
wrong-variable / unchanged / other, preservation of unedited behaviour (agreement on fixed points of
the edit) and edit size. **Evaluation-only post-training benchmarks** (never trained on): doc-22
readout fold; the registered upstream policy applied once at step 6,000; the full unbudgeted
interaction-removal edit.

## 24.9 Outcomes (supervised minus control at step 6,000, FINAL)

| outcome | criterion |
|---|---|
| **improved** | inversion edit intended ≥ +0.05 **and** at least one held-out edit ≥ +0.03, **and** ID ≥ −0.005, OOD_FINAL ≥ −0.02, block IIA(M) ≥ −0.01 |
| **traded off** | editability improves as above, but ID, OOD_FINAL or IIA worsens beyond those margins |
| **no effect** | inversion edit |Δ| < 0.05 and held-out edits |Δ| < 0.03 |
| **harmed** | anything else |
| **not tested** | no edit accepted during training (the two networks are then bitwise identical, which is checked) |

Also reported, without thresholds: whether accepted edits improved the VAL target immediately and
how much of it **remained at the next checkpoint** (retention); whether the shadow policy on the
control would have accepted at the same steps; and the post-training benchmarks (does editing
*during* training beat editing *after* it?).

## 24.10 Compute and storage

Separate CPU-second ledgers per trajectory for training, observation, weights/gradients, policy and
I/O, plus offline analysis and the sealed evaluation; bytes stored. Estimate before running:
≈ 15 CPU-minutes per trajectory (≈ 8 training, ≈ 6 audit) and ≈ 65 MB. The policy and atlas
measurements cost ≈ 4–6 CPU-s per checkpoint (measured on PR #21 networks).

## 24.11 Non-claims

One task, one arm, one seed, one candidate family, one matched pair. Decodability is not
representation; activation interchange is not a persistent edit; a readout fold is not an upstream
change; an LLM supervisor (this report's author) may interpret audit cards but **no** unregistered
edit is applied.
