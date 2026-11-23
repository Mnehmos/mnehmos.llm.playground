# 28. Registration: Adaptive Concept Lineage and Causal Supervision (spec v1.0)

*Implements the owner's spec v1.0. Committed with the frozen implementation (`nri/lineage.py`,
`nri/interventions.py`, `run_lineage.py`, `tests/test_lineage.py`) **before either study network is
trained past initialization**. Exploratory: one matched pair; descriptive comparisons only. Prior
registered studies are untouched.*

## 28.1 Questions

- **Q1 (supervision).** Can a scientific supervisor (an LLM) observing a network during learning
  design targeted, validated interventions that improve its final organization compared with
  ordinary training?
- **Q2 (lineage).** Can a meaningful developmental history of the network's concepts be reconstructed
  from its weights, gradients, activations and causal behaviour?

Q2 is valuable even if every intervention fails.

## 28.2 Networks (fixed)

| | A: control | B: adaptively supervised |
|---|---|---|
| recipe | doc-14 `mass_only` (IIT λ = 3 warm-up, 6,000 steps, batch 512, AdamW 2e-3, OneCycle), via `run_correction.new_state` / `train` | identical |
| seed | **701** (fresh; never used before) | 701 |
| observation | every 100 steps (61 checkpoints) | every 100 steps, plus a **mandatory supervisor review** at each of the 60 training checkpoints before training continues |
| interventions | none | validated declarative proposals only |

- **Matching.** Identical initialization, identical base and IIT-source batch streams (per-interval
  digests compared), identical optimizer and schedule. No other trajectories, except **temporary
  copies** used only for candidate evaluation (the 100-step lookahead), whose compute is booked
  separately.
- **Why a fresh seed.** In the doc-26 precursor the agent knew the network's future; seed 701 keeps
  the supervisor blind to it.
- **Tokenization is unchanged.**

## 28.3 Data (frozen; the doc-24 partition)

| use | split |
|---|---|
| policy fitting | FIT |
| validation (gate) | VAL |
| atlas / supervisor observation | OBS |
| development composition check | OOD_DEV (held-out values 9, 10) |
| **sealed** | **FINAL** (2,205 reserved ID questions) and **OOD_FINAL** (held-out values 11, 12) |

- The **matched diagnostic panel** (`lineage.matched_panel`, seed 31,337) has 48 base questions
  from OBS, each with five single-factor variants: equivalent mass in another unit, another mass,
  another acceleration, another output unit, swapped order. Every variant is a non-sealed ID
  question.
- Supervisor-facing commands run under a **sealed-split guard** that raises on any FINAL/OOD_FINAL
  access (tested).

## 28.4 Concept Lineage Engine (what is recorded at every checkpoint, both networks)

- **Checkpoint recorder:** full parameters (`model.pt`, all 61); full training state (model,
  optimizer, scheduler, generators, RNG) every 500 steps plus a rolling live state; state hash and
  full-state fingerprint; batch digest.
- **Causal audit engine:** the doc-24 atlas card (decodability κ for 11 factors × 14 sites;
  unit-transfer; geometry; activation interchange for M, A, S, raw values and units at the block,
  dims 16–63, each attn2 head and layer-1 positions; persistent pooled edit; copy; neuron ablation
  importance; neuron families and signature matching).
- **Token and example tracer** (matched panel):
  - per-position input-embedding gradient saliency of the correct answer;
  - block-gradient norm;
  - block and value-token similarity for each matched variant kind;
  - whether equivalent pairs receive the same prediction;
  - **semantic-group parameter gradients** (examples grouped by mass unit and by acceleration unit)
    for attn1, attn2, MLP-in, MLP-out and unembedding, with pairwise cosine agreement or conflict.
- Token-embedding gradients and example-level parameter gradients are reported separately.
- **Weight lineage:** per tensor norm, change from init and from the previous checkpoint, update
  direction cosine, update sparsity, singular spectrum and top-subspace overlap; task and IIT
  gradient norms; trapezoid path attribution of loss changes to tensors, heads and MLP neurons.
- **Feature and cluster lineage:** candidate features are attention-head roles (8), MLP neuron
  clusters (k-means, k = 10, on response signatures) and concept subspaces (6 concepts × 4 sites).
  - Matching: Hungarian on signature cosine, plus member Jaccard for clusters (0.7 / 0.3), plus
    subspace overlap.
  - Matches are confident at ≥ 0.8 and uncertain at 0.5–0.8; below 0.5 a candidate is new.
  - Events: emergence, consolidation, specialization (selectivity +0.1), splitting and merging
    (unmatched candidate ≥ 0.8 similar to a matched one), relocation, disappearance or weakening,
    persistence.
- **Concept records:** each lineage ID carries identity, location, evidence history, status history
  and match confidence.
  - Status: *hypothesized* → *correlationally supported* (κ ≥ 0.5 or η² ≥ 0.25) → *causally
    supported (partial / full)*.
  - The causal test is the matched interchange (≥ 0.5 / ≥ 0.9) or, for neuron clusters, summed
    mean-ablation loss (≥ 0.05 / ≥ 0.3 nats). A status that falls back is recorded.
- **An LLM may label; only these deterministic tests set causal status.**

## 28.5 The supervisor (B only)

At every training checkpoint (100 … 6,000) the supervisor receives the packet:
- B's audit next to A's at the same step, and changes since the previous checkpoint;
- token saliency, matched-panel similarities and group-gradient agreement;
- the largest weight updates, head attribution and lineage events;
- the previous question and proposals;
- the remaining budget.

It must file a **review** (`run_lineage.py review`; refused if a field is empty; training cannot
continue without it) with these fields:

| field | content |
|---|---|
| `what_changed` | |
| `hypothesis` | falsifiable |
| `evidence` | observation / correlation / causal / speculation separated |
| `key_question` | |
| `best_experiment` | |
| `intervene` | and `intervention_rationale` |
| `previous_interventions` | what persisted, disappeared or was compensated |
| `investigation_level` | 1 routine, 2 targeted, 3 intervention |

Rules:
- **Level 2 inspections are read-only** and capped at 8 per checkpoint: head anatomy, neuron/feature
  ablation, interchange at a chosen site and variable, Hessian sharpness, feature records, and an
  OBS dry run of a proposal.
- **Do not repeat a rejected hypothesis without new evidence.** The journal links each proposal to
  its review.

## 28.6 Interventions: primitives, budgets, validation (frozen in `nri/interventions.py`)

**Primitives:**
- `restricted_gradient_update`: Adam on FIT over allowed scopes, with a weighted sum of the
  registered terms: KL to the original predictions, task, interchange (IIT), block mass × route
  interaction, class-mean translation for random permutations (never inversion or add-one), KL on
  noise-perturbed block activations, frozen route probe, mass–acceleration separation; plus an L2
  leash.
- `low_rank_adjust`: rank ≤ 8, ridge fit to a registered activation target (block interaction
  removal; one head's canonical mass or interaction removal), SVD-truncated.
- `counterfactual_supervision`: interchange supervision of a named variable (M, A, S, m, a) at a named
  site (block, dims 16–63, one attn2 head).

**Scopes:** whole allowed tensors (embeddings, attention, MLP, unembedding), or one head's slice, or a
row range. No supervisor code is executed.

**Every proposal must state:** target concept, mechanism, scope, rationale, predicted behavioural and
causal effect, collateral risk, and a falsifying observation.

**Budgets:**

| budget | limit |
|---|---|
| per edit | ≤ 3% of the model norm and ≤ 25% of any tensor |
| cumulative | ≤ 15% of the model norm |
| candidate validations | ≤ 3 per checkpoint |
| extra gradient steps for compilation | ≤ 1,500 per checkpoint (lookahead booked separately) |
| inspections | ≤ 8 per checkpoint |

**Validation** runs on copies of the full training state. **Phase** is set from VAL before the
edit:
- **early:** block IIA < 0.5;
- **forming:** IIA < 0.9 or accuracy < 0.95;
- **mature:** otherwise.

| check | early | forming | mature |
|---|---|---|---|
| E0 eligible targets | `iia`, `ood_dev` only | all | all |
| G target gain (VAL) | `edit_inv` +0.01; `interaction` −0.02 (editability not falling); `unit_transfer` +0.03; `ood_dev` +0.01; `iia` +0.01 | same | same |
| S1 accuracy | ≥ before | ≥ before − 0.005 | ≥ before − 0.005 |
| S2 previously-correct kept | ≥ 0.95 | ≥ 0.98 | ≥ 0.99 |
| S3 OOD_DEV | ≥ before − 0.01 | same | same |
| S4 block IIA(M) | — | ≥ before − 0.01 | ≥ before − 0.01 |
| S5 route and unit information (probes) | ≥ before − 0.02 | same | same |
| S6 wrong-variable rate | ≤ before + 0.02 | same | same |
| S7 budget | as above | | |
| S8 sham | gain − mean of 3 random deltas (same per-tensor norms and slices) ≥ half the target threshold | | |
| **S9 stability** | **deterministic 100-step lookahead** on copies of the full state (the batches the live run will see), edited vs unedited: the edited run's max training loss ≤ 3 × the unedited max (floor 0.05) **and** its VAL accuracy at the end ≥ unedited − 0.02 | | |

- **Optimizer policy:** Adam moments are unchanged on installation (stale moments, declared).
- **Rejection:** model, optimizer, scheduler, generators and step are verified bitwise unchanged
  (fingerprint asserted; tested).

## 28.7 Final evaluation (once, step 6,000, A and B; `run_lineage.py final`)

The doc-24 sealed measures on FINAL and OOD_FINAL:
- ID and OOD_FINAL accuracy;
- block and DAS IIA, DAS–block overlap, copy;
- persistent pooled edits for inversion (the policy target) and the held-out **add-one**, at the
  block and at DAS (intended / wrong / preservation / size);
- post-training benchmark corrections.

Doubling is dropped (not variable-discriminating).

**Outcome (B − A):**

| outcome | criterion |
|---|---|
| **improved** | inversion Δ ≥ +0.05 **and** add-one Δ ≥ +0.03, **and** ΔID ≥ −0.005, ΔOOD_FINAL ≥ −0.02, Δ block IIA ≥ −0.01 |
| **traded off** | editability improves but a margin fails |
| **no effect** | inversion Δ < 0.05 and add-one Δ < 0.03 |
| **harmed** | anything else |
| **not tested** | no edit installed |

Also reported:
- lineage comparisons: formation timing, events, verified claims;
- intervention persistence: each accepted edit's effect at the next checkpoints;
- failure mechanisms;
- costs: base training vs observation vs supervision (inspections, compilation, validation,
  lookahead).

## 28.8 Lineage success criteria (Q2)

- 61 recorded checkpoints per network.
- Concept records with evidence status for every candidate.
- Lineage events with match confidence.
- The five visualizations (atlas, lineage graph, weight timeline, causal circuit atlas, intervention
  history) generated from the records.
- A journal entry for each of B's 60 training checkpoints.

## 28.9 Disclosures

- **Thresholds are inherited from docs 24 and 26.** S2 is phase-specific and S9 is new, both because
  of doc 27's findings. Doc 27 also shaped the primitives (`organize`- and `rewire`-style terms).
- **Development used non-study networks only:** PR #21's seed 601 and the doc-26 precursor (seed
  700). Seed 701 has never been trained.
- **Estimated compute:** base 2 × ~0.16 CPU-h; observation 2 × ~0.1 CPU-h; supervision ~0.5–1.5
  CPU-h. Measured values will be reported.
