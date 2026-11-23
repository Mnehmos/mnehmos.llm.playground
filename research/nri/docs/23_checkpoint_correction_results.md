# 23. Checkpoint-guided correction pilot: results (exploratory, n = 3)

*Registration: doc 22, commit `d7200fb`, made before any study network was trained (Issue #20).
Analysis: the frozen `run_correction.py report`. Full report: `results/correction/summary.md`.
Per-checkpoint cards: `results/correction/audit_cards.jsonl`. Networks and states:
`runs/zoo_correction/`.*

- **No deviations from the registered procedure.**
- **One post-hoc analysis** (§23.5), labeled as such.
- **One storage repack** (§23.8) that does not change any value.

## 23.1 Answer in plain English

**Can we observe and correct a developing representation during training, and does that give a
better final network than uninterrupted training?**

**Observing it: yes, cheaply and causally.**
- All three networks formed a causally identifiable mass representation **by step 1,500**: block
  interchange ≥ 0.9 with accuracy ≥ 0.95.
- **How that representation was organized was also settled by step 1,500** and barely changed
  afterwards.
- The three networks organized it in three different ways (§23.3).

**Correcting it: only in one of three networks, and it did not beat correcting afterwards.**
- The registered readout correction passed its safety and causal checks in **one** network (seed
  601).
- There it made the mass easier to edit at installation (pooled edit 0.651 → 0.732), without
  changing accuracy.
- That branch met the registered "improved" criterion at step 6,000, **narrowly**: edit +0.055
  against control, with identical accuracy and generalization.
- But applying **the same correction after training** gives at least as much: 0.722 vs 0.705
  (§23.5).
- Continued training slowly eroded the correction. It did **not** reorganize the upstream
  representation.
- **In this pilot, intervening during training gave no advantage over editing the finished
  network.**

**The other two networks:**
- **Seed 600:** nothing to correct. The mass was already easy to edit (0.87).
- **Seed 602:** canonicalizing the readout **destroys the calculation** (accuracy 1.00 → 0.52 at
  full strength). Its unit/position variation carries information the network needs. This
  confirms the owner's caution that route-specific representation is not necessarily a defect.

## 23.2 Gate and matching checks

- **Resume fidelity:** the seed-600 control branch, resumed from the full saved state at step
  3,000, is **bitwise identical** to an uninterrupted run of the original `run_swap.train_one`.
- **Identical batches:** in seed 601, every per-step batch hash in the correction branch matches
  the control branch (3,000 steps).
- **Tests:** `tests/test_correction.py` passes (3 tests: equivalence with the original trainer,
  split-run identity, fold scope).

## 23.3 Trajectories: three organizational regimes, settled by step 1,500

Frozen evaluation on reserved questions. "Edit" is the doc-15 pooled inversion edit at the block
(intended rate). "Without route variance" (D1) replaces the block by its mass class means.

| seed | checkpoint | ID | OOD | block IIA(M) | block edit | copy | route share (upstream / readout) | ID without route variance |
|---|---|---|---|---|---|---|---|---|
| 600 | 600 | 0.26 | 0.13 | 0.15 | 0.08 | 0.09 | 0.24 / 0.26 | 0.25 |
| 600 | 1,500 | **1.00** | 0.21 | **0.94** | **0.82** | 1.00 | 0.55 / 0.71 | **1.00** |
| 600 | 3,000 | 1.00 | 0.22 | 0.94 | 0.87 | 1.00 | 0.55 / 0.71 | 1.00 |
| 600 | 6,000 (control) | 1.00 | 0.22 | 0.94 | 0.89 | 1.00 | 0.56 / 0.71 | 1.00 |
| 601 | 600 | 0.24 | 0.15 | 0.16 | 0.09 | 0.08 | 0.15 / 0.17 | 0.24 |
| 601 | 1,500 | **1.00** | 0.49 | **0.95** | 0.64 | 1.00 | 0.75 / 0.84 | **1.00** |
| 601 | 3,000 | 1.00 | 0.52 | 0.95 | 0.65 | 1.00 | 0.82 / 0.87 | 1.00 |
| 601 | 6,000 (control) | 1.00 | 0.52 | 0.95 | 0.65 | 1.00 | 0.85 / 0.90 | 1.00 |
| 602 | 600 | 0.25 | 0.15 | 0.15 | 0.08 | 0.08 | 0.25 / 0.22 | 0.25 |
| 602 | 1,500 | **0.99** | 0.47 | **0.91** | **0.08** | 0.97 | 0.80 / 0.89 | **0.80** |
| 602 | 3,000 | 0.98 | 0.46 | 0.91 | 0.09 | 0.98 | 0.83 / 0.89 | 0.83 |
| 602 | 6,000 (control) | 1.00 | 0.53 | 0.95 | 0.10 | 1.00 | 0.85 / 0.90 | 0.85 |

**What the trajectories show.**
- **Formation:** in all three networks the mass representation forms **between steps 600 and
  1,500**. At 600, nothing is localized: interchange is about 0.15, accuracy about 0.25.
- **Seed 600, canonical-like.** The pooled editor already works (0.82–0.89). The unit/position
  variation is **not used**: removing it leaves accuracy at 1.00.
- **Seed 601, intermediate.** Moderately editable (0.65). The variation is not needed for ID
  accuracy (1.00 without it).
- **Seed 602, route-dependent.** The block is causally sufficient (interchange 0.91–0.95) and
  copying works (0.97–1.00), but the pooled editor fails (0.08–0.10). The unit/position
  **variation is used**: removing it drops accuracy to 0.80–0.85.
- **Probes vs causal evidence.** In all three networks the linear probe for mass at the block is
  only 0.47–0.68, even where interchange is ≥ 0.9. Probes would have under-reported these
  representations; the causal tests were decisive.
- **Upstream organization drifts slightly, but its regime is fixed early.** The upstream
  route-variance share rises a little from step 1,500 to 6,000 in every network (0.55 → 0.56,
  0.75 → 0.85, 0.80 → 0.85). Which regime a network is in never changes after step 1,500.

## 23.4 The correction decision at step 3,000 (calibration only), and D2

| seed | α tried | V1 edit gain | V2 IIA | V3 accuracy | V4 agreement | outcome |
|---|---|---|---|---|---|---|
| 600 | 1.0 / 0.5 / 0.25 | +0.008 / +0.008 / +0.003 | 0.000 | 0.000 | 1.000 | **not applied**: nothing to gain (V1 fails) |
| 601 | 1.0 | +0.146 | −0.016 | −0.006 | 0.994 | fails V2, V3 |
| 601 | **0.5** | **+0.073** | −0.001 | 0.000 | 1.000 | **applied** |
| 602 | 1.0 / 0.5 / 0.25 | +0.004 / −0.003 / 0.000 | −0.47 / −0.19 / −0.04 | −0.48 / −0.17 / −0.04 | 0.52 / 0.83 / 0.95 | **not applied**: destroys the calculation |

**Diagnostic D2.** The same validation was run at every trunk checkpoint, never applied:
- **At step 600** no network is fixable: there is nothing yet to canonicalize.
- **By step 1,500** each network's verdict is already the one it gets at 3,000 and 6,000:
  - seed 600: no gain;
  - seed 601: gain without harm;
  - seed 602: destructive.
- **D1 predicts the fold's safety.** It is harmless exactly where removing unit/position variation
  leaves accuracy intact (seeds 600, 601), and destructive where it does not (seed 602).

## 23.5 Did correcting *during* training help? (seed 601)

| network state | block edit intended | ID | OOD |
|---|---|---|---|
| control at 3,000 (before correction) | 0.651 | 1.000 | 0.516 |
| correction right after installation (3,000-edited) | **0.732** | 1.000 | 0.517 |
| correction at 4,500 | 0.708 | 1.000 | 0.522 |
| correction at 6,000 | 0.705 | 1.000 | 0.523 |
| control at 6,000 | 0.650 | 1.000 | 0.524 |
| **post hoc: same fold (α = 0.5) applied to the control at 6,000** | **0.722** | 1.000 | 0.522 |

**Against the registered criteria:**
- **C3 (persistent readout correction installed): yes.** Edit +0.081 on frozen evaluation; ID
  unchanged; 4,352 parameters changed (`W_in` columns 0–15 and `b_in`); ‖ΔW_in‖/‖W_in‖ = 0.50.
  That is a large relative change to those columns.
- **C4 at step 6,000: "improved"**, narrowly: Δ edit +0.055 (threshold +0.05), Δ ID 0.000, Δ OOD
  −0.001.
- **The weight change persisted:** ‖W_in(corr) − W_in(ctrl)‖/‖W_in(ctrl)‖ = 0.503 → 0.495 →
  0.494.
- **The behavioural benefit decayed** from +0.081 to +0.055.
- **C5 (upstream reorganization): no.** Upstream route share at 6,000 is 0.840 (correction) vs
  0.845 (control). Readout share is 0.877 vs 0.896. Continued training did not reorganize how the
  representation forms; it partly eroded the readout correction.

**Post-hoc comparison (not registered).** The same fold applied **after** training gives 0.722,
higher than the mid-training branch's 0.705, with the same accuracy. **Timing the correction at
mid-training bought nothing** in this network.

## 23.6 The four claims, separated

| seed | C1 representation identified | C2 behaviour changed causally | C3 persistent correction installed | C4 improved later training | C5 upstream changed |
|---|---|---|---|---|---|
| 600 | yes (step 1,500) | yes (copy 1.00 at 3,000) | no: nothing to gain | not run | — |
| 601 | yes (step 1,500) | yes (copy 1.00) | **yes** (α = 0.5) | **improved by the registered criterion (+0.055)**; but the post-hoc fold after training does as well or better | no |
| 602 | yes (step 1,500) | yes (copy 0.98) | no: destroys the calculation | not run | — |

## 23.7 Side observations

- **Non-declared acceleration:** never localized by DAS (IIA 0.14–0.17), in either branch.
  Unaffected by the correction.
- **Where held-out failures come from.** D1 replaces the block with the **true** mass class mean.
  For held-out unit–value questions that supplies the correct mass, and OOD accuracy rises from
  0.22 / 0.52 / 0.53 to 0.61 / 0.89 / 0.78. That diagnostic leaks the label, so the comparison is
  not about route variance. It does locate much of the held-out failure **upstream**, in forming
  mass for unseen unit–value combinations, rather than in the calculation.
- **The DAS-located subspace overlaps the designed block only 35–44%,** although both give the
  same interchange accuracy. Mass is also carried outside the block.

## 23.8 Compute, storage and housekeeping

- **CPU:** 0.69 CPU-hours in total (the registration estimated about 1.6). Training 0.49 h,
  audits 0.06 h, correction fitting and validation 0.04 h, fidelity reference 0.11 h. Per-phase
  numbers are in `results/correction/summary.md`.
- **Storage: 22 MB.** Each `acts.pt` was first written as a **view** of the full activation
  tensor (4.5 MB each, 114 MB in total). The files were re-saved as clones with **identical
  values**, verified before overwriting, and the run's `storage_bytes` field still records the
  original size. Full training states are stored at every trunk checkpoint and at step 6,000.

## 23.9 What this means, and what it does not

- **The central question, in this pilot:** a single verified correction at the midpoint did
  **not** produce a better final network than training uninterrupted and then editing.
  - Where the correction was safe, the same edit applied after training was at least as good.
  - Where the network needed its route-specific organization, the correction was unsafe and was
    correctly rejected.
- **The intervention window lies earlier than 50%.** Organization was settled by step 1,500 in
  every network and was absent at step 600. An intervention at 50% acts after the representation
  is fixed.
- **A readout correction is not a representation correction.** Training neither adopted it
  upstream nor fully kept it.
- **Exploratory, n = 3.** One task, one arm, one correction family, one branch point. Three
  networks are enough to show three distinct regimes. They are not enough to estimate how common
  each regime is.

## 23.10 Proposed next steps (supervisor proposals; see `results/correction/supervisor_notes.md`)

1. **Move the intervention into the formation window (steps 600–1,500),** with multiple
   checkpoints, as the owner anticipated. Test whether a correction **during** formation changes
   upstream organization; one applied after formation, as here, did not.
2. **Use D1 as a gate.** Canonicalization should be attempted only where removing route variance
   preserves accuracy. For route-dependent networks (like seed 602), a unit-aware correction
   (doc 17 style) is the right family, not canonicalization.
3. **Correct upstream directly.** Edit the representation-producing weights so the block itself
   changes, and measure whether training keeps it.
