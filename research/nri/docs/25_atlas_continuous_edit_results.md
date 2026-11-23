# 25. Two-trajectory concept-formation atlas and continuous upstream edit loop: results (exploratory, one matched pair)

*Registration: doc 24, commit `76c3dc2`, made before either network was trained (Issue #22).
Frozen analysis: `run_atlas.py analyse / final / report` (`report_atlas.py`); full tables in
`results/atlas/summary.md`. Post-hoc analyses (§25.8) are labeled and come from `posthoc_atlas.py`.
Networks, 61 snapshots each and activation records: `runs/zoo_atlas/`.*

- **No deviations from the registered procedure.**
- **One registered measure is undefined by construction:** the held-out *doubling* edit (§25.6).
- **Bottom line in one sentence:** the atlas worked and resolved *when, where and how* mass formed
  in this network; the registered upstream edit was proposed 51 times and **never accepted**, so the
  two networks stayed bitwise identical and the central claim was **not tested**.

## 25.1 Answers in plain English

**When did the network begin to represent mass?**
- **Decodable** (descriptive only) from step **400**, at the layer-1 residual of the *value token*.
- **Causally used** (block interchange ≥ 0.5) from step **1,000**.
- **Causally verified** (≥ 0.9) and behaviourally solved from step **1,200**: accuracy 0.99,
  block IIA 0.92.
- PR #21's 600–1,500 window is now resolved: formation happens between **900 and 1,200**.

**Where did it form?**
- **Inside the declared block.** 98% of the mass class-mean variance at h lies in dims 0–15.
- Swapping dims 16–63 never transfers mass (IIA 0.07).
- DAS finds an equally causal subspace (IIA 0.94) that overlaps the block only **37–44%**. A
  decodable/causal geometry partly outside the block exists, but **the outside dims alone are not
  sufficient**.

**How was it built?**
- **Layer 1 fuses unit into value.** By step 6,000, swapping only the layer-1 residual of the
  mass-*value* token makes the output follow the **source's canonical mass** 90% of the time (raw
  value: 38%). Swapping only the *unit* token leaves the answer unchanged 89% of the time (§25.8 P4).
  The network does not keep a separable "unit" variable that it later applies: it converts at the
  value token, inside layer 1.
- **Layer-2 attention head 0 assembles it.** It carries the most mass of any single head (IIA 0.23)
  and reduced the mass-interchange loss most during formation (−0.25 attribution over 500–2,000),
  with layer-1 heads 1 and 3. `attn1.qkv` is the parameter group whose change did most to reduce
  both the task loss and the interchange loss in the formation window.
- **The code is a discrete-log circle.** Mass class means at h span **two dimensions** (rank at 90%)
  and their geometry is **multiplicative**: 0.83 of the class-mean variance sits on one discrete-log
  frequency, against 0.37 additive. This circle forms at steps 800–900, **just before** mass becomes
  causal. Doc 20's calculator geometry has now been watched forming.

**Did equivalent quantities share one representation, or travel separate routes?**
- **Neither, cleanly: two routes, and they *split* at formation rather than coalesce.**
- Before formation (steps 200–800) the three units' mass configurations in the block were nearly
  identical (per-unit CKA 0.95–0.99), **but that shared code was not causal**.
- As mass became causal (900–1,000), unit 1 (scale 5) **split off**. From step 1,000 to 6,000:
  - units 0 and 2 share a configuration (CKA 0.98–0.99);
  - unit 1 differs (CKA 0.44–0.60 to the others).
- The mass × route **interaction** in the block rose from 0.05–0.20 to **0.45** at step 1,000 and
  **0.66** at 6,000. The unit-transfer ratio settled at 0.53–0.59 ("partially shared").
- **No registered coalescence occurred.** The trajectory moved *away* from a shared code once the
  code became causal.

**Did the interventions influence the subsequent organization?**
- **No intervention was ever installed.** All 51 eligible proposals (steps 1,000–6,000) failed E1:
  the targeted edit improved pooled editability by at most ≈ +0.005 against the registered +0.01.
- At steps 1,000–1,100 proposals also failed accuracy and behaviour checks. Steps 100–900 were
  skipped (no causal representation yet).
- Control and supervised are **bitwise identical** at every checkpoint (same state hashes, same batch
  digests; final hash `f412472d…`).

**How did the supervised model's development differ from the control?** It did not; see above.

## 25.2 Integrity

- **61 checkpoints per trajectory**, each with `model.pt`, `acts.pt` and a full JSON card. Full
  resumable training state at 0, 1,000, …, 6,000.
- Batch digests are identical at all 61 checkpoints.
- No edit was accepted, so the trajectories are bitwise identical, as the registered no-op test
  requires.
- **Tests:** `tests/test_atlas.py` (9) and `tests/test_correction.py` (3) pass. They cover:
  - reproducible initialization and identical batch streams;
  - instrumented training is bitwise `run_swap.train_one`;
  - a no-op supervised run equals the control;
  - exact checkpoint restore (model and optimizer);
  - the edit touches only `attn2.o` rows 0–15 and bias, within budget;
  - a decision never mutates model or optimizer;
  - the budget gate;
  - the data partition is disjoint and complete.

## 25.3 Formation timeline (control = supervised)

| step | ID (OBS) | OOD_DEV | block IIA(M) | copy | pooled edit | interaction share | per-unit CKA (block) | M mult. char. (h) | what changes |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.07 | 0.08 | 0.09 | 0.08 | 0.08 | 0.23 | 0.56 | 0.49 | — |
| 300 | 0.23 | 0.16 | 0.15 | 0.05 | 0.05 | 0.05 | 0.95 | 0.32 | MLP: product/answer-sensitive (M·A, F) neurons appear, not causal |
| 600 | 0.25 | 0.17 | 0.17 | 0.10 | 0.09 | 0.21 | 0.99 | 0.41 | mass decodable at the value token (κ 0.61); first M-family neurons (15) |
| 800 | 0.34 | 0.18 | 0.15 | 0.10 | 0.06 | 0.14 | 0.95 | 0.68 | discrete-log circle forming; neuron identities churn (40 relocated) |
| 900 | 0.51 | 0.26 | 0.27 | 0.27 | 0.02 | 0.33 | **0.75** | 0.82 | largest reorganization (CKA to previous h 0.57); routes split |
| 1,000 | 0.80 | 0.46 | **0.65** | 0.70 | 0.04 | 0.45 | 0.67 | 0.83 | **mass causal**; first proposals (rejected) |
| 1,100 | 0.95 | 0.53 | 0.83 | 0.90 | 0.07 | 0.50 | 0.71 | 0.82 | M-family neurons 57; identities settle (189 stable) |
| 1,200 | 0.99 | 0.66 | **0.92** | 0.98 | 0.07 | 0.57 | 0.68 | 0.81 | **identified (C1)** |
| 1,500 | 1.00 | 0.71 | 0.94 | 1.00 | 0.10 | 0.63 | 0.68 | 0.82 | organization fixed (h CKA to previous 0.999) |
| 3,000 | 1.00 | 0.71 | 0.94 | 1.00 | 0.12 | 0.65 | 0.66 | 0.83 | acceleration first causal (L1 tokens, 0.51) at 3,200 |
| 6,000 | 1.00 | 0.72 | 0.94 | 1.00 | 0.12 | 0.66 | 0.66 | 0.83 | — |

**Other concepts** (registered criteria; `results/atlas/atlas.json`, `events.json`):
- **Raw values and unit tokens:** decodable from initialization at their own token positions
  (embedding identity). Never causal as separable variables: unit-token swap IIA 0.00 after 1,100;
  raw-value swap 0.38. They are consumed into canonical mass at layer 1.
- **Acceleration (not declared):** decodable from 1,100. **Causal only from 3,200**, and only via the
  layer-1 tokens (0.51). Never localized in the block (IIA 0.07) or by DAS (0.19).
- **Answer F:** decodable only in the MLP, from step 1,000.
- **Product M·A:** never reaches κ 0.5 anywhere (best 0.37, MLP). **No linearly decodable product
  intermediate** was found.
- **Output unit:** decodable at its token from initialization. Causal interchange of dims 16–63 for
  S is 0.08: not localized there.

## 25.4 Features, clusters and weights

- **MLP neuron families** (argmax η² ≥ 0.25):
  - **Steps 0–300:** turnover from input-coded neurons (a, m, units) to product/answer-sensitive
    neurons (M·A 79, F 58 at step 300). These are **not causally important**: ablation importance is
    ≈ 0.
  - **Steps 600–1,500:** mass-selective neurons grow 15 → 76.
  - **End of training:** M-family 73, raw-m 40, raw-a 21, mixed 98, dead 20.
  - Mean-ablating the M family drops accuracy to 0.37.
  - **The most causally important single neurons (53, 37, 1) are *mixed*,** not mass-selective. A
    "mass neuron" label from selectivity would point at the wrong units.
- **Identity tracking:** Hungarian matching of response signatures shows the network's functional
  units **churn during formation and freeze afterwards**:
  - 40 relocated at step 800, 39 at 900;
  - stable neurons: 12 at 800 → 216 at 1,200 → 242 at 6,000.
- **Weights:** every tensor keeps moving after 1,500 (`fig_atlas_weights.png`), while h's geometry
  is frozen (CKA to previous ≥ 0.999). Weight change after formation is not reorganization.
- **Attribution:** first-order path attribution reproduces the measured loss changes
  (`attribution.*.actual_change` vs `total_first_order` in each card).
  - **Caution: the IIT cross-entropy on the fixed FIT batch rises again after step 1,100** while
    interchange accuracy stays at 0.94. Confidence on the residual wrong pairs grows.
  - So "IIT attribution" after formation measures that sharpening, not representation-building.

## 25.5 The continuous edit loop

| | control (shadow) | supervised (applied) |
|---|---|---|
| skipped (no causal representation, steps 100–900) | 10 | 10 |
| rejected | 51 | 51 |
| accepted | 0 | 0 |
| failing checks among rejections | E1 51, E2 1, E3 2, E4 2 | same |

**Why the edit could not get traction (post hoc, §25.8 P1–P2):**
- **The edit targeted the right tensor.** Before formation, half the block's interaction came from
  the layer-1 residual at the last position. From step 1,500 on, **95–97%** comes from the `attn2`
  output, which is exactly what rows 0–15 of `attn2.o` write.
- **But removing it requires a change 4–13× those rows' norm.** The full ridge solution needs
  3.7× the rows' norm at step 1,000 and 13.1× at step 6,000. Even then, only part of the
  interaction is removed (0.66 → 0.18), and accuracy collapses to 0.60–0.63.
- At the registered ≤ 25% budget, the edit removes ~2% of the interaction. That is far too little to
  change editability.
- **Applied whole at step 6,000 (evaluation only),** the full solution raises the pooled inversion
  edit to 0.21 (from 0.12) but cuts ID accuracy to 0.62, held-out accuracy to 0.25 and block IIA to
  0.53. That is a **destructive trade-off, not an improvement**.
- **Interpretation:** in this network the route split *is* the computation. Unit 1's separate
  geometry is how the network handles that unit; it is not removable noise.
  - D1 (block replaced by mass class means) keeps ID accuracy at 0.95, so the *exact* route
    variance is mostly not needed.
  - But no small linear change of the producer removes it without breaking the map from heads to
    block that the MLP depends on.

## 25.6 Sealed final evaluation (step 6,000, once)

| measure | control = supervised |
|---|---|
| ID accuracy (FINAL, 2,205 reserved questions) | 1.000 |
| held-out combinations, OOD_FINAL (values 11–12) | **0.388** |
| held-out combinations, legacy set / OOD_DEV | 0.529 / 0.719 |
| block IIA(M) / dims 16–63 IIA(M) / block IIA(A) | 0.953 / 0.071 / 0.077 |
| DAS IIA(M), DAS–block overlap | 0.954, 0.373 |
| copy real block (activation) | 1.000 |
| **persistent** pooled inversion edit, block: intended / wrong / other | 0.119 / 0.065 / 0.703 |
| same at DAS | 0.133 / 0.049 |
| held-out add-one edit, block / DAS | 0.131 / 0.114 |
| unedited-behaviour preservation of the inversion edit (fixed points M ∈ {0, 1, 12}) | 0.708 |
| ‖ΔW_in‖ / ‖W_in‖ of the fitted inversion fold | 3.70 |

- **Registered outcome: not tested** (no edit accepted).
- **The held-out doubling edit is undefined.** Doubling mass and doubling acceleration give the same
  answer (F = M·A·S), so no question is eligible under the variable-discriminating scorer. Doc 12
  had already identified this; including it in doc 24 was a design error, and it is reported, not
  replaced.
- **Activation vs persistent:** copying a real block pattern works perfectly (1.00), but every
  persistent linear edit fails (0.11–0.13). The fitted fold is huge (3.7× `W_in`) and damages 29% of
  answers it should leave alone.
- **Post-training benchmarks:**
  - the doc-22 readout fold passes no α;
  - the upstream policy applied once at step 6,000 is rejected (E1);
  - the unbudgeted upstream edit is destructive (§25.5).
- **This network sits in PR #21's route-dependent regime** (seed 602: pooled edit 0.10).

## 25.7 Compute and storage

| phase | control | supervised |
|---|---|---|
| training | 564 s | 564 s |
| observation (atlas, every 100 steps) | 114 s | 116 s |
| weights, gradients, attribution | 46 s | 46 s |
| policy (proposals, validation, shams, IIT comparator) | 79 s | 80 s |
| offline milestones (DAS, D1, ablations; both) | 472 s | |
| sealed evaluation (both) | 42 s | |
| **total** | **0.59 CPU-hours** (single thread) | |

- **Audit overhead:** 42% of training compute (observation 20%, weights 8%, policy 14%). Total compute
  was matched because the control ran the policy in shadow mode.
- **Storage:** 35.5 MB per trajectory: 61 × 280 KB snapshots, 7 full states, 61 activation records.
  Analysis outputs: 5 MB in `results/atlas/`, including `history_long.csv`, 64k rows queryable by
  trajectory × step × key.
- **Against the registration's estimate:** 15 CPU-min per trajectory and 65 MB were estimated; the
  actual was 13.4 min and 71 MB.

## 25.8 Post-hoc analyses (not registered)

- **P1 (where the interaction comes from):** the share of the block's mass × route interaction table
  carried by the `attn2` output rose 0.51 → 0.70 → 0.84 → 0.95 → 0.97 at steps 600 / 900 / 1,000 /
  1,500 / 6,000. The rest comes from layer 1 at the last position.
- **P2:** the norm needed by the full interaction-removal edit, as a multiple of the rows' norm,
  was 0.9× at step 600, 3.7× at 1,000, 9.9× at 1,500 and 13.1× at 6,000.
  - **The window in which a small upstream edit could reshape the code closes as the code forms.**
  - At step 600 a full-strength edit was only 0.9× the rows' norm, but there was no causal
    representation yet, so the registered readiness gate correctly skipped it.
- **P3 (per-unit CKA by pair at the block):** units 0–2 stay at 0.90–0.99 throughout. Units 0–1 and
  1–2 drop from 0.94–0.99 (step 800) to 0.44–0.60 (step 1,000 onwards).
- **P4 (fusion at layer 1):** after swapping only the mass-value token's layer-1 residual, the output
  follows the source's canonical M 0.26 → 0.71 → 0.90 (steps 1,000 / 1,500 / 6,000). It follows the
  raw value in the base's unit only 0.16 → 0.31 → 0.38. Swapping only the unit token leaves the
  output unchanged 0.69 → 0.91 → 0.89.

## 25.9 Verified, provisional, failed, open

**Verified (causal tests on held-out panels, this network):**
- mass becomes causally used at step 1,000 and causally verified at 1,200, inside the declared block;
- the unit is fused into the value token within layer 1;
- the block mass code is not sufficient outside dims 0–15;
- acceleration becomes weakly causal only late (3,200), via layer-1 tokens;
- organization froze by step 1,500.

**Provisional (descriptive geometry, single network):**
- discrete-log circle formation preceding causal use;
- the route split at formation (units 0/2 shared, unit 1 separate);
- neuron-identity churn then freeze;
- attribution to `attn1.qkv` and layer-2 head 0;
- "mixed" neurons as the most important units.

**Failed:**
- the registered upstream edit was never accepted (51/51 E1 failures), so there is **no evidence
  either way** on whether continuous upstream supervision helps;
- persistent linear edits fail on this network (0.12) although activation copying succeeds (1.00);
- the doubling held-out edit was an invalid measure.

**Open:**
1. Would a **non-linear or unit-aware** upstream edit (doc 17/19 style, acting per route) succeed where
   a linear ridge edit of one tensor cannot?
2. Is the **pre-formation window (600–900)**, when the full edit is cheap, the right time to act, with
   a readiness gate based on decodability or geometry rather than causal IIA?
3. Why does unit 1 split off? Scale 5 is a generator-like element whose discrete log is 9; units 0
   (scale 1, log 0) and 2 (scale 8, log 3) are not. A hypothesis only.
4. Held-out-combination accuracy is low (0.39 on values 11–12) and was never the edit's target. How
   much of it is lost upstream, at the fusion step?

## 25.10 Next step (owner-directed)

The owner has asked for a **separate, LLM-in-the-loop follow-up**. At every 100-step checkpoint the
research agent inspects the weights and activations itself and designs **bespoke** read-out and
edit methods, which may touch any weights. Each edit is still installed only if it passes fixed
deterministic safety checks on calibration data, and final test sets stay sealed. It will be
registered separately and will reuse this pilot's control trajectory (identical seed, data and
recipe) as its matched control.
