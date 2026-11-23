# 29. Adaptive Concept Lineage and Causal Supervision: pilot results (exploratory, one matched pair)

*Registration: doc 28, commit `41ae23e`, made before training. Spec v1.0. Auto-generated tables:
`results/lineage/summary.md`. Journal (60 reviews): `results/lineage/journal.jsonl`. Records,
trackers, inspections and snapshots: `runs/zoo_lineage/`. Figures: `results/lineage/fig_lineage_*.png`.*

**Bottom line.** The lineage engine reconstructed a detailed, evidence-linked developmental
history across 61 checkpoints (Q2). The supervisor filed 60 state-specific reviews and ran 21
targeted investigations, including two dry-run interventions. It **installed no edit**, because no
registered intervention had a credible benefit without a generalization cost (Q1). The two networks
are therefore bitwise identical, and the registered outcome is **not tested**.

The pilot's main scientific findings come from the supervisor's investigations:
- **Editability depends on the edit's algebra.** Inversion edits succeed (0.96); the held-out add-one
  edit fails (0.09).
- **Generalization fails upstream.** Held-out failures arise in layer-1 mass formation, with
  different failure modes for the declared mass and the undeclared acceleration.
- **The development split was optimistic.** Held-out values 11–12 generalize far worse (0.35) than
  values 9–10 (0.65).

## 29.1 Deviations and disclosures

1. **Read-only tools added mid-run** (the registered rules did not change; nothing could alter
   training):
   - `predictions` (step 900) and `ood_locus` (step 1,300) in `run_lineage.py`;
   - ad hoc analyses on copies or stored snapshots: margins (3,500), error structure (3,600), writer
     scaling (4,100), and `posthoc_bias_swap.py` (2,700). All are logged in `inspections.jsonl`.
2. **Stationary-phase helper (from step 3,300).** Reviews were filed with
   `reviews/stationary.py`. It fills `what_changed` from the actual record deltas and refuses to file
   when any metric leaves its stationary band. The hypothesis and key question are written per
   checkpoint.
3. **One review written before its packet was read (step 5,600).** It was chained with step 5,700
   in one command. All its deltas were inside the bands; this is flagged in the 5,700 review.
4. **Post-hoc lineage audit** (`report_lineage.audit_events`, below). It flags ambiguous head
   matches, threshold jitter and k-means repartitions. The registered matcher was not changed
   mid-run, so records stay consistent.

## 29.2 Q2: the reconstructed developmental history (A = B, seed 701)

| phase | steps | what the evidence shows |
|---|---|---|
| plateau: zero detector plus a constant | 100–1,000 | accuracy ≈ 0.25. Answers are 0 exactly when F = 0, otherwise mostly 8 or 12. No candidate function or discrete-log coset is above chance (`predictions`). It is **not** doc 12's sign trap (−F 0.23). No variable is causally used anywhere (interchange ≈ 0.06–0.17) |
| hidden structure on the plateau | 200–800 | the block keeps equivalent quantities closer than different masses (cosine 0.998 vs 0.95) **before** any causal use. Unit groups' gradients conflict at step 100, then reconverge. A large silent basis rotation at step 700 (attn1.qkv 46%, W_out 43% per 100 steps; MLP identities churn) does not start formation. Sharpness 15, far below Adam's edge (1,001) |
| discrete-log circle | 1,100 | mass's multiplicative character at h jumps 0.31 → 0.81. The mass conversion is learned **before** the acceleration conversion (answers agree with "acceleration unit ignored" as often as with F). The first causally supported neuron clusters appear (ablation) |
| formation | 1,100–1,500 | accuracy 0.31 → 0.995; block IIA 0.16 → 0.90; pooled edit 0.10 → 0.89; mass-family neurons 26 → 119. **Two balanced writer heads** emerge: attn2 head 0 (reads value and unit tokens) and head 3. Their unit splits differ (head 0 splits off unit 2) and partly cancel in the block (block interaction 0.39 vs 0.67–0.77 per head) |
| registered timings | | mass first causal **1,300**, identified (IIA ≥ 0.9, accuracy ≥ 0.95) **1,700**; answer F decodable 1,300; acceleration never causal (layer-1 token interchange ≤ 0.30); no product M·A intermediate; **no coalescence** (unit-transfer ratio ≤ 0.61) |
| spike and attractor | 1,600–1,700 | an ordinary loss spike (all metrics drop together; identities stable), then recovery to the same organization in 100 steps |
| converged, held-out jitter | 1,700–4,000 | OOD_DEV oscillates (mean 0.65, sd 0.03 over 1,500–6,000) while ID = 1.000. **Near-margin inputs flip** (held-out margin median 5.1 vs ID 11.0). Not carried by the attention biases (snapshot swap) but by the attention weight matrices jointly |
| frozen | 4,200–6,000 | updates < 0.002, then < 5e-4. Nothing changes |

**Persistent concept records.**
- B has 106 lineage IDs, 27 of them ever causally supported.
- The block mass subspace (B-su-0020) has a clean causal history: hypothesized → partial at 1,300 →
  verified from 1,700.
- Neuron-cluster lineages are fragmented into many short-lived IDs.
- No attention head ever reaches causal status: mass is written jointly, so single-head interchange
  is only 0.13–0.19.

**Lineage-event audit.** 209 non-persistence events:

| class | events |
|---|---|
| ambiguous head relocations (near-identical, diffuse heads) | 19 |
| status jitter at a threshold, reverting within two checkpoints | 45 |
| k-means repartitions | 8 |
| **retained** | **137 (65%)** |

**Instrument limitations found during the run** (for the next lineage engine):
- matches need a best-vs-runner-up margin, not an absolute similarity;
- causal-status thresholds need hysteresis;
- fixed-k clustering creates spurious splits and merges in a frozen network;
- **ID-panel lineage is blind to generalization-relevant changes.** Held-out mass placement moved
  while every ID feature was stable (step 2,500).

## 29.3 Q1: the supervisor's decisions

**Reviews:** 60 (L1 39, L2 21, L3 0); 25 logged inspections; 2 dry runs (600 extra gradient steps);
**0 gate submissions, 0 installed edits**.

**Why no edit was justified:**
- **100–1,200 (early phase).** No causal code existed, so there was nothing to organize. Pushing the
  network off its plateau would only be extra optimization, which the spec forbids crediting to
  supervision.
- **1,300 and 1,700 (`ood_locus`).** The held-out gap is upstream:
  - canonical mass is decodable at the value token for 0.86 of ID questions but 0.41 of held-out
    ones;
  - with the true mass in the block, held-out accuracy would be 0.92–0.94, so the downstream
    calculation is fine.

  No registered primitive targets layer-1 token binding.
- **1,500 dry run (interaction removal).** Interaction −0.22, editability +0.04, but
  **OOD_DEV −0.017**. The pre-written falsifier was met: the residual interaction carries
  generalization-relevant information. The precursor (seed 700) showed the same.
- **1,800 dry run (upstream, OOD-targeted).** OOD_DEV −0.026, as predicted: no registered term
  encodes compositional binding.
- **The network was already highly editable** (0.89–0.98 from step 1,500), so there was no
  editability defect to repair.

**Hypotheses tested and refuted during the run:**
- stable unit-1 gradient conflict (100 → 200);
- the circle forms before IIA > 0.3 by step 900 (falsified at 900);
- "unit 1 is special across seeds" (refuted at the writer level, 1,200);
- editability falls as the interaction grows (1,300);
- monotone generalization erosion (2,200);
- attention biases carry the held-out oscillation (2,700).

**Hypotheses supported:**
- the 1,600 spike recovers to the same organization (1,700);
- near-margin explanation of held-out jitter (3,500);
- writer balance supports editability (4,100: scaling either writer by 0.75× raises interaction and
  cuts editability 3–4× more than accuracy; the caveat is general damage).

**Proposed new primitives** (for a future registration; not added to this pilot):
1. layer-1 compositional binding supervision: interchange the value token and the unit token
   separately, with matched counterfactuals;
2. checkpoint-anchored upstream regularization: per-tensor early stopping on the development split;
3. held-out-panel lineage.

## 29.4 Sealed final evaluation (step 6,000, once; A = B)

| measure | value |
|---|---|
| ID accuracy (FINAL) | 1.000 |
| held-out combinations, values 11–12 (OOD_FINAL) | **0.354** |
| development held-out, values 9–10 (OOD_DEV) / legacy set | 0.647 / 0.481 |
| block IIA(M) / DAS IIA(M) / DAS–block overlap | 0.951 / 0.951 / 0.354 |
| copy into block | 1.000 |
| **persistent inversion edit** (block / DAS): intended / wrong / fixed points preserved | **0.960 / 0.000 / 0.988**; 0.953 / 0.000 / 0.977 |
| **held-out add-one edit** (block / DAS) | **0.094** / 0.099 (wrong 0.07) |
| ‖ΔW_in‖/‖W_in‖ of the folds (inversion / add-one) | 0.78 / 1.44 |

**Registered outcome (doc 28 §28.7): not tested.** No edit was installed.

**What the sealed numbers add:**
1. **Editability follows the representation's algebra.**
   - Mass is coded on a discrete-log circle (multiplicative character 0.97).
   - **Inversion** is negation of the discrete log, a reflection of that circle, and an affine fold
     implements it almost perfectly (0.96, no wrong-variable changes, 99% of fixed points
     preserved).
   - **Add-one** has no linear form in multiplicative coordinates, and the same editor fails (0.09).
   - So the "highly editable" network is editable for **edits that are linear in its own geometry**.
     Doc 25's network failed both (0.12 / 0.13) because its route split broke even the reflection.
   - This was the add-one edit's first and only measurement (it was held out from every target).
2. **The development split overstated generalization.** Values 11–12 (sealed) reach 0.35 against
   0.65 for values 9–10.
   - Combined with the error analysis (3,600), held-out mass errors match no unit's conversion,
     while held-out acceleration errors are a **default-unit fallback** (97.8% equal the answer with
     scale 1).
   - Generalization to unseen unit–value pairs degrades with the value's distance from the trained
     range.
3. **Benchmark (evaluation only).** The full, unbudgeted interaction-removal edit on this network
   costs 1.8× the producer rows (seed 700: 13×). It cuts the interaction 0.34 → 0.16 with ID 1.000,
   OOD_FINAL 0.361 (+0.007) and inversion 0.959. On this canonical-like network removal is almost
   free; on seed 700 it was destructive.

## 29.5 Costs

| item | CPU time / count |
|---|---|
| base training (both networks) | 1,074 s |
| observation (per network: atlas 114–161 s, tracer 6 s, lineage 6–7 s, weights 43 s) | ≈ 0.10 CPU-h total |
| supervisor inspections | 175 s |
| dry runs | 600 extra gradient steps |
| sealed evaluation | 29 s |
| **total** | **≈ 0.45 CPU-h** |

- **Storage:** 83 MB on disk. A and B are bitwise identical, so git stores about 42 MB of unique
  blobs.
- **Supervisor effort:** 60 reviews. That is the dominant cost, and it is not CPU.

## 29.6 Verified, provisional, failed, open

**Verified (causal tests; this network):**
- mass causal in the block from 1,300 and verified from 1,700;
- not causal in dims 16–63;
- acceleration never causal (≤ 0.30);
- persistent inversion edit 0.96 with no wrong-variable changes;
- add-one edit 0.09;
- held-out failure located upstream (true-mass diagnostic);
- the 1,500 / 1,800 interventions would have cost generalization (dry runs).

**Provisional:**
- writer balance as the reason for editability (supporting scaling test; confounded with general
  damage);
- algebraic explanation of edit-specific editability (consistent with the circle geometry; one
  network);
- hidden equivalence structure on the plateau;
- near-margin held-out jitter;
- the default-unit fallback for acceleration.

**Failed or negative:**
- no intervention was justified, so Q1 is not tested;
- lineage-event stream 35% artifacts;
- my 600 / 1,200 / 1,300 / 2,000 / 2,600 predictions were wrong;
- development OOD unrepresentative.

**Open:**
- Does the layer-1 binding primitive fix held-out mass formation?
- Is writer balance causal under a norm-preserving rebalance?
- Does an undeclared-mass network fail held-out mass with a default-unit fallback, as acceleration
  does?
- Across seeds, how often is the canonical-like regime reached?

## 29.7 Answer to the central question, for this pilot

**Watching the network learn worked.** The supervisor could follow, explain and falsify claims
about how this network built mass, at 100-step resolution, from weights, gradients, activations
and causal tests.

**Deliberately reorganizing it did not happen, for documented reasons.** The network organized mass
well on its own. Its real weakness, generalization to unseen unit–value pairs, lies where the
registered editing primitives cannot reach. Each registered reorganization that was tried
traded generalization for organization.

The pilot's case for "programming" networks therefore rests on two findings, and the evidence points
to layer-1 binding as the next target:
- precise **diagnosis**: where generalization fails, and why editability is edit-specific;
- a precise **specification** of which new primitives would be needed.
