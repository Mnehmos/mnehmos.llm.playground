# 19. Permanent-edit results: the unit-aware edit can be installed once, as an added layer or within the existing weights

*Registration: doc 18, commit `e4094d3`, made before any editor was fitted on the new networks.
Analysis: the frozen `run_perm.py analyse` / `report`. Full report: `results/perm/summary.md`.
Data: `results/perm/metrics.json`. 24 **new** networks (seeds 400–411); scoring on **1,011
reserved questions per network**, never used by any editor fit, location search or tuning, each
scored once. No deviations from the registration.*

## 19.1 Verdict in plain English

**Yes on both questions, with one caveat about the size of the change.**

1. **The ability can be installed once.** Write the unit-aware table into the network as a small
   added layer, with about 70 units, one per value-and-unit pattern. The edited network then
   gives the intended answer:
   - **0.90** of the time for mass and **0.87** for acceleration, on questions none of the
     editors had seen;
   - against **0.27 / 0.34** for the linear weight edit from doc 17;
   - indistinguishable from the stored table (0.905 / 0.870);
   - with **fewer** wrong-variable changes than the linear edit.

   Nobody tells the network which value or unit it is seeing; the layer reads that from the
   network's own activity.
2. **It can be installed within the original network.** Retrain only the existing weights after
   the edit site, with no new units, to reproduce the table's answers. That reaches the same
   **0.90 / 0.87**.
   - **So the doc-17 shortfall belonged to the *linear* edit, not to weight edits in general.**
   - **The caveat:** this is not a small, surgical change. The three weight matrices it touches
     move by **35–59% of their size**. It re-fits the network's output layers; it is not a patch
     to a few numbers. Whether a much smaller change to the existing weights could do the same
     is still open.
3. **Doc 17 replicates on new networks and unseen questions.** The unit-aware table beats the
   pooled table by +0.32 (mass) and +0.24 (acceleration), in 11/12 networks each.

## 19.2 What happened, separately for mass and acceleration (new networks, reserved questions)

Intended = gave the edited answer. Wrong = changed the other variable instead. Preserve = on
reserved questions whose correct edited answer equals the original answer, the share that keep
the original answer. The unedited networks score 0.979 (mass) and 0.949 (acceleration) there.

| editor | installed how | mass: intended / wrong / preserve | acceleration: intended / wrong / preserve |
|---|---|---|---|
| current editor (doc 15) | linear, written once | 0.15 / 0.04 / 0.76 | 0.24 / 0.05 / 0.73 |
| unit-aware linear edit (doc 17) | linear, written once | 0.27 / 0.05 / 0.72 | 0.34 / 0.05 / 0.70 |
| pooled table | stored table | 0.58 / 0.04 / 0.84 | 0.63 / 0.05 / 0.83 |
| unit-aware table | stored table | 0.91 / 0.01 / 0.96 | 0.87 / 0.02 / 0.95 |
| **added layer** (Q1) | **written once, ~70 added units** | **0.90 / 0.01 / 0.95** | **0.87 / 0.02 / 0.95** |
| **existing weights only** (Q2) | **written once, no new units** | **0.90 / 0.01 / 0.95** | **0.87 / 0.03 / 0.94** |
| per question (reference) | question's value and unit supplied | 0.91 / 0.01 / 0.98 | 0.87 / 0.02 / 0.95 |
| copy (benchmark) | a real example per question | 0.96 / 0.01 / 0.98 | 0.88 / 0.02 / 0.95 |

## 19.3 Registered tests

**Primary family** (Holm, 6 tests, 12 networks each, exact sign-flip). All six reject.

| test | population | Δ intended | 95% CI | p | networks > 0 | Δ wrong (p) |
|---|---|---|---|---|---|---|
| replication: unit-aware − pooled table | mass | **+0.320** | [+0.221, +0.440] | 0.001 | 11/12 | −0.025 (0.001) |
| replication: unit-aware − pooled table | acceleration | **+0.242** | [+0.166, +0.321] | 0.001 | 11/12 | −0.030 (0.002) |
| Q1: added layer − linear edit | mass | **+0.636** | [+0.520, +0.735] | 0.0005 | 12/12 | −0.035 (0.015) |
| Q1: added layer − linear edit | acceleration | **+0.524** | [+0.389, +0.646] | 0.0005 | 12/12 | −0.024 (0.023) |
| Q2: existing weights − linear edit | mass | **+0.633** | [+0.518, +0.732] | 0.0005 | 12/12 | −0.035 (0.013) |
| Q2: existing weights − linear edit | acceleration | **+0.524** | [+0.388, +0.647] | 0.0005 | 12/12 | −0.023 (0.035) |

Every gain comes with **fewer** wrong-variable changes, so all six are specific.

**Does the installed edit match the table?** The pre-registered criterion is a mean gap ≥ −0.05
and a CI lower bound ≥ −0.10.

| installed edit | mass: gap to table | acceleration: gap to table | matches? |
|---|---|---|---|
| added layer | −0.003 [−0.006, −0.001] | −0.003 [−0.006, −0.001] | **yes / yes** |
| existing weights | −0.006 [−0.009, −0.003] | −0.004 [−0.008, +0.000] | **yes / yes** |

The gaps are statistically detectable but tiny: less than 1 percentage point.

**Agreement with the table's answers** on all 2,205 reserved questions:
- added layer: 0.994 / 0.996;
- existing weights: 0.970 / 0.958;
- the linear edit, for comparison: 0.42 / 0.45.

## 19.4 Networks that trained well vs networks that struggled

Training converged in **11/12** mass-declared and **9/12** acceleration-declared networks
(ID accuracy ≥ 0.95; all networks count in every test).

| population | networks | n | linear | table | added layer | existing weights | copy |
|---|---|---|---|---|---|---|---|
| mass declared | converged | 11 | 0.27 | 0.95 | 0.95 | 0.94 | 1.00 |
| mass declared | struggling | 1 | 0.20 | 0.39 | 0.39 | 0.39 | 0.51 |
| acceleration declared | converged | 9 | 0.42 | 0.99 | 0.99 | 0.99 | 1.00 |
| acceleration declared | struggling | 3 | 0.10 | 0.50 | 0.50 | 0.49 | 0.52 |

- **Struggling networks:** even copying works only about half the time, so the network is the
  limit.
- **One exception among converged networks:** mass network 400 reaches only 0.60 with the table
  (and with both installed edits), while copying reaches 1.00. Its patterns for the same value
  and unit apparently vary in ways the 6-route table does not capture.

Per-network results are in `results/perm/summary.md`.

## 19.5 Against the pre-registered decision rules (doc 18 §18.5)

| rule | outcome |
|---|---|
| R holds | **Met.** The unit-aware advantage replicates on new networks and on questions no editor saw. |
| Q1 holds and the layer matches the table | **Met** for both variables: **the ability can be installed once**, as a small added layer, with no outside information per question. |
| Q2 holds and the existing-weights edit matches the table | **Met** for both variables: **it can be installed within the original network.** The doc-17 shortfall was the linear edit, not weight edits as such. |
| Any non-specific gain | None. Wrong-variable changes fall in all six comparisons. |

## 19.6 Caveats

- **The existing-weights edit is large.** Relative change: `W_in` 0.45 / 0.43, `W_out` 0.59 /
  0.56, `W_U` 0.35 / 0.43 (mass / acceleration).
  - It is a re-fit of everything after the edit site, learned from the table, not a minimal
    edit.
  - It keeps the answers that should not change about as well as the table does (preserve 0.95 /
    0.94).
  - Only this edit was tested. Other abilities of the network were not.
  - A registered test of *how small* a within-network edit can be, for example a rank or norm
    budget, is the natural next step.
- **Reserved from the editors, not from the network.** The networks were trained on every
  question (doc-14 procedure). "Unseen questions" means unseen by every editor, location search
  and tuning step.
- **The tuned settings came from different networks.** The layer sharpness (β = 100) and the
  distillation settings were chosen on the doc-15 networks, using non-reserved questions and
  agreement with the table only. They then transferred to new networks without retuning.
- **Toy scale:** one task, one edit (inversion mod 13), networks with 69.8K parameters. The unit
  of evidence is the trained network.

## 19.7 Where this leaves the program

- **Declaration matters** (doc 15): training a variable into a designated place makes *that*
  variable the editable one.
- **The expression matters** (docs 17, 19): the same value leaves a different pattern per unit,
  and an edit must account for that.
- **The ability can be installed permanently** (doc 19): as a small added layer or within the
  existing weights, at table-level reliability (0.87–0.90; 0.94–0.99 on networks that trained
  well), on new networks and unseen questions.

**Open:**
- the smallest within-network edit that works;
- whether an independent AI interpreter can produce such an installed edit from the network
  alone (the R1 track; it should run under the sandboxed checker of PR #8);
- the deferred paraphrase experiment, which can now use an edit that reliably follows the
  declared variable.
