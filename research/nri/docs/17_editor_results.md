# 17. Editor comparison results: accounting for the unit closes most of the gap to copying, but not yet as a weight edit

*Registration: doc 16, commit `c24f1ae`, made before the fresh sample was scored. Analysis: the
frozen `run_editors.py --report`. Full auto-generated report: `results/editors/summary.md`. Data:
`results/editors/metrics_test.json`, from a fresh 16,384-example sample (seed 72727), on the same
36 fixed networks. No deviations from the registration.*

## 17.1 Verdict in plain English

**The hypothesis holds.** The same mass or acceleration leaves a **different internal pattern
depending on the unit it was written in**, like "1 kg" vs "1,000 g". Inside the located subspace,
how the value was expressed (unit and position) accounts for about 80% of the variation among
examples with the same value, and the tests below show the unit is the part that matters. The current
editor averages those patterns together, which is why it misses.

An editor that moves each example from **its own unit's pattern** to **the target value's pattern
in the same unit** gives the intended answer:
- **0.76** of the time for mass (current editor: 0.18) and **0.89** for acceleration (current:
  0.31);
- close to copying a real pattern (0.80 and 0.91);
- **without changing the wrong variable more often** (it actually does so slightly less).

On the networks that trained well, it reaches **0.97–0.98**, against copying's 1.00.

**The catch is what the editor needs.** That result comes from an editor that picks a pattern on
every question.
- It works whether the question's value and unit are supplied from outside or the network's own
  activation is used to look them up in a stored table. Both reach the same accuracy.
- A **single change written into the weights once** gains only a little: 0.27 vs 0.21 for mass
  and 0.40 vs 0.35 for acceleration. That closes about a tenth of the gap.
- A single linear weight change cannot apply different adjustments to different units.

**Position doesn't matter; the unit does.**
- Accounting for the unit alone does as well as unit + position.
- Accounting for position alone does no better than the pooled editor.

## 17.2 What happened, separately for mass and acceleration (fresh sample, all 12 networks each)

Intended = gave the edited answer. Wrong = changed the other variable instead. Eligible examples:
7,371 per network.

| editor | what it needs | mass declared: intended | wrong | acceleration declared: intended | wrong |
|---|---|---|---|---|---|
| current editor (doc 15) | weight edit, written once | 0.18 | 0.06 | 0.31 | 0.04 |
| pooled, full-activation input | weight edit, written once | 0.21 | 0.06 | 0.35 | 0.04 |
| **unit-aware, full-activation input** | weight edit, written once | **0.27** | 0.07 | **0.40** | 0.04 |
| pooled table | stored table, entry picked by the network's activation | 0.45 | 0.07 | 0.55 | 0.04 |
| **unit-aware table** | stored table, entry picked by the network's activation | **0.76** | **0.04** | **0.89** | **0.02** |
| pooled, per question | the question's value, supplied | 0.45 | 0.06 | 0.55 | 0.04 |
| **unit-aware, per question** | the question's value and unit, supplied | **0.76** | **0.04** | **0.89** | **0.02** |
| unit only, per question | value and unit | 0.74 | 0.04 | 0.89 | 0.02 |
| position only, per question | value and position | 0.45 | 0.05 | 0.56 | 0.04 |
| copy a real pattern (benchmark) | a real example per question | 0.80 | 0.04 | 0.91 | 0.02 |

"Unit-aware" is the registered route-aware editor (unit + position).

## 17.3 Registered tests

**Primary family:** unit-aware − pooled, intended-answer rate, paired by network, 12 networks,
Holm-corrected across 6 tests. All six reject.

| tier | population | Δ intended | 95% CI | p | networks > 0 | Δ wrong (p) |
|---|---|---|---|---|---|---|
| 1 weight edit | mass declared | **+0.066** | [+0.038, +0.096] | 0.0005 | 12/12 | +0.010 (0.37) |
| 1 weight edit | acceleration declared | **+0.049** | [+0.022, +0.079] | 0.004 | 11/12 | +0.006 (0.10) |
| 2 stored table | mass declared | **+0.312** | [+0.199, +0.417] | 0.0005 | 12/12 | −0.032 (0.03) |
| 2 stored table | acceleration declared | **+0.342** | [+0.189, +0.510] | 0.0005 | 12/12 | −0.024 (0.002) |
| 3 per question | mass declared | **+0.312** | [+0.195, +0.420] | 0.0005 | 12/12 | −0.015 (0.17) |
| 3 per question | acceleration declared | **+0.333** | [+0.188, +0.491] | 0.0005 | 12/12 | −0.020 (0.02) |

None of the gains comes with a significant rise in wrong-variable changes. Under the registered
rule, all six are **specific**.

**Secondaries (uncorrected):**
- **unit-aware weight edit − current editor:** +0.088 (mass) and +0.095 (acceleration), p = 0.0005
  for both, 12/12 networks.
- **unit only − pooled** (per question): +0.29 and +0.33, p = 0.0005, 12/12.
- **position only − pooled:** −0.003 and +0.001, p = 0.75 and 0.27. No effect.
- **copy − unit-aware** (gap left): +0.043 and +0.025, p = 0.02. A small but real gap remains.
- **Share of the pooled → copy gap closed:** tier 1: 0.11 / 0.09; tier 2: 0.88 / 0.94; tier 3:
  0.88 / 0.93.

## 17.4 Networks that trained well vs networks that struggled

Mean intended-answer rate (pooled → unit-aware):

| population | networks | n | weight edit | stored table | per question | copy |
|---|---|---|---|---|---|---|
| mass declared | converged | 7 | 0.14 → 0.23 | 0.52 → **0.97** | 0.52 → **0.97** | 1.00 |
| mass declared | struggling | 5 | 0.30 → 0.33 | 0.35 → 0.47 | 0.35 → 0.47 | 0.53 |
| acceleration declared | converged | 9 | 0.41 → 0.48 | 0.61 → **0.98** | 0.62 → **0.98** | 1.00 |
| acceleration declared | struggling | 3 | 0.16 → 0.17 | 0.37 → 0.61 | 0.34 → 0.60 | 0.65 |

On struggling networks even copying works only about half the time, so the network is the limit.
On mass networks 303, 306 and 308, the editors and copying all land at or below about 0.5. Mass network
309 is the one where the unit-aware editor falls well short of copying (0.26 vs 0.51). Per-network
results are in `results/editors/summary.md`.

## 17.5 Against the pre-registered decision rules (doc 16 §16.6)

| rule | outcome |
|---|---|
| Tier 3 unit-aware > pooled, specific | **Met** (both variables). Different expressions of a value leave different patterns, and targeting the right one works. |
| … within 0.05 of copy | **Met:** 0.043 (mass) and 0.025 (acceleration) below copy. Unit-specific patterns explain most of the copy advantage. |
| Tier 2 also succeeds | **Met:** the stored table matches the per-question editor (0.76 / 0.89). The network's own activation carries enough to pick the right entry; **no outside information per question is needed**, but a lookup still runs on every question. |
| Tier 1 succeeds but stays far below tiers 2–3 | **Met:** +0.05 to +0.07, closing about a tenth of the gap. **A single linear weight change cannot express unit-specific moves.** |
| Unit-only ≈ unit-aware; position-only ≈ pooled | **Met.** The **unit** is the part of the expression that matters. |

## 17.6 Caveats

- **Same networks as the development run.** The evaluation examples are fresh, but the 36
  networks were also used for development (doc 16 §16.4), and the fresh results match the
  development results to within 0.01. That consistency shows the result does not depend on the
  particular sample. It does **not** show it holds on newly trained networks.
- **Side effects where no variable is declared.** Unit-aware moves are larger. In the `neither`
  networks, and for the undeclared variable, they push answers away from "unchanged" without
  hitting the intended answer much more often. For example, mass in `neither` networks:
  unchanged 0.82 → 0.57, intended 0.01 → 0.03. The editor helps only where the variable has a
  place to be edited.
- **Toy scale:** one task, one edit (inversion mod 13), networks with 69.8K parameters. The unit
  of evidence is the trained network.

## 17.7 Next step

The remaining question is whether the unit-aware edit can be **written into the weights once**
rather than looked up per question. The stored-table result shows the required information is
already in the network's activation. What a single linear change lacks is the ability to apply a
different adjustment per unit.

The natural next registration is a **non-linear weight edit**: added neurons that implement the
stored table, each detecting one (value, unit) pattern and writing its adjustment. It would be
scored against the same tiers, scorer and copy benchmark. **It should also be replicated on newly
trained networks.**

The paraphrase experiment remains deferred. An edit now reliably follows the declared variable
(0.97–0.98 on converged networks), but only with a per-question lookup.
