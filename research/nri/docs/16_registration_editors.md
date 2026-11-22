# 16. Registration: does an editor that accounts for *how* a value was expressed build better replacements?

*This document, together with `run_editors.py` in the same commit, is the registration. Both
were committed **before** the fresh evaluation sample was scored. Earlier registrations and
results (docs 11–15) are preserved unchanged.*

## 16.1 Question

Doc 15 found that the role-swap networks carry the declared variable in a usable place: copying a
real internal pattern into the located subspace gives the intended answer 0.80 (mass) and 0.91
(acceleration) of the time. The current editor, fitted at the same place, reaches only 0.18 and
0.31.

The current editor learns **one pooled adjustment per value**. It moves the average pattern of
value *v* to the average pattern of the edited value, averaging over every way the value was
expressed.

**Hypothesis.** The same canonical value leaves *different* patterns depending on how it was
expressed. Think of "1 kg" and "1,000 g": in this task, the same mass written with a different
unit, or in a different position. If so, each expression (**route**) needs its own adjustment,
and an editor that accounts for the route should get closer to copying.

## 16.2 What is held fixed, so a gain can be credited to the editor's construction

| fixed | how |
|---|---|
| networks | the 36 role-swap networks in `runs/zoo_swap/` (3 groups × seeds 300–311), loaded read-only, never retrained |
| target locations | the frozen DAS subspace from `run_swap.das_basis` (same seeds, so the identical subspace) is primary; the known block (dims 0–15) is secondary |
| scoring | the frozen `run_swap.score`: the same eligible set (unchanged, M-edited and A-edited answers pairwise distinct), the same four outcomes (intended / wrong-variable / unchanged / other), and the same edit (inversion mod 13, 0 → 0) |
| fitting examples | every editor is fitted on the **same** 8,192 examples (`run_swap.FIT_S`), with the same labels available: each example's value and route |
| evaluation | a **fresh** 16,384-example sample, `TEST_S = 72727`, never used before this commit |
| unit of evidence | the trained network: 12 seeds per population, paired by seed |

## 16.3 Editors

The **route** of an example is how the edited variable was expressed: its **unit** (3 options)
× its **position** in the sentence (mass first or second), giving 6 routes. This was fixed before
any development run. Unit-only (3 routes) and position-only (2 routes) are pre-specified
secondaries.

Editors come in **pooled / route-aware pairs that differ only in whether they use the route**.
- **Pooled:** move value *v*'s mean pattern to the edited value's mean pattern.
- **Route-aware:** move the (value *v*, route *r*) cluster's mean pattern to the (edited value, *r*)
  cluster's mean pattern.
- **Missing target cluster:** sometimes the edited value never occurs on route *r* in training.
  This happens when the edited value in that unit would be a held-out unit–value combination, or
  when the cluster has fewer than 5 fitting examples. The target is then the mean of the edited
  value's available clusters in the same position.

All moves are projected onto the target location.

These are **different capabilities**, so they are reported in three separate tiers:

| tier | what it is | what it needs at run time | pooled | route-aware |
|---|---|---|---|---|
| **1 weight edit** | one change folded into the existing weights (`W_in`), written once | nothing | affine map from the full activation into the subspace, fitted by least squares to the pooled moves | the same map, fitted to the route-aware moves |
| **2 stored table** | a fixed table of moves, added once | the network's **own activation** picks the entry (nearest fitted centroid inside the located subspace); no outside information | table of 13 value entries | table of 13 × 6 (value, route) entries |
| **3 per-question** | the move is chosen from each question's **known** value and route | the question's value and unit/position, supplied from outside | pooled move for the known value | route move for the known value and route |

- **Benchmark (tier 3):** copy the real pattern of a source example. This is the frozen
  `run_swap.copy_test` construction, re-run on the fresh sample. It also needs a source example
  per question.
- **Also reported:**
  - the **current editor exactly as frozen in doc 15** (affine map from the subspace coordinates);
  - copying along the weight edits' write path (into the MLP input only).

**Why the tier-1 pair uses the full activation as input.** A single affine map cannot apply
different moves to different routes unless it can *see* the route. Route information is not
guaranteed to lie inside the 16-dim subspace, so both tier-1 editors read all 64 dims. That
keeps the pair identical except for the targets. The narrower current editor is reported
alongside.

## 16.4 Development sample (disclosed)

Before this registration, every editor was run on a **development sample** (`DEV_S = 91919`,
16,384 examples) on the same 36 networks. Results: `results/editors/metrics_dev.json` and
`summary_dev.md`, committed here.

The development run was used for:
1. **Locking the tier-2 lookup space, by a classification rule, not by edit success.** Use the space
   (full 64-dim activation, or the located subspace) where the nearest-centroid lookup identifies
   (value, route) more accurately on the declared variable. Subspace: 0.67 / 0.74; full: 0.33 /
   0.36. **Locked: subspace.** The development edit results were visible at the time, and point
   the same way.
2. Checking that the report code runs.

An earlier scratch exploration used the same development seed on 4 of the networks. It showed
that the write path (MLP input only vs everywhere) changes copying and editing little, so the
frozen write paths are kept: weight edits on the MLP input, as in doc 15; activation edits and
copy everywhere, as in doc 15.

No other choice was changed after the development run: routes, fallback rule, editors, tests and
populations are as first written.

**What this registration does and does not protect against.** The evaluation examples are fresh.
The **networks are not**: they are the same 36 networks as the development run, by design (the
networks are held fixed). A gain that held only for these particular networks would not be caught
here. Replication on newly trained networks is the follow-up.

**Development results** (mean intended / wrong over 12 seeds, DAS location, declared variable),
for transparency:

| editor | tier | mass declared | acceleration declared |
|---|---|---|---|
| current (doc 15) | 1 | 0.18 / 0.06 | 0.31 / 0.04 |
| pooled → route-aware | 1 | 0.20 → 0.27 | 0.35 → 0.40 |
| pooled → route-aware | 2 | 0.45 → 0.76 | 0.55 → 0.89 |
| pooled → route-aware | 3 | 0.45 → 0.76 | 0.56 → 0.89 |
| copy (benchmark) | 3 | 0.80 / 0.04 | 0.91 / 0.02 |

## 16.5 Hypotheses and tests

**Populations (primary):** the declared variable in declared networks. That is the mass edit in
`mass_only` networks and the acceleration edit in `acc_only` networks, **all 12 seeds each,
including networks that did not converge.** The undeclared variable and the `neither` networks are
reported as secondary.

**Primary family** (6 tests, Holm-corrected, α = 0.05, two-sided):
- For each tier (1, 2, 3) × population (mass, acceleration): Δ = intended(route-aware) −
  intended(pooled), per seed.
- Test: an exact sign-flip permutation over 12 seeds, with a 95% percentile bootstrap CI over
  seeds.
- **Prediction** (from the development sample): Δ > 0 in all three tiers. It should be large in
  tiers 2–3 (≈ +0.3) and small in tier 1 (≈ +0.05).

**Wrong-variable check, for each primary pair:** Δwrong = wrong(route-aware) − wrong(pooled),
same test, uncorrected. A gain counts as **specific** only if Δwrong is not significantly
positive (p < 0.05 with mean > 0).

**Pre-specified secondaries** (uncorrected):
- tier 1: route-aware − current (the frozen doc-15 editor);
- tier 3: unit-only − pooled, and position-only − pooled. This asks *which* part of the
  expression matters;
- tier 3: copy − route-aware, the gap left to the benchmark;
- the share of the pooled → copy gap closed, per tier;
- all four outcome proportions for every editor and population;
- the per-network table with convergence status;
- converged vs struggling networks shown separately;
- the known-block location;
- tier-2 lookup accuracy;
- the share of within-value variance (inside the subspace) explained by route.

## 16.6 Decision rules, fixed in advance

| outcome | conclusion |
|---|---|
| Tier 3 route-aware > pooled (Holm), specific | Different expressions of the same value leave different patterns, and an edit that targets the right route's pattern works better. **The hypothesis is supported.** |
| … and tier 3 route-aware is within 0.05 of copy | Route-specific patterns explain **most of** the copy advantage. |
| Tier 2 also succeeds | The network's own activation carries enough to pick the right route. **No outside information per question is needed**, but a lookup still runs on every question. |
| Tier 1 succeeds but stays far below tiers 2–3 | A **single linear weight change cannot express route-specific moves**. Writing the route-aware edit into weights needs a non-linear edit, e.g. added neurons implementing the table. That is the next registration. |
| Unit-only ≈ route-aware and position-only ≈ pooled | The **unit** is the part of the expression that matters ("1 kg" vs "1,000 g"); position does not. |
| A tier fails (Holm does not reject, or the gain is non-specific) | Report it directly. If tier 3 fails, the development result did not replicate on fresh examples. |

## 16.7 Compute and procedure

- About 2.5 minutes of wall-clock time on 4 CPU cores for all 36 networks, in one pass.
- **No restarts:** all 36 networks are analysed and every seed counts.
- Procedure: commit this registration → `python run_editors.py --split test` →
  `python run_editors.py --report` → docs/17.
