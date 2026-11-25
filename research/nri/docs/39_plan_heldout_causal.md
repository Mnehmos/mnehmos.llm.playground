# 39 — Plan (committed before running): held-out causal test of the canonical code, and seed-clustered M2

Retrospective analysis of the archived factorial checkpoints (doc 37/38). Prompted by an owner review of PR #32:

1. `patch_semantics` (the registered "canonical utilisation" measure) draws **both base and donor from trained
   combinations**. It shows what the x1 number-token code carries in trained contexts; it does not show that this
   code *implements* the answers on unseen combinations. The causal account in doc 38 §38.1 (Q3) therefore lacks
   held-out intervention evidence.
2. M2's binomial test treats 127 checkpoint transitions as independent although they come from 12 trajectories.

Nothing below changes the registered primary endpoint or its five contrasts. Everything below is **retrospective
and exploratory**, and is labelled so in the results. The reserved test combinations were used once (doc 38) and
are **excluded from every stratum** here (no base, donor or implied combination may be a reserved test combination).

## 39.1 Instrument: `patch_semantics_split` (sci/heldout_causal.py)

Same intervention as `patch_semantics` (copy the donor's residual at the source-number token after block `site`
into the base; base and donor share the dimension and differ in source unit), but every pair is classified by:

* **base pool**: base combination trained vs development held-out;
* **implied combination of each prediction** on the base route — number prediction implies (base route, v_d);
  canonical prediction implies (base route, v′) with v′ = q_d · s_{u_b}⁻¹ (the base-route input whose answer equals
  the canonical prediction) — each either trained or development held-out.

The decisive stratum is **canonical-implied held-out**: following the canonical prediction there means the
downstream circuit converts the patched quantity along a route on which that input–output combination was never
trained. A route-specific lookup over (quantity, route) cannot do this; a reusable conversion of the canonical code
can. Pairs where the two predictions coincide or equal the base answer are excluded. Rates: canonical, number,
unchanged, other.

Controls with known answers (must hold before any result is interpreted): at site x0, swapping the number *token*
must follow the number prediction and swapping number + unit tokens must follow the canonical prediction in every
stratum where the model answers the implied combination correctly (input-level swaps are exact re-inputs); at
step 0 (untrained) all rates must be at chance level (≤ 0.10 for canonical in every stratum).

## 39.2 Analyses and predictions

All at site x1 (number token), n ≥ 200 pairs per stratum per network where available; seed is the unit.

* **R1 (main).** At step 6,000, canonical-following rate in the stratum *trained base, canonical-implied held-out*.
  Prediction if the canonical code implements composition: in C wd 1.0 this rate is at least half of the same
  network's rate in the *canonical-implied trained* stratum (median ratio ≥ 0.5 over 12 seeds) and exceeds C wd 0.1
  and D wd 1.0 (paired sign-flip, one-sided). Falsifier: C wd 1.0 implied-held-out rate ≤ 0.10, or ratio < 0.25 —
  then the canonical code is read through a route-specific lookup and composition is implemented elsewhere.
* **R2.** Same comparison with **held-out bases** (base combination held-out): canonical-following on held-out
  bases vs trained bases in C wd 1.0.
* **R3.** Across the 12 C wd 1.0 networks at 6,000: Spearman correlation between the R1 rate and development
  held-out accuracy (prediction > 0).
* **R4 (timing, descriptive).** R1 rate for C wd 1.0 every 500 steps vs the registered compositional onset.

## 39.3 M2, seed-clustered

Per network (C networks with a compositional onset, i.e. the 12 C wd 1.0 networks): fraction of post-onset
transitions with |Δ held-out| ≥ 0.05 in which Δ x1 canonical semantics has the same sign. Report the mean of the
per-network fractions, a seed-bootstrap 95% CI (10,000 resamples), and an exact sign-flip test of mean(fraction −
0.5) > 0 over networks. The registered transition-level binomial result stays reported, marked as not accounting
for clustering.

---

# 39R — Results (added after running; the plan above was committed first, 51c546e)

Inputs: model weights extracted from the 96 verified archives (288 sampled files re-hashed against the Git
manifests: 288 match, 0 mismatch). Output: `runs/concept_factorial/heldout_causal.json`. Instrument:
`sci/heldout_causal.py` (tested: no base, donor or implied combination is a reserved test combination; implied
combinations reproduce the canonical prediction exactly). Pairs available per stratum (base | canonical-implied):
trained|trained 2,229; trained|held-out 424; held-out|trained 388; held-out|held-out 33 (small — reported only
descriptively).

## Deviation from the plan (control failure)

The pre-specified untrained-network control ("canonical ≤ 0.10 in every stratum at step 0") **failed** in three
cells (0.18, 0.13, 0.13): an untrained network gives a near-constant answer, and where that answer coincides with the
canonical prediction the raw rate is inflated. Fix, applied to all results below: each rate is reported as an
**excess over the no-intervention baseline** (the fraction of the same pairs for which the *unpatched* model already
outputs the predicted answer). With this correction the step-0 control holds (excess −0.007…+0.007 in every cell of
four seeds). The input-level controls held without correction: swapping the number token gives number-following
0.82-1.00; swapping number + unit tokens gives canonical-following 0.52-0.92, in every stratum.

## R1 — the canonical code is read correctly on never-trained implied combinations

Canonical-following excess at x1, step 6,000 (mean of 12 seeds):

| | trained base, implied trained | **trained base, implied held-out** | held-out base, implied trained | held-out base, implied held-out (n ≤ 33) |
|---|---|---|---|---|
| C wd 1.0 | 0.625 | **0.469** | 0.610 | 0.331 |
| C wd 0.1 | 0.379 | 0.144 | 0.310 | 0.068 |
| D wd 1.0 | 0.124 | 0.048 | 0.049 | 0.008 |
| B wd 1.0 | 0.092 | 0.058 | 0.012 | 0.015 |
| A wd 1.0 | 0.020 | 0.024 | 0.012 | −0.023 |

Prediction (doc 39 §39.2): in C wd 1.0 the implied-held-out rate is at least half of the implied-trained rate —
**met** (median per-seed ratio 0.76, range 0.53-0.99) — and exceeds C wd 0.1, D wd 1.0: **confirmed**, paired
differences +0.32 (95% CI 0.23-0.43), +0.42 (0.33-0.53) and, added, B wd 1.0 +0.41 (0.32-0.51), each exact sign-flip
p = 0.00024. Falsifiers (rate ≤ 0.10 or ratio < 0.25) not met. **Interpretation:** when a quantity is patched into the
number token, C wd 1.0 networks convert it along the base route even when that (input, route) combination never
occurred in training — a route-specific lookup over trained pairs cannot produce this; a conversion operation that
reads the canonical code can. This is the held-out intervention evidence the doc-38 causal account lacked.

## R2 — held-out base questions

C wd 1.0: canonical-following excess on held-out bases 0.610 vs trained bases 0.625 (paired two-sided p = 0.34): the
code is read the same way inside questions the network never trained on.

## R3 — link to accuracy across networks

Spearman between the R1 rate and development held-out accuracy across the 12 C wd 1.0 networks: **0.38** — positive
as predicted, but weak with n = 12; network-level held-out accuracy is not well explained by this one site.

## R4 — timing (descriptive; threshold and grid chosen at analysis time)

First 500-step checkpoint at which the R1 excess ≥ 0.20, vs the registered compositional onset: **before** in 11/12
networks, **same** checkpoint in 1 (seed 2002), **after** in 0. The registered trained-context measure (M1) had given
9 before / 3 after; the three "after" networks (2002, 2007, 2008) do read the code canonically on held-out implied
combinations at or before composition. So M1's three exceptions were produced by the trained-context limitation of
the registered measure.

## M2, seed-clustered

Per-network agreement fractions (12 C wd 1.0 networks): 0.40-1.00, **mean 0.70**, seed-bootstrap 95% CI
**0.60-0.80**, exact sign-flip test of mean(fraction − 0.5) > 0: **p = 0.003**. The registered transition-level
binomial (88/127, p ≈ 8 × 10⁻⁶) overstated the precision; the clustered analysis still supports M2, with the point
estimate at the predicted 70% and a lower bound of 0.60.

## What this changes in doc 38

* The causal claim is upgraded from "the code carries the quantity in trained contexts" to "the code is read as a
  quantity and converted along routes whose (input, route) combinations were never trained" — at one site (x1,
  number token), C wd 1.0, step 6,000, and over training (R4).
* C wd 0.1 also reads the code partly canonically (0.38 trained, 0.14 implied held-out), but far less on unseen
  combinations; consistent with M3, now on held-out evidence.
* Still not shown: that this site is *necessary* for composition (no ablation of the pathway was found that
  discriminates — doc 37 pre-registration note), or that it is the only route; R3 is weak.
