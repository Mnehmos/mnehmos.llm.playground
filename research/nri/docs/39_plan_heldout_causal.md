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
