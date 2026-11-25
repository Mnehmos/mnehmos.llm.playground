# Cohort review — step 200 (96/96)

**What changed.** B/C/D own-trained accuracy rose to 0.84-0.86 (wd 0.1) and 0.74-0.75 (wd 1.0): weight decay
already slows fitting of the 120-combination set. A is at 1.00. Held-out stays ≤ 0.03 everywhere, and held-out
task loss rose sharply in every population (B: 3.6 → 9.5-11.9): all populations are becoming confidently wrong on
unseen combinations while fitting trained ones — memorisation dominates early in all four.

**Geometry.** C's x1m geometry moved strongly and identically at both weight decays: raw-grouped dlog circle 0.25
→ 0.37-0.38 (null 0.21), canonical-grouped 0.52 → 0.61-0.62 (null 0.45). A/B/D unchanged from initialisation.
Canonical decodability on held-out combinations: C 0.30-0.32 vs 0.20-0.25 elsewhere (still far below the 0.5
onset criterion).

**Causal.** x1 canonical − number semantics fell to −0.21…−0.29 in B/C/D (−0.13 A): at this stage the
number-token code at x1 is read as the *raw number* in every population including C — C's auxiliary target has not
yet changed how the downstream circuit reads x1.

**Update dynamics.** W_in adaptive-vs-decay cosine became more negative (−0.22…−0.26): Adam is actively growing
the readout against the decay in all conditions. Weight norm: wd 1.0 → 96, wd 0.1 → 110.

**Anomalies.** All 24 A networks flagged by the registered spike rule (max > 5× interval median). Investigated:
the maxima are at steps 101-104 (just after warm-up ends at 100) while A finishes fitting; after that the median
gradient norm is ~0.01, so any ordinary fluctuation exceeds 5× the median. **Instrument limitation**: the
registered relative spike rule is uninformative for converged networks. It is kept as registered; for
investigation I add (not as a registered measure) an absolute criterion — a spike must also exceed 1.0 and coincide
with a task-loss rise — before individual follow-up.

**Previous predictions.** (i) B/C/D own-trained ≥ 0.95 by 300 — open. (iii) |Δ x1_cmn| between weight decays
in C small early — consistent so far (−0.24 vs −0.21).

**Prediction for 300-500.** C's raw-grouped circle keeps rising at both weight decays with no weight-decay
difference > 0.05 before step 500; x1 semantics in C remains number-dominant (canonical − number < 0) through
step 500 at both weight decays — i.e. the representation forms before the downstream circuit reads it as a
quantity.
