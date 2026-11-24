# Cohort review — step 100 (94/96 audited at writing; remaining 2 audited later, same pattern)

**What changed.** A (42 own combinations, one frame, canonical style) already answers its own trained items (0.99 ±
0.02); B, C, D (120 combinations, four frames, styled surface forms) are at 0.30-0.33 on theirs. Held-out is at or
below 0.02 everywhere (best constant 0.17). A's held-out task loss is 9.0-9.7 versus 3.4-3.8 for B/C/D: A is already
*confidently wrong* on unseen combinations — a memorisation signature. Weight decay is visible only in the weight
norm so far (wd 1.0: 110.8 → 105.6; wd 0.1: 110.8 → 110.5).

**Geometry.** C's canonical-grouped circle rose 0.45 → 0.52 (init null 0.45) and its raw-grouped circle 0.21 →
0.24-0.25 at both weight decays; D, B, A unchanged. The auxiliary target is shaping x1m within 100 steps, while
canonical decodability on held-out combinations is still near its initial 0.23-0.28.

**Update dynamics.** Over this interval the summed adaptive (Adam) update in the readout input W_in is
anti-aligned with the summed weight-decay update (cos −0.09 B/C/D; −0.16 A): Adam is growing W_in against the
decay. Identical cosines at wd 0.1 and 1.0 (direction of the decay update is fixed by the weights; only its size
differs).

**Anomalies investigated.** Gradient-norm spikes flagged for seed 2011 in several conditions: the maxima are at
steps 1-4 (≈10-11 vs median ≈1.9) in every condition, because seed 2011's initialisation is poorly calibrated
(initial loss 3.95 vs ln 17 = 2.83). Initialisation effect shared across conditions, not a training event. Not a
concern for the contrasts (paired by seed).

**Previous prediction (step 0).** Own-trained < 0.9 for every condition at step 100 — **falsified for A** (0.99);
correct for B/C/D. Held-out below 0.17 everywhere — confirmed. A's narrow set is learned an order of magnitude
faster, as one would expect from 42 vs 120 combinations and a single frame.

**Hypotheses / predictions for step 300.** (i) B/C/D reach own-trained ≥ 0.95 by step 300 (pilot timing).
(ii) C's canonical decodability on held-out combinations exceeds 0.5 by step 400-700 at both weight decays (pilot),
while held-out accuracy stays below 0.17 — the "representation before use" regime begins. (iii) Competing: if the
weight decay effect acts early, wd 1.0 C would already show higher x1 canonical semantics than wd 0.1 C by step
500; the pilot suggests the difference only appears late (prediction: |Δ x1_cmn| < 0.05 between weight decays in C
at step 500).
