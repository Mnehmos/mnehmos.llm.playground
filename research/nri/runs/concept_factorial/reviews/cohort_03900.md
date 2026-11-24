# Cohort review — step 3,900 (96/96)

**Largest collapse of the best composer, with a different signature (C wd 1.0 seed 2004, measured):**

| | 3,800 | 3,900 |
|---|---|---|
| own-trained / held-out | 1.00 / 0.83 | 0.59 / 0.42 |
| canonical decodable (x1m, kNN) | 1.00 | **0.69** |
| x1 patch canonical / number | 0.85 / 0.07 | 0.47 / 0.13 |
| auxiliary loss | 0.024 | **0.34** |
| raw dlog circle (x1m) | 0.64 | 0.69 |
| W_in cos(adaptive, decay) / adaptive path | −0.91 / 1.00 | **−0.85** / 1.61 |

Unlike its step-2,500 collapse (readout equilibrium broke: cosine → −0.07, path 4.7; representation spared), this
time the **representation at x1m is hit** (decodability 1.00 → 0.69, auxiliary loss ×14) while the readout's
block-level equilibrium largely holds (−0.85). So there are at least two collapse types: readout-first (2,500; and
the earlier in-progress captures) and representation-first (3,900). The "readout-first" generalisation from steps
1,200-1,700 is therefore **not universal** — corrected here.

**Step-3,800 prediction status.** Seed 2010 0.36 → 0.39 (onset by 4,100 open); seed 2005 0.24.

**Other.** C wd 1.0 held-out 0.50; 14 sustained flags (all wd 1.0); A 2006 own-trained 0.36.

**Prediction.** Seed 2004 recovers held-out ≥ 0.70 by 4,200 and decodability ≥ 0.95; if a representation-first
collapse is harder to recover from than a readout-first one, recovery will take ≥ 2 cohorts (vs 1 at 2,500).
