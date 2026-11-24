# Cohort review — step 400 (96/96)

**What changed.** Registered **decodable onsets** (two consecutive audits, kNN ≥ 0.5) now recorded at step 300 for
C in 6/12 seeds at wd 0.1 and 2/12 at wd 1.0, and for one D network (wd 0.1, seed 2005). C's held-out decodability
0.57 (wd 0.1) / 0.58 (wd 1.0); D 0.40-0.44 (D's per-unit scrambled labels still require reading value and unit, so
q is partly recoverable). C's raw-grouped dlog circle 0.50-0.54 — the canonical code is forming its circle.

**Causal reading of x1.** C canonical − number −0.14 (wd 0.1) and −0.11 (wd 1.0), up from −0.19/−0.18: the
downstream reading of x1 is beginning to drift from "raw number" toward "quantity", slightly faster at wd 1.0
(first weight-decay difference in the causal measure; 0.03, within noise for now). Held-out accuracy still ≤ 0.03.

**Anomalies investigated.** Five B (wd 0.1) networks exceed the registered relative spike rule; three exceed my
supplementary absolute criterion (> 1.0). Each is a single-step event (steps 301, 303, 318) caused by one hard
item in the batch (worst item loss 1.4-3.0), with the next step back at baseline: batch-composition noise among the
last unconsolidated trained combinations, not instability.

**Previous predictions.** Step 300: task-aux cosine at x1m in C within ±0.05 — holding (0.01 / −0.00).
Step 200: C remains number-dominant through 500 — holding at 400.

**Prediction for 500-1,000.** If the weight-decay hypothesis is right, the x1 canonical − number gap between
weight decays in C opens progressively (wd 1.0 > wd 0.1 by ≥ 0.05 at step 1,000), *before* any held-out gain;
if instead it stays < 0.05 while composition later appears only at wd 1.0, the weight-decay effect acts
downstream of x1 (M3 would fail).
