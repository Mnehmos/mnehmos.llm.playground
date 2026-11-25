# Cohort review — step 700 (96/96)

**What changed.** C at wd 1.0 crossed into "quantity" reading of the x1 number-token code: canonical − number ≥ 0 in
**7/12 seeds** (values −0.13…+0.12, mean −0.01); C at wd 0.1 **0/12** (−0.23…−0.05, mean −0.11). Step-600
prediction (≥ 6/12 vs ≤ 3/12 by step 1,000) **met early**. Held-out decodability at x1m also separates by weight
decay in C: 0.79 (wd 1.0, range 0.39-1.00) vs 0.65 (wd 0.1, 0.49-0.83). Held-out accuracy still ≤ 0.08 in every
network — C wd 1.0 seeds 2006 and 2011, the two with the highest x1 canonical reading (+0.12, +0.04), have the two
highest held-out values (0.08, 0.07): first hint of the causal-utilisation → composition link at seed level.
D decodable onset recorded at 600 for seed 2009 (wd 1.0).

**No sustained anomalies.** Isolated single-batch spikes only.

**Interpretation so far (exploratory).** The weight-decay effect in C appears first in *how x1 is read downstream*
(a causal property), not in whether x1 carries the quantity (decodability exists at both weight decays). This
fits the hypothesis that weight decay weakens the competing raw-number route and leaves the canonical code as the
cheaper one to read, but it cannot yet distinguish "decay suppresses lookup" from "decay changes the readout
geometry generally" — B and D (which have no canonical code to switch to) show no comparable shift, which favours
the former.

**Prediction (step 1,000-2,000).** Within C wd 1.0, the seeds whose canonical − number crosses +0.2 first are the
first to reach compositional onset (Spearman between the two onset steps > 0.5 across seeds that reach both).
