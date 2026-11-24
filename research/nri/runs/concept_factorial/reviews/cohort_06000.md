# Cohort review — step 6,000 (96/96) — final development audit

**Training complete:** 96/96 networks reached 6,000 steps with 61 complete checkpoints each (5,856 audited
checkpoints); no network failed or was restarted from a checkpoint.

**Final development picture (dev combinations; the registered endpoint is the reserved test, evaluated next).**
C wd 1.0 held-out **0.62** (seeds 0.35-0.90), new frame **0.32**, decodability 0.97, canonical − number 0.47. Every
other condition ≤ 0.08 held-out (B wd 1.0 0.08, A wd 1.0 0.06; all wd 0.1 ≤ 0.04). 12/12 C wd 1.0 networks reached
compositional onset during training; no network in any other condition did. C wd 0.1: decodability 0.84,
canonical − number +0.08 (from −0.09 at 1,000), 2/12 utilisation onsets (both after episodes), 0/12 composition.

**Predictions resolved at the end.** Step 5,900 (seed 2012 < 0.37 at 6,000): **confirmed** (0.35, recovering from
0.08). Step 5,800 (mean 0.50-0.62; seed 2005 most likely below 0.37; new frame ≥ 0.25): mean **confirmed** (0.62),
new frame **confirmed** (0.32), seed-2005 part **falsified** (0.57; seed 2012 was the one below). Step 4,000 (C wd
0.1 canonical − number < 0.10 at 6,000, no wd 0.1 composer): **confirmed** (0.08; 0/12).

**Final weight norms.** wd 1.0 ≈ 17-21 (flat since ~3,500); wd 0.1 ≈ 65-68 (still shrinking).

Next: reserved-test evaluation (once), registered analysis, archives.
