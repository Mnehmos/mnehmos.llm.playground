# Cohort review — step 4,100 (96/96)

**What changed.** C wd 1.0 held-out **0.55** (new frame 0.22, held-out task loss 2.87 — lowest). Seed 2010 0.47
(first audit ≥ 0.37 since its step-3,700 episode rebound; onset needs 4,200), seed 2005 0.32 (rising: 0.18 → 0.32;
it had a sustained episode this block — another episode coinciding with a held-out rise). 19 sustained flags, all
wd 1.0; A 2007 own-trained 0.40 at the audit.

**Readout equilibrium at wd 1.0 has tightened over the run.** W_in adaptive/decay cosine: C −0.85, B −0.82, D −0.81
(from ≈ −0.6 at 1,000-2,000); A −0.57 (A's episodes are so frequent that its block-level equilibrium is repeatedly
broken). Hypothesis: A's equilibrium is unstable because 42 combinations over-determine little of a 129K-parameter
readout — many directions are free for decay to shrink and Adam to restore, so equilibrium is reached on a thin
margin; predicted A has the highest fraction of episode blocks with own-trained drops at the audit (end count).

**Prediction.** Seed 2010 records compositional onset at 4,100 (needs ≥ 0.37 at 4,200); seed 2005 composes by 4,500
(earlier prediction said not before 4,500 — I now expect it to be wrong, given 0.32 and rising).
