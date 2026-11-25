# Cohort review — step 2,600 (96/96)

**Recovery of the first composer (step-2,500 prediction, partly met already).** C wd 1.0 seed 2004 at 2,600:
own-trained 1.00, held-out 0.26 → **0.53**, x1 canonical − number 0.24 → **0.66** (higher than before the collapse);
readout cosine −0.07 → −0.53 (the ≤ −0.7 part is still open, due by 2,800).

**Two new compositional onsets — without a registered utilisation onset.** Seeds 2002 and 2007 (C wd 1.0) reach
compositional onset at 2,500 with canonical − number 0.19 and 0.17 (below the 0.20 criterion; never above it).
Together with seed 2008's mixed code (2,200), this establishes that composition in C does **not require** the x1
number-token code to be read predominantly as the canonical quantity. M1, as registered, counts only networks with
both onsets (2/2 in the predicted order so far); the networks composing without utilisation will be reported
alongside it, since they bear on its interpretation.

**First episode at weight decay 0.1 (after 26 cohorts).** D wd 0.1 seed 2006 (measured): median gradient
0.014 (very converged), sudden cascade from step 2,581 (gradient 1.7 → 13.3, batch task loss to 0.90, item losses to
16.5); audit own-trained 1.00 → 0.89; auxiliary loss 0.12 → 0.46; readout cosine −0.46 → −0.01. This one fits
explanation (a) (Adam instability at tiny gradients, here after long convergence) rather than weight-decay drift:
**both mechanisms exist** — decay-driven episodes begin early at wd 1.0, small-gradient episodes can appear late at
wd 0.1. The equilibrium-break signature (readout cosine → 0) appears in both.

**What changed.** C wd 1.0 held-out mean **0.28** (largest single-cohort gain, +0.06), 4/12 compositional. 15
sustained flags (14 at wd 1.0).

**Prediction.** Within the next 1,000 steps at least one further episode appears at wd 0.1, and every wd 0.1
episode is preceded by a block with median gradient < 0.02 (small-gradient mechanism).
