# Cohort review — step 4,300 (96/96)

**Second wd 0.1 episode — and the most violent of the run (C wd 0.1 seed 2012, measured).** Preceding block median
gradient 0.005; in this block the median is 0.007 until step 4,265, then gradient 4.2 (4,266) → **67-73** (4,267-4,268),
batch task loss 5.0-11.5, single items to loss 46, auxiliary loss 0.00 → 0.28. Own-trained recovered to 0.93 by the
audit. **Step-2,600 prediction confirmed for both wd 0.1 episodes** (each preceded by a block with median gradient
< 0.02): the small-gradient (Adam) mechanism at low weight decay, after long convergence, with explosive onset
(no 25-step escalation, unlike the wd 1.0 episodes).

**Two episode regimes (exploratory synthesis).** wd 1.0: frequent, escalating over ~25-30 steps, moderate magnitude
(gradients 5-50), at a decay/Adam equilibrium with thin margins. wd 0.1: rare (2 networks in 4,300 steps), late,
explosive (gradients to 73 within 2 steps), after the gradient has decayed to ~0.005.

**C wd 1.0.** Held-out 0.50; seed 2012 did not recover yet (0.24 → 0.15; canonical semantics 0.30 → 0.51 rising
first) — step-4,200 prediction (≥ 0.50 by 4,400) at risk. Seed 2002 0.42 (recovering). A at wd 1.0: own-trained
0.21 (2011) and 0.36 (2010) at the audit.
