# Cohort review — step 4,900 (96/96)

**What changed.** Seed 2006 recovered in one cohort (0.24 → 0.60, canonical semantics 0.36 → 0.52). Seed 2007
collapsed (0.71 → 0.40, own-trained 0.74). Seed 2005 caught in an episode again (held-out 0.07, own-trained 0.76) —
the step-4,600 prediction (2005 composes before 5,500) is at risk. Own-trained losses at the audit across A (2001
0.62, 2010 0.57), B (2009 0.60), C (2005, 2007), D (2002 0.88, 2009 0.85).

**Collapse-recovery bookkeeping (for the final analysis):** collapses ≥ 0.15 in held-out after onset observed so far
in 2002 (×3+), 2004 (×5), 2006, 2007 (×2), 2008, 2010, 2012, 2001; each recovered within 1-2 cohorts except 2012
(1,700 → fast; 4,200 → ≥ 3 cohorts, now 0.42). The automated count in `factorial_analysis.exploratory` will replace
this hand tally.
