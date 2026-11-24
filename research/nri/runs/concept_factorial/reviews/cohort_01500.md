# Cohort review — step 1,500 (96/96)

**First registered causal-utilisation onsets** (canonical − number ≥ 0.20 at two consecutive audits), both C wd 1.0:
seed 2012 at **1,200** and seed 2010 at **1,400**; both decodable since 300. No compositional onset anywhere yet
(C wd 1.0 held-out mean 0.07, best seeds 0.11-0.17).

*Process note:* seed 2012's onset was flagged by the cohort tool at cohort 1,300 but did not appear in my
truncated screen output, so the 1,300 review missed it. From this cohort, onsets are additionally recomputed
directly for all 96 networks each cohort (`cohort.onsets_so_far`) and the view is no longer truncated.

**What changed.** C wd 1.0 canonical − number +0.18 (rising ~0.02-0.03 per cohort), x2 canonical 0.15, held-out
task loss 8.2 (falling). C wd 0.1 completely flat since ~step 700 (−0.09 / 0.01 / 0.72). M4 holding.

**Episodes.** B wd 1.0 seed 2001 caught mid-episode (own-trained 0.88, gradient to 47); C 2001/2003, D 2008
(gradient to 71) sustained. All at wd 1.0 (zero at wd 0.1 for 18 cohorts).

**Check of step-1,400 prediction** (first compositional onsets 1,800-3,000, led by 2009 and 2012): utilisation is
led by 2012 (as predicted) and 2010 (not predicted); 2009 has the highest held-out (0.17) but canonical − number
only 0.14. Prediction stands.
