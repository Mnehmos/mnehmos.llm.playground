# Cohort review — step 3,300 (96/96)

**What changed.** C wd 1.0 held-out mean 0.43; seed 2004 0.85, seed 2001 0.58 (up from 0.46 after its own episode
at 3,200 — another post-episode gain), seed 2006 0.40 (first audit above criterion; onset needs 3,400). No new
onsets. Episodes: 13 sustained flags, all wd 1.0; trained-item losses at the audit in A 2012 (0.69), D 2001 (0.69);
no C network lost own-trained accuracy at this audit (C's three sustained flags 2003/2006/2012 resolved within
their blocks).

**Observation.** C at wd 1.0 has had as many episode blocks as B and D (counter at 3,000: 48 vs 50 / 47) but, since
~2,700, its episodes rarely leave own-trained accuracy reduced at the audit, while A, B, D episodes still do
(A 0.21-0.69, D 0.69 this run). Possible reason: a composing network has a second, rule-like route to the answers,
so it re-fits faster. Prediction (end-of-run count): among episode blocks after step 2,500, the fraction with an
own-trained drop ≥ 0.05 at the audit is lower in C wd 1.0 than in each of A, B, D at wd 1.0.
