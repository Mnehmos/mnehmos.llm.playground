# Cohort review — step 3,500 (96/96)

**What changed.** C wd 1.0 held-out mean 0.47 (new frame 0.21). Seed 2004 dipped 0.79 → 0.64 during a sustained
episode while its trained answers stayed intact at the audit and its canonical reading *rose* (0.73 → 0.83): a
held-out-only dip, smaller than its earlier collapses. Seed 2009 reached 0.69 (best yet). Seed 2011 0.36 (just
below criterion). Non-composers among C wd 1.0 with utilisation: 2005 (0.11) and 2010 (0.22) — the step-3,200
prediction that they do not compose before 4,000 stands so far.

**Episodes.** 14 sustained (all wd 1.0); trained-item loss at the audit only in B 2008 (0.85).

**Weight norms.** wd 1.0 networks have nearly stopped shrinking (C 20.1 → 19.8 over 100 steps); wd 0.1 continue
(≈ 0.75 per cohort). Equilibrium-norm regime at wd 1.0 reached.

**Prediction.** If episodes are driven by norm shrinkage (explanation b), episode frequency at wd 1.0 should now
decline as the norm stabilises: sustained flags per cohort at wd 1.0 average ≤ 8 over 4,500-6,000 (vs ≈ 13 over
3,000-3,500).
