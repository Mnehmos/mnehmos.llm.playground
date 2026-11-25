# Cohort review — step 4,700 (96/96)

**What changed.** Seed 2004 — the canonical composer — collapsed for the fifth time (0.89 → 0.38; own-trained 0.76;
x1 canonical semantics 0.87 → 0.66): at the 100-step resolution its collapses are now roughly every 400-500 steps
(2,500, 3,000, 3,500, 3,900, 4,700). Seed 2001 0.72 → 0.53. C wd 1.0 mean held-out 0.49 (new frame 0.22). B 2002
own-trained 0.51. 17 sustained (all wd 1.0).

**Interpretation update.** The canonical-vs-mixed robustness hypothesis (step 2,800) is weakening: the strongest
canonical composer (2004) collapses at least as often as the mixed-code composers. What distinguishes networks is
the *level* they return to, not the frequency of collapse. Revised prediction for the end count: collapse frequency
per composing audit does not differ by code type (canonical vs mixed), while post-recovery held-out is higher for
canonical composers.

**Primary-endpoint implication.** The registered endpoint is a single snapshot at 6,000; with collapses every
~5 cohorts per network at wd 1.0, roughly 15-25% of C wd 1.0 networks will be caught mid-collapse at the final audit.
This lowers the C mean but cannot reverse the C vs D sign (D has never exceeded 0.20 held-out in any network).
