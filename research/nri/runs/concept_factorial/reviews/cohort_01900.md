# Cohort review — step 1,900 (96/96)

**What changed.** Utilisation onset at 1,800 for C wd 1.0 seed 2003 (6/12). C wd 1.0 seed 2004 held-out **0.36**
(criterion 0.37; 0.31 at 1,800) — on the threshold. Seeds 2009 0.28, 2007 0.22. C wd 1.0 new-frame held-out 0.09.

**Largest episodes so far (measured).** A wd 1.0 seed 2010: own-trained 1.00 → 0.45. D wd 1.0 seed 2009: own-trained
1.00 → 0.48. Eleven sustained flags, all at wd 1.0 (A, B, D; no C this block).

**Asymmetry worth tracking (Q5).** In D seed 2009's episode the **auxiliary loss rose 27-fold (0.016 → 0.43)**
although its decodability barely moved (0.56 → 0.50); in every captured C episode the auxiliary loss rose only
0.003-0.04. One D case vs four C cases: the canonical code (a quantity shared across units) may be more robust to
episodes than D's per-unit scrambled code (which must be maintained separately for each unit). Prediction, to be
tested over all episode blocks at the end: the median relative rise of auxiliary loss in episode blocks is larger in
D than in C (paired by seed where both have episodes).

**Prediction (re-stated).** Seed 2004 records a compositional onset at 1,900 or 2,000 (needs ≥ 0.37 at two
consecutive audits).
