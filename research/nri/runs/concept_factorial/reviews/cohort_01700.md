# Cohort review — step 1,700 (96/96)

**What changed.** Utilisation onset at 1,600 for C wd 1.0 seed 2001 (5/12 now) — half of the step-1,600 prediction
(2001 and 2009 by 2,000) met. **Episodes reach A** for the first time (wd 1.0 seeds 2005, 2009; seed 2005 caught at
own-trained 0.79): every population at wd 1.0 now shows episodes, none at wd 0.1 (20 cohorts). A at wd 1.0 has
the smallest weight norm (29.8) of the A/B/C/D wd 1.0 group trend; a narrow lookup also reaches its boundaries
once the norm is small enough.

**Fourth mid-episode capture — C wd 1.0 seed 2012 (measured 1,600 → 1,700):** own-trained 1.00 → 0.875, held-out
0.111 → 0.056, canonical decodability 0.93 → 0.92, raw circle 0.65 → 0.66, x1 canonical − number 0.36 → 0.25, aux
loss 0.054 → 0.063. Readout-first again (4/4: trained answers and held-out break while the representation stays),
and this time the episode **costs held-out accuracy and part of the causal canonical reading** while leaving
decodability and geometry intact — the pilot's "collapse" signature (held-out and x1 canonical reading falling
together) appearing at the moment of an instability episode.

**Q5 working account (to be tested on the whole run).** Collapse = an instability episode in a weight-decay-limited
network: the readout lookup breaks, downstream use of the canonical code at x1 weakens, held-out accuracy falls; the
representation itself (decodability, circle) is spared; recovery re-fits the readout. Prediction for seed 2012:
own-trained ≥ 0.99 and held-out ≥ 0.10 by step 1,900, with canonical − number back ≥ 0.30.
