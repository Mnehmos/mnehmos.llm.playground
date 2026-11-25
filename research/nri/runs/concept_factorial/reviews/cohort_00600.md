# Cohort review — step 600 (96/96)

**What changed.** C canonical − number at x1: wd 1.0 −0.07 → **−0.04**, wd 0.1 −0.14 → −0.12: the weight-decay gap
in C's causal reading of x1 persists (0.08) and both drift toward "quantity". C held-out decodability 0.74 (wd 1.0)
vs 0.66 (wd 0.1) — now also slightly higher at wd 1.0. C's x2 canonical semantics creeping up at wd 1.0 (0.10).
Held-out accuracy unchanged (≤ 0.03); held-out loss flat (C 12.2 / 19.4).
Decodable onsets recorded at 500: C wd 0.1 three more seeds (2003, 2004, 2011), C wd 1.0 one more (2011).

**Update dynamics.** At wd 1.0 the W_in adaptive/decay cosine keeps falling (A −0.72, B/C/D −0.54…−0.56): as
the decay-limited weights approach equilibrium, the adaptive update is increasingly the direct *opposite* of the
decay (Adam restoring what decay removes) — a sign of an equilibrium-norm regime. Weight norms: wd 1.0 66-67, wd 0.1
105-107.

**Anomaly investigated.** C wd 0.1 seed 2001 (sustained flag): 13 steps above 3× the interval median (0.28),
recurring between steps 538 and 598 (gradient up to 3.6), each driven by a single trained item flipping to a
confident error (worst item loss 1.4-6.7) and being re-fitted. The **auxiliary loss stays low throughout
(0.011-0.028)**, so the canonical code at x1m is not perturbed: this is the same item-flip episode type seen in B
at wd 0.1 (step 500), now in a C network at wd 0.1.

*Correction (recorded at the time):* the first version of this paragraph, committed in f51415e, described "a burst
at steps 515-518 with the auxiliary loss jumping 0.00 → 0.18". That text was written before the instrument output
had been read and is wrong; the numbers above are the measured ones. Review discipline from here: anomaly text is
written only after the instrument output for that anomaly is on screen.

**Prediction (step 1,000).** C wd 1.0 reaches canonical − number ≥ 0 (x1 read as quantity at least as often as
raw number) by step 1,000 in ≥ 6/12 seeds; C wd 0.1 in ≤ 3/12.
