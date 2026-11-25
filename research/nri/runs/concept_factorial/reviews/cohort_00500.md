# Cohort review — step 500 (96/96)

**What changed.** Decodable onsets continue (recorded at 400: C 2 more seeds at wd 0.1, 4 at wd 1.0; D 2 at wd 0.1,
2 at wd 1.0). C held-out decodability 0.65 (wd 0.1) / 0.68 (wd 1.0). Held-out accuracy still ≤ 0.03 everywhere.

**Causal reading of x1 — first weight-decay separation.** C canonical − number: −0.14 (wd 0.1) vs **−0.07
(wd 1.0)**; the gap (0.07) already exceeds the 0.05 I predicted for step 1,000 (step-400 prediction, early
confirmation; still a mean of 12, to be re-checked as noise). Weight decay is shifting how the downstream circuit
reads x1 in C before any held-out gain — the ordering the weight-decay hypothesis requires. B/D unchanged
(≈ −0.27…−0.34), A −0.12.

**Anomalies — new instrument.** The registered relative spike rule now fires on 11+ networks per cohort, mostly
isolated single-batch events. I added a deterministic classifier to `sci/cohort.py` (longest run of consecutive
steps above 3× the interval median; ≥ 3 = "sustained"), checked against the step-400 cases inspected by hand (it
labels seed 2009's single spike "isolated"; it labels seed 2001's single spike + decay "sustained" — conservative).
Two sustained cases at step 500, both **B at wd 0.1** (seeds 2002, 2005): 30-40-step episodes in which a few
*trained* items repeatedly flip to confident errors (item loss 3.6-4.8) and are re-fitted, with own-trained
accuracy back at 1.00 by the audit. Candidate interpretation: Adam instability with very small gradients on a
fitted lookup (cf. the "slingshot" late-training spikes reported for adaptive optimisers) — a candidate mechanism
for Q5 collapses.

**Prediction.** Sustained gradient episodes are more frequent at wd 0.1 than at wd 1.0 across all populations
(counted over the whole run from interval logs), and in C they will precede the held-out collapses seen in the
pilot. Falsifier: equal frequency across weight decays, or C collapses without a preceding sustained episode.

**Instrument to build:** a whole-run episode counter over interval logs (per network, per 100-step block), to be
run retrospectively at the end and replayed on every cohort.
