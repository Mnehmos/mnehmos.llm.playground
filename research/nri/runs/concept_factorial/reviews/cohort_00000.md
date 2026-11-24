# Cohort review — step 0 (initialisation), 96/96 audited

**What changed / evidence.** Nothing has trained. All 8 conditions are numerically identical (every metric equal
to the last digit across conditions for each seed), confirming the registered matched initialisation empirically
(also tested in `tests/test_factorial.py`). Behaviour at chance: own-trained 0.04, held-out 0.04 (best constant
0.17); canonical quantity decodable at x1m 0.23 (kNN, chance 1/13 = 0.08 — partly decodable from random features
because the source-number token's identity is already in the embedding and q is a function of (v, unit) where the
unit is adjacent).

**Instrument finding (important for interpretation).** The discrete-log circle score for codes grouped by the
CANONICAL quantity is 0.45 at initialisation versus its shuffled-order baseline 0.26; for codes grouped by the RAW
value it is 0.21 (shuffled 0.21). Cause: each canonical class averages the codes of values that differ by a unit
scale factor (multiplication), so canonical grouping builds multiplicative (discrete-log) structure into the class
means even from random embeddings. Consequence: the null for `circ_can` is the matched-initialisation value
(~0.45), not the shuffled mean. Pilot doc 36 reported only raw-grouped circles (null 0.21), so its 0.52-0.74
values are unaffected; all factorial analyses of canonical geometry will be expressed relative to step 0 of the
same network.

**Competing explanations.** None needed for behaviour. For the artifact: alternatively the random network could
genuinely carry multiplicative structure through the unit token; this is excluded because x0 (pure embeddings
before any mixing) shows the same 0.45.

**Prediction for step 100.** All populations reach high own-trained accuracy quickly (pilot: practiced ≥ 0.95 by
step 300 for B/C/D); at step 100 own-trained accuracy will be < 0.9 for every condition and held-out will stay
below the 0.17 best constant everywhere.

**New instrument needed?** A random-code null for the canonical circle score (replace each value's code by an
independent random vector, then group by q) — to be added to the analysis, not to the registered audit.
