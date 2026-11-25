# Cohort review — step 1,000 (96/96)

**What changed.** Slow consolidation. C wd 1.0: canonical − number +0.07 (per seed 0.00…+0.16, all ≥ 0), held-out
0.04 (seeds 2006, 2009, 2011 at 0.10-0.14 — the same three that led at step 700), decodability 0.84. C wd 0.1:
−0.10, 0.02, 0.71. Weight norms: wd 1.0 ≈ 47-48, wd 0.1 ≈ 102.

**Episodes (measured).** Sustained episodes now in B wd 1.0 seed 2004 (steps 942-954: batch loss to 0.75, gradient
to 12.9), C wd 1.0 seeds 2001/2005/2012 (gradients to 4.4-8.4) and D wd 1.0 seed 2004. In C episodes the task
loss spikes (batch to 0.29, single items to 9.3) while the **auxiliary loss rises only slightly** (2012: 0.03 →
0.06; 2005: 0.02 → 0.04): the canonical code is nudged but not disrupted. Neither C network lost held-out accuracy
or x1 canonical reading across the episode (2012: canonical − number 0.11 → 0.16; 2005: 0.00 → 0.05). All
sustained episodes this cohort are at **wd 1.0**; none at wd 0.1. This pattern now spans B, C and D and favours
explanation (b) (decay-driven drift of fitted lookups to their decision boundary).

**Mid-run status of registered predictions.** M3 (wd 1.0 > wd 0.1 in x1 canonical − number for C) — holding at
every cohort since 500 (current +0.07 vs −0.10). M4 (C wd 0.1 decodable but not compositional) — holding
(decodable 0.71, held-out 0.02). M1/M2 — no compositional onsets yet (pilot: 2,900-3,900).

**Step-700 prediction** (first to cross canonical − number +0.2 are first to compose) — cannot be tested until
onsets occur; current leaders in held-out (2006, 2009, 2011) are also among the highest x1 readers (0.12, 0.09,
0.09) but 2012 (0.16) has held-out 0.00, a counterexample in the making.
