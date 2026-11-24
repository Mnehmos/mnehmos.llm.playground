# 36 — Birth of a Concept, Stage 2: qualification pilot (exploratory)

24 networks = 4 populations (A narrow, B diverse, C canonical-quantity auxiliary, D route-scrambled control) ×
3 seeds (101-103) × 2 weight-decay settings (0.1, 1.0); 6,000 AdamW steps, batch 128, identical architecture
(SemStudent, 129,280 inference parameters). Code `sci/concept.py`, summary `sci/concept_report.py`, runs
`runs/concept_pilot/`. Task: modular unit conversion (mod 13, fictional units — not real-world arithmetic).
Split: 12 routes × 13 values; 36 combinations withheld (18 dev, 18 reserved test, never evaluated here); every
withheld value and route is trained elsewhere. Held-out best-constant baseline 0.17. **Everything here is pilot
evidence: n = 3 per cell, thresholds and several instruments were defined or corrected during the pilot.**

## 36.1 Qualification answer: yes — one population crosses into composition

Final step 6,000 (mean ± sd over 3 seeds; best = best audited held-out value):

| wd | Pop | practiced | held-out | best held-out | new frame | canonical decodable at x1 (kNN) | dlog-circle R² at x1 |
|---|---|---|---|---|---|---|---|
| 0.1 | A | 0.43* | 0.05 | 0.06 | 0.06 | 0.19 | 0.21 |
| 0.1 | B | 1.00 | 0.02 | 0.06 | 0.02 | 0.26 | 0.21 |
| 0.1 | C | 0.97 | 0.00 | 0.06 | 0.04 | 0.88 | 0.52 |
| 0.1 | D | 1.00 | 0.03 | 0.06 | 0.04 | 0.38 | 0.26 |
| 1.0 | A | 0.41* | 0.03 | 0.10 | 0.02 | 0.18 | 0.25 |
| 1.0 | B | 0.81 | 0.07 | 0.12 | 0.06 | 0.19 | 0.32 |
| **1.0** | **C** | **1.00** | **0.59 ± 0.06** | **0.74** | 0.26 | **0.96** | **0.74** |
| 1.0 | D | 1.00 | 0.02 | 0.09 | 0.00 | 0.39 | 0.26 |

\* A's "practiced" counts all 120 trained combinations, of which A trains 42; on its own 42 it is ~1.00 (rescore).
Shuffled-order baseline for the circle R² ≈ 0.21-0.24.

* **C at weight decay 1.0 generalises in all three seeds** (0.64, 0.50, 0.62 at step 6,000). **D — the same
  auxiliary head, loss weight, classes and entropy, but labels that break cross-unit equivalence — does not
  (0.00, 0.06, 0.00).** The pilot therefore favours "the semantic structure of the auxiliary target", not "extra
  gradients", as the active ingredient — the question the C/D contrast was built for.
* Weight decay is necessary in this regime: C at 0.1 has the representation (decodable 0.88, circle 0.52) but
  never composes (0.00).
* Diversity alone (B) did not produce composition at either setting; narrow training (A) did not either.
* Paraphrase transfer of the new skill is partial: C's held-out accuracy in an unseen frame is 0.26 (0.11-0.44).

## 36.2 Three distinct transitions (C, wd 1.0)

Provisional onsets (two consecutive audits): canonical quantity **decodable** on held-out combinations at steps
400 / 700 / 300; **compositional** (held-out ≥ constant + 0.20) at 2,900 / 3,900 / 3,400. The representation
precedes its compositional use by ~2,500-3,400 steps in every seed. The original "causal" onset in the scorecard
used a flawed test (§36.4) and is not reported.

## 36.3 What the patched code means — layer 1 vs layer 2 (exploratory instrument `patch_semantics`)

Copy the residual at the source-number token from a donor (same dimension, different source unit) into a base,
and score two competing predictions on pairs where they differ: **number** (donor's raw value converted along the
base route) vs **canonical** (donor's canonical quantity expressed in the base's target unit); "unchanged" = base
answer kept. Seed 101 shown; n = 300 pairs.

| Site | B wd1 @6000 num / can / unch | C wd1 @6000 | D wd1 @6000 |
|---|---|---|---|
| x0 number token (input control) | .67 / .07 / .03 | .89 / .03 / .01 | .68 / .04 / .04 |
| x0 number + unit tokens (input control) | .22 / .40 / .03 | .15 / .58 / .01 | .12 / .43 / .08 |
| **x1 number token** | **.48 / .08 / .04** | **.10 / .57 / .06** | .15 / .12 / .20 |
| x2 number token | .09 / .06 / .44 | .07 / .27 / .32 | .11 / .07 / .27 |

The input controls behave as defined (swapping the token = number semantics; token + unit = canonical semantics).
After layer 1, **B's number token carries the raw number** (route-specific lookup downstream), **C's carries the
canonical quantity** — layer 1 has already folded the unit in — and D's carries neither consistently. After layer
2, B's code has largely left the token (unchanged 0.44), whereas C still carries the canonical quantity at 0.27.
On the question raised about PR #27: in this student family the number at layer 2 is not simply destroyed; in the
composing population it is *transformed* into a different variable, and in the non-composing ones it is consumed
or moved off the token.

**Coupling with behaviour.** Across the 12 saved checkpoints of each C (wd 1.0) seed, x1 canonical semantics and
held-out accuracy correlate at Spearman 0.90 / 0.97 / 0.97, and in **19 of 22** checkpoint-to-checkpoint held-out
changes larger than 0.05 the canonical semantics moved in the same direction — **including the transient
collapses** (seed 103 at step 4,500: held-out 0.64 → 0.22 with canonical 0.67 → 0.41; both recover). Canonical
semantics starts rising (0.26 at step 500) long before held-out accuracy leaves 0. Common time trends inflate
the level correlation; the change-direction agreement is the stronger evidence. At 500-step resolution, lead/lag
cannot be resolved.

## 36.4 Instrument defects found and fixed during the pilot

1. **Held-out causal patch scored mostly trained combinations** (donor number on a route where that combination
   was trained) — a lookup passes it. Replaced by `novel_combo_patch` (donor value on a route where that exact
   combination was withheld).
2. `novel_combo_patch` itself assumes **number semantics**; it under-reads C (0.03-0.16) because C's code is the
   canonical quantity. Superseded by `patch_semantics`, which tests both readings.
3. A's practiced score counted combinations A never trained on; `own_practiced` added.
4. **Checkpoints were saved every 500 steps (weights only) — not the required complete state every 100 steps.**
   Fixed after the pilot: every 100 steps the trainer now saves weights, auxiliary head, full AdamW state, torch and
   Python RNG and token-level gradient traces (per-token gradient norm and gradient × input on a fixed 48-item
   probe panel), plus a per-step interval log; a test verifies bit-exact resume (`tests/test_concept.py`). A
   name collision (`aux.*` used by both the student's role head and the new auxiliary head) silently corrupted
   resumed optimizer state and was caught by that test.

## 36.5 Recommendations for the registered experiment (to be frozen in doc 37 after owner review)

* **Primary endpoint**: held-out accuracy on the reserved test combinations at the final step, seed as the unit.
  Planned contrasts: C vs D (semantic structure vs extra supervision), C vs B, B vs A. Weight decay 1.0.
  With pilot effects of ~0.55 (C−D) and between-seed sd ~0.06, 12 seeds per population are ample for these
  contrasts; more seeds mainly buy precision on transition timing.
* **Mechanistic endpoints** (separate): `patch_semantics` at x1 per checkpoint; onset definitions frozen in
  advance; the canonical-semantics lead over behaviour tested at 100-step resolution on the registered seeds,
  with the lead estimate chosen on the pilot and only *tested* on new seeds.
* **Keep the weight-decay factor in the registered design?** The pilot shows it is decisive; options are fixing it
  at 1.0 (cleaner population contrast) or crossing it (a 4×2 design; 96 networks at 12 seeds).
* Retain the A population, but report its own-trained accuracy alongside.
* Compute: ~45 min per network on one CPU thread with the audit; 48 networks ≈ 9 h on 4 cores; storage ≈ 5 GB
  for complete 100-step checkpoints.
