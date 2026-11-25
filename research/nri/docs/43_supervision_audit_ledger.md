# 43 — Supervision audit ledger (Birth of a Concept, docs 35–42)

Every methodological decision, deviation, defect and correction in the program, with:
- **who** made it: O = owner, A = supervising agent;
- **when**, relative to the data it concerns: **pre** = before the data were seen, **post** = after;
- its **effect** on the claims.

**Provenance.** The supervising agent is Claude Code running in a cloud container. No specific model version is
inferred from attribution metadata.

**Purpose.** To let a reader check that no registered hypothesis, threshold or endpoint was changed after its data
were seen, and that every post-hoc change is labelled.

| # | date (2026) | item | by | pre/post | effect on claims | where |
|---|---|---|---|---|---|---|
| 1 | Nov | Pilot scorecard defect: held-out patch used trained combinations → `novel_combo_patch` | A | post (pilot) | pilot only; fixed before registration | doc 36 |
| 2 | Nov | Pilot: patch assumed number semantics → `patch_semantics` (number vs canonical) | A | post (pilot) | pilot only | doc 36 |
| 3 | Nov | Pilot: A's practiced score measured on others' combinations → `own_practiced` | A | post (pilot) | pilot only | doc 36 |
| 4 | Nov | Checkpoint requirement: full state every 100 steps (weights, optimizer, RNG, update decomposition) | O | pre (factorial) | enabled all retrospective work | doc 37 |
| 5 | Nov | `aux.*` parameter-name collision corrupted resume → `auxhead.*` names + uniqueness assert; name-sorted digest | A | pre (factorial) | none (caught by resume test) | doc 37 |
| 6 | Nov | Registration of populations, endpoints, H1–H5, M1–M4, onset thresholds (commit 1b54b0b) | A, O-approved | pre | defines confirmatory scope | doc 37 |
| 7 | Nov | Wrong `.gitignore` path committed 68 checkpoint files (37829c5); untracked in fa8ec10 (still in history) | A | — | none on results; repository size | git |
| 8 | Nov | Cohort-600 review written before reading output; corrected; rule: never write commentary before reading the data | A | during | review text only | reviews/ |
| 9 | Nov | Truncated output hid seed 2012's onset; onsets now recomputed directly | A | during | corrected | doc 38 |
| 10 | Nov | Doc-38 number errors corrected: new-frame 0.44→0.37; C ≥ 0.5 on 15/18 (not 16); 11/257 audits (not 10) | A | post | corrected before PR review | doc 38 |
| 11 | Nov | Owner review: `patch_semantics` uses trained contexts only; M2 ignores clustering | O | post | causal claim was limited to trained contexts; M2's precision overstated | doc 39 |
| 12 | Nov | Plan for held-out causal test + clustered M2 committed before running (51c546e) | A | pre (doc 39) | — | doc 39 |
| 13 | Nov | Doc-39 step-0 control failed (0.13–0.18) → excess over the no-intervention baseline | A | post (control) | reported as a deviation; all doc-39 rates are excess | doc 39 |
| 14 | Nov | "Archive preservation is the highest priority; do not launch training" | O | — | training paused; 96 archives delivered; owner verification pending | PRESERVATION.md |
| 15 | Nov | Independent re-derivation of all registered results from archived weights (separate code) | A | post | no discrepancy | doc 40 |
| 16 | Nov 25 | Phases II–III authorised with owner-side verification still pending; originals untouched | O | — | scope of this work | doc 41 |
| 17 | Nov 25 | Protocol doc 41 (instruments, H-A–H-E, replay ladder) committed before running (0b7c6b1) | A | pre | — | doc 41 |
| 18 | Nov 25 | Planted P3 failed the excess criterion (0.88): excess is bounded by 1 − baseline → switch rate added; P3 criterion re-stated on the switch rate | A | post (validation) | validation-driven instrument change, before any real-network result was read | doc 41R |
| 19 | Nov 25 | Probes fitted on ID inputs read 0.00 on OOD for networks whose block causally carries OOD values (0.99) → probes flagged invalid for OOD; causal OOD-source IIA used instead | A | post (validation) | probes not used as evidence about OOD | doc 41R |
| 20 | Nov 25 | Replay ladder Q1–Q3: EXACT before any reconstruction was launched | A | pre (Q4) | reconstruction claims carry the EXACT label | qualification.json |
| 21 | Nov 25 | Prospective experiments (doc 42 §42.2) written as proposals only; none run | A | — | requires owner approval | doc 42 |

## Invariants checked

* No registered hypothesis, threshold, endpoint or analysis in doc 37 was changed after training began. The
  reserved test was evaluated once (doc 38). Doc 40 re-derived it from weights with no discrepancy.
* No seed was excluded in any analysis.
* Exploratory results stay labelled: docs 39 and 41R are retrospective and exploratory.
* No original trajectory, checkpoint or measurement was modified. Archives and extracted weights are read-only.
  Their SHA-256 values are re-verified at the end of Phases II–III (doc 41R).
