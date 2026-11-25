# 40 — Independent verification of the registered concept factorial (Phase I.3)

Implementation: `sci/verify_factorial.py`, written separately from `sci/factorial_analysis.py` (numpy statistics;
reserved-test items rebuilt from the list registered in doc 37, not from the training code's split; answers from an
independent formula; models re-evaluated from the archived step-6,000 weights after checking each file's SHA-256
against its Git manifest; onset code reimplemented). Output: `runs/concept_factorial/verification.json`. Original
results were read only for comparison and are unchanged.

## Result: no discrepancies in the registered results

| Check | Result |
|---|---|
| Checkpoint completeness | 96 × 61 = **5,856** checkpoints; every required file present in every manifest; 61 audits and 6,000 interval records per run |
| Archive integrity | 96/96 archives match `SHA256SUMS`; deep content check of one archive 673/673 files |
| Initial parameter matching | step-0 weights identical within each seed across all 8 conditions; distinct across the 12 seeds |
| Training-stream matching | identical batch-hash streams across weight decays for all 48 population × seed pairs |
| Partitions | train/dev/test disjoint (120/18/18); registered reserved list = split's test set |
| Reserved test re-evaluated from weights | all 72 × 96 item outcomes identical to `final_test.json` (max accuracy difference 0.0) |
| Best constant | 0.1667 |
| Registered onsets (C wd 1.0) | recomputed independently: identical for all 12 networks |

| Condition mean (reserved test) | A | B | C | D |
|---|---|---|---|---|
| wd 0.1 | 0.0613 | 0.0984 | **0.1343** | 0.0579 |
| wd 1.0 | 0.0648 | **0.3067** | **0.6551** | **0.1007** |

| Contrast | mean (independent) | original | exact sign-flip p (indep. / orig.) | Holm p (indep. / orig.) | 95% CI independent (numpy, 20,000) | original (10,000) |
|---|---|---|---|---|---|---|
| H1 C − D @1.0 | +0.5544 | +0.5544 | 0.000244 / 0.000244 | 0.00122 / 0.00122 | 0.450-0.662 | 0.450-0.659 |
| H2 interaction | +0.4780 | +0.4780 | 0.000244 / 0.000244 | 0.00122 / 0.00122 | 0.372-0.588 | 0.372-0.587 |
| H3 C@1.0 − C@0.1 | +0.5208 | +0.5208 | 0.000244 / 0.000244 | 0.00122 / 0.00122 | 0.431-0.615 | 0.431-0.613 |
| H4 C − B @1.0 | +0.3484 | +0.3484 | 0.00146 / 0.00146 | 0.00146 / 0.00146 | 0.199-0.480 | 0.199-0.479 |
| H5 B − A @1.0 | +0.2419 | +0.2419 | 0.00049 / 0.00049 | 0.00122 / 0.00122 | 0.177-0.331 | 0.176-0.332 |

Bootstrap intervals differ only in the third decimal (different resampling generator and count) — expected Monte
Carlo variation, not a discrepancy. The minimum attainable one-sided p with 12 paired seeds is 1/4096 = 0.000244;
H1-H3 sit at that floor (every seed's difference positive).

Scope: this verifies the computation and data integrity within this environment; it is internal, not an external
independent replication.
