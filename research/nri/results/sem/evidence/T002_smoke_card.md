# Evidence card — T002_swap_reconciler v1 (NON-STUDY smoke, seed 999)

**Question.** Block-swap IIA reads 1.0 but T001 exemplar copy ~0.55. Which attribute explains the gap?
**Registered prediction (manifest, before running):** canonical-style sources ≥ 0.9, non-canonical much lower;
competing: source pool (train vs dev tuples) or cross-family.
**Controls (sandbox, all pass):** self-swap identity 1.0; full-h swap = source prediction 1.0; empty block
identity 1.0; strata reweight to the total.

| ckpt | overall | canonical-style src | non-canonical src | word-number src | abbr-unit src | fit vs dev pool | same vs cross family |
|---|---|---|---|---|---|---|---|
| s2800 | .571 | **1.000** | .141 | .080 | .180 | .573 / .570 | .556 / .578 |
| s3100 | .569 | **1.000** | .137 | .083 | .172 | .571 / .567 | .561 / .573 |

n = 1,365 scored pairs per checkpoint. Prediction confirmed; competing explanations falsified. Consequence
for the curriculum: a pretrained student that cannot bind word-numbers/abbreviations upstream will fail
every task presented in those forms; the style axis is a first-order lesson target.
