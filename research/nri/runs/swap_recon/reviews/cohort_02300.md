# Cohort review — step 2,300

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.787 | 0.544 | 6 | 0.777 | 0.503 |
| acc_only | 12 | 0.754 | 0.525 | 5 | 0.747 | 0.548 |
| neither | 12 | 0.935 | 0.814 | 10 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.98 | 0.32 | 0.57 |
| mass_only_s311 | focus | 0.999 | 0.993 | 0.99 | 1.00 | 1.00 | 0.99 | 0.27 | 0.34 |
| mass_only_s304 | comparator | 1.000 | 0.254 | 0.09 | 0.48 | 1.00 | 0.09 | 0.62 | 0.66 |
| mass_only_s310 | comparator | 1.000 | 0.249 | 0.10 | 0.46 | 1.00 | 0.09 | 0.73 | 0.82 |
| mass_only_s305 | intermediate | 0.996 | 0.787 | 0.61 | 0.99 | 0.96 | 0.44 | 0.39 | 0.58 |

## Events since step 2,200

- **acc_only_s301**: block carries unseen inputs (IIA-OOD >= 0.5)
- **acc_only_s306**: block carries unseen inputs (IIA-OOD >= 0.5)
- **acc_only_s310**: ID >= 0.95; declared block formed (IIA >= 0.9)
- **mass_only_s305**: block LOST unseen inputs (IIA-OOD < 0.5)
- **mass_only_s306**: block carries unseen inputs (IIA-OOD >= 0.5); OOD jump 0.23->0.46
- **mass_only_s309**: block carries unseen inputs (IIA-OOD >= 0.5)
- **neither_s302**: ID >= 0.95; OOD jump 0.59->0.82

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.320 | 0.31 | 0.42 | 1.00 | 0.44 | 0.59 | 0.91 |
| acc_only_s301 | 0.605 | 0.531 | 0.52 | 0.49 | 0.59 | 0.51 | 0.84 | 0.62 |
| acc_only_s302 | 0.593 | 0.538 | 0.54 | 0.52 | 0.59 | 0.53 | 0.61 | 0.40 |
| acc_only_s303 | 0.579 | 0.542 | 0.52 | 0.53 | 0.59 | 0.53 | 0.37 | 0.60 |
| acc_only_s304 | 0.257 | 0.147 | 0.15 | 0.15 | 0.24 | 0.16 | 0.89 | 0.94 |
| acc_only_s305 | 1.000 | 0.820 | 0.90 | 0.82 | 1.00 | 0.81 | 0.56 | 0.56 |
| acc_only_s306 | 0.844 | 0.455 | 0.28 | 0.65 | 0.83 | 0.60 | 0.86 | 0.93 |
| acc_only_s307 | 0.997 | 0.726 | 0.55 | 0.98 | 0.99 | 0.94 | 0.41 | 0.56 |
| acc_only_s308 | 1.000 | 0.919 | 0.99 | 0.88 | 1.00 | 0.92 | 0.37 | 0.25 |
| acc_only_s309 | 0.594 | 0.534 | 0.53 | 0.53 | 0.58 | 0.52 | 0.57 | 0.45 |
| acc_only_s310 | 0.988 | 0.308 | 0.41 | 0.23 | 0.99 | 0.24 | 0.78 | 0.93 |
| acc_only_s311 | 0.591 | 0.455 | 0.50 | 0.39 | 0.58 | 0.38 | 0.90 | 0.83 |
| mass_only_s300 | 1.000 | 0.351 | 0.51 | 0.24 | 0.99 | 0.47 | 0.38 | 0.75 |
| mass_only_s301 | 0.593 | 0.536 | 0.53 | 0.54 | 0.58 | 0.54 | 0.15 | 0.46 |
| mass_only_s302 | 0.593 | 0.465 | 0.48 | 0.44 | 0.59 | 0.47 | 0.96 | 0.75 |
| mass_only_s303 | 0.494 | 0.395 | 0.41 | 0.36 | 0.45 | 0.42 | 0.81 | 0.50 |
| mass_only_s304 | 1.000 | 0.254 | 0.09 | 0.48 | 1.00 | 0.09 | 0.62 | 0.66 |
| mass_only_s305 | 0.996 | 0.787 | 0.61 | 0.99 | 0.96 | 0.44 | 0.39 | 0.58 |
| mass_only_s306 | 0.589 | 0.458 | 0.53 | 0.42 | 0.58 | 0.52 | 0.97 | 0.81 |
| mass_only_s307 | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.98 | 0.32 | 0.57 |
| mass_only_s308 | 0.591 | 0.516 | 0.49 | 0.50 | 0.59 | 0.50 | 0.92 | 0.48 |
| mass_only_s309 | 0.587 | 0.526 | 0.53 | 0.51 | 0.58 | 0.52 | 0.18 | 0.46 |
| mass_only_s310 | 1.000 | 0.249 | 0.10 | 0.46 | 1.00 | 0.09 | 0.73 | 0.82 |
| mass_only_s311 | 0.999 | 0.993 | 0.99 | 1.00 | 1.00 | 0.99 | 0.27 | 0.34 |
| neither_s300 | 1.000 | 0.860 | 0.88 | 0.90 | — | — | 0.26 | 0.28 |
| neither_s301 | 1.000 | 0.954 | 0.96 | 0.99 | — | — | 0.43 | 0.75 |
| neither_s302 | 0.991 | 0.821 | 0.92 | 0.82 | — | — | 0.50 | 0.47 |
| neither_s303 | 1.000 | 0.927 | 0.96 | 0.96 | — | — | 0.56 | 0.79 |
| neither_s304 | 0.601 | 0.521 | 0.50 | 0.51 | — | — | 0.27 | 0.57 |
| neither_s305 | 1.000 | 0.804 | 0.85 | 0.85 | — | — | 0.60 | 0.45 |
| neither_s306 | 0.633 | 0.501 | 0.44 | 0.45 | — | — | 0.47 | 0.54 |
| neither_s307 | 1.000 | 0.937 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.812 | 0.86 | 0.83 | — | — | 0.43 | 0.34 |
| neither_s309 | 1.000 | 0.825 | 0.85 | 0.96 | — | — | 0.54 | 0.80 |
| neither_s310 | 1.000 | 0.883 | 0.91 | 0.93 | — | — | 0.53 | 0.39 |
| neither_s311 | 1.000 | 0.929 | 0.94 | 0.95 | — | — | 0.43 | 0.39 |

## Scientific review (supervisor)

**Measured state.** mass_only: 6/12 fit (ID ≥ 0.95), mean ID 0.79, mean OOD 0.52, block IIA 0.78, unseen-source IIA 0.50, factorisation R²_add 0.71; acc_only: 5/12 fit (ID ≥ 0.95), mean ID 0.75, mean OOD 0.55, block IIA 0.75, unseen-source IIA 0.55, factorisation R²_add 0.78; neither: 10/12 fit (ID ≥ 0.95), mean ID 0.94, mean OOD 0.81.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 1.00 / 0.98 / 1.00; 311: 1.00 / 0.99 / 0.99 / 0.98; 305: 1.00 / 0.61 / 0.44 / 0.97; 304: 1.00 / 0.09 / 0.09 / 0.71; 310: 1.00 / 0.10 / 0.09 / 0.69.
**Events since previous cohort:** 7 — acc_only_s301: block carries unseen inputs (IIA-OOD >= 0.5); acc_only_s306: block carries unseen inputs (IIA-OOD >= 0.5); acc_only_s310: ID >= 0.95, declared block formed (IIA >= 0.9); mass_only_s305: block LOST unseen inputs (IIA-OOD < 0.5); mass_only_s306: block carries unseen inputs (IIA-OOD >= 0.5), OOD jump 0.23->0.46; mass_only_s309: block carries unseen inputs (IIA-OOD >= 0.5); neither_s302: ID >= 0.95, OOD jump 0.59->0.82.
**Interpretation.** Stable phase with slow drift. Failures are frozen at chance on unseen masses (304, 310: 0.09). Generalisers stay at 0.97-1.00. 305 erodes slowly: per-value accuracy on unit 2 with values 9 and 11 falls from 0.51/0.57 at 1,800 to 0.25/0.35 at 3,000, and its unseen-source interchange from 0.55 to 0.41, while route-specificity stays fixed at 0.39. Unconstrained unseen pairings drift once no loss anchors them. Undeclared networks keep improving OOD slowly.
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): HELD (changes in unseen-mass accuracy: {307: 0.01, 311: 0.01, 305: -0.07, 304: 0.0, 310: 0.0}).
**Prediction for step 2,400:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

