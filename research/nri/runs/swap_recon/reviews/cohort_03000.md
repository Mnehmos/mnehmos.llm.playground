# Cohort review — step 3,000

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.810 | 0.548 | 6 | 0.804 | 0.493 |
| acc_only | 12 | 0.772 | 0.514 | 6 | 0.766 | 0.522 |
| neither | 12 | 0.935 | 0.818 | 10 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.999 | 0.980 | 0.97 | 1.00 | 1.00 | 0.97 | 0.32 | 0.56 |
| mass_only_s311 | focus | 1.000 | 0.994 | 1.00 | 1.00 | 1.00 | 0.99 | 0.25 | 0.33 |
| mass_only_s304 | comparator | 1.000 | 0.249 | 0.09 | 0.47 | 1.00 | 0.09 | 0.63 | 0.67 |
| mass_only_s310 | comparator | 1.000 | 0.258 | 0.09 | 0.48 | 1.00 | 0.09 | 0.73 | 0.82 |
| mass_only_s305 | intermediate | 1.000 | 0.750 | 0.53 | 1.00 | 1.00 | 0.41 | 0.39 | 0.57 |

## Events since step 2,900

- **acc_only_s303**: block LOST unseen inputs (IIA-OOD < 0.5)
- **mass_only_s308**: block LOST unseen inputs (IIA-OOD < 0.5)

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.329 | 0.32 | 0.42 | 1.00 | 0.43 | 0.59 | 0.92 |
| acc_only_s301 | 0.650 | 0.421 | 0.44 | 0.34 | 0.64 | 0.34 | 0.82 | 0.62 |
| acc_only_s302 | 0.605 | 0.531 | 0.53 | 0.51 | 0.58 | 0.53 | 0.61 | 0.40 |
| acc_only_s303 | 0.586 | 0.532 | 0.50 | 0.49 | 0.59 | 0.49 | 0.36 | 0.58 |
| acc_only_s304 | 0.251 | 0.144 | 0.15 | 0.15 | 0.23 | 0.17 | 0.88 | 0.94 |
| acc_only_s305 | 1.000 | 0.821 | 0.91 | 0.82 | 1.00 | 0.82 | 0.55 | 0.56 |
| acc_only_s306 | 1.000 | 0.484 | 0.48 | 0.55 | 1.00 | 0.56 | 0.78 | 0.94 |
| acc_only_s307 | 1.000 | 0.769 | 0.61 | 1.00 | 1.00 | 0.99 | 0.41 | 0.57 |
| acc_only_s308 | 0.988 | 0.890 | 0.92 | 0.87 | 0.97 | 0.86 | 0.38 | 0.26 |
| acc_only_s309 | 0.590 | 0.522 | 0.48 | 0.51 | 0.58 | 0.52 | 0.57 | 0.45 |
| acc_only_s310 | 1.000 | 0.283 | 0.43 | 0.17 | 1.00 | 0.19 | 0.78 | 0.95 |
| acc_only_s311 | 0.592 | 0.441 | 0.50 | 0.38 | 0.59 | 0.37 | 0.90 | 0.84 |
| mass_only_s300 | 1.000 | 0.381 | 0.54 | 0.26 | 1.00 | 0.54 | 0.39 | 0.76 |
| mass_only_s301 | 0.604 | 0.531 | 0.52 | 0.53 | 0.59 | 0.53 | 0.15 | 0.46 |
| mass_only_s302 | 0.731 | 0.357 | 0.28 | 0.43 | 0.73 | 0.28 | 0.95 | 0.77 |
| mass_only_s303 | 0.596 | 0.531 | 0.52 | 0.52 | 0.58 | 0.53 | 0.94 | 0.44 |
| mass_only_s304 | 1.000 | 0.249 | 0.09 | 0.47 | 1.00 | 0.09 | 0.63 | 0.67 |
| mass_only_s305 | 1.000 | 0.750 | 0.53 | 1.00 | 1.00 | 0.41 | 0.39 | 0.57 |
| mass_only_s306 | 0.603 | 0.486 | 0.46 | 0.47 | 0.59 | 0.46 | 0.98 | 0.82 |
| mass_only_s307 | 0.999 | 0.980 | 0.97 | 1.00 | 1.00 | 0.97 | 0.32 | 0.56 |
| mass_only_s308 | 0.586 | 0.531 | 0.49 | 0.49 | 0.59 | 0.50 | 0.93 | 0.49 |
| mass_only_s309 | 0.601 | 0.533 | 0.52 | 0.50 | 0.58 | 0.54 | 0.17 | 0.46 |
| mass_only_s310 | 1.000 | 0.258 | 0.09 | 0.48 | 1.00 | 0.09 | 0.73 | 0.82 |
| mass_only_s311 | 1.000 | 0.994 | 1.00 | 1.00 | 1.00 | 0.99 | 0.25 | 0.33 |
| neither_s300 | 1.000 | 0.852 | 0.87 | 0.91 | — | — | 0.25 | 0.28 |
| neither_s301 | 1.000 | 0.971 | 0.98 | 1.00 | — | — | 0.42 | 0.75 |
| neither_s302 | 0.999 | 0.831 | 0.91 | 0.86 | — | — | 0.57 | 0.47 |
| neither_s303 | 1.000 | 0.935 | 0.96 | 0.96 | — | — | 0.54 | 0.79 |
| neither_s304 | 0.610 | 0.507 | 0.46 | 0.47 | — | — | 0.28 | 0.57 |
| neither_s305 | 1.000 | 0.806 | 0.85 | 0.86 | — | — | 0.59 | 0.44 |
| neither_s306 | 0.617 | 0.503 | 0.43 | 0.46 | — | — | 0.47 | 0.53 |
| neither_s307 | 1.000 | 0.939 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.816 | 0.87 | 0.84 | — | — | 0.43 | 0.34 |
| neither_s309 | 1.000 | 0.828 | 0.85 | 0.96 | — | — | 0.54 | 0.80 |
| neither_s310 | 1.000 | 0.883 | 0.91 | 0.93 | — | — | 0.53 | 0.39 |
| neither_s311 | 1.000 | 0.938 | 0.96 | 0.97 | — | — | 0.43 | 0.38 |

## Scientific review (supervisor)

**Measured state.** mass_only: 6/12 fit (ID ≥ 0.95), mean ID 0.81, mean OOD 0.50, block IIA 0.80, unseen-source IIA 0.49, factorisation R²_add 0.68; acc_only: 6/12 fit (ID ≥ 0.95), mean ID 0.77, mean OOD 0.52, block IIA 0.77, unseen-source IIA 0.52, factorisation R²_add 0.78; neither: 10/12 fit (ID ≥ 0.95), mean ID 0.94, mean OOD 0.82.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 0.97 / 0.97 / 1.00; 311: 1.00 / 1.00 / 0.99 / 0.98; 305: 1.00 / 0.53 / 0.41 / 0.96; 304: 1.00 / 0.09 / 0.09 / 0.71; 310: 1.00 / 0.09 / 0.09 / 0.69.
**Events since previous cohort:** 2 — acc_only_s303: block LOST unseen inputs (IIA-OOD < 0.5); mass_only_s308: block LOST unseen inputs (IIA-OOD < 0.5).
**Interpretation.** Stable phase with slow drift. Failures are frozen at chance on unseen masses (304, 310: 0.09). Generalisers stay at 0.97-1.00. 305 erodes slowly: per-value accuracy on unit 2 with values 9 and 11 falls from 0.51/0.57 at 1,800 to 0.25/0.35 at 3,000, and its unseen-source interchange from 0.55 to 0.41, while route-specificity stays fixed at 0.39. Unconstrained unseen pairings drift once no loss anchors them. Undeclared networks keep improving OOD slowly.
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): HELD (changes in unseen-mass accuracy: {307: -0.02, 311: -0.0, 305: -0.04, 304: 0.0, 310: 0.0}).
**Prediction for step 3,100:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

