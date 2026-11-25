# Cohort review — step 2,600

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.798 | 0.558 | 6 | 0.795 | 0.510 |
| acc_only | 12 | 0.768 | 0.526 | 6 | 0.765 | 0.536 |
| neither | 12 | 0.930 | 0.811 | 10 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 1.000 | 0.996 | 0.99 | 1.00 | 1.00 | 0.98 | 0.32 | 0.56 |
| mass_only_s311 | focus | 0.999 | 0.997 | 0.99 | 1.00 | 1.00 | 0.99 | 0.26 | 0.33 |
| mass_only_s304 | comparator | 1.000 | 0.271 | 0.09 | 0.52 | 1.00 | 0.09 | 0.63 | 0.67 |
| mass_only_s310 | comparator | 1.000 | 0.271 | 0.10 | 0.51 | 1.00 | 0.10 | 0.73 | 0.82 |
| mass_only_s305 | intermediate | 0.999 | 0.752 | 0.54 | 1.00 | 1.00 | 0.41 | 0.39 | 0.58 |

## Events since step 2,500

- **acc_only_s303**: block LOST unseen inputs (IIA-OOD < 0.5)
- **mass_only_s308**: block carries unseen inputs (IIA-OOD >= 0.5)

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.327 | 0.33 | 0.42 | 1.00 | 0.43 | 0.59 | 0.92 |
| acc_only_s301 | 0.585 | 0.517 | 0.51 | 0.46 | 0.59 | 0.48 | 0.84 | 0.62 |
| acc_only_s302 | 0.601 | 0.541 | 0.54 | 0.55 | 0.58 | 0.53 | 0.61 | 0.40 |
| acc_only_s303 | 0.593 | 0.547 | 0.49 | 0.49 | 0.59 | 0.50 | 0.37 | 0.59 |
| acc_only_s304 | 0.256 | 0.143 | 0.16 | 0.15 | 0.23 | 0.16 | 0.89 | 0.94 |
| acc_only_s305 | 1.000 | 0.823 | 0.90 | 0.82 | 1.00 | 0.82 | 0.55 | 0.56 |
| acc_only_s306 | 1.000 | 0.485 | 0.48 | 0.55 | 1.00 | 0.55 | 0.78 | 0.94 |
| acc_only_s307 | 1.000 | 0.753 | 0.58 | 0.99 | 1.00 | 0.97 | 0.41 | 0.57 |
| acc_only_s308 | 1.000 | 0.921 | 0.99 | 0.88 | 1.00 | 0.92 | 0.37 | 0.25 |
| acc_only_s309 | 0.579 | 0.534 | 0.52 | 0.53 | 0.59 | 0.53 | 0.56 | 0.45 |
| acc_only_s310 | 1.000 | 0.271 | 0.43 | 0.14 | 1.00 | 0.15 | 0.77 | 0.95 |
| acc_only_s311 | 0.602 | 0.455 | 0.47 | 0.39 | 0.60 | 0.39 | 0.90 | 0.84 |
| mass_only_s300 | 1.000 | 0.376 | 0.55 | 0.24 | 0.99 | 0.53 | 0.39 | 0.78 |
| mass_only_s301 | 0.582 | 0.510 | 0.51 | 0.54 | 0.60 | 0.54 | 0.15 | 0.46 |
| mass_only_s302 | 0.600 | 0.459 | 0.43 | 0.41 | 0.60 | 0.41 | 0.96 | 0.75 |
| mass_only_s303 | 0.602 | 0.539 | 0.54 | 0.52 | 0.59 | 0.52 | 0.93 | 0.44 |
| mass_only_s304 | 1.000 | 0.271 | 0.09 | 0.52 | 1.00 | 0.09 | 0.63 | 0.67 |
| mass_only_s305 | 0.999 | 0.752 | 0.54 | 1.00 | 1.00 | 0.41 | 0.39 | 0.58 |
| mass_only_s306 | 0.604 | 0.463 | 0.51 | 0.44 | 0.59 | 0.51 | 0.98 | 0.80 |
| mass_only_s307 | 1.000 | 0.996 | 0.99 | 1.00 | 1.00 | 0.98 | 0.32 | 0.56 |
| mass_only_s308 | 0.585 | 0.540 | 0.51 | 0.50 | 0.59 | 0.50 | 0.92 | 0.49 |
| mass_only_s309 | 0.610 | 0.523 | 0.53 | 0.52 | 0.58 | 0.53 | 0.17 | 0.46 |
| mass_only_s310 | 1.000 | 0.271 | 0.10 | 0.51 | 1.00 | 0.10 | 0.73 | 0.82 |
| mass_only_s311 | 0.999 | 0.997 | 0.99 | 1.00 | 1.00 | 0.99 | 0.26 | 0.33 |
| neither_s300 | 1.000 | 0.855 | 0.87 | 0.91 | — | — | 0.26 | 0.28 |
| neither_s301 | 1.000 | 0.970 | 0.97 | 0.99 | — | — | 0.42 | 0.75 |
| neither_s302 | 0.951 | 0.748 | 0.82 | 0.77 | — | — | 0.53 | 0.45 |
| neither_s303 | 0.998 | 0.931 | 0.96 | 0.96 | — | — | 0.55 | 0.79 |
| neither_s304 | 0.595 | 0.515 | 0.45 | 0.47 | — | — | 0.27 | 0.57 |
| neither_s305 | 1.000 | 0.804 | 0.85 | 0.85 | — | — | 0.60 | 0.44 |
| neither_s306 | 0.615 | 0.515 | 0.45 | 0.47 | — | — | 0.50 | 0.57 |
| neither_s307 | 1.000 | 0.940 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.818 | 0.87 | 0.83 | — | — | 0.43 | 0.34 |
| neither_s309 | 1.000 | 0.827 | 0.85 | 0.96 | — | — | 0.54 | 0.80 |
| neither_s310 | 1.000 | 0.883 | 0.91 | 0.93 | — | — | 0.53 | 0.39 |
| neither_s311 | 1.000 | 0.920 | 0.93 | 0.95 | — | — | 0.43 | 0.38 |

## Scientific review (supervisor)

**Measured state.** mass_only: 6/12 fit (ID ≥ 0.95), mean ID 0.80, mean OOD 0.52, block IIA 0.80, unseen-source IIA 0.51, factorisation R²_add 0.69; acc_only: 6/12 fit (ID ≥ 0.95), mean ID 0.77, mean OOD 0.53, block IIA 0.77, unseen-source IIA 0.54, factorisation R²_add 0.78; neither: 10/12 fit (ID ≥ 0.95), mean ID 0.93, mean OOD 0.81.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 0.99 / 0.98 / 1.00; 311: 1.00 / 0.99 / 0.99 / 0.98; 305: 1.00 / 0.54 / 0.41 / 0.96; 304: 1.00 / 0.09 / 0.09 / 0.71; 310: 1.00 / 0.10 / 0.10 / 0.69.
**Events since previous cohort:** 2 — acc_only_s303: block LOST unseen inputs (IIA-OOD < 0.5); mass_only_s308: block carries unseen inputs (IIA-OOD >= 0.5).
**Interpretation.** Stable phase with slow drift. Failures are frozen at chance on unseen masses (304, 310: 0.09). Generalisers stay at 0.97-1.00. 305 erodes slowly: per-value accuracy on unit 2 with values 9 and 11 falls from 0.51/0.57 at 1,800 to 0.25/0.35 at 3,000, and its unseen-source interchange from 0.55 to 0.41, while route-specificity stays fixed at 0.39. Unconstrained unseen pairings drift once no loss anchors them. Undeclared networks keep improving OOD slowly.
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): HELD (changes in unseen-mass accuracy: {307: -0.0, 311: -0.0, 305: -0.05, 304: 0.0, 310: 0.01}).
**Prediction for step 2,700:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

