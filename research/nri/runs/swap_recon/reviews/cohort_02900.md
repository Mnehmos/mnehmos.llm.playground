# Cohort review — step 2,900

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.802 | 0.553 | 6 | 0.797 | 0.501 |
| acc_only | 12 | 0.769 | 0.522 | 6 | 0.766 | 0.534 |
| neither | 12 | 0.936 | 0.816 | 10 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 1.000 | 0.995 | 0.99 | 1.00 | 1.00 | 0.98 | 0.32 | 0.56 |
| mass_only_s311 | focus | 1.000 | 0.997 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.33 |
| mass_only_s304 | comparator | 1.000 | 0.252 | 0.09 | 0.47 | 1.00 | 0.09 | 0.63 | 0.67 |
| mass_only_s310 | comparator | 1.000 | 0.258 | 0.09 | 0.48 | 1.00 | 0.09 | 0.73 | 0.82 |
| mass_only_s305 | intermediate | 1.000 | 0.774 | 0.58 | 1.00 | 1.00 | 0.46 | 0.39 | 0.58 |

## Events since step 2,800

- **mass_only_s306**: block LOST unseen inputs (IIA-OOD < 0.5)

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.337 | 0.34 | 0.42 | 1.00 | 0.43 | 0.59 | 0.92 |
| acc_only_s301 | 0.604 | 0.463 | 0.46 | 0.39 | 0.62 | 0.40 | 0.83 | 0.62 |
| acc_only_s302 | 0.588 | 0.535 | 0.52 | 0.52 | 0.58 | 0.54 | 0.62 | 0.40 |
| acc_only_s303 | 0.602 | 0.540 | 0.51 | 0.51 | 0.59 | 0.52 | 0.37 | 0.59 |
| acc_only_s304 | 0.255 | 0.142 | 0.15 | 0.16 | 0.23 | 0.15 | 0.88 | 0.94 |
| acc_only_s305 | 1.000 | 0.822 | 0.90 | 0.82 | 1.00 | 0.82 | 0.55 | 0.56 |
| acc_only_s306 | 1.000 | 0.478 | 0.47 | 0.55 | 1.00 | 0.56 | 0.78 | 0.94 |
| acc_only_s307 | 1.000 | 0.769 | 0.60 | 1.00 | 1.00 | 0.98 | 0.41 | 0.57 |
| acc_only_s308 | 1.000 | 0.927 | 0.99 | 0.89 | 1.00 | 0.93 | 0.36 | 0.25 |
| acc_only_s309 | 0.582 | 0.535 | 0.52 | 0.52 | 0.58 | 0.52 | 0.56 | 0.45 |
| acc_only_s310 | 1.000 | 0.277 | 0.43 | 0.16 | 1.00 | 0.19 | 0.78 | 0.95 |
| acc_only_s311 | 0.601 | 0.442 | 0.47 | 0.35 | 0.59 | 0.37 | 0.91 | 0.84 |
| mass_only_s300 | 1.000 | 0.382 | 0.50 | 0.31 | 1.00 | 0.50 | 0.38 | 0.77 |
| mass_only_s301 | 0.614 | 0.529 | 0.52 | 0.53 | 0.58 | 0.52 | 0.15 | 0.46 |
| mass_only_s302 | 0.654 | 0.354 | 0.35 | 0.35 | 0.63 | 0.33 | 0.95 | 0.78 |
| mass_only_s303 | 0.581 | 0.550 | 0.53 | 0.53 | 0.58 | 0.52 | 0.94 | 0.44 |
| mass_only_s304 | 1.000 | 0.252 | 0.09 | 0.47 | 1.00 | 0.09 | 0.63 | 0.67 |
| mass_only_s305 | 1.000 | 0.774 | 0.58 | 1.00 | 1.00 | 0.46 | 0.39 | 0.58 |
| mass_only_s306 | 0.598 | 0.463 | 0.49 | 0.43 | 0.60 | 0.48 | 0.98 | 0.81 |
| mass_only_s307 | 1.000 | 0.995 | 0.99 | 1.00 | 1.00 | 0.98 | 0.32 | 0.56 |
| mass_only_s308 | 0.597 | 0.539 | 0.51 | 0.50 | 0.59 | 0.51 | 0.92 | 0.49 |
| mass_only_s309 | 0.580 | 0.540 | 0.53 | 0.54 | 0.58 | 0.53 | 0.17 | 0.46 |
| mass_only_s310 | 1.000 | 0.258 | 0.09 | 0.48 | 1.00 | 0.09 | 0.73 | 0.82 |
| mass_only_s311 | 1.000 | 0.997 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.33 |
| neither_s300 | 1.000 | 0.852 | 0.87 | 0.91 | — | — | 0.25 | 0.28 |
| neither_s301 | 1.000 | 0.969 | 0.97 | 0.99 | — | — | 0.42 | 0.75 |
| neither_s302 | 0.998 | 0.824 | 0.91 | 0.86 | — | — | 0.57 | 0.47 |
| neither_s303 | 1.000 | 0.935 | 0.96 | 0.96 | — | — | 0.54 | 0.79 |
| neither_s304 | 0.616 | 0.511 | 0.45 | 0.48 | — | — | 0.27 | 0.57 |
| neither_s305 | 1.000 | 0.804 | 0.85 | 0.85 | — | — | 0.59 | 0.44 |
| neither_s306 | 0.619 | 0.500 | 0.43 | 0.46 | — | — | 0.48 | 0.54 |
| neither_s307 | 1.000 | 0.941 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.815 | 0.86 | 0.83 | — | — | 0.43 | 0.34 |
| neither_s309 | 1.000 | 0.828 | 0.86 | 0.96 | — | — | 0.54 | 0.80 |
| neither_s310 | 1.000 | 0.884 | 0.91 | 0.93 | — | — | 0.53 | 0.39 |
| neither_s311 | 1.000 | 0.930 | 0.94 | 0.96 | — | — | 0.42 | 0.38 |

## Scientific review (supervisor)

**Measured state.** mass_only: 6/12 fit (ID ≥ 0.95), mean ID 0.80, mean OOD 0.51, block IIA 0.80, unseen-source IIA 0.50, factorisation R²_add 0.68; acc_only: 6/12 fit (ID ≥ 0.95), mean ID 0.77, mean OOD 0.52, block IIA 0.77, unseen-source IIA 0.53, factorisation R²_add 0.78; neither: 10/12 fit (ID ≥ 0.95), mean ID 0.94, mean OOD 0.82.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 0.99 / 0.98 / 1.00; 311: 1.00 / 1.00 / 1.00 / 0.98; 305: 1.00 / 0.58 / 0.46 / 0.96; 304: 1.00 / 0.09 / 0.09 / 0.71; 310: 1.00 / 0.09 / 0.09 / 0.69.
**Events since previous cohort:** 1 — mass_only_s306: block LOST unseen inputs (IIA-OOD < 0.5).
**Interpretation.** Stable phase with slow drift. Failures are frozen at chance on unseen masses (304, 310: 0.09). Generalisers stay at 0.97-1.00. 305 erodes slowly: per-value accuracy on unit 2 with values 9 and 11 falls from 0.51/0.57 at 1,800 to 0.25/0.35 at 3,000, and its unseen-source interchange from 0.55 to 0.41, while route-specificity stays fixed at 0.39. Unconstrained unseen pairings drift once no loss anchors them. Undeclared networks keep improving OOD slowly.
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): HELD (changes in unseen-mass accuracy: {307: -0.01, 311: 0.0, 305: 0.03, 304: 0.0, 310: -0.01}).
**Prediction for step 3,000:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

