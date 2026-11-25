# Cohort review — step 2,500

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.792 | 0.560 | 6 | 0.791 | 0.512 |
| acc_only | 12 | 0.771 | 0.521 | 6 | 0.762 | 0.541 |
| neither | 12 | 0.933 | 0.818 | 10 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.98 | 0.32 | 0.56 |
| mass_only_s311 | focus | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.99 | 0.26 | 0.33 |
| mass_only_s304 | comparator | 0.993 | 0.266 | 0.09 | 0.51 | 0.99 | 0.09 | 0.63 | 0.65 |
| mass_only_s310 | comparator | 1.000 | 0.252 | 0.09 | 0.47 | 1.00 | 0.09 | 0.73 | 0.82 |
| mass_only_s305 | intermediate | 1.000 | 0.774 | 0.59 | 1.00 | 1.00 | 0.45 | 0.39 | 0.58 |

## Events since step 2,400

- **acc_only_s306**: ID >= 0.95
- **mass_only_s300**: block carries unseen inputs (IIA-OOD >= 0.5)

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.318 | 0.32 | 0.42 | 1.00 | 0.43 | 0.59 | 0.91 |
| acc_only_s301 | 0.580 | 0.504 | 0.49 | 0.46 | 0.60 | 0.47 | 0.84 | 0.62 |
| acc_only_s302 | 0.606 | 0.529 | 0.53 | 0.53 | 0.58 | 0.54 | 0.61 | 0.40 |
| acc_only_s303 | 0.615 | 0.529 | 0.51 | 0.51 | 0.58 | 0.52 | 0.36 | 0.59 |
| acc_only_s304 | 0.254 | 0.150 | 0.16 | 0.15 | 0.23 | 0.16 | 0.89 | 0.94 |
| acc_only_s305 | 1.000 | 0.818 | 0.90 | 0.81 | 1.00 | 0.82 | 0.56 | 0.56 |
| acc_only_s306 | 1.000 | 0.512 | 0.46 | 0.62 | 1.00 | 0.61 | 0.79 | 0.94 |
| acc_only_s307 | 0.998 | 0.718 | 0.52 | 0.98 | 0.99 | 0.96 | 0.41 | 0.57 |
| acc_only_s308 | 1.000 | 0.920 | 0.98 | 0.89 | 1.00 | 0.92 | 0.37 | 0.25 |
| acc_only_s309 | 0.599 | 0.521 | 0.51 | 0.52 | 0.58 | 0.54 | 0.57 | 0.45 |
| acc_only_s310 | 1.000 | 0.264 | 0.44 | 0.11 | 1.00 | 0.13 | 0.77 | 0.94 |
| acc_only_s311 | 0.600 | 0.467 | 0.50 | 0.38 | 0.59 | 0.39 | 0.91 | 0.84 |
| mass_only_s300 | 0.985 | 0.384 | 0.51 | 0.29 | 0.96 | 0.51 | 0.39 | 0.76 |
| mass_only_s301 | 0.584 | 0.524 | 0.50 | 0.51 | 0.59 | 0.53 | 0.15 | 0.46 |
| mass_only_s302 | 0.614 | 0.477 | 0.43 | 0.45 | 0.60 | 0.44 | 0.97 | 0.75 |
| mass_only_s303 | 0.571 | 0.538 | 0.53 | 0.52 | 0.58 | 0.53 | 0.92 | 0.45 |
| mass_only_s304 | 0.993 | 0.266 | 0.09 | 0.51 | 0.99 | 0.09 | 0.63 | 0.65 |
| mass_only_s305 | 1.000 | 0.774 | 0.59 | 1.00 | 1.00 | 0.45 | 0.39 | 0.58 |
| mass_only_s306 | 0.570 | 0.461 | 0.52 | 0.42 | 0.60 | 0.52 | 0.98 | 0.80 |
| mass_only_s307 | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.98 | 0.32 | 0.56 |
| mass_only_s308 | 0.608 | 0.528 | 0.49 | 0.48 | 0.59 | 0.49 | 0.92 | 0.48 |
| mass_only_s309 | 0.580 | 0.521 | 0.51 | 0.51 | 0.59 | 0.53 | 0.17 | 0.46 |
| mass_only_s310 | 1.000 | 0.252 | 0.09 | 0.47 | 1.00 | 0.09 | 0.73 | 0.82 |
| mass_only_s311 | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.99 | 0.26 | 0.33 |
| neither_s300 | 1.000 | 0.858 | 0.88 | 0.91 | — | — | 0.26 | 0.28 |
| neither_s301 | 1.000 | 0.967 | 0.97 | 1.00 | — | — | 0.42 | 0.75 |
| neither_s302 | 0.995 | 0.839 | 0.91 | 0.86 | — | — | 0.52 | 0.45 |
| neither_s303 | 1.000 | 0.929 | 0.96 | 0.95 | — | — | 0.55 | 0.79 |
| neither_s304 | 0.603 | 0.522 | 0.46 | 0.50 | — | — | 0.27 | 0.57 |
| neither_s305 | 1.000 | 0.803 | 0.84 | 0.85 | — | — | 0.60 | 0.44 |
| neither_s306 | 0.599 | 0.512 | 0.50 | 0.49 | — | — | 0.52 | 0.57 |
| neither_s307 | 1.000 | 0.939 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.813 | 0.86 | 0.83 | — | — | 0.43 | 0.34 |
| neither_s309 | 1.000 | 0.826 | 0.85 | 0.96 | — | — | 0.54 | 0.80 |
| neither_s310 | 1.000 | 0.884 | 0.91 | 0.93 | — | — | 0.53 | 0.39 |
| neither_s311 | 1.000 | 0.923 | 0.94 | 0.95 | — | — | 0.43 | 0.38 |

## Scientific review (supervisor)

**Measured state.** mass_only: 6/12 fit (ID ≥ 0.95), mean ID 0.79, mean OOD 0.52, block IIA 0.79, unseen-source IIA 0.51, factorisation R²_add 0.69; acc_only: 6/12 fit (ID ≥ 0.95), mean ID 0.77, mean OOD 0.53, block IIA 0.76, unseen-source IIA 0.54, factorisation R²_add 0.78; neither: 10/12 fit (ID ≥ 0.95), mean ID 0.93, mean OOD 0.82.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 1.00 / 0.98 / 1.00; 311: 1.00 / 1.00 / 0.99 / 0.98; 305: 1.00 / 0.59 / 0.45 / 0.96; 304: 0.99 / 0.09 / 0.09 / 0.71; 310: 1.00 / 0.09 / 0.09 / 0.69.
**Events since previous cohort:** 2 — acc_only_s306: ID >= 0.95; mass_only_s300: block carries unseen inputs (IIA-OOD >= 0.5).
**Interpretation.** Stable phase with slow drift. Failures are frozen at chance on unseen masses (304, 310: 0.09). Generalisers stay at 0.97-1.00. 305 erodes slowly: per-value accuracy on unit 2 with values 9 and 11 falls from 0.51/0.57 at 1,800 to 0.25/0.35 at 3,000, and its unseen-source interchange from 0.55 to 0.41, while route-specificity stays fixed at 0.39. Unconstrained unseen pairings drift once no loss anchors them. Undeclared networks keep improving OOD slowly.
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): HELD (changes in unseen-mass accuracy: {307: 0.01, 311: 0.0, 305: 0.04, 304: 0.0, 310: -0.0}).
**Prediction for step 2,600:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

