# Cohort review — step 6,000

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.843 | 0.526 | 7 | 0.842 | 0.446 |
| acc_only | 12 | 0.929 | 0.502 | 9 | 0.921 | 0.541 |
| neither | 12 | 0.978 | 0.842 | 11 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.99 | 0.31 | 0.55 |
| mass_only_s311 | focus | 1.000 | 0.997 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.32 |
| mass_only_s304 | comparator | 1.000 | 0.248 | 0.09 | 0.47 | 1.00 | 0.09 | 0.63 | 0.66 |
| mass_only_s310 | comparator | 1.000 | 0.263 | 0.09 | 0.49 | 1.00 | 0.09 | 0.73 | 0.83 |
| mass_only_s305 | intermediate | 1.000 | 0.711 | 0.46 | 1.00 | 1.00 | 0.36 | 0.39 | 0.57 |

## Events since step 5,900

- none

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.341 | 0.35 | 0.42 | 1.00 | 0.43 | 0.59 | 0.92 |
| acc_only_s301 | 1.000 | 0.235 | 0.29 | 0.18 | 1.00 | 0.18 | 0.66 | 0.66 |
| acc_only_s302 | 0.583 | 0.536 | 0.48 | 0.48 | 0.58 | 0.49 | 0.61 | 0.39 |
| acc_only_s303 | 0.678 | 0.432 | 0.35 | 0.40 | 0.66 | 0.39 | 0.36 | 0.62 |
| acc_only_s304 | 0.887 | 0.275 | 0.28 | 0.31 | 0.81 | 0.30 | 0.60 | 0.89 |
| acc_only_s305 | 1.000 | 0.824 | 0.91 | 0.82 | 1.00 | 0.84 | 0.55 | 0.56 |
| acc_only_s306 | 1.000 | 0.479 | 0.47 | 0.54 | 1.00 | 0.54 | 0.78 | 0.94 |
| acc_only_s307 | 1.000 | 0.771 | 0.62 | 1.00 | 1.00 | 1.00 | 0.41 | 0.56 |
| acc_only_s308 | 1.000 | 0.926 | 0.99 | 0.89 | 1.00 | 0.94 | 0.36 | 0.24 |
| acc_only_s309 | 1.000 | 0.674 | 0.44 | 0.97 | 1.00 | 0.98 | 0.52 | 0.45 |
| acc_only_s310 | 1.000 | 0.287 | 0.43 | 0.18 | 1.00 | 0.21 | 0.78 | 0.95 |
| acc_only_s311 | 1.000 | 0.245 | 0.30 | 0.21 | 1.00 | 0.19 | 0.88 | 0.91 |
| mass_only_s300 | 1.000 | 0.416 | 0.57 | 0.32 | 1.00 | 0.54 | 0.38 | 0.76 |
| mass_only_s301 | 0.603 | 0.510 | 0.46 | 0.50 | 0.60 | 0.47 | 0.14 | 0.45 |
| mass_only_s302 | 1.000 | 0.354 | 0.19 | 0.57 | 1.00 | 0.17 | 0.91 | 0.79 |
| mass_only_s303 | 0.596 | 0.525 | 0.46 | 0.47 | 0.59 | 0.46 | 0.96 | 0.43 |
| mass_only_s304 | 1.000 | 0.248 | 0.09 | 0.47 | 1.00 | 0.09 | 0.63 | 0.66 |
| mass_only_s305 | 1.000 | 0.711 | 0.46 | 1.00 | 1.00 | 0.36 | 0.39 | 0.57 |
| mass_only_s306 | 0.660 | 0.338 | 0.27 | 0.40 | 0.67 | 0.26 | 0.98 | 0.80 |
| mass_only_s307 | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.99 | 0.31 | 0.55 |
| mass_only_s308 | 0.651 | 0.436 | 0.38 | 0.38 | 0.66 | 0.43 | 0.93 | 0.49 |
| mass_only_s309 | 0.601 | 0.518 | 0.47 | 0.47 | 0.59 | 0.48 | 0.17 | 0.46 |
| mass_only_s310 | 1.000 | 0.263 | 0.09 | 0.49 | 1.00 | 0.09 | 0.73 | 0.83 |
| mass_only_s311 | 1.000 | 0.997 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.32 |
| neither_s300 | 1.000 | 0.847 | 0.87 | 0.90 | — | — | 0.25 | 0.27 |
| neither_s301 | 1.000 | 0.973 | 0.98 | 1.00 | — | — | 0.41 | 0.74 |
| neither_s302 | 1.000 | 0.846 | 0.92 | 0.88 | — | — | 0.59 | 0.48 |
| neither_s303 | 1.000 | 0.942 | 0.97 | 0.97 | — | — | 0.54 | 0.79 |
| neither_s304 | 0.741 | 0.473 | 0.39 | 0.45 | — | — | 0.23 | 0.52 |
| neither_s305 | 1.000 | 0.808 | 0.85 | 0.86 | — | — | 0.59 | 0.43 |
| neither_s306 | 1.000 | 0.781 | 0.80 | 0.80 | — | — | 0.32 | 0.53 |
| neither_s307 | 1.000 | 0.941 | 0.96 | 0.96 | — | — | 0.54 | 0.64 |
| neither_s308 | 1.000 | 0.821 | 0.87 | 0.85 | — | — | 0.43 | 0.33 |
| neither_s309 | 1.000 | 0.837 | 0.86 | 0.97 | — | — | 0.54 | 0.79 |
| neither_s310 | 1.000 | 0.890 | 0.92 | 0.93 | — | — | 0.52 | 0.38 |
| neither_s311 | 1.000 | 0.940 | 0.96 | 0.96 | — | — | 0.42 | 0.37 |

## Scientific review (supervisor)

**Measured state.** mass_only: 7/12 fit (ID ≥ 0.95), mean ID 0.84, mean OOD 0.45, block IIA 0.84, unseen-source IIA 0.45, factorisation R²_add 0.68; acc_only: 9/12 fit (ID ≥ 0.95), mean ID 0.93, mean OOD 0.53, block IIA 0.92, unseen-source IIA 0.54, factorisation R²_add 0.78; neither: 11/12 fit (ID ≥ 0.95), mean ID 0.98, mean OOD 0.84.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 1.00 / 0.99 / 1.00; 311: 1.00 / 1.00 / 1.00 / 0.98; 305: 1.00 / 0.46 / 0.36 / 0.96; 304: 1.00 / 0.09 / 0.09 / 0.71; 310: 1.00 / 0.09 / 0.09 / 0.68.
**Events since previous cohort:** 0.
**Interpretation.** Final state. Outcomes are as frozen since about step 2,000 for the focus networks: 307, 311 = 1.00; 305 = 0.46 (eroded from a peak of 0.75); 304, 310 = 0.09. Final factorisation: 307 1.00, 311 0.98, 305 0.96, 304 0.71, 310 0.68. Across all 24 declared networks, final factorisation ≥ 0.98 is necessary for near-perfect generalisation (4/4) but not sufficient (doc 46 §46.1).
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): HELD (changes in unseen-mass accuracy: {307: 0.0, 311: 0.0, 305: 0.0, 304: 0.0, 310: 0.0}).
**Final cohort.** No further prediction.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

