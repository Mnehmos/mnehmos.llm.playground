# Cohort review — step 5,200

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.839 | 0.529 | 7 | 0.841 | 0.452 |
| acc_only | 12 | 0.924 | 0.500 | 9 | 0.918 | 0.540 |
| neither | 12 | 0.977 | 0.842 | 11 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.99 | 0.31 | 0.55 |
| mass_only_s311 | focus | 1.000 | 0.997 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.32 |
| mass_only_s304 | comparator | 1.000 | 0.246 | 0.09 | 0.46 | 1.00 | 0.09 | 0.63 | 0.66 |
| mass_only_s310 | comparator | 1.000 | 0.265 | 0.09 | 0.50 | 1.00 | 0.09 | 0.73 | 0.83 |
| mass_only_s305 | intermediate | 1.000 | 0.713 | 0.47 | 1.00 | 1.00 | 0.36 | 0.39 | 0.57 |

## Events since step 5,100

- **mass_only_s309**: block LOST unseen inputs (IIA-OOD < 0.5)

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.337 | 0.34 | 0.42 | 1.00 | 0.43 | 0.59 | 0.92 |
| acc_only_s301 | 1.000 | 0.238 | 0.29 | 0.19 | 1.00 | 0.18 | 0.66 | 0.66 |
| acc_only_s302 | 0.573 | 0.529 | 0.49 | 0.49 | 0.59 | 0.49 | 0.61 | 0.39 |
| acc_only_s303 | 0.662 | 0.438 | 0.35 | 0.40 | 0.66 | 0.42 | 0.36 | 0.61 |
| acc_only_s304 | 0.853 | 0.250 | 0.26 | 0.29 | 0.77 | 0.27 | 0.61 | 0.89 |
| acc_only_s305 | 1.000 | 0.825 | 0.91 | 0.82 | 1.00 | 0.84 | 0.55 | 0.56 |
| acc_only_s306 | 1.000 | 0.477 | 0.46 | 0.54 | 1.00 | 0.54 | 0.78 | 0.94 |
| acc_only_s307 | 1.000 | 0.771 | 0.62 | 1.00 | 1.00 | 1.00 | 0.41 | 0.56 |
| acc_only_s308 | 1.000 | 0.926 | 0.99 | 0.89 | 1.00 | 0.94 | 0.36 | 0.24 |
| acc_only_s309 | 1.000 | 0.674 | 0.44 | 0.97 | 1.00 | 0.97 | 0.52 | 0.45 |
| acc_only_s310 | 1.000 | 0.287 | 0.43 | 0.18 | 1.00 | 0.21 | 0.78 | 0.95 |
| acc_only_s311 | 1.000 | 0.247 | 0.29 | 0.21 | 1.00 | 0.19 | 0.88 | 0.90 |
| mass_only_s300 | 1.000 | 0.411 | 0.56 | 0.32 | 1.00 | 0.53 | 0.38 | 0.76 |
| mass_only_s301 | 0.598 | 0.518 | 0.47 | 0.51 | 0.61 | 0.48 | 0.14 | 0.45 |
| mass_only_s302 | 1.000 | 0.354 | 0.19 | 0.57 | 1.00 | 0.18 | 0.91 | 0.79 |
| mass_only_s303 | 0.589 | 0.533 | 0.49 | 0.49 | 0.59 | 0.48 | 0.96 | 0.43 |
| mass_only_s304 | 1.000 | 0.246 | 0.09 | 0.46 | 1.00 | 0.09 | 0.63 | 0.66 |
| mass_only_s305 | 1.000 | 0.713 | 0.47 | 1.00 | 1.00 | 0.36 | 0.39 | 0.57 |
| mass_only_s306 | 0.656 | 0.343 | 0.30 | 0.40 | 0.65 | 0.29 | 0.98 | 0.80 |
| mass_only_s307 | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.99 | 0.31 | 0.55 |
| mass_only_s308 | 0.629 | 0.453 | 0.41 | 0.38 | 0.65 | 0.44 | 0.93 | 0.49 |
| mass_only_s309 | 0.600 | 0.516 | 0.47 | 0.48 | 0.59 | 0.50 | 0.17 | 0.46 |
| mass_only_s310 | 1.000 | 0.265 | 0.09 | 0.50 | 1.00 | 0.09 | 0.73 | 0.83 |
| mass_only_s311 | 1.000 | 0.997 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.32 |
| neither_s300 | 1.000 | 0.847 | 0.88 | 0.90 | — | — | 0.25 | 0.27 |
| neither_s301 | 1.000 | 0.975 | 0.98 | 0.99 | — | — | 0.41 | 0.74 |
| neither_s302 | 1.000 | 0.846 | 0.92 | 0.88 | — | — | 0.59 | 0.48 |
| neither_s303 | 1.000 | 0.942 | 0.97 | 0.97 | — | — | 0.54 | 0.79 |
| neither_s304 | 0.722 | 0.479 | 0.41 | 0.46 | — | — | 0.24 | 0.52 |
| neither_s305 | 1.000 | 0.809 | 0.85 | 0.86 | — | — | 0.59 | 0.43 |
| neither_s306 | 1.000 | 0.779 | 0.80 | 0.80 | — | — | 0.32 | 0.53 |
| neither_s307 | 1.000 | 0.941 | 0.96 | 0.96 | — | — | 0.54 | 0.64 |
| neither_s308 | 1.000 | 0.820 | 0.87 | 0.85 | — | — | 0.43 | 0.33 |
| neither_s309 | 1.000 | 0.837 | 0.86 | 0.97 | — | — | 0.54 | 0.79 |
| neither_s310 | 1.000 | 0.890 | 0.92 | 0.93 | — | — | 0.52 | 0.38 |
| neither_s311 | 1.000 | 0.939 | 0.96 | 0.97 | — | — | 0.42 | 0.37 |

## Scientific review (supervisor)

**Measured state.** mass_only: 7/12 fit (ID ≥ 0.95), mean ID 0.84, mean OOD 0.46, block IIA 0.84, unseen-source IIA 0.45; acc_only: 9/12 fit (ID ≥ 0.95), mean ID 0.92, mean OOD 0.53, block IIA 0.92, unseen-source IIA 0.54; neither: 11/12 fit (ID ≥ 0.95), mean ID 0.98, mean OOD 0.84.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 1.00 / 0.99 / —; 311: 1.00 / 1.00 / 1.00 / —; 305: 1.00 / 0.47 / 0.36 / —; 304: 1.00 / 0.09 / 0.09 / —; 310: 1.00 / 0.09 / 0.09 / —.
**Events since previous cohort:** 1 — mass_only_s309: block LOST unseen inputs (IIA-OOD < 0.5).
**Interpretation.** Stable phase. Focus networks unchanged. 305 continues its slow erosion (OOD-mass about 0.46-0.48), consistent with drift of unanchored unseen pairings.
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): HELD (changes in unseen-mass accuracy: {307: 0.0, 311: 0.0, 305: -0.02, 304: 0.0, 310: 0.0}).
**Prediction for step 5,300:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

