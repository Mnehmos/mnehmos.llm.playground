# Cohort review — step 5,500

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.841 | 0.527 | 7 | 0.841 | 0.450 |
| acc_only | 12 | 0.928 | 0.501 | 9 | 0.919 | 0.543 |
| neither | 12 | 0.978 | 0.842 | 11 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.99 | 0.31 | 0.55 |
| mass_only_s311 | focus | 1.000 | 0.997 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.32 |
| mass_only_s304 | comparator | 1.000 | 0.247 | 0.09 | 0.46 | 1.00 | 0.09 | 0.63 | 0.66 |
| mass_only_s310 | comparator | 1.000 | 0.263 | 0.09 | 0.50 | 1.00 | 0.09 | 0.73 | 0.83 |
| mass_only_s305 | intermediate | 1.000 | 0.709 | 0.46 | 1.00 | 1.00 | 0.36 | 0.39 | 0.57 |

## Events since step 5,400

- **mass_only_s309**: block carries unseen inputs (IIA-OOD >= 0.5)

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.342 | 0.35 | 0.42 | 1.00 | 0.43 | 0.59 | 0.92 |
| acc_only_s301 | 1.000 | 0.233 | 0.29 | 0.18 | 1.00 | 0.18 | 0.66 | 0.66 |
| acc_only_s302 | 0.595 | 0.529 | 0.47 | 0.47 | 0.58 | 0.50 | 0.61 | 0.39 |
| acc_only_s303 | 0.672 | 0.432 | 0.35 | 0.41 | 0.66 | 0.41 | 0.36 | 0.62 |
| acc_only_s304 | 0.874 | 0.270 | 0.27 | 0.31 | 0.79 | 0.30 | 0.60 | 0.89 |
| acc_only_s305 | 1.000 | 0.825 | 0.91 | 0.82 | 1.00 | 0.84 | 0.55 | 0.56 |
| acc_only_s306 | 1.000 | 0.478 | 0.46 | 0.54 | 1.00 | 0.54 | 0.78 | 0.94 |
| acc_only_s307 | 1.000 | 0.771 | 0.62 | 1.00 | 1.00 | 1.00 | 0.41 | 0.56 |
| acc_only_s308 | 1.000 | 0.923 | 0.99 | 0.89 | 1.00 | 0.94 | 0.36 | 0.24 |
| acc_only_s309 | 1.000 | 0.677 | 0.44 | 0.98 | 1.00 | 0.98 | 0.52 | 0.45 |
| acc_only_s310 | 1.000 | 0.288 | 0.43 | 0.18 | 1.00 | 0.21 | 0.78 | 0.95 |
| acc_only_s311 | 1.000 | 0.246 | 0.29 | 0.21 | 1.00 | 0.20 | 0.88 | 0.91 |
| mass_only_s300 | 1.000 | 0.414 | 0.57 | 0.32 | 1.00 | 0.54 | 0.38 | 0.76 |
| mass_only_s301 | 0.599 | 0.519 | 0.46 | 0.51 | 0.61 | 0.48 | 0.14 | 0.45 |
| mass_only_s302 | 1.000 | 0.354 | 0.19 | 0.57 | 1.00 | 0.17 | 0.91 | 0.79 |
| mass_only_s303 | 0.593 | 0.517 | 0.46 | 0.48 | 0.59 | 0.48 | 0.96 | 0.43 |
| mass_only_s304 | 1.000 | 0.247 | 0.09 | 0.46 | 1.00 | 0.09 | 0.63 | 0.66 |
| mass_only_s305 | 1.000 | 0.709 | 0.46 | 1.00 | 1.00 | 0.36 | 0.39 | 0.57 |
| mass_only_s306 | 0.658 | 0.350 | 0.28 | 0.41 | 0.66 | 0.27 | 0.98 | 0.80 |
| mass_only_s307 | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.99 | 0.31 | 0.55 |
| mass_only_s308 | 0.642 | 0.442 | 0.40 | 0.38 | 0.65 | 0.44 | 0.93 | 0.49 |
| mass_only_s309 | 0.602 | 0.517 | 0.47 | 0.47 | 0.59 | 0.50 | 0.17 | 0.46 |
| mass_only_s310 | 1.000 | 0.263 | 0.09 | 0.50 | 1.00 | 0.09 | 0.73 | 0.83 |
| mass_only_s311 | 1.000 | 0.997 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.32 |
| neither_s300 | 1.000 | 0.847 | 0.87 | 0.90 | — | — | 0.25 | 0.27 |
| neither_s301 | 1.000 | 0.973 | 0.98 | 1.00 | — | — | 0.41 | 0.74 |
| neither_s302 | 1.000 | 0.846 | 0.92 | 0.88 | — | — | 0.59 | 0.48 |
| neither_s303 | 1.000 | 0.942 | 0.97 | 0.97 | — | — | 0.54 | 0.79 |
| neither_s304 | 0.739 | 0.481 | 0.40 | 0.44 | — | — | 0.23 | 0.52 |
| neither_s305 | 1.000 | 0.809 | 0.85 | 0.86 | — | — | 0.59 | 0.43 |
| neither_s306 | 1.000 | 0.781 | 0.80 | 0.80 | — | — | 0.32 | 0.53 |
| neither_s307 | 1.000 | 0.941 | 0.96 | 0.96 | — | — | 0.54 | 0.64 |
| neither_s308 | 1.000 | 0.821 | 0.87 | 0.85 | — | — | 0.43 | 0.33 |
| neither_s309 | 1.000 | 0.837 | 0.86 | 0.97 | — | — | 0.54 | 0.79 |
| neither_s310 | 1.000 | 0.891 | 0.92 | 0.93 | — | — | 0.52 | 0.38 |
| neither_s311 | 1.000 | 0.940 | 0.96 | 0.96 | — | — | 0.42 | 0.37 |

## Scientific review (supervisor)

**Measured state.** mass_only: 7/12 fit (ID ≥ 0.95), mean ID 0.84, mean OOD 0.46, block IIA 0.84, unseen-source IIA 0.45; acc_only: 9/12 fit (ID ≥ 0.95), mean ID 0.93, mean OOD 0.53, block IIA 0.92, unseen-source IIA 0.54; neither: 11/12 fit (ID ≥ 0.95), mean ID 0.98, mean OOD 0.84.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 1.00 / 0.99 / —; 311: 1.00 / 1.00 / 1.00 / —; 305: 1.00 / 0.46 / 0.36 / —; 304: 1.00 / 0.09 / 0.09 / —; 310: 1.00 / 0.09 / 0.09 / —.
**Events since previous cohort:** 1 — mass_only_s309: block carries unseen inputs (IIA-OOD >= 0.5).
**Interpretation.** Stable phase. Focus networks unchanged. 305 continues its slow erosion (OOD-mass about 0.46-0.48), consistent with drift of unanchored unseen pairings.
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): HELD (changes in unseen-mass accuracy: {307: 0.0, 311: 0.0, 305: 0.0, 304: 0.0, 310: -0.0}).
**Prediction for step 5,600:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

