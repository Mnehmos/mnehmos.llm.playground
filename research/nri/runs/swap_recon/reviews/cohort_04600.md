# Cohort review — step 4,600

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.836 | 0.538 | 7 | 0.835 | 0.467 |
| acc_only | 12 | 0.908 | 0.494 | 9 | 0.901 | 0.539 |
| neither | 12 | 0.973 | 0.842 | 11 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.99 | 0.31 | 0.55 |
| mass_only_s311 | focus | 1.000 | 0.996 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.32 |
| mass_only_s304 | comparator | 1.000 | 0.251 | 0.09 | 0.47 | 1.00 | 0.09 | 0.63 | 0.66 |
| mass_only_s310 | comparator | 1.000 | 0.264 | 0.10 | 0.49 | 1.00 | 0.09 | 0.73 | 0.83 |
| mass_only_s305 | intermediate | 1.000 | 0.710 | 0.47 | 1.00 | 1.00 | 0.36 | 0.39 | 0.57 |

## Events since step 4,500

- none

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.334 | 0.34 | 0.42 | 1.00 | 0.43 | 0.59 | 0.92 |
| acc_only_s301 | 1.000 | 0.233 | 0.29 | 0.18 | 1.00 | 0.17 | 0.66 | 0.66 |
| acc_only_s302 | 0.588 | 0.529 | 0.49 | 0.49 | 0.58 | 0.49 | 0.61 | 0.39 |
| acc_only_s303 | 0.621 | 0.470 | 0.38 | 0.45 | 0.62 | 0.46 | 0.36 | 0.61 |
| acc_only_s304 | 0.692 | 0.186 | 0.17 | 0.24 | 0.61 | 0.23 | 0.63 | 0.90 |
| acc_only_s305 | 1.000 | 0.825 | 0.91 | 0.82 | 1.00 | 0.84 | 0.55 | 0.56 |
| acc_only_s306 | 1.000 | 0.483 | 0.47 | 0.54 | 1.00 | 0.54 | 0.78 | 0.94 |
| acc_only_s307 | 1.000 | 0.771 | 0.62 | 1.00 | 1.00 | 1.00 | 0.41 | 0.56 |
| acc_only_s308 | 1.000 | 0.924 | 0.99 | 0.90 | 1.00 | 0.94 | 0.36 | 0.24 |
| acc_only_s309 | 1.000 | 0.643 | 0.37 | 0.98 | 1.00 | 0.97 | 0.52 | 0.45 |
| acc_only_s310 | 1.000 | 0.288 | 0.43 | 0.18 | 1.00 | 0.21 | 0.78 | 0.95 |
| acc_only_s311 | 1.000 | 0.241 | 0.28 | 0.21 | 1.00 | 0.19 | 0.88 | 0.90 |
| mass_only_s300 | 1.000 | 0.418 | 0.58 | 0.32 | 1.00 | 0.54 | 0.38 | 0.76 |
| mass_only_s301 | 0.590 | 0.531 | 0.49 | 0.51 | 0.59 | 0.52 | 0.14 | 0.45 |
| mass_only_s302 | 1.000 | 0.349 | 0.18 | 0.56 | 1.00 | 0.17 | 0.91 | 0.79 |
| mass_only_s303 | 0.600 | 0.522 | 0.50 | 0.52 | 0.59 | 0.51 | 0.95 | 0.43 |
| mass_only_s304 | 1.000 | 0.251 | 0.09 | 0.47 | 1.00 | 0.09 | 0.63 | 0.66 |
| mass_only_s305 | 1.000 | 0.710 | 0.47 | 1.00 | 1.00 | 0.36 | 0.39 | 0.57 |
| mass_only_s306 | 0.628 | 0.394 | 0.33 | 0.40 | 0.63 | 0.35 | 0.98 | 0.80 |
| mass_only_s307 | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.99 | 0.31 | 0.55 |
| mass_only_s308 | 0.614 | 0.489 | 0.45 | 0.43 | 0.61 | 0.45 | 0.93 | 0.49 |
| mass_only_s309 | 0.602 | 0.530 | 0.50 | 0.50 | 0.60 | 0.52 | 0.17 | 0.46 |
| mass_only_s310 | 1.000 | 0.264 | 0.10 | 0.49 | 1.00 | 0.09 | 0.73 | 0.83 |
| mass_only_s311 | 1.000 | 0.996 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.32 |
| neither_s300 | 1.000 | 0.854 | 0.88 | 0.91 | — | — | 0.25 | 0.27 |
| neither_s301 | 1.000 | 0.973 | 0.98 | 1.00 | — | — | 0.41 | 0.74 |
| neither_s302 | 1.000 | 0.844 | 0.92 | 0.88 | — | — | 0.59 | 0.48 |
| neither_s303 | 1.000 | 0.942 | 0.97 | 0.97 | — | — | 0.54 | 0.79 |
| neither_s304 | 0.676 | 0.483 | 0.40 | 0.45 | — | — | 0.26 | 0.55 |
| neither_s305 | 1.000 | 0.809 | 0.85 | 0.86 | — | — | 0.59 | 0.43 |
| neither_s306 | 1.000 | 0.775 | 0.79 | 0.80 | — | — | 0.32 | 0.53 |
| neither_s307 | 1.000 | 0.940 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.817 | 0.86 | 0.85 | — | — | 0.43 | 0.33 |
| neither_s309 | 1.000 | 0.836 | 0.86 | 0.97 | — | — | 0.54 | 0.79 |
| neither_s310 | 1.000 | 0.885 | 0.91 | 0.93 | — | — | 0.52 | 0.38 |
| neither_s311 | 1.000 | 0.939 | 0.96 | 0.96 | — | — | 0.42 | 0.37 |

## Scientific review (supervisor)

**Measured state.** mass_only: 7/12 fit (ID ≥ 0.95), mean ID 0.84, mean OOD 0.47, block IIA 0.83, unseen-source IIA 0.47; acc_only: 9/12 fit (ID ≥ 0.95), mean ID 0.91, mean OOD 0.53, block IIA 0.90, unseen-source IIA 0.54; neither: 11/12 fit (ID ≥ 0.95), mean ID 0.97, mean OOD 0.84.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 1.00 / 0.99 / —; 311: 1.00 / 1.00 / 1.00 / —; 305: 1.00 / 0.47 / 0.36 / —; 304: 1.00 / 0.09 / 0.09 / —; 310: 1.00 / 0.10 / 0.09 / —.
**Events since previous cohort:** 0.
**Interpretation.** Stable phase. Focus networks unchanged. 305 continues its slow erosion (OOD-mass about 0.46-0.48), consistent with drift of unanchored unseen pairings.
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): HELD (changes in unseen-mass accuracy: {307: 0.01, 311: 0.0, 305: -0.01, 304: 0.0, 310: 0.0}).
**Prediction for step 4,700:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

