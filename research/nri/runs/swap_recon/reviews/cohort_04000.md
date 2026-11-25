# Cohort review — step 4,000

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.834 | 0.541 | 7 | 0.828 | 0.473 |
| acc_only | 12 | 0.863 | 0.480 | 7 | 0.852 | 0.527 |
| neither | 12 | 0.968 | 0.842 | 11 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.99 | 0.32 | 0.56 |
| mass_only_s311 | focus | 1.000 | 0.997 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.32 |
| mass_only_s304 | comparator | 1.000 | 0.247 | 0.09 | 0.46 | 1.00 | 0.09 | 0.63 | 0.66 |
| mass_only_s310 | comparator | 1.000 | 0.264 | 0.09 | 0.49 | 1.00 | 0.09 | 0.73 | 0.83 |
| mass_only_s305 | intermediate | 1.000 | 0.720 | 0.48 | 1.00 | 1.00 | 0.37 | 0.39 | 0.58 |

## Events since step 3,900

- **mass_only_s303**: block LOST unseen inputs (IIA-OOD < 0.5)

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.337 | 0.34 | 0.42 | 1.00 | 0.43 | 0.59 | 0.92 |
| acc_only_s301 | 1.000 | 0.246 | 0.30 | 0.19 | 1.00 | 0.19 | 0.66 | 0.66 |
| acc_only_s302 | 0.611 | 0.529 | 0.51 | 0.51 | 0.58 | 0.52 | 0.62 | 0.39 |
| acc_only_s303 | 0.602 | 0.523 | 0.45 | 0.49 | 0.59 | 0.50 | 0.36 | 0.59 |
| acc_only_s304 | 0.385 | 0.150 | 0.13 | 0.18 | 0.34 | 0.18 | 0.54 | 0.91 |
| acc_only_s305 | 1.000 | 0.825 | 0.91 | 0.82 | 1.00 | 0.83 | 0.55 | 0.56 |
| acc_only_s306 | 1.000 | 0.480 | 0.46 | 0.54 | 1.00 | 0.54 | 0.78 | 0.94 |
| acc_only_s307 | 1.000 | 0.774 | 0.61 | 1.00 | 1.00 | 0.99 | 0.41 | 0.57 |
| acc_only_s308 | 1.000 | 0.919 | 0.99 | 0.89 | 1.00 | 0.94 | 0.36 | 0.25 |
| acc_only_s309 | 0.916 | 0.478 | 0.21 | 0.83 | 0.88 | 0.80 | 0.55 | 0.46 |
| acc_only_s310 | 1.000 | 0.285 | 0.43 | 0.18 | 1.00 | 0.20 | 0.78 | 0.95 |
| acc_only_s311 | 0.846 | 0.219 | 0.22 | 0.23 | 0.83 | 0.21 | 0.89 | 0.89 |
| mass_only_s300 | 1.000 | 0.407 | 0.55 | 0.31 | 1.00 | 0.53 | 0.38 | 0.76 |
| mass_only_s301 | 0.599 | 0.520 | 0.50 | 0.51 | 0.59 | 0.51 | 0.15 | 0.46 |
| mass_only_s302 | 1.000 | 0.343 | 0.17 | 0.56 | 1.00 | 0.16 | 0.91 | 0.79 |
| mass_only_s303 | 0.604 | 0.536 | 0.51 | 0.50 | 0.59 | 0.50 | 0.95 | 0.43 |
| mass_only_s304 | 1.000 | 0.247 | 0.09 | 0.46 | 1.00 | 0.09 | 0.63 | 0.66 |
| mass_only_s305 | 1.000 | 0.720 | 0.48 | 1.00 | 1.00 | 0.37 | 0.39 | 0.58 |
| mass_only_s306 | 0.598 | 0.431 | 0.44 | 0.39 | 0.59 | 0.44 | 0.98 | 0.80 |
| mass_only_s307 | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.99 | 0.32 | 0.56 |
| mass_only_s308 | 0.613 | 0.503 | 0.46 | 0.47 | 0.59 | 0.46 | 0.93 | 0.49 |
| mass_only_s309 | 0.595 | 0.528 | 0.50 | 0.51 | 0.58 | 0.53 | 0.17 | 0.46 |
| mass_only_s310 | 1.000 | 0.264 | 0.09 | 0.49 | 1.00 | 0.09 | 0.73 | 0.83 |
| mass_only_s311 | 1.000 | 0.997 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.32 |
| neither_s300 | 1.000 | 0.853 | 0.88 | 0.91 | — | — | 0.25 | 0.27 |
| neither_s301 | 1.000 | 0.975 | 0.98 | 1.00 | — | — | 0.41 | 0.74 |
| neither_s302 | 1.000 | 0.844 | 0.92 | 0.88 | — | — | 0.58 | 0.48 |
| neither_s303 | 1.000 | 0.940 | 0.97 | 0.96 | — | — | 0.54 | 0.79 |
| neither_s304 | 0.628 | 0.496 | 0.45 | 0.46 | — | — | 0.26 | 0.55 |
| neither_s305 | 1.000 | 0.807 | 0.85 | 0.86 | — | — | 0.59 | 0.44 |
| neither_s306 | 0.983 | 0.763 | 0.79 | 0.77 | — | — | 0.35 | 0.53 |
| neither_s307 | 1.000 | 0.940 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.819 | 0.87 | 0.85 | — | — | 0.43 | 0.34 |
| neither_s309 | 1.000 | 0.835 | 0.86 | 0.97 | — | — | 0.54 | 0.80 |
| neither_s310 | 1.000 | 0.888 | 0.92 | 0.93 | — | — | 0.52 | 0.38 |
| neither_s311 | 1.000 | 0.939 | 0.96 | 0.97 | — | — | 0.42 | 0.37 |

## Scientific review (supervisor)

**Measured state.** mass_only: 7/12 fit (ID ≥ 0.95), mean ID 0.83, mean OOD 0.48, block IIA 0.83, unseen-source IIA 0.47, factorisation R²_add 0.68; acc_only: 7/12 fit (ID ≥ 0.95), mean ID 0.86, mean OOD 0.52, block IIA 0.85, unseen-source IIA 0.53, factorisation R²_add 0.78; neither: 11/12 fit (ID ≥ 0.95), mean ID 0.97, mean OOD 0.84.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 1.00 / 0.99 / 1.00; 311: 1.00 / 1.00 / 1.00 / 0.98; 305: 1.00 / 0.48 / 0.37 / 0.96; 304: 1.00 / 0.09 / 0.09 / 0.71; 310: 1.00 / 0.09 / 0.09 / 0.69.
**Events since previous cohort:** 1 — mass_only_s303: block LOST unseen inputs (IIA-OOD < 0.5).
**Interpretation.** Stable phase. No focus network changes by more than the noise of the audit panel. The remaining population changes are late settlements in the acceleration group (see events).
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): HELD (changes in unseen-mass accuracy: {307: 0.0, 311: 0.0, 305: -0.03, 304: 0.0, 310: -0.0}).
**Prediction for step 4,100:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

