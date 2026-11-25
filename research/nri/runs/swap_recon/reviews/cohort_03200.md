# Cohort review — step 3,200

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.817 | 0.547 | 6 | 0.819 | 0.482 |
| acc_only | 12 | 0.783 | 0.510 | 6 | 0.777 | 0.524 |
| neither | 12 | 0.937 | 0.815 | 10 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.98 | 0.32 | 0.56 |
| mass_only_s311 | focus | 0.999 | 0.990 | 0.99 | 1.00 | 1.00 | 0.99 | 0.25 | 0.32 |
| mass_only_s304 | comparator | 1.000 | 0.244 | 0.09 | 0.45 | 1.00 | 0.09 | 0.63 | 0.66 |
| mass_only_s310 | comparator | 1.000 | 0.258 | 0.09 | 0.48 | 1.00 | 0.09 | 0.73 | 0.83 |
| mass_only_s305 | intermediate | 1.000 | 0.750 | 0.54 | 1.00 | 1.00 | 0.37 | 0.39 | 0.57 |

## Events since step 3,100

- **mass_only_s300**: block carries unseen inputs (IIA-OOD >= 0.5)

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.335 | 0.34 | 0.42 | 1.00 | 0.43 | 0.59 | 0.92 |
| acc_only_s301 | 0.758 | 0.350 | 0.38 | 0.31 | 0.74 | 0.30 | 0.75 | 0.64 |
| acc_only_s302 | 0.591 | 0.531 | 0.52 | 0.51 | 0.57 | 0.52 | 0.62 | 0.39 |
| acc_only_s303 | 0.607 | 0.524 | 0.46 | 0.49 | 0.59 | 0.51 | 0.36 | 0.59 |
| acc_only_s304 | 0.245 | 0.142 | 0.15 | 0.15 | 0.24 | 0.16 | 0.83 | 0.93 |
| acc_only_s305 | 1.000 | 0.820 | 0.90 | 0.82 | 1.00 | 0.83 | 0.55 | 0.56 |
| acc_only_s306 | 1.000 | 0.489 | 0.47 | 0.55 | 1.00 | 0.55 | 0.78 | 0.94 |
| acc_only_s307 | 1.000 | 0.769 | 0.61 | 1.00 | 1.00 | 0.99 | 0.41 | 0.57 |
| acc_only_s308 | 1.000 | 0.918 | 0.99 | 0.89 | 1.00 | 0.93 | 0.37 | 0.25 |
| acc_only_s309 | 0.592 | 0.537 | 0.50 | 0.51 | 0.58 | 0.51 | 0.57 | 0.45 |
| acc_only_s310 | 1.000 | 0.284 | 0.43 | 0.17 | 1.00 | 0.19 | 0.78 | 0.95 |
| acc_only_s311 | 0.602 | 0.425 | 0.46 | 0.35 | 0.60 | 0.36 | 0.90 | 0.85 |
| mass_only_s300 | 1.000 | 0.394 | 0.53 | 0.31 | 0.99 | 0.51 | 0.38 | 0.76 |
| mass_only_s301 | 0.578 | 0.526 | 0.52 | 0.51 | 0.59 | 0.53 | 0.15 | 0.46 |
| mass_only_s302 | 0.878 | 0.341 | 0.22 | 0.49 | 0.88 | 0.22 | 0.95 | 0.78 |
| mass_only_s303 | 0.578 | 0.552 | 0.52 | 0.52 | 0.58 | 0.51 | 0.95 | 0.44 |
| mass_only_s304 | 1.000 | 0.244 | 0.09 | 0.45 | 1.00 | 0.09 | 0.63 | 0.66 |
| mass_only_s305 | 1.000 | 0.750 | 0.54 | 1.00 | 1.00 | 0.37 | 0.39 | 0.57 |
| mass_only_s306 | 0.597 | 0.462 | 0.47 | 0.45 | 0.61 | 0.48 | 0.98 | 0.81 |
| mass_only_s307 | 1.000 | 0.998 | 1.00 | 1.00 | 1.00 | 0.98 | 0.32 | 0.56 |
| mass_only_s308 | 0.578 | 0.528 | 0.51 | 0.49 | 0.60 | 0.49 | 0.93 | 0.49 |
| mass_only_s309 | 0.595 | 0.522 | 0.50 | 0.50 | 0.59 | 0.53 | 0.17 | 0.46 |
| mass_only_s310 | 1.000 | 0.258 | 0.09 | 0.48 | 1.00 | 0.09 | 0.73 | 0.83 |
| mass_only_s311 | 0.999 | 0.990 | 0.99 | 1.00 | 1.00 | 0.99 | 0.25 | 0.32 |
| neither_s300 | 1.000 | 0.853 | 0.88 | 0.91 | — | — | 0.25 | 0.28 |
| neither_s301 | 1.000 | 0.971 | 0.98 | 1.00 | — | — | 0.42 | 0.75 |
| neither_s302 | 1.000 | 0.820 | 0.91 | 0.85 | — | — | 0.57 | 0.47 |
| neither_s303 | 1.000 | 0.936 | 0.96 | 0.96 | — | — | 0.54 | 0.79 |
| neither_s304 | 0.613 | 0.509 | 0.46 | 0.47 | — | — | 0.27 | 0.57 |
| neither_s305 | 1.000 | 0.807 | 0.85 | 0.86 | — | — | 0.59 | 0.44 |
| neither_s306 | 0.626 | 0.490 | 0.42 | 0.47 | — | — | 0.46 | 0.53 |
| neither_s307 | 1.000 | 0.941 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.817 | 0.86 | 0.84 | — | — | 0.43 | 0.34 |
| neither_s309 | 1.000 | 0.829 | 0.86 | 0.96 | — | — | 0.54 | 0.80 |
| neither_s310 | 1.000 | 0.881 | 0.91 | 0.93 | — | — | 0.52 | 0.39 |
| neither_s311 | 1.000 | 0.933 | 0.94 | 0.96 | — | — | 0.42 | 0.38 |

## Scientific review (supervisor)

**Measured state.** mass_only: 6/12 fit (ID ≥ 0.95), mean ID 0.82, mean OOD 0.50, block IIA 0.82, unseen-source IIA 0.48; acc_only: 6/12 fit (ID ≥ 0.95), mean ID 0.78, mean OOD 0.51, block IIA 0.78, unseen-source IIA 0.52; neither: 10/12 fit (ID ≥ 0.95), mean ID 0.94, mean OOD 0.82.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 1.00 / 0.98 / —; 311: 1.00 / 0.99 / 0.99 / —; 305: 1.00 / 0.54 / 0.37 / —; 304: 1.00 / 0.09 / 0.09 / —; 310: 1.00 / 0.09 / 0.09 / —.
**Events since previous cohort:** 1 — mass_only_s300: block carries unseen inputs (IIA-OOD >= 0.5).
**Interpretation.** Stable phase. No focus network changes by more than the noise of the audit panel. The remaining population changes are late settlements in the acceleration group (see events).
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): HELD (changes in unseen-mass accuracy: {307: 0.0, 311: -0.0, 305: 0.05, 304: 0.0, 310: -0.0}).
**Prediction for step 3,300:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

