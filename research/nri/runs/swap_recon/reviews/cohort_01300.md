# Cohort review — step 1,300

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.675 | 0.490 | 3 | 0.615 | 0.435 |
| acc_only | 12 | 0.600 | 0.439 | 2 | 0.580 | 0.457 |
| neither | 12 | 0.849 | 0.710 | 7 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.852 | 0.714 | 0.66 | 0.81 | 0.80 | 0.74 | 0.34 | 0.59 |
| mass_only_s311 | focus | 0.978 | 0.901 | 0.92 | 0.93 | 0.90 | 0.82 | 0.29 | 0.30 |
| mass_only_s304 | comparator | 0.991 | 0.285 | 0.14 | 0.47 | 0.99 | 0.14 | 0.62 | 0.69 |
| mass_only_s310 | comparator | 0.311 | 0.161 | 0.16 | 0.18 | 0.24 | 0.15 | 0.85 | 0.73 |
| mass_only_s305 | intermediate | 0.527 | 0.431 | 0.44 | 0.46 | 0.44 | 0.26 | 0.45 | 0.68 |

## Events since step 1,200

- **acc_only_s300**: OOD jump 0.42->0.26
- **acc_only_s301**: block carries unseen inputs (IIA-OOD >= 0.5)
- **mass_only_s300**: ID >= 0.95
- **mass_only_s302**: block carries unseen inputs (IIA-OOD >= 0.5)
- **mass_only_s307**: ID fell below 0.95; OOD jump 0.94->0.71
- **mass_only_s311**: ID >= 0.95; block carries unseen inputs (IIA-OOD >= 0.5); OOD jump 0.59->0.90
- **neither_s300**: ID >= 0.95; OOD jump 0.67->0.85
- **neither_s305**: ID >= 0.95
- **neither_s309**: OOD jump 0.44->0.65
- **neither_s311**: ID >= 0.95; OOD jump 0.61->0.85

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 0.741 | 0.260 | 0.29 | 0.27 | 0.68 | 0.28 | 0.52 | 0.81 |
| acc_only_s301 | 0.585 | 0.512 | 0.49 | 0.51 | 0.58 | 0.51 | 0.80 | 0.61 |
| acc_only_s302 | 0.269 | 0.168 | 0.18 | 0.18 | 0.19 | 0.14 | 0.77 | 0.88 |
| acc_only_s303 | 0.588 | 0.532 | 0.52 | 0.51 | 0.59 | 0.52 | 0.41 | 0.59 |
| acc_only_s304 | 0.239 | 0.155 | 0.16 | 0.17 | 0.23 | 0.16 | 0.93 | 0.94 |
| acc_only_s305 | 0.435 | 0.293 | 0.14 | 0.49 | 0.41 | 0.49 | 0.52 | 0.68 |
| acc_only_s306 | 0.579 | 0.503 | 0.54 | 0.51 | 0.58 | 0.50 | 0.89 | 0.94 |
| acc_only_s307 | 0.999 | 0.661 | 0.47 | 0.93 | 0.98 | 0.91 | 0.43 | 0.58 |
| acc_only_s308 | 0.989 | 0.880 | 0.94 | 0.86 | 0.96 | 0.81 | 0.39 | 0.30 |
| acc_only_s309 | 0.579 | 0.531 | 0.53 | 0.54 | 0.58 | 0.52 | 0.56 | 0.57 |
| acc_only_s310 | 0.616 | 0.292 | 0.39 | 0.23 | 0.59 | 0.24 | 0.78 | 0.88 |
| acc_only_s311 | 0.579 | 0.479 | 0.54 | 0.44 | 0.59 | 0.41 | 0.88 | 0.78 |
| mass_only_s300 | 0.964 | 0.356 | 0.38 | 0.37 | 0.89 | 0.37 | 0.40 | 0.80 |
| mass_only_s301 | 0.583 | 0.501 | 0.51 | 0.50 | 0.60 | 0.52 | 0.85 | 0.53 |
| mass_only_s302 | 0.576 | 0.472 | 0.53 | 0.43 | 0.58 | 0.52 | 0.94 | 0.74 |
| mass_only_s303 | 0.601 | 0.519 | 0.51 | 0.50 | 0.59 | 0.52 | 0.79 | 0.77 |
| mass_only_s304 | 0.991 | 0.285 | 0.14 | 0.47 | 0.99 | 0.14 | 0.62 | 0.69 |
| mass_only_s305 | 0.527 | 0.431 | 0.44 | 0.46 | 0.44 | 0.26 | 0.45 | 0.68 |
| mass_only_s306 | 0.558 | 0.473 | 0.48 | 0.44 | 0.54 | 0.48 | 0.94 | 0.79 |
| mass_only_s307 | 0.852 | 0.714 | 0.66 | 0.81 | 0.80 | 0.74 | 0.34 | 0.59 |
| mass_only_s308 | 0.575 | 0.529 | 0.53 | 0.50 | 0.59 | 0.52 | 0.91 | 0.49 |
| mass_only_s309 | 0.585 | 0.531 | 0.53 | 0.51 | 0.25 | 0.17 | 0.39 | 0.51 |
| mass_only_s310 | 0.311 | 0.161 | 0.16 | 0.18 | 0.24 | 0.15 | 0.85 | 0.73 |
| mass_only_s311 | 0.978 | 0.901 | 0.92 | 0.93 | 0.90 | 0.82 | 0.29 | 0.30 |
| neither_s300 | 0.998 | 0.851 | 0.87 | 0.90 | — | — | 0.29 | 0.31 |
| neither_s301 | 0.979 | 0.889 | 0.93 | 0.91 | — | — | 0.58 | 0.78 |
| neither_s302 | 0.585 | 0.460 | 0.46 | 0.44 | — | — | 0.45 | 0.59 |
| neither_s303 | 0.591 | 0.489 | 0.48 | 0.51 | — | — | 0.59 | 0.73 |
| neither_s304 | 0.575 | 0.491 | 0.48 | 0.47 | — | — | 0.29 | 0.55 |
| neither_s305 | 0.999 | 0.776 | 0.81 | 0.82 | — | — | 0.61 | 0.45 |
| neither_s306 | 0.591 | 0.510 | 0.46 | 0.48 | — | — | 0.43 | 0.56 |
| neither_s307 | 1.000 | 0.928 | 0.96 | 0.96 | — | — | 0.52 | 0.63 |
| neither_s308 | 1.000 | 0.795 | 0.84 | 0.81 | — | — | 0.42 | 0.33 |
| neither_s309 | 0.912 | 0.654 | 0.68 | 0.76 | — | — | 0.53 | 0.76 |
| neither_s310 | 0.995 | 0.837 | 0.89 | 0.91 | — | — | 0.56 | 0.43 |
| neither_s311 | 0.966 | 0.846 | 0.83 | 0.87 | — | — | 0.35 | 0.33 |

## Scientific review (supervisor)

**Measured state.** mass_only: 3/12 fit (ID ≥ 0.95), mean ID 0.67, mean OOD 0.48, block IIA 0.62, unseen-source IIA 0.43, factorisation R²_add 0.67; acc_only: 2/12 fit (ID ≥ 0.95), mean ID 0.60, mean OOD 0.47, block IIA 0.58, unseen-source IIA 0.46, factorisation R²_add 0.80; neither: 7/12 fit (ID ≥ 0.95), mean ID 0.85, mean OOD 0.71.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 0.85 / 0.66 / 0.74 / 1.00; 311: 0.98 / 0.92 / 0.82 / 0.98; 305: 0.53 / 0.44 / 0.26 / 0.98; 304: 0.99 / 0.14 / 0.14 / 0.71; 310: 0.31 / 0.16 / 0.15 / 0.99.
**Events since previous cohort:** 10 — acc_only_s300: OOD jump 0.42->0.26; acc_only_s301: block carries unseen inputs (IIA-OOD >= 0.5); mass_only_s300: ID >= 0.95; mass_only_s302: block carries unseen inputs (IIA-OOD >= 0.5); mass_only_s307: ID fell below 0.95, OOD jump 0.94->0.71; mass_only_s311: ID >= 0.95, block carries unseen inputs (IIA-OOD >= 0.5), OOD jump 0.59->0.90; neither_s300: ID >= 0.95, OOD jump 0.67->0.85; neither_s305: ID >= 0.95; neither_s309: OOD jump 0.44->0.65; neither_s311: ID >= 0.95, OOD jump 0.61->0.85.
**Interpretation.** Generalisers settle. 307 (ID 0.97, OOD-mass 0.92) and 311 (settles at 1,300) reach near-perfect unseen-mass accuracy at the same time as they fit the training data. No memorise-then-generalise delay is seen. 305 escapes late with partial OOD and mild factorisation loss (0.96).
**Anomaly investigated.** Anomaly check: 307's ID drops to 0.85 and OOD-mass to 0.66 at step 1,300, then recovers by 1,400. A transient instability during consolidation (task-loss spike in the per-step log at about 1,300-1,400). Not a collapse.
**Previous prediction** ('304 and 310 stay ≤ 0.25 on unseen masses at the next cohort, whatever their ID (failures never rise on unseen masses)'): HELD (304/310 unseen-mass: [0.14, 0.16]).
**Prediction for step 1,400:** 304 and 310 stay ≤ 0.25 on unseen masses at the next cohort, whatever their ID (failures never rise on unseen masses).

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

