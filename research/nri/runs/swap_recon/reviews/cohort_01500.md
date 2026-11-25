# Cohort review — step 1,500

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.702 | 0.503 | 4 | 0.648 | 0.442 |
| acc_only | 12 | 0.706 | 0.480 | 5 | 0.693 | 0.470 |
| neither | 12 | 0.862 | 0.736 | 8 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.992 | 0.971 | 0.95 | 0.99 | 0.98 | 0.94 | 0.33 | 0.57 |
| mass_only_s311 | focus | 0.950 | 0.936 | 0.92 | 0.95 | 0.93 | 0.90 | 0.27 | 0.30 |
| mass_only_s304 | comparator | 0.988 | 0.253 | 0.09 | 0.49 | 0.98 | 0.09 | 0.62 | 0.70 |
| mass_only_s310 | comparator | 0.429 | 0.183 | 0.16 | 0.23 | 0.37 | 0.16 | 0.73 | 0.56 |
| mass_only_s305 | intermediate | 0.840 | 0.606 | 0.44 | 0.81 | 0.71 | 0.28 | 0.41 | 0.64 |

## Events since step 1,400

- **acc_only_s303**: OOD jump 0.34->0.53
- **acc_only_s309**: block carries unseen inputs (IIA-OOD >= 0.5)
- **acc_only_s310**: ID >= 0.95
- **mass_only_s300**: block carries unseen inputs (IIA-OOD >= 0.5)
- **mass_only_s303**: block carries unseen inputs (IIA-OOD >= 0.5)
- **neither_s301**: ID >= 0.95

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.376 | 0.41 | 0.42 | 1.00 | 0.42 | 0.54 | 0.88 |
| acc_only_s301 | 0.570 | 0.515 | 0.53 | 0.52 | 0.59 | 0.51 | 0.82 | 0.61 |
| acc_only_s302 | 0.380 | 0.295 | 0.33 | 0.25 | 0.26 | 0.18 | 0.79 | 0.72 |
| acc_only_s303 | 0.601 | 0.530 | 0.54 | 0.52 | 0.58 | 0.47 | 0.33 | 0.60 |
| acc_only_s304 | 0.242 | 0.152 | 0.16 | 0.16 | 0.23 | 0.16 | 0.92 | 0.94 |
| acc_only_s305 | 0.995 | 0.508 | 0.49 | 0.61 | 0.99 | 0.63 | 0.53 | 0.63 |
| acc_only_s306 | 0.587 | 0.522 | 0.51 | 0.50 | 0.59 | 0.50 | 0.90 | 0.94 |
| acc_only_s307 | 0.992 | 0.745 | 0.64 | 0.92 | 0.96 | 0.87 | 0.43 | 0.57 |
| acc_only_s308 | 0.990 | 0.847 | 0.91 | 0.84 | 0.97 | 0.82 | 0.38 | 0.27 |
| acc_only_s309 | 0.569 | 0.531 | 0.53 | 0.54 | 0.59 | 0.52 | 0.57 | 0.46 |
| acc_only_s310 | 0.970 | 0.265 | 0.48 | 0.13 | 0.96 | 0.15 | 0.82 | 0.96 |
| acc_only_s311 | 0.582 | 0.468 | 0.52 | 0.42 | 0.59 | 0.41 | 0.89 | 0.78 |
| mass_only_s300 | 0.973 | 0.356 | 0.53 | 0.23 | 0.92 | 0.51 | 0.39 | 0.79 |
| mass_only_s301 | 0.268 | 0.156 | 0.19 | 0.17 | 0.24 | 0.17 | 0.87 | 0.89 |
| mass_only_s302 | 0.593 | 0.486 | 0.50 | 0.45 | 0.57 | 0.52 | 0.95 | 0.75 |
| mass_only_s303 | 0.598 | 0.515 | 0.51 | 0.51 | 0.58 | 0.52 | 0.81 | 0.76 |
| mass_only_s304 | 0.988 | 0.253 | 0.09 | 0.49 | 0.98 | 0.09 | 0.62 | 0.70 |
| mass_only_s305 | 0.840 | 0.606 | 0.44 | 0.81 | 0.71 | 0.28 | 0.41 | 0.64 |
| mass_only_s306 | 0.602 | 0.508 | 0.51 | 0.49 | 0.60 | 0.51 | 0.97 | 0.78 |
| mass_only_s307 | 0.992 | 0.971 | 0.95 | 0.99 | 0.98 | 0.94 | 0.33 | 0.57 |
| mass_only_s308 | 0.596 | 0.531 | 0.51 | 0.51 | 0.60 | 0.51 | 0.91 | 0.48 |
| mass_only_s309 | 0.590 | 0.533 | 0.52 | 0.52 | 0.28 | 0.20 | 0.25 | 0.46 |
| mass_only_s310 | 0.429 | 0.183 | 0.16 | 0.23 | 0.37 | 0.16 | 0.73 | 0.56 |
| mass_only_s311 | 0.950 | 0.936 | 0.92 | 0.95 | 0.93 | 0.90 | 0.27 | 0.30 |
| neither_s300 | 1.000 | 0.844 | 0.86 | 0.89 | — | — | 0.27 | 0.29 |
| neither_s301 | 0.999 | 0.961 | 0.98 | 0.96 | — | — | 0.54 | 0.76 |
| neither_s302 | 0.601 | 0.498 | 0.49 | 0.49 | — | — | 0.45 | 0.62 |
| neither_s303 | 0.622 | 0.453 | 0.41 | 0.53 | — | — | 0.61 | 0.73 |
| neither_s304 | 0.561 | 0.509 | 0.51 | 0.50 | — | — | 0.26 | 0.56 |
| neither_s305 | 0.999 | 0.771 | 0.81 | 0.82 | — | — | 0.61 | 0.44 |
| neither_s306 | 0.610 | 0.516 | 0.46 | 0.49 | — | — | 0.42 | 0.56 |
| neither_s307 | 1.000 | 0.929 | 0.96 | 0.96 | — | — | 0.52 | 0.64 |
| neither_s308 | 1.000 | 0.805 | 0.86 | 0.82 | — | — | 0.43 | 0.33 |
| neither_s309 | 0.988 | 0.800 | 0.83 | 0.93 | — | — | 0.57 | 0.79 |
| neither_s310 | 0.978 | 0.852 | 0.91 | 0.91 | — | — | 0.57 | 0.46 |
| neither_s311 | 0.987 | 0.894 | 0.91 | 0.91 | — | — | 0.39 | 0.35 |

## Scientific review (supervisor)

**Measured state.** mass_only: 4/12 fit (ID ≥ 0.95), mean ID 0.70, mean OOD 0.49, block IIA 0.65, unseen-source IIA 0.44, factorisation R²_add 0.70; acc_only: 5/12 fit (ID ≥ 0.95), mean ID 0.71, mean OOD 0.49, block IIA 0.69, unseen-source IIA 0.47, factorisation R²_add 0.78; neither: 8/12 fit (ID ≥ 0.95), mean ID 0.86, mean OOD 0.74.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 0.99 / 0.95 / 0.94 / 1.00; 311: 0.95 / 0.92 / 0.90 / 0.98; 305: 0.84 / 0.44 / 0.28 / 0.98; 304: 0.99 / 0.09 / 0.09 / 0.72; 310: 0.43 / 0.16 / 0.16 / 0.74.
**Events since previous cohort:** 6 — acc_only_s303: OOD jump 0.34->0.53; acc_only_s309: block carries unseen inputs (IIA-OOD >= 0.5); acc_only_s310: ID >= 0.95; mass_only_s300: block carries unseen inputs (IIA-OOD >= 0.5); mass_only_s303: block carries unseen inputs (IIA-OOD >= 0.5); neither_s301: ID >= 0.95.
**Interpretation.** Late escapes. 310 escapes now (ID 0.38 → 0.92 by 1,700) and loses factorisation in the same window (1.00 at 1,300 → 0.74 at 1,500). This replicates 304's pattern about 600 steps later, so failure is not tied to escaping early or late.
**Anomaly investigated.** Anomaly check: 304 shows a single-step task-loss spike near 1,450 (100-step mean task loss 0.84) with no lasting change to its factorisation (0.72) or OOD (0.09). Instability without reorganisation, consistent with lock-in.
**Previous prediction** ('304 and 310 stay ≤ 0.25 on unseen masses at the next cohort, whatever their ID (failures never rise on unseen masses)'): HELD (304/310 unseen-mass: [0.09, 0.16]).
**Prediction for step 1,600:** 304 and 310 stay ≤ 0.25 on unseen masses at the next cohort, whatever their ID (failures never rise on unseen masses).

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

