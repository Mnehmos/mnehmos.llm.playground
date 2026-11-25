# Cohort review — step 2,000

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.771 | 0.532 | 6 | 0.735 | 0.456 |
| acc_only | 12 | 0.741 | 0.515 | 5 | 0.728 | 0.525 |
| neither | 12 | 0.922 | 0.804 | 9 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.998 | 0.986 | 0.97 | 1.00 | 1.00 | 0.95 | 0.32 | 0.57 |
| mass_only_s311 | focus | 0.989 | 0.947 | 0.93 | 0.98 | 0.96 | 0.86 | 0.26 | 0.31 |
| mass_only_s304 | comparator | 1.000 | 0.252 | 0.09 | 0.48 | 1.00 | 0.09 | 0.62 | 0.66 |
| mass_only_s310 | comparator | 1.000 | 0.250 | 0.10 | 0.45 | 1.00 | 0.09 | 0.72 | 0.81 |
| mass_only_s305 | intermediate | 1.000 | 0.813 | 0.67 | 0.99 | 0.98 | 0.52 | 0.39 | 0.58 |

## Events since step 1,900

- **mass_only_s301**: block carries unseen inputs (IIA-OOD >= 0.5)
- **mass_only_s305**: ID >= 0.95; declared block formed (IIA >= 0.9); block carries unseen inputs (IIA-OOD >= 0.5); OOD jump 0.61->0.81
- **neither_s303**: ID >= 0.95

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.300 | 0.28 | 0.41 | 1.00 | 0.42 | 0.58 | 0.91 |
| acc_only_s301 | 0.598 | 0.524 | 0.54 | 0.50 | 0.59 | 0.50 | 0.84 | 0.62 |
| acc_only_s302 | 0.595 | 0.514 | 0.52 | 0.51 | 0.57 | 0.51 | 0.61 | 0.41 |
| acc_only_s303 | 0.621 | 0.540 | 0.50 | 0.51 | 0.59 | 0.52 | 0.37 | 0.60 |
| acc_only_s304 | 0.254 | 0.141 | 0.16 | 0.15 | 0.23 | 0.17 | 0.89 | 0.94 |
| acc_only_s305 | 1.000 | 0.808 | 0.90 | 0.80 | 1.00 | 0.80 | 0.56 | 0.57 |
| acc_only_s306 | 0.612 | 0.464 | 0.46 | 0.43 | 0.59 | 0.44 | 0.92 | 0.94 |
| acc_only_s307 | 0.999 | 0.775 | 0.63 | 0.99 | 1.00 | 0.96 | 0.41 | 0.57 |
| acc_only_s308 | 0.999 | 0.898 | 0.97 | 0.87 | 0.99 | 0.91 | 0.37 | 0.26 |
| acc_only_s309 | 0.612 | 0.540 | 0.53 | 0.52 | 0.59 | 0.54 | 0.57 | 0.46 |
| acc_only_s310 | 1.000 | 0.214 | 0.31 | 0.15 | 1.00 | 0.16 | 0.82 | 0.95 |
| acc_only_s311 | 0.606 | 0.466 | 0.52 | 0.39 | 0.59 | 0.38 | 0.90 | 0.83 |
| mass_only_s300 | 1.000 | 0.350 | 0.50 | 0.23 | 0.99 | 0.48 | 0.39 | 0.76 |
| mass_only_s301 | 0.573 | 0.518 | 0.53 | 0.52 | 0.56 | 0.52 | 0.18 | 0.49 |
| mass_only_s302 | 0.607 | 0.496 | 0.51 | 0.45 | 0.59 | 0.51 | 0.96 | 0.75 |
| mass_only_s303 | 0.310 | 0.189 | 0.23 | 0.17 | 0.25 | 0.17 | 0.82 | 0.82 |
| mass_only_s304 | 1.000 | 0.252 | 0.09 | 0.48 | 1.00 | 0.09 | 0.62 | 0.66 |
| mass_only_s305 | 1.000 | 0.813 | 0.67 | 0.99 | 0.98 | 0.52 | 0.39 | 0.58 |
| mass_only_s306 | 0.600 | 0.509 | 0.51 | 0.48 | 0.59 | 0.51 | 0.98 | 0.79 |
| mass_only_s307 | 0.998 | 0.986 | 0.97 | 1.00 | 1.00 | 0.95 | 0.32 | 0.57 |
| mass_only_s308 | 0.594 | 0.527 | 0.52 | 0.49 | 0.59 | 0.50 | 0.92 | 0.48 |
| mass_only_s309 | 0.582 | 0.540 | 0.53 | 0.53 | 0.32 | 0.25 | 0.24 | 0.43 |
| mass_only_s310 | 1.000 | 0.250 | 0.10 | 0.45 | 1.00 | 0.09 | 0.72 | 0.81 |
| mass_only_s311 | 0.989 | 0.947 | 0.93 | 0.98 | 0.96 | 0.86 | 0.26 | 0.31 |
| neither_s300 | 1.000 | 0.863 | 0.88 | 0.91 | — | — | 0.26 | 0.29 |
| neither_s301 | 0.988 | 0.933 | 0.96 | 0.93 | — | — | 0.53 | 0.75 |
| neither_s302 | 0.900 | 0.803 | 0.90 | 0.80 | — | — | 0.40 | 0.45 |
| neither_s303 | 0.982 | 0.844 | 0.91 | 0.90 | — | — | 0.55 | 0.78 |
| neither_s304 | 0.589 | 0.528 | 0.50 | 0.51 | — | — | 0.27 | 0.57 |
| neither_s305 | 1.000 | 0.805 | 0.85 | 0.86 | — | — | 0.60 | 0.45 |
| neither_s306 | 0.608 | 0.508 | 0.45 | 0.46 | — | — | 0.47 | 0.55 |
| neither_s307 | 1.000 | 0.932 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.813 | 0.86 | 0.83 | — | — | 0.43 | 0.33 |
| neither_s309 | 1.000 | 0.823 | 0.85 | 0.96 | — | — | 0.54 | 0.80 |
| neither_s310 | 1.000 | 0.880 | 0.91 | 0.92 | — | — | 0.53 | 0.40 |
| neither_s311 | 0.999 | 0.914 | 0.92 | 0.94 | — | — | 0.40 | 0.36 |

## Scientific review (supervisor)

**Measured state.** mass_only: 6/12 fit (ID ≥ 0.95), mean ID 0.77, mean OOD 0.51, block IIA 0.74, unseen-source IIA 0.46, factorisation R²_add 0.74; acc_only: 5/12 fit (ID ≥ 0.95), mean ID 0.74, mean OOD 0.52, block IIA 0.73, unseen-source IIA 0.53, factorisation R²_add 0.78; neither: 9/12 fit (ID ≥ 0.95), mean ID 0.92, mean OOD 0.80.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 0.97 / 0.95 / 1.00; 311: 0.99 / 0.93 / 0.86 / 0.98; 305: 1.00 / 0.67 / 0.52 / 0.97; 304: 1.00 / 0.09 / 0.09 / 0.71; 310: 1.00 / 0.10 / 0.09 / 0.71.
**Events since previous cohort:** 3 — mass_only_s301: block carries unseen inputs (IIA-OOD >= 0.5); mass_only_s305: ID >= 0.95, declared block formed (IIA >= 0.9), block carries unseen inputs (IIA-OOD >= 0.5), OOD jump 0.61->0.81; neither_s303: ID >= 0.95.
**Interpretation.** Post-fit consolidation. All five focus networks fit the training data. Group means change little. The mass group's mean factorisation is about 0.74 and stays nearly fixed thereafter: with weight decay 0.01 and losses near zero, no measurable pressure reverses a locked-in interaction. 305's partial generalisation (OOD-mass about 0.75 at 1,800) starts to erode.
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): FAILED (changes in unseen-mass accuracy: {307: -0.01, 311: -0.04, 305: 0.21, 304: 0.0, 310: -0.01}).
**Prediction for step 2,100:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

