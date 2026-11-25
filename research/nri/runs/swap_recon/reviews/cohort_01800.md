# Cohort review — step 1,800

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.758 | 0.517 | 6 | 0.705 | 0.434 |
| acc_only | 12 | 0.734 | 0.512 | 5 | 0.725 | 0.521 |
| neither | 12 | 0.878 | 0.751 | 8 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.990 | 0.971 | 0.95 | 1.00 | 0.99 | 0.95 | 0.33 | 0.57 |
| mass_only_s311 | focus | 0.996 | 0.977 | 0.99 | 0.98 | 0.99 | 0.98 | 0.26 | 0.31 |
| mass_only_s304 | comparator | 1.000 | 0.250 | 0.09 | 0.48 | 1.00 | 0.09 | 0.62 | 0.68 |
| mass_only_s310 | comparator | 0.998 | 0.229 | 0.09 | 0.42 | 1.00 | 0.09 | 0.69 | 0.81 |
| mass_only_s305 | intermediate | 0.995 | 0.850 | 0.75 | 0.98 | 0.97 | 0.55 | 0.39 | 0.59 |

## Events since step 1,700

- **acc_only_s302**: block carries unseen inputs (IIA-OOD >= 0.5)
- **acc_only_s308**: ID >= 0.95; declared block formed (IIA >= 0.9); block carries unseen inputs (IIA-OOD >= 0.5); OOD jump 0.47->0.88
- **mass_only_s305**: block carries unseen inputs (IIA-OOD >= 0.5)
- **mass_only_s310**: ID >= 0.95; declared block formed (IIA >= 0.9)
- **neither_s301**: ID >= 0.95

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 0.998 | 0.324 | 0.32 | 0.44 | 1.00 | 0.45 | 0.57 | 0.91 |
| acc_only_s301 | 0.603 | 0.500 | 0.51 | 0.46 | 0.58 | 0.47 | 0.83 | 0.63 |
| acc_only_s302 | 0.574 | 0.504 | 0.51 | 0.50 | 0.55 | 0.53 | 0.62 | 0.43 |
| acc_only_s303 | 0.592 | 0.530 | 0.52 | 0.52 | 0.58 | 0.52 | 0.37 | 0.60 |
| acc_only_s304 | 0.252 | 0.144 | 0.16 | 0.16 | 0.24 | 0.16 | 0.89 | 0.95 |
| acc_only_s305 | 1.000 | 0.799 | 0.89 | 0.79 | 1.00 | 0.78 | 0.56 | 0.58 |
| acc_only_s306 | 0.613 | 0.482 | 0.46 | 0.45 | 0.58 | 0.44 | 0.90 | 0.94 |
| acc_only_s307 | 0.999 | 0.770 | 0.65 | 0.98 | 1.00 | 0.95 | 0.41 | 0.57 |
| acc_only_s308 | 0.998 | 0.882 | 0.92 | 0.87 | 0.99 | 0.89 | 0.37 | 0.27 |
| acc_only_s309 | 0.587 | 0.537 | 0.54 | 0.54 | 0.58 | 0.53 | 0.57 | 0.46 |
| acc_only_s310 | 1.000 | 0.217 | 0.32 | 0.13 | 1.00 | 0.14 | 0.82 | 0.95 |
| acc_only_s311 | 0.593 | 0.458 | 0.51 | 0.37 | 0.59 | 0.38 | 0.89 | 0.83 |
| mass_only_s300 | 0.978 | 0.314 | 0.40 | 0.29 | 0.95 | 0.43 | 0.39 | 0.77 |
| mass_only_s301 | 0.520 | 0.397 | 0.44 | 0.38 | 0.26 | 0.19 | 0.47 | 0.65 |
| mass_only_s302 | 0.589 | 0.494 | 0.54 | 0.47 | 0.60 | 0.52 | 0.96 | 0.75 |
| mass_only_s303 | 0.271 | 0.160 | 0.18 | 0.16 | 0.24 | 0.16 | 0.85 | 0.92 |
| mass_only_s304 | 1.000 | 0.250 | 0.09 | 0.48 | 1.00 | 0.09 | 0.62 | 0.68 |
| mass_only_s305 | 0.995 | 0.850 | 0.75 | 0.98 | 0.97 | 0.55 | 0.39 | 0.59 |
| mass_only_s306 | 0.595 | 0.509 | 0.51 | 0.48 | 0.59 | 0.51 | 0.98 | 0.79 |
| mass_only_s307 | 0.990 | 0.971 | 0.95 | 1.00 | 0.99 | 0.95 | 0.33 | 0.57 |
| mass_only_s308 | 0.574 | 0.529 | 0.50 | 0.50 | 0.59 | 0.51 | 0.92 | 0.48 |
| mass_only_s309 | 0.592 | 0.528 | 0.50 | 0.49 | 0.30 | 0.24 | 0.22 | 0.43 |
| mass_only_s310 | 0.998 | 0.229 | 0.09 | 0.42 | 1.00 | 0.09 | 0.69 | 0.81 |
| mass_only_s311 | 0.996 | 0.977 | 0.99 | 0.98 | 0.99 | 0.98 | 0.26 | 0.31 |
| neither_s300 | 0.982 | 0.829 | 0.85 | 0.87 | — | — | 0.27 | 0.30 |
| neither_s301 | 0.995 | 0.951 | 0.97 | 0.96 | — | — | 0.54 | 0.74 |
| neither_s302 | 0.680 | 0.603 | 0.63 | 0.67 | — | — | 0.43 | 0.57 |
| neither_s303 | 0.681 | 0.461 | 0.40 | 0.55 | — | — | 0.57 | 0.72 |
| neither_s304 | 0.590 | 0.503 | 0.48 | 0.49 | — | — | 0.27 | 0.57 |
| neither_s305 | 1.000 | 0.791 | 0.84 | 0.84 | — | — | 0.61 | 0.45 |
| neither_s306 | 0.606 | 0.527 | 0.50 | 0.48 | — | — | 0.46 | 0.55 |
| neither_s307 | 1.000 | 0.929 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.812 | 0.86 | 0.83 | — | — | 0.43 | 0.33 |
| neither_s309 | 1.000 | 0.824 | 0.85 | 0.96 | — | — | 0.54 | 0.80 |
| neither_s310 | 0.999 | 0.854 | 0.87 | 0.89 | — | — | 0.54 | 0.41 |
| neither_s311 | 0.997 | 0.923 | 0.94 | 0.94 | — | — | 0.40 | 0.36 |

## Scientific review (supervisor)

**Measured state.** mass_only: 6/12 fit (ID ≥ 0.95), mean ID 0.76, mean OOD 0.50, block IIA 0.70, unseen-source IIA 0.43, factorisation R²_add 0.74; acc_only: 5/12 fit (ID ≥ 0.95), mean ID 0.73, mean OOD 0.52, block IIA 0.72, unseen-source IIA 0.52, factorisation R²_add 0.78; neither: 8/12 fit (ID ≥ 0.95), mean ID 0.88, mean OOD 0.75.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 0.99 / 0.95 / 0.95 / 1.00; 311: 1.00 / 0.99 / 0.98 / 0.98; 305: 1.00 / 0.75 / 0.55 / 0.97; 304: 1.00 / 0.09 / 0.09 / 0.72; 310: 1.00 / 0.09 / 0.09 / 0.71.
**Events since previous cohort:** 5 — acc_only_s302: block carries unseen inputs (IIA-OOD >= 0.5); acc_only_s308: ID >= 0.95, declared block formed (IIA >= 0.9), block carries unseen inputs (IIA-OOD >= 0.5), OOD jump 0.47->0.88; mass_only_s305: block carries unseen inputs (IIA-OOD >= 0.5); mass_only_s310: ID >= 0.95, declared block formed (IIA >= 0.9); neither_s301: ID >= 0.95.
**Interpretation.** Post-fit consolidation. All five focus networks fit the training data. Group means change little. The mass group's mean factorisation is about 0.74 and stays nearly fixed thereafter: with weight decay 0.01 and losses near zero, no measurable pressure reverses a locked-in interaction. 305's partial generalisation (OOD-mass about 0.75 at 1,800) starts to erode.
**Previous prediction** ('304 and 310 stay ≤ 0.25 on unseen masses at the next cohort, whatever their ID (failures never rise on unseen masses)'): HELD (304/310 unseen-mass: [0.09, 0.09]).
**Prediction for step 1,900:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

