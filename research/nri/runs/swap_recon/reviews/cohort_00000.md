# Cohort review — step 0

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.075 | 0.074 | 0 | 0.078 | 0.076 |
| acc_only | 12 | 0.075 | 0.074 | 0 | 0.077 | 0.077 |
| neither | 12 | 0.075 | 0.074 | 0 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.061 | 0.073 | 0.08 | 0.08 | 0.07 | 0.08 | 0.21 | 0.28 |
| mass_only_s311 | focus | 0.073 | 0.072 | 0.08 | 0.07 | 0.07 | 0.08 | 0.22 | 0.15 |
| mass_only_s304 | comparator | 0.149 | 0.066 | 0.08 | 0.08 | 0.15 | 0.07 | 0.29 | 0.27 |
| mass_only_s310 | comparator | 0.061 | 0.077 | 0.08 | 0.07 | 0.07 | 0.07 | 0.19 | 0.18 |
| mass_only_s305 | intermediate | 0.065 | 0.071 | 0.07 | 0.08 | 0.07 | 0.08 | 0.15 | 0.18 |

## Events

- none

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 0.077 | 0.070 | 0.07 | 0.07 | 0.07 | 0.08 | 0.27 | 0.34 |
| acc_only_s301 | 0.066 | 0.074 | 0.08 | 0.09 | 0.06 | 0.08 | 0.23 | 0.21 |
| acc_only_s302 | 0.069 | 0.082 | 0.08 | 0.08 | 0.08 | 0.08 | 0.22 | 0.26 |
| acc_only_s303 | 0.068 | 0.073 | 0.07 | 0.07 | 0.06 | 0.08 | 0.20 | 0.20 |
| acc_only_s304 | 0.149 | 0.066 | 0.08 | 0.08 | 0.15 | 0.08 | 0.29 | 0.27 |
| acc_only_s305 | 0.065 | 0.071 | 0.07 | 0.08 | 0.07 | 0.08 | 0.15 | 0.18 |
| acc_only_s306 | 0.061 | 0.075 | 0.08 | 0.08 | 0.07 | 0.08 | 0.15 | 0.25 |
| acc_only_s307 | 0.061 | 0.073 | 0.08 | 0.08 | 0.07 | 0.08 | 0.21 | 0.28 |
| acc_only_s308 | 0.070 | 0.083 | 0.08 | 0.07 | 0.08 | 0.07 | 0.16 | 0.19 |
| acc_only_s309 | 0.078 | 0.073 | 0.07 | 0.07 | 0.08 | 0.07 | 0.26 | 0.24 |
| acc_only_s310 | 0.061 | 0.077 | 0.08 | 0.07 | 0.07 | 0.07 | 0.19 | 0.18 |
| acc_only_s311 | 0.073 | 0.072 | 0.08 | 0.07 | 0.06 | 0.07 | 0.22 | 0.15 |
| mass_only_s300 | 0.077 | 0.070 | 0.07 | 0.07 | 0.08 | 0.08 | 0.27 | 0.34 |
| mass_only_s301 | 0.066 | 0.074 | 0.08 | 0.09 | 0.07 | 0.08 | 0.23 | 0.21 |
| mass_only_s302 | 0.069 | 0.082 | 0.08 | 0.08 | 0.07 | 0.07 | 0.22 | 0.26 |
| mass_only_s303 | 0.068 | 0.073 | 0.07 | 0.07 | 0.07 | 0.08 | 0.20 | 0.20 |
| mass_only_s304 | 0.149 | 0.066 | 0.08 | 0.08 | 0.15 | 0.07 | 0.29 | 0.27 |
| mass_only_s305 | 0.065 | 0.071 | 0.07 | 0.08 | 0.07 | 0.08 | 0.15 | 0.18 |
| mass_only_s306 | 0.061 | 0.075 | 0.08 | 0.08 | 0.06 | 0.08 | 0.15 | 0.25 |
| mass_only_s307 | 0.061 | 0.073 | 0.08 | 0.08 | 0.07 | 0.08 | 0.21 | 0.28 |
| mass_only_s308 | 0.070 | 0.083 | 0.08 | 0.07 | 0.07 | 0.07 | 0.16 | 0.19 |
| mass_only_s309 | 0.078 | 0.073 | 0.07 | 0.07 | 0.08 | 0.07 | 0.26 | 0.24 |
| mass_only_s310 | 0.061 | 0.077 | 0.08 | 0.07 | 0.07 | 0.07 | 0.19 | 0.18 |
| mass_only_s311 | 0.073 | 0.072 | 0.08 | 0.07 | 0.07 | 0.08 | 0.22 | 0.15 |
| neither_s300 | 0.077 | 0.070 | 0.07 | 0.07 | — | — | 0.27 | 0.34 |
| neither_s301 | 0.066 | 0.074 | 0.08 | 0.09 | — | — | 0.23 | 0.21 |
| neither_s302 | 0.069 | 0.082 | 0.08 | 0.08 | — | — | 0.22 | 0.26 |
| neither_s303 | 0.068 | 0.073 | 0.07 | 0.07 | — | — | 0.20 | 0.20 |
| neither_s304 | 0.149 | 0.066 | 0.08 | 0.08 | — | — | 0.29 | 0.27 |
| neither_s305 | 0.065 | 0.071 | 0.07 | 0.08 | — | — | 0.15 | 0.18 |
| neither_s306 | 0.061 | 0.075 | 0.08 | 0.08 | — | — | 0.15 | 0.25 |
| neither_s307 | 0.061 | 0.073 | 0.08 | 0.08 | — | — | 0.21 | 0.28 |
| neither_s308 | 0.070 | 0.083 | 0.08 | 0.07 | — | — | 0.16 | 0.19 |
| neither_s309 | 0.078 | 0.073 | 0.07 | 0.07 | — | — | 0.26 | 0.24 |
| neither_s310 | 0.061 | 0.077 | 0.08 | 0.07 | — | — | 0.19 | 0.18 |
| neither_s311 | 0.073 | 0.072 | 0.08 | 0.07 | — | — | 0.22 | 0.15 |

## Scientific review (supervisor)

**Measured state.** mass_only: 0/12 fit (ID ≥ 0.95), mean ID 0.07, mean OOD 0.08, block IIA 0.08, unseen-source IIA 0.08, factorisation R²_add 0.99; acc_only: 0/12 fit (ID ≥ 0.95), mean ID 0.07, mean OOD 0.08, block IIA 0.08, unseen-source IIA 0.08, factorisation R²_add 0.99; neither: 0/12 fit (ID ≥ 0.95), mean ID 0.07, mean OOD 0.07.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 0.06 / 0.08 / 0.08 / 0.99; 311: 0.07 / 0.08 / 0.08 / 0.99; 305: 0.07 / 0.07 / 0.08 / 0.99; 304: 0.15 / 0.08 / 0.07 / 0.99; 310: 0.06 / 0.08 / 0.07 / 0.99.
**Events since previous cohort:** 0.
**Interpretation.** Initialisation. All 36 networks predict near chance on familiar (ID) and unseen (OOD) inputs. Every declared block is fully factorised (value + unit; R²_add ≈ 0.99), as expected from random embeddings mixed linearly by attention. Interchange accuracy is at chance. Nothing distinguishes the later generalisers 307/311 from the failures 304/310.
**Prediction for step 100:** every network's ID stays in [0.20, 0.30] at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

