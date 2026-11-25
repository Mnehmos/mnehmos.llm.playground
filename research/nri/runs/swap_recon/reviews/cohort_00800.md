# Cohort review — step 800

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.388 | 0.256 | 0 | 0.282 | 0.199 |
| acc_only | 12 | 0.333 | 0.198 | 0 | 0.247 | 0.165 |
| neither | 12 | 0.814 | 0.679 | 7 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.346 | 0.172 | 0.18 | 0.25 | 0.25 | 0.15 | 0.83 | 0.90 |
| mass_only_s311 | focus | 0.459 | 0.248 | 0.26 | 0.27 | 0.24 | 0.16 | 0.68 | 0.60 |
| mass_only_s304 | comparator | 0.500 | 0.249 | 0.19 | 0.37 | 0.27 | 0.13 | 0.61 | 0.77 |
| mass_only_s310 | comparator | 0.253 | 0.146 | 0.16 | 0.17 | 0.23 | 0.16 | 0.96 | 0.93 |
| mass_only_s305 | intermediate | 0.262 | 0.140 | 0.14 | 0.16 | 0.24 | 0.15 | 0.96 | 0.93 |

## Events since step 700

- **acc_only_s303**: OOD jump 0.49->0.20
- **neither_s300**: ID >= 0.95
- **neither_s305**: ID >= 0.95
- **neither_s307**: ID >= 0.95; OOD jump 0.64->0.88
- **neither_s308**: ID >= 0.95; OOD jump 0.59->0.75
- **neither_s310**: ID >= 0.95; OOD jump 0.62->0.78

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 0.249 | 0.145 | 0.16 | 0.16 | 0.24 | 0.17 | 0.95 | 0.94 |
| acc_only_s301 | 0.561 | 0.471 | 0.50 | 0.44 | 0.30 | 0.19 | 0.58 | 0.77 |
| acc_only_s302 | 0.247 | 0.146 | 0.16 | 0.16 | 0.24 | 0.16 | 0.92 | 0.97 |
| acc_only_s303 | 0.318 | 0.204 | 0.22 | 0.20 | 0.32 | 0.23 | 0.43 | 0.61 |
| acc_only_s304 | 0.172 | 0.107 | 0.12 | 0.12 | 0.16 | 0.12 | 0.94 | 0.93 |
| acc_only_s305 | 0.245 | 0.145 | 0.15 | 0.15 | 0.24 | 0.15 | 0.91 | 0.93 |
| acc_only_s306 | 0.363 | 0.199 | 0.24 | 0.23 | 0.24 | 0.15 | 0.69 | 0.79 |
| acc_only_s307 | 0.564 | 0.307 | 0.30 | 0.38 | 0.29 | 0.17 | 0.75 | 0.84 |
| acc_only_s308 | 0.415 | 0.198 | 0.17 | 0.26 | 0.24 | 0.17 | 0.70 | 0.57 |
| acc_only_s309 | 0.274 | 0.141 | 0.17 | 0.14 | 0.23 | 0.16 | 0.92 | 0.93 |
| acc_only_s310 | 0.263 | 0.137 | 0.14 | 0.16 | 0.24 | 0.15 | 0.92 | 0.83 |
| acc_only_s311 | 0.322 | 0.179 | 0.22 | 0.17 | 0.23 | 0.16 | 0.78 | 0.89 |
| mass_only_s300 | 0.250 | 0.144 | 0.14 | 0.16 | 0.24 | 0.14 | 0.96 | 0.97 |
| mass_only_s301 | 0.555 | 0.451 | 0.47 | 0.40 | 0.44 | 0.34 | 0.68 | 0.64 |
| mass_only_s302 | 0.261 | 0.150 | 0.17 | 0.15 | 0.24 | 0.16 | 0.96 | 0.94 |
| mass_only_s303 | 0.570 | 0.468 | 0.49 | 0.42 | 0.53 | 0.50 | 0.70 | 0.73 |
| mass_only_s304 | 0.500 | 0.249 | 0.19 | 0.37 | 0.27 | 0.13 | 0.61 | 0.77 |
| mass_only_s305 | 0.262 | 0.140 | 0.14 | 0.16 | 0.24 | 0.15 | 0.96 | 0.93 |
| mass_only_s306 | 0.326 | 0.232 | 0.27 | 0.23 | 0.24 | 0.16 | 0.74 | 0.78 |
| mass_only_s307 | 0.346 | 0.172 | 0.18 | 0.25 | 0.25 | 0.15 | 0.83 | 0.90 |
| mass_only_s308 | 0.317 | 0.177 | 0.23 | 0.19 | 0.24 | 0.18 | 0.72 | 0.84 |
| mass_only_s309 | 0.560 | 0.500 | 0.51 | 0.49 | 0.24 | 0.16 | 0.45 | 0.56 |
| mass_only_s310 | 0.253 | 0.146 | 0.16 | 0.17 | 0.23 | 0.16 | 0.96 | 0.93 |
| mass_only_s311 | 0.459 | 0.248 | 0.26 | 0.27 | 0.24 | 0.16 | 0.68 | 0.60 |
| neither_s300 | 0.952 | 0.790 | 0.79 | 0.86 | — | — | 0.29 | 0.32 |
| neither_s301 | 0.981 | 0.856 | 0.85 | 0.93 | — | — | 0.65 | 0.83 |
| neither_s302 | 0.595 | 0.523 | 0.51 | 0.49 | — | — | 0.43 | 0.64 |
| neither_s303 | 0.577 | 0.489 | 0.51 | 0.51 | — | — | 0.61 | 0.72 |
| neither_s304 | 0.589 | 0.482 | 0.50 | 0.43 | — | — | 0.53 | 0.76 |
| neither_s305 | 0.951 | 0.731 | 0.76 | 0.79 | — | — | 0.62 | 0.46 |
| neither_s306 | 0.593 | 0.504 | 0.51 | 0.48 | — | — | 0.45 | 0.58 |
| neither_s307 | 0.984 | 0.881 | 0.89 | 0.92 | — | — | 0.54 | 0.63 |
| neither_s308 | 0.985 | 0.754 | 0.76 | 0.81 | — | — | 0.42 | 0.31 |
| neither_s309 | 0.597 | 0.508 | 0.45 | 0.48 | — | — | 0.47 | 0.48 |
| neither_s310 | 0.981 | 0.778 | 0.83 | 0.85 | — | — | 0.55 | 0.41 |
| neither_s311 | 0.987 | 0.848 | 0.87 | 0.90 | — | — | 0.40 | 0.37 |

## Scientific review (supervisor)

**Measured state.** mass_only: 0/12 fit (ID ≥ 0.95), mean ID 0.39, mean OOD 0.27, block IIA 0.28, unseen-source IIA 0.20, factorisation R²_add 0.89; acc_only: 0/12 fit (ID ≥ 0.95), mean ID 0.33, mean OOD 0.21, block IIA 0.25, unseen-source IIA 0.16, factorisation R²_add 0.98; neither: 7/12 fit (ID ≥ 0.95), mean ID 0.81, mean OOD 0.68.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 0.35 / 0.18 / 0.15 / 0.98; 311: 0.46 / 0.26 / 0.16 / 0.95; 305: 0.26 / 0.14 / 0.15 / 1.00; 304: 0.50 / 0.19 / 0.13 / 0.93; 310: 0.25 / 0.16 / 0.16 / 1.00.
**Events since previous cohort:** 6 — acc_only_s303: OOD jump 0.49->0.20; neither_s300: ID >= 0.95; neither_s305: ID >= 0.95; neither_s307: ID >= 0.95, OOD jump 0.64->0.88; neither_s308: ID >= 0.95, OOD jump 0.59->0.75; neither_s310: ID >= 0.95, OOD jump 0.62->0.78.
**Interpretation.** Escape under way for the earliest declared networks. 304 leads (ID 0.50 at 800) and has already lost factorisation (0.93 → 0.84 by 850). Its unseen-mass accuracy does not rise with ID. 307 and 311 escape with unseen-mass accuracy rising together with ID. This is the first behavioural divergence.
**Previous prediction** ('304 and 310 stay ≤ 0.25 on unseen masses at the next cohort, whatever their ID (failures never rise on unseen masses)'): HELD (304/310 unseen-mass: [0.19, 0.16]).
**Prediction for step 900:** 304 and 310 stay ≤ 0.25 on unseen masses at the next cohort, whatever their ID (failures never rise on unseen masses).

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

