# Cohort review — step 200

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.248 | 0.148 | 0 | 0.228 | 0.149 |
| acc_only | 12 | 0.248 | 0.147 | 0 | 0.222 | 0.149 |
| neither | 12 | 0.248 | 0.148 | 0 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.247 | 0.149 | 0.17 | 0.16 | 0.23 | 0.15 | 0.98 | 0.99 |
| mass_only_s311 | focus | 0.246 | 0.151 | 0.16 | 0.16 | 0.23 | 0.15 | 0.97 | 0.97 |
| mass_only_s304 | comparator | 0.245 | 0.146 | 0.15 | 0.17 | 0.23 | 0.15 | 0.97 | 0.97 |
| mass_only_s310 | comparator | 0.250 | 0.145 | 0.16 | 0.15 | 0.23 | 0.15 | 0.97 | 0.97 |
| mass_only_s305 | intermediate | 0.254 | 0.144 | 0.15 | 0.16 | 0.23 | 0.15 | 0.95 | 0.94 |

## Events since step 100

- none

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 0.256 | 0.149 | 0.16 | 0.15 | 0.22 | 0.15 | 0.96 | 0.97 |
| acc_only_s301 | 0.238 | 0.145 | 0.16 | 0.15 | 0.23 | 0.15 | 0.98 | 0.98 |
| acc_only_s302 | 0.247 | 0.146 | 0.16 | 0.16 | 0.22 | 0.15 | 0.96 | 0.95 |
| acc_only_s303 | 0.257 | 0.150 | 0.16 | 0.16 | 0.22 | 0.15 | 0.95 | 0.95 |
| acc_only_s304 | 0.246 | 0.157 | 0.15 | 0.16 | 0.22 | 0.15 | 0.97 | 0.97 |
| acc_only_s305 | 0.250 | 0.143 | 0.15 | 0.16 | 0.22 | 0.15 | 0.95 | 0.94 |
| acc_only_s306 | 0.246 | 0.147 | 0.15 | 0.16 | 0.22 | 0.14 | 0.98 | 0.98 |
| acc_only_s307 | 0.245 | 0.149 | 0.17 | 0.16 | 0.22 | 0.16 | 0.99 | 0.99 |
| acc_only_s308 | 0.252 | 0.152 | 0.16 | 0.15 | 0.22 | 0.15 | 0.96 | 0.97 |
| acc_only_s309 | 0.247 | 0.145 | 0.16 | 0.16 | 0.22 | 0.15 | 0.97 | 0.97 |
| acc_only_s310 | 0.248 | 0.140 | 0.15 | 0.16 | 0.22 | 0.15 | 0.97 | 0.97 |
| acc_only_s311 | 0.245 | 0.144 | 0.16 | 0.15 | 0.22 | 0.15 | 0.97 | 0.97 |
| mass_only_s300 | 0.253 | 0.140 | 0.15 | 0.15 | 0.23 | 0.15 | 0.96 | 0.97 |
| mass_only_s301 | 0.240 | 0.148 | 0.16 | 0.16 | 0.23 | 0.15 | 0.98 | 0.98 |
| mass_only_s302 | 0.251 | 0.157 | 0.17 | 0.16 | 0.23 | 0.16 | 0.96 | 0.95 |
| mass_only_s303 | 0.252 | 0.150 | 0.15 | 0.16 | 0.23 | 0.14 | 0.95 | 0.95 |
| mass_only_s304 | 0.245 | 0.146 | 0.15 | 0.17 | 0.23 | 0.15 | 0.97 | 0.97 |
| mass_only_s305 | 0.254 | 0.144 | 0.15 | 0.16 | 0.23 | 0.15 | 0.95 | 0.94 |
| mass_only_s306 | 0.243 | 0.143 | 0.15 | 0.15 | 0.23 | 0.14 | 0.98 | 0.98 |
| mass_only_s307 | 0.247 | 0.149 | 0.17 | 0.16 | 0.23 | 0.15 | 0.98 | 0.99 |
| mass_only_s308 | 0.251 | 0.156 | 0.16 | 0.15 | 0.23 | 0.15 | 0.96 | 0.97 |
| mass_only_s309 | 0.249 | 0.142 | 0.16 | 0.15 | 0.23 | 0.15 | 0.96 | 0.97 |
| mass_only_s310 | 0.250 | 0.145 | 0.16 | 0.15 | 0.23 | 0.15 | 0.97 | 0.97 |
| mass_only_s311 | 0.246 | 0.151 | 0.16 | 0.16 | 0.23 | 0.15 | 0.97 | 0.97 |
| neither_s300 | 0.250 | 0.146 | 0.17 | 0.16 | — | — | 0.97 | 0.97 |
| neither_s301 | 0.238 | 0.144 | 0.16 | 0.15 | — | — | 0.98 | 0.98 |
| neither_s302 | 0.249 | 0.149 | 0.17 | 0.15 | — | — | 0.95 | 0.94 |
| neither_s303 | 0.262 | 0.144 | 0.16 | 0.16 | — | — | 0.96 | 0.96 |
| neither_s304 | 0.241 | 0.152 | 0.15 | 0.15 | — | — | 0.98 | 0.98 |
| neither_s305 | 0.259 | 0.146 | 0.15 | 0.16 | — | — | 0.95 | 0.95 |
| neither_s306 | 0.248 | 0.150 | 0.16 | 0.16 | — | — | 0.98 | 0.98 |
| neither_s307 | 0.242 | 0.140 | 0.16 | 0.16 | — | — | 0.98 | 0.98 |
| neither_s308 | 0.251 | 0.157 | 0.17 | 0.16 | — | — | 0.97 | 0.97 |
| neither_s309 | 0.245 | 0.151 | 0.17 | 0.16 | — | — | 0.97 | 0.98 |
| neither_s310 | 0.249 | 0.147 | 0.17 | 0.15 | — | — | 0.97 | 0.97 |
| neither_s311 | 0.242 | 0.146 | 0.16 | 0.15 | — | — | 0.98 | 0.98 |

## Scientific review (supervisor)

**Measured state.** mass_only: 0/12 fit (ID ≥ 0.95), mean ID 0.25, mean OOD 0.16, block IIA 0.23, unseen-source IIA 0.15, factorisation R²_add 1.00; acc_only: 0/12 fit (ID ≥ 0.95), mean ID 0.25, mean OOD 0.16, block IIA 0.22, unseen-source IIA 0.15, factorisation R²_add 1.00; neither: 0/12 fit (ID ≥ 0.95), mean ID 0.25, mean OOD 0.15.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 0.25 / 0.17 / 0.15 / 1.00; 311: 0.25 / 0.16 / 0.15 / 1.00; 305: 0.25 / 0.15 / 0.15 / 1.00; 304: 0.24 / 0.15 / 0.15 / 1.00; 310: 0.25 / 0.16 / 0.15 / 1.00.
**Events since previous cohort:** 0.
**Interpretation.** Shared plateau. Every network, declared or not, sits at ID ≈ 0.24-0.25. A direct check on 307 at step 400 shows this is a zero-detection shortcut: questions where mass or acceleration is 0 (answer 0; 16.8% of questions) are answered perfectly and the rest at chance (0.083). Declared blocks remain factorised (R²_add 1.00). Interchange accuracy rises slightly above chance (≈ 0.2) because the zero case transfers through the block.
**Previous prediction** ('every network's ID stays in [0.20, 0.30] at the next cohort'): HELD (focus ID at next cohort: [0.25, 0.25, 0.25, 0.24, 0.25]).
**Prediction for step 300:** every network's ID stays in [0.20, 0.30] at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

