# Cohort review — step 700

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.350 | 0.229 | 0 | 0.246 | 0.161 |
| acc_only | 12 | 0.320 | 0.208 | 0 | 0.242 | 0.163 |
| neither | 12 | 0.730 | 0.590 | 2 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.271 | 0.132 | 0.13 | 0.16 | 0.25 | 0.15 | 0.92 | 0.93 |
| mass_only_s311 | focus | 0.328 | 0.152 | 0.15 | 0.19 | 0.25 | 0.16 | 0.87 | 0.87 |
| mass_only_s304 | comparator | 0.332 | 0.158 | 0.15 | 0.21 | 0.24 | 0.15 | 0.81 | 0.90 |
| mass_only_s310 | comparator | 0.273 | 0.132 | 0.13 | 0.15 | 0.24 | 0.16 | 0.92 | 0.91 |
| mass_only_s305 | intermediate | 0.258 | 0.147 | 0.16 | 0.15 | 0.24 | 0.16 | 0.97 | 0.95 |

## Events since step 600

- **mass_only_s309**: OOD jump 0.32->0.49
- **neither_s300**: OOD jump 0.37->0.68
- **neither_s301**: ID >= 0.95; OOD jump 0.60->0.82
- **neither_s304**: OOD jump 0.23->0.44
- **neither_s305**: OOD jump 0.42->0.63
- **neither_s307**: OOD jump 0.18->0.64
- **neither_s308**: OOD jump 0.23->0.59
- **neither_s311**: ID >= 0.95

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 0.252 | 0.142 | 0.16 | 0.16 | 0.23 | 0.16 | 0.96 | 0.97 |
| acc_only_s301 | 0.573 | 0.489 | 0.50 | 0.48 | 0.26 | 0.18 | 0.54 | 0.78 |
| acc_only_s302 | 0.244 | 0.145 | 0.16 | 0.16 | 0.23 | 0.16 | 0.90 | 0.97 |
| acc_only_s303 | 0.575 | 0.489 | 0.47 | 0.45 | 0.31 | 0.22 | 0.41 | 0.67 |
| acc_only_s304 | 0.258 | 0.152 | 0.16 | 0.16 | 0.23 | 0.17 | 0.92 | 0.91 |
| acc_only_s305 | 0.248 | 0.144 | 0.14 | 0.15 | 0.24 | 0.16 | 0.91 | 0.95 |
| acc_only_s306 | 0.260 | 0.139 | 0.16 | 0.15 | 0.24 | 0.15 | 0.91 | 0.92 |
| acc_only_s307 | 0.362 | 0.160 | 0.14 | 0.22 | 0.24 | 0.14 | 0.88 | 0.93 |
| acc_only_s308 | 0.296 | 0.166 | 0.18 | 0.21 | 0.23 | 0.15 | 0.74 | 0.66 |
| acc_only_s309 | 0.257 | 0.149 | 0.15 | 0.16 | 0.23 | 0.16 | 0.98 | 0.97 |
| acc_only_s310 | 0.250 | 0.155 | 0.16 | 0.15 | 0.24 | 0.16 | 0.96 | 0.95 |
| acc_only_s311 | 0.266 | 0.162 | 0.17 | 0.17 | 0.23 | 0.15 | 0.88 | 0.95 |
| mass_only_s300 | 0.256 | 0.128 | 0.12 | 0.17 | 0.24 | 0.15 | 0.94 | 0.95 |
| mass_only_s301 | 0.583 | 0.472 | 0.48 | 0.47 | 0.27 | 0.18 | 0.57 | 0.75 |
| mass_only_s302 | 0.251 | 0.151 | 0.16 | 0.17 | 0.23 | 0.15 | 0.98 | 0.97 |
| mass_only_s303 | 0.562 | 0.489 | 0.51 | 0.45 | 0.29 | 0.21 | 0.48 | 0.63 |
| mass_only_s304 | 0.332 | 0.158 | 0.15 | 0.21 | 0.24 | 0.15 | 0.81 | 0.90 |
| mass_only_s305 | 0.258 | 0.147 | 0.16 | 0.15 | 0.24 | 0.16 | 0.97 | 0.95 |
| mass_only_s306 | 0.267 | 0.143 | 0.17 | 0.15 | 0.24 | 0.15 | 0.89 | 0.94 |
| mass_only_s307 | 0.271 | 0.132 | 0.13 | 0.16 | 0.25 | 0.15 | 0.92 | 0.93 |
| mass_only_s308 | 0.270 | 0.164 | 0.19 | 0.16 | 0.25 | 0.16 | 0.88 | 0.94 |
| mass_only_s309 | 0.549 | 0.485 | 0.50 | 0.46 | 0.23 | 0.15 | 0.52 | 0.51 |
| mass_only_s310 | 0.273 | 0.132 | 0.13 | 0.15 | 0.24 | 0.16 | 0.92 | 0.91 |
| mass_only_s311 | 0.328 | 0.152 | 0.15 | 0.19 | 0.25 | 0.16 | 0.87 | 0.87 |
| neither_s300 | 0.786 | 0.677 | 0.69 | 0.73 | — | — | 0.31 | 0.34 |
| neither_s301 | 0.955 | 0.825 | 0.84 | 0.87 | — | — | 0.61 | 0.82 |
| neither_s302 | 0.584 | 0.509 | 0.50 | 0.47 | — | — | 0.37 | 0.58 |
| neither_s303 | 0.536 | 0.383 | 0.42 | 0.38 | — | — | 0.61 | 0.72 |
| neither_s304 | 0.509 | 0.440 | 0.49 | 0.41 | — | — | 0.61 | 0.81 |
| neither_s305 | 0.852 | 0.633 | 0.68 | 0.71 | — | — | 0.68 | 0.54 |
| neither_s306 | 0.535 | 0.462 | 0.50 | 0.42 | — | — | 0.47 | 0.71 |
| neither_s307 | 0.844 | 0.637 | 0.70 | 0.67 | — | — | 0.56 | 0.65 |
| neither_s308 | 0.808 | 0.587 | 0.59 | 0.67 | — | — | 0.44 | 0.32 |
| neither_s309 | 0.605 | 0.528 | 0.50 | 0.50 | — | — | 0.42 | 0.48 |
| neither_s310 | 0.782 | 0.617 | 0.63 | 0.69 | — | — | 0.59 | 0.48 |
| neither_s311 | 0.964 | 0.782 | 0.82 | 0.83 | — | — | 0.39 | 0.35 |

## Scientific review (supervisor)

**Measured state.** mass_only: 0/12 fit (ID ≥ 0.95), mean ID 0.35, mean OOD 0.24, block IIA 0.25, unseen-source IIA 0.16, factorisation R²_add 0.95; acc_only: 0/12 fit (ID ≥ 0.95), mean ID 0.32, mean OOD 0.22, block IIA 0.24, unseen-source IIA 0.16, factorisation R²_add 0.99; neither: 2/12 fit (ID ≥ 0.95), mean ID 0.73, mean OOD 0.59.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 0.27 / 0.13 / 0.15 / 1.00; 311: 0.33 / 0.15 / 0.16 / 0.98; 305: 0.26 / 0.16 / 0.16 / 1.00; 304: 0.33 / 0.15 / 0.15 / 0.94; 310: 0.27 / 0.13 / 0.16 / 1.00.
**Events since previous cohort:** 8 — mass_only_s309: OOD jump 0.32->0.49; neither_s300: OOD jump 0.37->0.68; neither_s301: ID >= 0.95, OOD jump 0.60->0.82; neither_s304: OOD jump 0.23->0.44; neither_s305: OOD jump 0.42->0.63; neither_s307: OOD jump 0.18->0.64; neither_s308: OOD jump 0.23->0.59; neither_s311: ID >= 0.95.
**Interpretation.** Critical window opens. The `neither` group escapes rapidly. In declared networks the block starts to lose factorisation (mass group mean R²_add below 0.98) before ID rises. In the 10-step replay, 304 is already at 0.97 at step 650, the earliest measured sign of its later failure. The focus networks are not yet behaviourally distinguishable.
**Previous prediction** ('304 and 310 stay ≤ 0.25 on unseen masses at the next cohort, whatever their ID (failures never rise on unseen masses)'): HELD (304/310 unseen-mass: [0.15, 0.13]).
**Prediction for step 800:** 304 and 310 stay ≤ 0.25 on unseen masses at the next cohort, whatever their ID (failures never rise on unseen masses).

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

