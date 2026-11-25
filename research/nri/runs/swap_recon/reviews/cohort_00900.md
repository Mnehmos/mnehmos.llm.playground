# Cohort review — step 900

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.446 | 0.299 | 0 | 0.334 | 0.223 |
| acc_only | 12 | 0.426 | 0.286 | 0 | 0.345 | 0.245 |
| neither | 12 | 0.710 | 0.572 | 3 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.526 | 0.355 | 0.36 | 0.40 | 0.32 | 0.15 | 0.63 | 0.83 |
| mass_only_s311 | focus | 0.597 | 0.317 | 0.33 | 0.34 | 0.25 | 0.16 | 0.63 | 0.47 |
| mass_only_s304 | comparator | 0.628 | 0.297 | 0.20 | 0.40 | 0.55 | 0.20 | 0.64 | 0.73 |
| mass_only_s310 | comparator | 0.254 | 0.149 | 0.17 | 0.16 | 0.23 | 0.16 | 0.96 | 0.92 |
| mass_only_s305 | intermediate | 0.279 | 0.143 | 0.14 | 0.19 | 0.24 | 0.12 | 0.93 | 0.91 |

## Events since step 800

- **acc_only_s303**: OOD jump 0.20->0.40
- **acc_only_s306**: OOD jump 0.20->0.44
- **acc_only_s309**: OOD jump 0.14->0.36
- **mass_only_s301**: block carries unseen inputs (IIA-OOD >= 0.5)
- **mass_only_s303**: block carries unseen inputs (IIA-OOD >= 0.5)
- **mass_only_s307**: OOD jump 0.17->0.35
- **neither_s307**: ID fell below 0.95; OOD jump 0.88->0.64
- **neither_s308**: ID fell below 0.95; OOD jump 0.75->0.41
- **neither_s310**: ID fell below 0.95; OOD jump 0.78->0.21
- **neither_s311**: ID fell below 0.95

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 0.276 | 0.142 | 0.15 | 0.18 | 0.23 | 0.14 | 0.86 | 0.89 |
| acc_only_s301 | 0.596 | 0.479 | 0.53 | 0.39 | 0.53 | 0.34 | 0.69 | 0.65 |
| acc_only_s302 | 0.244 | 0.144 | 0.16 | 0.16 | 0.23 | 0.16 | 0.93 | 0.97 |
| acc_only_s303 | 0.461 | 0.400 | 0.39 | 0.39 | 0.45 | 0.40 | 0.41 | 0.61 |
| acc_only_s304 | 0.245 | 0.144 | 0.16 | 0.16 | 0.23 | 0.16 | 0.95 | 0.96 |
| acc_only_s305 | 0.254 | 0.151 | 0.14 | 0.14 | 0.23 | 0.16 | 0.93 | 0.94 |
| acc_only_s306 | 0.590 | 0.443 | 0.52 | 0.40 | 0.56 | 0.38 | 0.86 | 0.92 |
| acc_only_s307 | 0.748 | 0.411 | 0.30 | 0.57 | 0.59 | 0.48 | 0.54 | 0.65 |
| acc_only_s308 | 0.554 | 0.312 | 0.29 | 0.38 | 0.31 | 0.22 | 0.60 | 0.50 |
| acc_only_s309 | 0.448 | 0.355 | 0.35 | 0.36 | 0.25 | 0.17 | 0.66 | 0.74 |
| acc_only_s310 | 0.279 | 0.145 | 0.14 | 0.18 | 0.24 | 0.15 | 0.96 | 0.94 |
| acc_only_s311 | 0.420 | 0.303 | 0.37 | 0.27 | 0.29 | 0.17 | 0.61 | 0.72 |
| mass_only_s300 | 0.260 | 0.128 | 0.13 | 0.14 | 0.24 | 0.14 | 0.95 | 0.97 |
| mass_only_s301 | 0.600 | 0.493 | 0.49 | 0.48 | 0.58 | 0.51 | 0.74 | 0.57 |
| mass_only_s302 | 0.275 | 0.152 | 0.18 | 0.16 | 0.25 | 0.16 | 0.95 | 0.92 |
| mass_only_s303 | 0.589 | 0.536 | 0.52 | 0.51 | 0.58 | 0.53 | 0.76 | 0.78 |
| mass_only_s304 | 0.628 | 0.297 | 0.20 | 0.40 | 0.55 | 0.20 | 0.64 | 0.73 |
| mass_only_s305 | 0.279 | 0.143 | 0.14 | 0.19 | 0.24 | 0.12 | 0.93 | 0.91 |
| mass_only_s306 | 0.331 | 0.200 | 0.23 | 0.21 | 0.24 | 0.16 | 0.86 | 0.93 |
| mass_only_s307 | 0.526 | 0.355 | 0.36 | 0.40 | 0.32 | 0.15 | 0.63 | 0.83 |
| mass_only_s308 | 0.417 | 0.303 | 0.33 | 0.30 | 0.28 | 0.21 | 0.63 | 0.71 |
| mass_only_s309 | 0.591 | 0.515 | 0.52 | 0.50 | 0.23 | 0.16 | 0.39 | 0.49 |
| mass_only_s310 | 0.254 | 0.149 | 0.17 | 0.16 | 0.23 | 0.16 | 0.96 | 0.92 |
| mass_only_s311 | 0.597 | 0.317 | 0.33 | 0.34 | 0.25 | 0.16 | 0.63 | 0.47 |
| neither_s300 | 0.960 | 0.778 | 0.80 | 0.86 | — | — | 0.30 | 0.33 |
| neither_s301 | 0.951 | 0.833 | 0.85 | 0.85 | — | — | 0.58 | 0.80 |
| neither_s302 | 0.591 | 0.502 | 0.49 | 0.49 | — | — | 0.46 | 0.64 |
| neither_s303 | 0.584 | 0.460 | 0.47 | 0.44 | — | — | 0.58 | 0.71 |
| neither_s304 | 0.567 | 0.495 | 0.50 | 0.48 | — | — | 0.55 | 0.74 |
| neither_s305 | 0.979 | 0.757 | 0.79 | 0.82 | — | — | 0.60 | 0.44 |
| neither_s306 | 0.607 | 0.512 | 0.51 | 0.49 | — | — | 0.40 | 0.55 |
| neither_s307 | 0.758 | 0.640 | 0.66 | 0.63 | — | — | 0.53 | 0.60 |
| neither_s308 | 0.507 | 0.405 | 0.42 | 0.43 | — | — | 0.40 | 0.33 |
| neither_s309 | 0.616 | 0.511 | 0.43 | 0.47 | — | — | 0.42 | 0.47 |
| neither_s310 | 0.473 | 0.208 | 0.18 | 0.28 | — | — | 0.57 | 0.45 |
| neither_s311 | 0.924 | 0.768 | 0.80 | 0.83 | — | — | 0.38 | 0.34 |

## Scientific review (supervisor)

**Measured state.** mass_only: 0/12 fit (ID ≥ 0.95), mean ID 0.45, mean OOD 0.30, block IIA 0.33, unseen-source IIA 0.22, factorisation R²_add 0.84; acc_only: 0/12 fit (ID ≥ 0.95), mean ID 0.43, mean OOD 0.30, block IIA 0.35, unseen-source IIA 0.25, factorisation R²_add 0.94; neither: 3/12 fit (ID ≥ 0.95), mean ID 0.71, mean OOD 0.57.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 0.53 / 0.36 / 0.15 / 0.93; 311: 0.60 / 0.33 / 0.16 / 0.93; 305: 0.28 / 0.14 / 0.12 / 0.99; 304: 0.63 / 0.20 / 0.20 / 0.76; 310: 0.25 / 0.17 / 0.16 / 1.00.
**Events since previous cohort:** 10 — acc_only_s303: OOD jump 0.20->0.40; acc_only_s306: OOD jump 0.20->0.44; acc_only_s309: OOD jump 0.14->0.36; mass_only_s301: block carries unseen inputs (IIA-OOD >= 0.5); mass_only_s303: block carries unseen inputs (IIA-OOD >= 0.5); mass_only_s307: OOD jump 0.17->0.35; neither_s307: ID fell below 0.95, OOD jump 0.88->0.64; neither_s308: ID fell below 0.95, OOD jump 0.75->0.41; neither_s310: ID fell below 0.95, OOD jump 0.78->0.21; neither_s311: ID fell below 0.95.
**Interpretation.** Escape under way for the earliest declared networks. 304 leads (ID 0.50 at 800) and has already lost factorisation (0.93 → 0.84 by 850). Its unseen-mass accuracy does not rise with ID. 307 and 311 escape with unseen-mass accuracy rising together with ID. This is the first behavioural divergence.
**Anomaly investigated.** Anomaly check: 307 has a transient ID dip in the 10-step replay (0.84 at 1,100 → 0.58 at 1,150 → 0.97 at 1,200), coinciding with its factorisation recovery. Instability during reorganisation, not collapse.
**Previous prediction** ('304 and 310 stay ≤ 0.25 on unseen masses at the next cohort, whatever their ID (failures never rise on unseen masses)'): HELD (304/310 unseen-mass: [0.2, 0.17]).
**Prediction for step 1,000:** 304 and 310 stay ≤ 0.25 on unseen masses at the next cohort, whatever their ID (failures never rise on unseen masses).

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

