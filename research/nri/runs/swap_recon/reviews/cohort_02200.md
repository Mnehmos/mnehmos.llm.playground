# Cohort review — step 2,200

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.761 | 0.518 | 6 | 0.740 | 0.471 |
| acc_only | 12 | 0.693 | 0.511 | 4 | 0.691 | 0.539 |
| neither | 12 | 0.918 | 0.793 | 9 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 1.000 | 0.992 | 0.98 | 1.00 | 1.00 | 0.96 | 0.32 | 0.57 |
| mass_only_s311 | focus | 1.000 | 0.980 | 0.98 | 1.00 | 1.00 | 0.97 | 0.28 | 0.36 |
| mass_only_s304 | comparator | 1.000 | 0.251 | 0.09 | 0.47 | 1.00 | 0.09 | 0.62 | 0.67 |
| mass_only_s310 | comparator | 1.000 | 0.243 | 0.10 | 0.44 | 1.00 | 0.09 | 0.73 | 0.82 |
| mass_only_s305 | intermediate | 1.000 | 0.818 | 0.68 | 1.00 | 0.99 | 0.54 | 0.39 | 0.58 |

## Events since step 2,100

- **acc_only_s310**: ID fell below 0.95
- **mass_only_s306**: block LOST unseen inputs (IIA-OOD < 0.5); OOD jump 0.50->0.23
- **neither_s302**: ID fell below 0.95; OOD jump 0.78->0.59

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.323 | 0.32 | 0.43 | 1.00 | 0.44 | 0.58 | 0.91 |
| acc_only_s301 | 0.601 | 0.512 | 0.52 | 0.47 | 0.60 | 0.48 | 0.84 | 0.62 |
| acc_only_s302 | 0.572 | 0.542 | 0.52 | 0.52 | 0.58 | 0.52 | 0.61 | 0.40 |
| acc_only_s303 | 0.593 | 0.530 | 0.52 | 0.52 | 0.59 | 0.52 | 0.37 | 0.60 |
| acc_only_s304 | 0.241 | 0.139 | 0.15 | 0.16 | 0.23 | 0.16 | 0.90 | 0.94 |
| acc_only_s305 | 1.000 | 0.817 | 0.90 | 0.81 | 1.00 | 0.81 | 0.56 | 0.57 |
| acc_only_s306 | 0.653 | 0.402 | 0.31 | 0.46 | 0.65 | 0.47 | 0.92 | 0.93 |
| acc_only_s307 | 1.000 | 0.736 | 0.59 | 0.98 | 1.00 | 0.96 | 0.41 | 0.57 |
| acc_only_s308 | 1.000 | 0.929 | 0.99 | 0.90 | 1.00 | 0.93 | 0.37 | 0.25 |
| acc_only_s309 | 0.574 | 0.528 | 0.53 | 0.51 | 0.59 | 0.52 | 0.57 | 0.46 |
| acc_only_s310 | 0.492 | 0.221 | 0.23 | 0.24 | 0.47 | 0.24 | 0.82 | 0.91 |
| acc_only_s311 | 0.592 | 0.456 | 0.51 | 0.40 | 0.59 | 0.42 | 0.90 | 0.84 |
| mass_only_s300 | 1.000 | 0.366 | 0.49 | 0.28 | 0.99 | 0.48 | 0.38 | 0.75 |
| mass_only_s301 | 0.594 | 0.536 | 0.53 | 0.53 | 0.59 | 0.54 | 0.15 | 0.46 |
| mass_only_s302 | 0.609 | 0.482 | 0.48 | 0.44 | 0.60 | 0.49 | 0.96 | 0.76 |
| mass_only_s303 | 0.371 | 0.281 | 0.33 | 0.28 | 0.26 | 0.20 | 0.73 | 0.61 |
| mass_only_s304 | 1.000 | 0.251 | 0.09 | 0.47 | 1.00 | 0.09 | 0.62 | 0.67 |
| mass_only_s305 | 1.000 | 0.818 | 0.68 | 1.00 | 0.99 | 0.54 | 0.39 | 0.58 |
| mass_only_s306 | 0.380 | 0.228 | 0.30 | 0.21 | 0.36 | 0.31 | 0.91 | 0.87 |
| mass_only_s307 | 1.000 | 0.992 | 0.98 | 1.00 | 1.00 | 0.96 | 0.32 | 0.57 |
| mass_only_s308 | 0.614 | 0.536 | 0.51 | 0.50 | 0.60 | 0.52 | 0.92 | 0.48 |
| mass_only_s309 | 0.562 | 0.504 | 0.53 | 0.52 | 0.49 | 0.47 | 0.19 | 0.45 |
| mass_only_s310 | 1.000 | 0.243 | 0.10 | 0.44 | 1.00 | 0.09 | 0.73 | 0.82 |
| mass_only_s311 | 1.000 | 0.980 | 0.98 | 1.00 | 1.00 | 0.97 | 0.28 | 0.36 |
| neither_s300 | 1.000 | 0.863 | 0.88 | 0.91 | — | — | 0.26 | 0.29 |
| neither_s301 | 0.994 | 0.942 | 0.96 | 0.97 | — | — | 0.43 | 0.75 |
| neither_s302 | 0.829 | 0.593 | 0.68 | 0.61 | — | — | 0.51 | 0.48 |
| neither_s303 | 0.998 | 0.925 | 0.95 | 0.96 | — | — | 0.56 | 0.79 |
| neither_s304 | 0.600 | 0.520 | 0.50 | 0.49 | — | — | 0.27 | 0.57 |
| neither_s305 | 1.000 | 0.808 | 0.85 | 0.86 | — | — | 0.60 | 0.45 |
| neither_s306 | 0.604 | 0.515 | 0.45 | 0.46 | — | — | 0.48 | 0.54 |
| neither_s307 | 1.000 | 0.940 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.817 | 0.86 | 0.83 | — | — | 0.43 | 0.33 |
| neither_s309 | 1.000 | 0.823 | 0.85 | 0.96 | — | — | 0.54 | 0.80 |
| neither_s310 | 1.000 | 0.876 | 0.90 | 0.92 | — | — | 0.53 | 0.39 |
| neither_s311 | 0.998 | 0.896 | 0.91 | 0.93 | — | — | 0.40 | 0.36 |

## Scientific review (supervisor)

**Measured state.** mass_only: 6/12 fit (ID ≥ 0.95), mean ID 0.76, mean OOD 0.50, block IIA 0.74, unseen-source IIA 0.47, factorisation R²_add 0.76; acc_only: 4/12 fit (ID ≥ 0.95), mean ID 0.69, mean OOD 0.53, block IIA 0.69, unseen-source IIA 0.54, factorisation R²_add 0.78; neither: 9/12 fit (ID ≥ 0.95), mean ID 0.92, mean OOD 0.79.
**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): 307: 1.00 / 0.98 / 0.96 / 1.00; 311: 1.00 / 0.98 / 0.97 / 0.97; 305: 1.00 / 0.68 / 0.54 / 0.97; 304: 1.00 / 0.09 / 0.09 / 0.71; 310: 1.00 / 0.10 / 0.09 / 0.70.
**Events since previous cohort:** 3 — acc_only_s310: ID fell below 0.95; mass_only_s306: block LOST unseen inputs (IIA-OOD < 0.5), OOD jump 0.50->0.23; neither_s302: ID fell below 0.95, OOD jump 0.78->0.59.
**Interpretation.** Post-fit consolidation. All five focus networks fit the training data. Group means change little. The mass group's mean factorisation is about 0.74 and stays nearly fixed thereafter: with weight decay 0.01 and losses near zero, no measurable pressure reverses a locked-in interaction. 305's partial generalisation (OOD-mass about 0.75 at 1,800) starts to erode.
**Previous prediction** ('no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort'): HELD (changes in unseen-mass accuracy: {307: -0.01, 311: 0.0, 305: -0.0, 304: 0.0, 310: 0.01}).
**Prediction for step 2,300:** no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort.

*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an experimental demonstration (doc 46 labels).*

