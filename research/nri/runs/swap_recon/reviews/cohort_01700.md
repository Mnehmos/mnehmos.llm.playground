# Cohort review — step 1,700

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.744 | 0.503 | 5 | 0.690 | 0.425 |
| acc_only | 12 | 0.696 | 0.467 | 4 | 0.680 | 0.456 |
| neither | 12 | 0.867 | 0.736 | 7 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.980 | 0.953 | 0.94 | 0.99 | 0.97 | 0.92 | 0.33 | 0.56 |
| mass_only_s311 | focus | 0.984 | 0.943 | 0.94 | 0.98 | 0.96 | 0.89 | 0.26 | 0.31 |
| mass_only_s304 | comparator | 0.998 | 0.259 | 0.09 | 0.48 | 1.00 | 0.09 | 0.61 | 0.68 |
| mass_only_s310 | comparator | 0.914 | 0.243 | 0.13 | 0.39 | 0.90 | 0.15 | 0.65 | 0.77 |
| mass_only_s305 | intermediate | 0.981 | 0.783 | 0.68 | 0.95 | 0.93 | 0.49 | 0.40 | 0.59 |

## Events since step 1,600

- **acc_only_s301**: block LOST unseen inputs (IIA-OOD < 0.5)
- **acc_only_s302**: OOD jump 0.33->0.48
- **acc_only_s306**: block LOST unseen inputs (IIA-OOD < 0.5)
- **acc_only_s308**: ID fell below 0.95; block LOST unseen inputs (IIA-OOD < 0.5); OOD jump 0.89->0.47
- **mass_only_s303**: OOD jump 0.50->0.15
- **mass_only_s306**: block carries unseen inputs (IIA-OOD >= 0.5)

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.375 | 0.42 | 0.41 | 1.00 | 0.43 | 0.55 | 0.89 |
| acc_only_s301 | 0.577 | 0.425 | 0.51 | 0.36 | 0.54 | 0.34 | 0.78 | 0.65 |
| acc_only_s302 | 0.544 | 0.480 | 0.52 | 0.48 | 0.52 | 0.46 | 0.64 | 0.45 |
| acc_only_s303 | 0.594 | 0.541 | 0.52 | 0.51 | 0.58 | 0.51 | 0.37 | 0.61 |
| acc_only_s304 | 0.249 | 0.144 | 0.16 | 0.16 | 0.24 | 0.17 | 0.90 | 0.96 |
| acc_only_s305 | 0.999 | 0.770 | 0.87 | 0.75 | 1.00 | 0.71 | 0.57 | 0.58 |
| acc_only_s306 | 0.590 | 0.490 | 0.51 | 0.49 | 0.58 | 0.48 | 0.90 | 0.94 |
| acc_only_s307 | 0.998 | 0.735 | 0.62 | 0.97 | 0.98 | 0.92 | 0.42 | 0.57 |
| acc_only_s308 | 0.673 | 0.470 | 0.46 | 0.51 | 0.56 | 0.40 | 0.42 | 0.32 |
| acc_only_s309 | 0.552 | 0.514 | 0.54 | 0.53 | 0.58 | 0.52 | 0.57 | 0.47 |
| acc_only_s310 | 1.000 | 0.217 | 0.30 | 0.15 | 1.00 | 0.15 | 0.81 | 0.94 |
| acc_only_s311 | 0.579 | 0.445 | 0.54 | 0.39 | 0.58 | 0.37 | 0.89 | 0.83 |
| mass_only_s300 | 0.996 | 0.326 | 0.48 | 0.22 | 0.97 | 0.45 | 0.39 | 0.77 |
| mass_only_s301 | 0.441 | 0.308 | 0.36 | 0.30 | 0.24 | 0.17 | 0.60 | 0.75 |
| mass_only_s302 | 0.585 | 0.485 | 0.52 | 0.47 | 0.59 | 0.52 | 0.96 | 0.74 |
| mass_only_s303 | 0.260 | 0.154 | 0.16 | 0.15 | 0.24 | 0.16 | 0.86 | 0.92 |
| mass_only_s304 | 0.998 | 0.259 | 0.09 | 0.48 | 1.00 | 0.09 | 0.61 | 0.68 |
| mass_only_s305 | 0.981 | 0.783 | 0.68 | 0.95 | 0.93 | 0.49 | 0.40 | 0.59 |
| mass_only_s306 | 0.592 | 0.515 | 0.52 | 0.48 | 0.59 | 0.52 | 0.97 | 0.78 |
| mass_only_s307 | 0.980 | 0.953 | 0.94 | 0.99 | 0.97 | 0.92 | 0.33 | 0.56 |
| mass_only_s308 | 0.603 | 0.530 | 0.49 | 0.49 | 0.60 | 0.52 | 0.91 | 0.48 |
| mass_only_s309 | 0.593 | 0.536 | 0.52 | 0.53 | 0.30 | 0.23 | 0.25 | 0.43 |
| mass_only_s310 | 0.914 | 0.243 | 0.13 | 0.39 | 0.90 | 0.15 | 0.65 | 0.77 |
| mass_only_s311 | 0.984 | 0.943 | 0.94 | 0.98 | 0.96 | 0.89 | 0.26 | 0.31 |
| neither_s300 | 0.997 | 0.833 | 0.86 | 0.88 | — | — | 0.26 | 0.28 |
| neither_s301 | 0.945 | 0.867 | 0.90 | 0.87 | — | — | 0.55 | 0.73 |
| neither_s302 | 0.648 | 0.537 | 0.57 | 0.60 | — | — | 0.44 | 0.57 |
| neither_s303 | 0.633 | 0.450 | 0.37 | 0.53 | — | — | 0.55 | 0.70 |
| neither_s304 | 0.578 | 0.520 | 0.53 | 0.49 | — | — | 0.27 | 0.57 |
| neither_s305 | 1.000 | 0.783 | 0.83 | 0.83 | — | — | 0.61 | 0.45 |
| neither_s306 | 0.609 | 0.517 | 0.50 | 0.49 | — | — | 0.50 | 0.55 |
| neither_s307 | 1.000 | 0.929 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.809 | 0.86 | 0.82 | — | — | 0.43 | 0.33 |
| neither_s309 | 1.000 | 0.825 | 0.85 | 0.96 | — | — | 0.55 | 0.80 |
| neither_s310 | 0.999 | 0.850 | 0.87 | 0.89 | — | — | 0.55 | 0.42 |
| neither_s311 | 0.995 | 0.910 | 0.93 | 0.94 | — | — | 0.40 | 0.36 |
