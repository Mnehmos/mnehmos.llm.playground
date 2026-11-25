# Cohort review — step 1,600

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.743 | 0.527 | 5 | 0.693 | 0.445 |
| acc_only | 12 | 0.716 | 0.487 | 5 | 0.703 | 0.478 |
| neither | 12 | 0.859 | 0.733 | 7 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.997 | 0.988 | 0.98 | 1.00 | 0.98 | 0.94 | 0.33 | 0.57 |
| mass_only_s311 | focus | 0.998 | 0.973 | 0.97 | 0.99 | 0.98 | 0.93 | 0.28 | 0.31 |
| mass_only_s304 | comparator | 0.999 | 0.248 | 0.09 | 0.48 | 1.00 | 0.09 | 0.62 | 0.70 |
| mass_only_s310 | comparator | 0.656 | 0.165 | 0.10 | 0.26 | 0.61 | 0.10 | 0.66 | 0.67 |
| mass_only_s305 | intermediate | 0.977 | 0.772 | 0.64 | 0.95 | 0.91 | 0.43 | 0.39 | 0.62 |

## Events since step 1,500

- **acc_only_s303**: block carries unseen inputs (IIA-OOD >= 0.5)
- **acc_only_s305**: OOD jump 0.51->0.69
- **acc_only_s306**: block carries unseen inputs (IIA-OOD >= 0.5)
- **mass_only_s300**: block LOST unseen inputs (IIA-OOD < 0.5)
- **mass_only_s303**: block LOST unseen inputs (IIA-OOD < 0.5)
- **mass_only_s305**: ID >= 0.95; declared block formed (IIA >= 0.9); OOD jump 0.61->0.77
- **mass_only_s306**: block LOST unseen inputs (IIA-OOD < 0.5)
- **neither_s301**: ID fell below 0.95

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.374 | 0.41 | 0.42 | 1.00 | 0.43 | 0.54 | 0.89 |
| acc_only_s301 | 0.583 | 0.521 | 0.50 | 0.50 | 0.59 | 0.51 | 0.82 | 0.61 |
| acc_only_s302 | 0.429 | 0.325 | 0.37 | 0.30 | 0.36 | 0.27 | 0.66 | 0.56 |
| acc_only_s303 | 0.589 | 0.519 | 0.52 | 0.52 | 0.58 | 0.50 | 0.36 | 0.59 |
| acc_only_s304 | 0.250 | 0.142 | 0.16 | 0.16 | 0.23 | 0.16 | 0.92 | 0.95 |
| acc_only_s305 | 0.988 | 0.688 | 0.82 | 0.64 | 0.98 | 0.58 | 0.58 | 0.59 |
| acc_only_s306 | 0.597 | 0.522 | 0.52 | 0.51 | 0.58 | 0.52 | 0.90 | 0.94 |
| acc_only_s307 | 1.000 | 0.706 | 0.55 | 0.97 | 0.99 | 0.90 | 0.41 | 0.57 |
| acc_only_s308 | 0.996 | 0.892 | 0.94 | 0.88 | 0.99 | 0.88 | 0.38 | 0.27 |
| acc_only_s309 | 0.588 | 0.520 | 0.53 | 0.53 | 0.58 | 0.52 | 0.57 | 0.46 |
| acc_only_s310 | 0.980 | 0.200 | 0.28 | 0.13 | 0.97 | 0.14 | 0.81 | 0.93 |
| acc_only_s311 | 0.591 | 0.432 | 0.52 | 0.34 | 0.58 | 0.33 | 0.87 | 0.81 |
| mass_only_s300 | 0.989 | 0.381 | 0.51 | 0.31 | 0.94 | 0.48 | 0.39 | 0.79 |
| mass_only_s301 | 0.350 | 0.235 | 0.28 | 0.25 | 0.25 | 0.17 | 0.65 | 0.78 |
| mass_only_s302 | 0.590 | 0.499 | 0.54 | 0.47 | 0.59 | 0.51 | 0.95 | 0.75 |
| mass_only_s303 | 0.609 | 0.499 | 0.46 | 0.48 | 0.60 | 0.48 | 0.81 | 0.77 |
| mass_only_s304 | 0.999 | 0.248 | 0.09 | 0.48 | 1.00 | 0.09 | 0.62 | 0.70 |
| mass_only_s305 | 0.977 | 0.772 | 0.64 | 0.95 | 0.91 | 0.43 | 0.39 | 0.62 |
| mass_only_s306 | 0.576 | 0.495 | 0.48 | 0.45 | 0.59 | 0.48 | 0.97 | 0.78 |
| mass_only_s307 | 0.997 | 0.988 | 0.98 | 1.00 | 0.98 | 0.94 | 0.33 | 0.57 |
| mass_only_s308 | 0.594 | 0.533 | 0.52 | 0.51 | 0.60 | 0.52 | 0.92 | 0.48 |
| mass_only_s309 | 0.584 | 0.539 | 0.52 | 0.54 | 0.28 | 0.21 | 0.24 | 0.42 |
| mass_only_s310 | 0.656 | 0.165 | 0.10 | 0.26 | 0.61 | 0.10 | 0.66 | 0.67 |
| mass_only_s311 | 0.998 | 0.973 | 0.97 | 0.99 | 0.98 | 0.93 | 0.28 | 0.31 |
| neither_s300 | 1.000 | 0.841 | 0.86 | 0.89 | — | — | 0.26 | 0.28 |
| neither_s301 | 0.924 | 0.904 | 0.92 | 0.91 | — | — | 0.59 | 0.72 |
| neither_s302 | 0.616 | 0.502 | 0.50 | 0.52 | — | — | 0.46 | 0.61 |
| neither_s303 | 0.619 | 0.490 | 0.45 | 0.54 | — | — | 0.54 | 0.68 |
| neither_s304 | 0.594 | 0.525 | 0.51 | 0.52 | — | — | 0.27 | 0.56 |
| neither_s305 | 1.000 | 0.789 | 0.83 | 0.83 | — | — | 0.61 | 0.45 |
| neither_s306 | 0.582 | 0.522 | 0.48 | 0.48 | — | — | 0.44 | 0.56 |
| neither_s307 | 1.000 | 0.930 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.804 | 0.85 | 0.82 | — | — | 0.43 | 0.33 |
| neither_s309 | 0.998 | 0.812 | 0.84 | 0.96 | — | — | 0.55 | 0.81 |
| neither_s310 | 0.982 | 0.762 | 0.79 | 0.82 | — | — | 0.57 | 0.45 |
| neither_s311 | 0.997 | 0.911 | 0.93 | 0.94 | — | — | 0.39 | 0.35 |
