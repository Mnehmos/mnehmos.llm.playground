# Cohort review — step 1,200

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.621 | 0.449 | 2 | 0.527 | 0.353 |
| acc_only | 12 | 0.604 | 0.435 | 2 | 0.564 | 0.435 |
| neither | 12 | 0.788 | 0.643 | 4 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.969 | 0.936 | 0.92 | 0.97 | 0.88 | 0.82 | 0.36 | 0.62 |
| mass_only_s311 | focus | 0.643 | 0.586 | 0.56 | 0.66 | 0.55 | 0.47 | 0.38 | 0.40 |
| mass_only_s304 | comparator | 0.998 | 0.305 | 0.12 | 0.52 | 0.99 | 0.11 | 0.62 | 0.70 |
| mass_only_s310 | comparator | 0.293 | 0.137 | 0.13 | 0.16 | 0.23 | 0.15 | 0.94 | 0.81 |
| mass_only_s305 | intermediate | 0.469 | 0.300 | 0.32 | 0.31 | 0.32 | 0.17 | 0.54 | 0.86 |

## Events since step 1,100

- **acc_only_s300**: declared block formed (IIA >= 0.9)
- **acc_only_s308**: declared block formed (IIA >= 0.9)
- **acc_only_s309**: block carries unseen inputs (IIA-OOD >= 0.5)
- **mass_only_s302**: OOD jump 0.17->0.34
- **mass_only_s307**: ID >= 0.95; OOD jump 0.71->0.94
- **mass_only_s311**: OOD jump 0.42->0.59
- **neither_s300**: ID fell below 0.95; OOD jump 0.84->0.67
- **neither_s301**: ID >= 0.95
- **neither_s305**: ID fell below 0.95
- **neither_s311**: ID fell below 0.95; OOD jump 0.89->0.61

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 0.949 | 0.418 | 0.44 | 0.47 | 0.92 | 0.43 | 0.47 | 0.82 |
| acc_only_s301 | 0.596 | 0.512 | 0.52 | 0.51 | 0.58 | 0.50 | 0.79 | 0.61 |
| acc_only_s302 | 0.268 | 0.160 | 0.18 | 0.17 | 0.23 | 0.16 | 0.88 | 0.95 |
| acc_only_s303 | 0.586 | 0.531 | 0.53 | 0.53 | 0.58 | 0.54 | 0.40 | 0.59 |
| acc_only_s304 | 0.253 | 0.143 | 0.16 | 0.16 | 0.23 | 0.16 | 0.93 | 0.94 |
| acc_only_s305 | 0.444 | 0.171 | 0.16 | 0.23 | 0.31 | 0.20 | 0.72 | 0.79 |
| acc_only_s306 | 0.591 | 0.511 | 0.52 | 0.49 | 0.58 | 0.49 | 0.89 | 0.94 |
| acc_only_s307 | 0.986 | 0.642 | 0.51 | 0.88 | 0.95 | 0.83 | 0.42 | 0.58 |
| acc_only_s308 | 0.987 | 0.832 | 0.85 | 0.85 | 0.94 | 0.80 | 0.38 | 0.29 |
| acc_only_s309 | 0.597 | 0.532 | 0.51 | 0.51 | 0.58 | 0.51 | 0.58 | 0.56 |
| acc_only_s310 | 0.399 | 0.296 | 0.27 | 0.37 | 0.26 | 0.18 | 0.66 | 0.74 |
| acc_only_s311 | 0.595 | 0.474 | 0.52 | 0.41 | 0.59 | 0.41 | 0.88 | 0.78 |
| mass_only_s300 | 0.833 | 0.322 | 0.37 | 0.31 | 0.73 | 0.33 | 0.40 | 0.80 |
| mass_only_s301 | 0.597 | 0.503 | 0.49 | 0.49 | 0.59 | 0.51 | 0.83 | 0.53 |
| mass_only_s302 | 0.399 | 0.335 | 0.38 | 0.30 | 0.32 | 0.28 | 0.71 | 0.65 |
| mass_only_s303 | 0.574 | 0.534 | 0.52 | 0.50 | 0.59 | 0.52 | 0.79 | 0.77 |
| mass_only_s304 | 0.998 | 0.305 | 0.12 | 0.52 | 0.99 | 0.11 | 0.62 | 0.70 |
| mass_only_s305 | 0.469 | 0.300 | 0.32 | 0.31 | 0.32 | 0.17 | 0.54 | 0.86 |
| mass_only_s306 | 0.492 | 0.388 | 0.40 | 0.36 | 0.31 | 0.21 | 0.75 | 0.85 |
| mass_only_s307 | 0.969 | 0.936 | 0.92 | 0.97 | 0.88 | 0.82 | 0.36 | 0.62 |
| mass_only_s308 | 0.600 | 0.517 | 0.50 | 0.50 | 0.57 | 0.51 | 0.89 | 0.49 |
| mass_only_s309 | 0.592 | 0.520 | 0.50 | 0.49 | 0.24 | 0.17 | 0.49 | 0.49 |
| mass_only_s310 | 0.293 | 0.137 | 0.13 | 0.16 | 0.23 | 0.15 | 0.94 | 0.81 |
| mass_only_s311 | 0.643 | 0.586 | 0.56 | 0.66 | 0.55 | 0.47 | 0.38 | 0.40 |
| neither_s300 | 0.855 | 0.665 | 0.68 | 0.74 | — | — | 0.34 | 0.39 |
| neither_s301 | 0.978 | 0.871 | 0.88 | 0.89 | — | — | 0.59 | 0.80 |
| neither_s302 | 0.605 | 0.521 | 0.49 | 0.48 | — | — | 0.50 | 0.66 |
| neither_s303 | 0.597 | 0.485 | 0.45 | 0.50 | — | — | 0.59 | 0.73 |
| neither_s304 | 0.490 | 0.384 | 0.41 | 0.37 | — | — | 0.35 | 0.57 |
| neither_s305 | 0.926 | 0.661 | 0.70 | 0.71 | — | — | 0.62 | 0.46 |
| neither_s306 | 0.597 | 0.516 | 0.49 | 0.49 | — | — | 0.42 | 0.57 |
| neither_s307 | 1.000 | 0.923 | 0.95 | 0.96 | — | — | 0.52 | 0.64 |
| neither_s308 | 1.000 | 0.794 | 0.84 | 0.81 | — | — | 0.43 | 0.33 |
| neither_s309 | 0.704 | 0.440 | 0.39 | 0.48 | — | — | 0.48 | 0.54 |
| neither_s310 | 0.999 | 0.849 | 0.90 | 0.91 | — | — | 0.57 | 0.45 |
| neither_s311 | 0.702 | 0.607 | 0.62 | 0.65 | — | — | 0.42 | 0.36 |
