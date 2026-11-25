# Cohort review — step 1,100

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.561 | 0.384 | 1 | 0.472 | 0.313 |
| acc_only | 12 | 0.559 | 0.402 | 2 | 0.519 | 0.400 |
| neither | 12 | 0.824 | 0.686 | 6 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.830 | 0.705 | 0.72 | 0.69 | 0.74 | 0.64 | 0.36 | 0.59 |
| mass_only_s311 | focus | 0.571 | 0.419 | 0.35 | 0.53 | 0.46 | 0.36 | 0.38 | 0.31 |
| mass_only_s304 | comparator | 0.963 | 0.271 | 0.11 | 0.49 | 0.93 | 0.10 | 0.61 | 0.72 |
| mass_only_s310 | comparator | 0.259 | 0.144 | 0.15 | 0.16 | 0.24 | 0.15 | 0.95 | 0.89 |
| mass_only_s305 | intermediate | 0.389 | 0.248 | 0.25 | 0.28 | 0.24 | 0.15 | 0.78 | 0.92 |

## Events since step 1,000

- **acc_only_s307**: ID >= 0.95; declared block formed (IIA >= 0.9)
- **acc_only_s308**: ID >= 0.95; block carries unseen inputs (IIA-OOD >= 0.5); OOD jump 0.19->0.79
- **mass_only_s304**: ID >= 0.95; declared block formed (IIA >= 0.9)
- **mass_only_s307**: block carries unseen inputs (IIA-OOD >= 0.5); OOD jump 0.49->0.71
- **mass_only_s308**: block carries unseen inputs (IIA-OOD >= 0.5); OOD jump 0.35->0.52
- **neither_s301**: ID fell below 0.95

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 0.676 | 0.316 | 0.34 | 0.34 | 0.59 | 0.31 | 0.42 | 0.77 |
| acc_only_s301 | 0.582 | 0.520 | 0.52 | 0.50 | 0.58 | 0.49 | 0.79 | 0.61 |
| acc_only_s302 | 0.252 | 0.142 | 0.16 | 0.15 | 0.24 | 0.16 | 0.89 | 0.96 |
| acc_only_s303 | 0.590 | 0.521 | 0.50 | 0.52 | 0.58 | 0.52 | 0.41 | 0.60 |
| acc_only_s304 | 0.250 | 0.149 | 0.16 | 0.16 | 0.23 | 0.16 | 0.94 | 0.95 |
| acc_only_s305 | 0.290 | 0.134 | 0.13 | 0.16 | 0.23 | 0.14 | 0.70 | 0.81 |
| acc_only_s306 | 0.594 | 0.506 | 0.52 | 0.49 | 0.58 | 0.49 | 0.89 | 0.94 |
| acc_only_s307 | 0.977 | 0.621 | 0.47 | 0.85 | 0.91 | 0.74 | 0.44 | 0.58 |
| acc_only_s308 | 0.960 | 0.791 | 0.84 | 0.83 | 0.89 | 0.73 | 0.40 | 0.31 |
| acc_only_s309 | 0.594 | 0.514 | 0.51 | 0.53 | 0.56 | 0.50 | 0.59 | 0.54 |
| acc_only_s310 | 0.361 | 0.152 | 0.16 | 0.19 | 0.25 | 0.16 | 0.80 | 0.77 |
| acc_only_s311 | 0.586 | 0.454 | 0.52 | 0.41 | 0.59 | 0.40 | 0.87 | 0.77 |
| mass_only_s300 | 0.689 | 0.238 | 0.27 | 0.22 | 0.55 | 0.25 | 0.44 | 0.86 |
| mass_only_s301 | 0.601 | 0.504 | 0.51 | 0.50 | 0.59 | 0.53 | 0.82 | 0.54 |
| mass_only_s302 | 0.260 | 0.166 | 0.17 | 0.17 | 0.25 | 0.16 | 0.72 | 0.75 |
| mass_only_s303 | 0.589 | 0.521 | 0.52 | 0.52 | 0.59 | 0.53 | 0.78 | 0.76 |
| mass_only_s304 | 0.963 | 0.271 | 0.11 | 0.49 | 0.93 | 0.10 | 0.61 | 0.72 |
| mass_only_s305 | 0.389 | 0.248 | 0.25 | 0.28 | 0.24 | 0.15 | 0.78 | 0.92 |
| mass_only_s306 | 0.418 | 0.338 | 0.39 | 0.31 | 0.25 | 0.18 | 0.65 | 0.84 |
| mass_only_s307 | 0.830 | 0.705 | 0.72 | 0.69 | 0.74 | 0.64 | 0.36 | 0.59 |
| mass_only_s308 | 0.584 | 0.521 | 0.50 | 0.50 | 0.57 | 0.51 | 0.90 | 0.49 |
| mass_only_s309 | 0.580 | 0.530 | 0.52 | 0.51 | 0.25 | 0.17 | 0.43 | 0.40 |
| mass_only_s310 | 0.259 | 0.144 | 0.15 | 0.16 | 0.24 | 0.15 | 0.95 | 0.89 |
| mass_only_s311 | 0.571 | 0.419 | 0.35 | 0.53 | 0.46 | 0.36 | 0.38 | 0.31 |
| neither_s300 | 0.996 | 0.837 | 0.87 | 0.89 | — | — | 0.27 | 0.30 |
| neither_s301 | 0.940 | 0.769 | 0.82 | 0.78 | — | — | 0.65 | 0.81 |
| neither_s302 | 0.601 | 0.532 | 0.49 | 0.49 | — | — | 0.50 | 0.67 |
| neither_s303 | 0.579 | 0.488 | 0.50 | 0.52 | — | — | 0.59 | 0.72 |
| neither_s304 | 0.516 | 0.434 | 0.45 | 0.41 | — | — | 0.41 | 0.62 |
| neither_s305 | 0.998 | 0.768 | 0.80 | 0.83 | — | — | 0.62 | 0.45 |
| neither_s306 | 0.604 | 0.521 | 0.50 | 0.49 | — | — | 0.41 | 0.57 |
| neither_s307 | 1.000 | 0.922 | 0.96 | 0.95 | — | — | 0.52 | 0.63 |
| neither_s308 | 1.000 | 0.808 | 0.85 | 0.83 | — | — | 0.43 | 0.33 |
| neither_s309 | 0.662 | 0.453 | 0.34 | 0.45 | — | — | 0.48 | 0.49 |
| neither_s310 | 0.997 | 0.816 | 0.86 | 0.89 | — | — | 0.56 | 0.44 |
| neither_s311 | 0.996 | 0.886 | 0.92 | 0.92 | — | — | 0.39 | 0.35 |
