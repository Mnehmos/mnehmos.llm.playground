# Cohort review — step 1,400

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.706 | 0.507 | 4 | 0.643 | 0.442 |
| acc_only | 12 | 0.698 | 0.448 | 4 | 0.670 | 0.451 |
| neither | 12 | 0.862 | 0.730 | 7 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.997 | 0.987 | 0.98 | 1.00 | 0.99 | 0.95 | 0.33 | 0.58 |
| mass_only_s311 | focus | 0.994 | 0.919 | 0.95 | 0.96 | 0.96 | 0.90 | 0.28 | 0.30 |
| mass_only_s304 | comparator | 0.976 | 0.307 | 0.17 | 0.49 | 0.97 | 0.15 | 0.62 | 0.69 |
| mass_only_s310 | comparator | 0.380 | 0.196 | 0.20 | 0.23 | 0.25 | 0.15 | 0.76 | 0.57 |
| mass_only_s305 | intermediate | 0.898 | 0.622 | 0.51 | 0.81 | 0.76 | 0.27 | 0.41 | 0.62 |

## Events since step 1,300

- **acc_only_s300**: ID >= 0.95; declared block formed (IIA >= 0.9)
- **acc_only_s303**: block LOST unseen inputs (IIA-OOD < 0.5); OOD jump 0.53->0.34
- **acc_only_s305**: ID >= 0.95; declared block formed (IIA >= 0.9); block carries unseen inputs (IIA-OOD >= 0.5); OOD jump 0.29->0.46
- **acc_only_s309**: block LOST unseen inputs (IIA-OOD < 0.5)
- **acc_only_s310**: declared block formed (IIA >= 0.9)
- **mass_only_s300**: declared block formed (IIA >= 0.9)
- **mass_only_s301**: block LOST unseen inputs (IIA-OOD < 0.5); OOD jump 0.50->0.15
- **mass_only_s303**: block LOST unseen inputs (IIA-OOD < 0.5)
- **mass_only_s305**: OOD jump 0.43->0.62
- **mass_only_s306**: block carries unseen inputs (IIA-OOD >= 0.5)
- **mass_only_s307**: ID >= 0.95; declared block formed (IIA >= 0.9); OOD jump 0.71->0.99
- **mass_only_s311**: declared block formed (IIA >= 0.9)
- **neither_s301**: ID fell below 0.95
- **neither_s309**: ID >= 0.95; OOD jump 0.65->0.81

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.312 | 0.33 | 0.39 | 1.00 | 0.38 | 0.53 | 0.87 |
| acc_only_s301 | 0.591 | 0.521 | 0.52 | 0.49 | 0.59 | 0.50 | 0.82 | 0.61 |
| acc_only_s302 | 0.340 | 0.222 | 0.24 | 0.22 | 0.24 | 0.16 | 0.79 | 0.76 |
| acc_only_s303 | 0.499 | 0.338 | 0.36 | 0.35 | 0.37 | 0.27 | 0.44 | 0.67 |
| acc_only_s304 | 0.241 | 0.144 | 0.16 | 0.15 | 0.24 | 0.16 | 0.92 | 0.94 |
| acc_only_s305 | 0.987 | 0.462 | 0.43 | 0.57 | 0.98 | 0.57 | 0.53 | 0.64 |
| acc_only_s306 | 0.596 | 0.500 | 0.52 | 0.49 | 0.58 | 0.49 | 0.90 | 0.94 |
| acc_only_s307 | 0.998 | 0.698 | 0.53 | 0.95 | 0.99 | 0.92 | 0.43 | 0.57 |
| acc_only_s308 | 0.994 | 0.879 | 0.93 | 0.87 | 0.97 | 0.85 | 0.39 | 0.28 |
| acc_only_s309 | 0.590 | 0.510 | 0.51 | 0.49 | 0.57 | 0.50 | 0.59 | 0.49 |
| acc_only_s310 | 0.949 | 0.312 | 0.48 | 0.16 | 0.94 | 0.19 | 0.79 | 0.93 |
| acc_only_s311 | 0.591 | 0.478 | 0.52 | 0.44 | 0.58 | 0.43 | 0.88 | 0.79 |
| mass_only_s300 | 0.995 | 0.356 | 0.48 | 0.26 | 0.93 | 0.47 | 0.39 | 0.78 |
| mass_only_s301 | 0.256 | 0.148 | 0.17 | 0.16 | 0.24 | 0.17 | 0.91 | 0.90 |
| mass_only_s302 | 0.599 | 0.490 | 0.51 | 0.46 | 0.58 | 0.52 | 0.95 | 0.75 |
| mass_only_s303 | 0.592 | 0.486 | 0.48 | 0.47 | 0.59 | 0.49 | 0.80 | 0.77 |
| mass_only_s304 | 0.976 | 0.307 | 0.17 | 0.49 | 0.97 | 0.15 | 0.62 | 0.69 |
| mass_only_s305 | 0.898 | 0.622 | 0.51 | 0.81 | 0.76 | 0.27 | 0.41 | 0.62 |
| mass_only_s306 | 0.584 | 0.512 | 0.51 | 0.48 | 0.60 | 0.53 | 0.97 | 0.78 |
| mass_only_s307 | 0.997 | 0.987 | 0.98 | 1.00 | 0.99 | 0.95 | 0.33 | 0.58 |
| mass_only_s308 | 0.588 | 0.532 | 0.54 | 0.53 | 0.60 | 0.53 | 0.91 | 0.48 |
| mass_only_s309 | 0.606 | 0.528 | 0.52 | 0.51 | 0.26 | 0.19 | 0.31 | 0.50 |
| mass_only_s310 | 0.380 | 0.196 | 0.20 | 0.23 | 0.25 | 0.15 | 0.76 | 0.57 |
| mass_only_s311 | 0.994 | 0.919 | 0.95 | 0.96 | 0.96 | 0.90 | 0.28 | 0.30 |
| neither_s300 | 1.000 | 0.845 | 0.87 | 0.89 | — | — | 0.27 | 0.30 |
| neither_s301 | 0.935 | 0.869 | 0.84 | 0.93 | — | — | 0.56 | 0.78 |
| neither_s302 | 0.598 | 0.505 | 0.46 | 0.46 | — | — | 0.43 | 0.62 |
| neither_s303 | 0.631 | 0.457 | 0.40 | 0.50 | — | — | 0.60 | 0.73 |
| neither_s304 | 0.595 | 0.522 | 0.51 | 0.51 | — | — | 0.30 | 0.54 |
| neither_s305 | 0.999 | 0.782 | 0.82 | 0.83 | — | — | 0.61 | 0.45 |
| neither_s306 | 0.610 | 0.508 | 0.45 | 0.47 | — | — | 0.43 | 0.57 |
| neither_s307 | 1.000 | 0.927 | 0.96 | 0.95 | — | — | 0.52 | 0.63 |
| neither_s308 | 1.000 | 0.797 | 0.85 | 0.81 | — | — | 0.42 | 0.33 |
| neither_s309 | 0.990 | 0.811 | 0.82 | 0.94 | — | — | 0.56 | 0.79 |
| neither_s310 | 0.997 | 0.868 | 0.92 | 0.92 | — | — | 0.56 | 0.44 |
| neither_s311 | 0.984 | 0.871 | 0.87 | 0.90 | — | — | 0.36 | 0.33 |
