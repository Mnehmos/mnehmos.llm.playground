# Cohort review — step 2,400

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.796 | 0.554 | 6 | 0.793 | 0.509 |
| acc_only | 12 | 0.761 | 0.520 | 5 | 0.753 | 0.533 |
| neither | 12 | 0.934 | 0.816 | 10 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.999 | 0.995 | 0.99 | 1.00 | 1.00 | 0.97 | 0.32 | 0.56 |
| mass_only_s311 | focus | 1.000 | 0.982 | 1.00 | 0.99 | 1.00 | 0.99 | 0.26 | 0.34 |
| mass_only_s304 | comparator | 0.999 | 0.250 | 0.09 | 0.48 | 1.00 | 0.09 | 0.62 | 0.67 |
| mass_only_s310 | comparator | 1.000 | 0.258 | 0.09 | 0.48 | 1.00 | 0.09 | 0.73 | 0.82 |
| mass_only_s305 | intermediate | 1.000 | 0.755 | 0.54 | 1.00 | 0.99 | 0.42 | 0.39 | 0.58 |

## Events since step 2,300

- **acc_only_s301**: block LOST unseen inputs (IIA-OOD < 0.5)
- **acc_only_s306**: declared block formed (IIA >= 0.9)
- **mass_only_s303**: block carries unseen inputs (IIA-OOD >= 0.5)
- **mass_only_s308**: block LOST unseen inputs (IIA-OOD < 0.5)

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 1.000 | 0.323 | 0.32 | 0.42 | 1.00 | 0.43 | 0.59 | 0.91 |
| acc_only_s301 | 0.592 | 0.507 | 0.50 | 0.46 | 0.59 | 0.48 | 0.85 | 0.62 |
| acc_only_s302 | 0.588 | 0.525 | 0.53 | 0.54 | 0.57 | 0.53 | 0.61 | 0.40 |
| acc_only_s303 | 0.576 | 0.528 | 0.53 | 0.53 | 0.58 | 0.51 | 0.37 | 0.59 |
| acc_only_s304 | 0.252 | 0.139 | 0.16 | 0.15 | 0.23 | 0.16 | 0.90 | 0.94 |
| acc_only_s305 | 1.000 | 0.820 | 0.91 | 0.82 | 1.00 | 0.82 | 0.55 | 0.56 |
| acc_only_s306 | 0.932 | 0.462 | 0.40 | 0.57 | 0.92 | 0.55 | 0.82 | 0.93 |
| acc_only_s307 | 0.999 | 0.749 | 0.57 | 0.99 | 0.97 | 0.93 | 0.43 | 0.58 |
| acc_only_s308 | 1.000 | 0.915 | 0.98 | 0.89 | 1.00 | 0.92 | 0.36 | 0.25 |
| acc_only_s309 | 0.610 | 0.527 | 0.52 | 0.53 | 0.58 | 0.53 | 0.57 | 0.45 |
| acc_only_s310 | 1.000 | 0.280 | 0.44 | 0.14 | 1.00 | 0.15 | 0.77 | 0.94 |
| acc_only_s311 | 0.577 | 0.461 | 0.50 | 0.40 | 0.59 | 0.38 | 0.90 | 0.84 |
| mass_only_s300 | 0.998 | 0.361 | 0.52 | 0.25 | 0.99 | 0.48 | 0.38 | 0.76 |
| mass_only_s301 | 0.585 | 0.524 | 0.52 | 0.53 | 0.58 | 0.53 | 0.15 | 0.46 |
| mass_only_s302 | 0.596 | 0.469 | 0.45 | 0.44 | 0.60 | 0.47 | 0.96 | 0.76 |
| mass_only_s303 | 0.594 | 0.512 | 0.51 | 0.50 | 0.57 | 0.51 | 0.89 | 0.46 |
| mass_only_s304 | 0.999 | 0.250 | 0.09 | 0.48 | 1.00 | 0.09 | 0.62 | 0.67 |
| mass_only_s305 | 1.000 | 0.755 | 0.54 | 1.00 | 0.99 | 0.42 | 0.39 | 0.58 |
| mass_only_s306 | 0.586 | 0.467 | 0.52 | 0.43 | 0.59 | 0.52 | 0.98 | 0.81 |
| mass_only_s307 | 0.999 | 0.995 | 0.99 | 1.00 | 1.00 | 0.97 | 0.32 | 0.56 |
| mass_only_s308 | 0.602 | 0.545 | 0.50 | 0.49 | 0.60 | 0.49 | 0.92 | 0.48 |
| mass_only_s309 | 0.586 | 0.529 | 0.53 | 0.53 | 0.59 | 0.53 | 0.18 | 0.46 |
| mass_only_s310 | 1.000 | 0.258 | 0.09 | 0.48 | 1.00 | 0.09 | 0.73 | 0.82 |
| mass_only_s311 | 1.000 | 0.982 | 1.00 | 0.99 | 1.00 | 0.99 | 0.26 | 0.34 |
| neither_s300 | 1.000 | 0.859 | 0.88 | 0.91 | — | — | 0.26 | 0.28 |
| neither_s301 | 1.000 | 0.968 | 0.97 | 0.99 | — | — | 0.43 | 0.75 |
| neither_s302 | 0.987 | 0.812 | 0.90 | 0.84 | — | — | 0.49 | 0.47 |
| neither_s303 | 1.000 | 0.929 | 0.96 | 0.96 | — | — | 0.55 | 0.79 |
| neither_s304 | 0.601 | 0.523 | 0.50 | 0.50 | — | — | 0.27 | 0.57 |
| neither_s305 | 1.000 | 0.810 | 0.85 | 0.86 | — | — | 0.60 | 0.45 |
| neither_s306 | 0.617 | 0.518 | 0.42 | 0.46 | — | — | 0.48 | 0.55 |
| neither_s307 | 1.000 | 0.940 | 0.96 | 0.96 | — | — | 0.53 | 0.64 |
| neither_s308 | 1.000 | 0.815 | 0.86 | 0.83 | — | — | 0.43 | 0.34 |
| neither_s309 | 1.000 | 0.826 | 0.85 | 0.96 | — | — | 0.54 | 0.80 |
| neither_s310 | 1.000 | 0.880 | 0.91 | 0.92 | — | — | 0.53 | 0.39 |
| neither_s311 | 0.999 | 0.918 | 0.93 | 0.94 | — | — | 0.43 | 0.39 |
