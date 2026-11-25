# Cohort review — step 600

Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.

| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |
|---|---|---|---|---|---|---|
| mass_only | 12 | 0.314 | 0.203 | 0 | 0.242 | 0.160 |
| acc_only | 12 | 0.302 | 0.195 | 0 | 0.236 | 0.159 |
| neither | 12 | 0.587 | 0.425 | 0 | — | — |

## Focus, comparator and intermediate networks (mass_only)

| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|---|
| mass_only_s307 | focus | 0.258 | 0.142 | 0.14 | 0.14 | 0.24 | 0.15 | 0.96 | 0.96 |
| mass_only_s311 | focus | 0.272 | 0.138 | 0.14 | 0.17 | 0.24 | 0.16 | 0.91 | 0.88 |
| mass_only_s304 | comparator | 0.260 | 0.142 | 0.16 | 0.16 | 0.24 | 0.16 | 0.87 | 0.93 |
| mass_only_s310 | comparator | 0.256 | 0.134 | 0.15 | 0.15 | 0.24 | 0.16 | 0.96 | 0.96 |
| mass_only_s305 | intermediate | 0.243 | 0.138 | 0.16 | 0.15 | 0.24 | 0.16 | 0.97 | 0.95 |

## Events since step 500

- **acc_only_s301**: OOD jump 0.18->0.45
- **acc_only_s303**: OOD jump 0.20->0.44
- **mass_only_s301**: OOD jump 0.24->0.43
- **mass_only_s303**: OOD jump 0.16->0.39
- **mass_only_s309**: OOD jump 0.14->0.32
- **neither_s300**: OOD jump 0.14->0.37
- **neither_s301**: OOD jump 0.43->0.60
- **neither_s302**: OOD jump 0.32->0.48
- **neither_s305**: OOD jump 0.14->0.42
- **neither_s306**: OOD jump 0.14->0.31
- **neither_s309**: OOD jump 0.34->0.53
- **neither_s310**: OOD jump 0.27->0.64
- **neither_s311**: OOD jump 0.28->0.63

## All networks

| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |
|---|---|---|---|---|---|---|---|---|
| acc_only_s300 | 0.250 | 0.141 | 0.15 | 0.16 | 0.24 | 0.16 | 0.92 | 0.95 |
| acc_only_s301 | 0.542 | 0.451 | 0.48 | 0.42 | 0.24 | 0.17 | 0.60 | 0.80 |
| acc_only_s302 | 0.254 | 0.142 | 0.15 | 0.14 | 0.23 | 0.15 | 0.94 | 0.95 |
| acc_only_s303 | 0.532 | 0.442 | 0.47 | 0.42 | 0.25 | 0.16 | 0.46 | 0.69 |
| acc_only_s304 | 0.258 | 0.147 | 0.15 | 0.15 | 0.23 | 0.15 | 0.95 | 0.95 |
| acc_only_s305 | 0.250 | 0.148 | 0.15 | 0.15 | 0.23 | 0.15 | 0.93 | 0.96 |
| acc_only_s306 | 0.249 | 0.146 | 0.15 | 0.15 | 0.24 | 0.17 | 0.98 | 0.95 |
| acc_only_s307 | 0.268 | 0.144 | 0.14 | 0.15 | 0.23 | 0.15 | 0.95 | 0.96 |
| acc_only_s308 | 0.269 | 0.144 | 0.15 | 0.16 | 0.23 | 0.16 | 0.88 | 0.90 |
| acc_only_s309 | 0.241 | 0.157 | 0.16 | 0.15 | 0.23 | 0.16 | 0.97 | 0.99 |
| acc_only_s310 | 0.258 | 0.143 | 0.15 | 0.15 | 0.23 | 0.15 | 0.97 | 0.96 |
| acc_only_s311 | 0.250 | 0.140 | 0.14 | 0.14 | 0.24 | 0.16 | 0.93 | 0.97 |
| mass_only_s300 | 0.252 | 0.150 | 0.16 | 0.17 | 0.23 | 0.16 | 0.97 | 0.97 |
| mass_only_s301 | 0.548 | 0.429 | 0.45 | 0.43 | 0.25 | 0.17 | 0.59 | 0.77 |
| mass_only_s302 | 0.258 | 0.153 | 0.16 | 0.16 | 0.24 | 0.16 | 0.95 | 0.96 |
| mass_only_s303 | 0.482 | 0.389 | 0.42 | 0.37 | 0.26 | 0.18 | 0.47 | 0.66 |
| mass_only_s304 | 0.260 | 0.142 | 0.16 | 0.16 | 0.24 | 0.16 | 0.87 | 0.93 |
| mass_only_s305 | 0.243 | 0.138 | 0.16 | 0.15 | 0.24 | 0.16 | 0.97 | 0.95 |
| mass_only_s306 | 0.256 | 0.144 | 0.14 | 0.15 | 0.24 | 0.15 | 0.93 | 0.94 |
| mass_only_s307 | 0.258 | 0.142 | 0.14 | 0.14 | 0.24 | 0.15 | 0.96 | 0.96 |
| mass_only_s308 | 0.279 | 0.156 | 0.18 | 0.16 | 0.24 | 0.16 | 0.85 | 0.89 |
| mass_only_s309 | 0.400 | 0.323 | 0.39 | 0.28 | 0.24 | 0.16 | 0.66 | 0.67 |
| mass_only_s310 | 0.256 | 0.134 | 0.15 | 0.15 | 0.24 | 0.16 | 0.96 | 0.96 |
| mass_only_s311 | 0.272 | 0.138 | 0.14 | 0.17 | 0.24 | 0.16 | 0.91 | 0.88 |
| neither_s300 | 0.559 | 0.375 | 0.41 | 0.43 | — | — | 0.40 | 0.45 |
| neither_s301 | 0.806 | 0.596 | 0.63 | 0.65 | — | — | 0.59 | 0.79 |
| neither_s302 | 0.566 | 0.483 | 0.52 | 0.45 | — | — | 0.31 | 0.61 |
| neither_s303 | 0.597 | 0.465 | 0.49 | 0.48 | — | — | 0.64 | 0.78 |
| neither_s304 | 0.361 | 0.231 | 0.30 | 0.21 | — | — | 0.93 | 0.96 |
| neither_s305 | 0.661 | 0.425 | 0.43 | 0.48 | — | — | 0.66 | 0.49 |
| neither_s306 | 0.396 | 0.315 | 0.38 | 0.26 | — | — | 0.78 | 0.89 |
| neither_s307 | 0.362 | 0.182 | 0.23 | 0.20 | — | — | 0.76 | 0.85 |
| neither_s308 | 0.402 | 0.227 | 0.21 | 0.30 | — | — | 0.57 | 0.46 |
| neither_s309 | 0.599 | 0.526 | 0.47 | 0.49 | — | — | 0.41 | 0.44 |
| neither_s310 | 0.882 | 0.643 | 0.69 | 0.69 | — | — | 0.59 | 0.48 |
| neither_s311 | 0.858 | 0.632 | 0.67 | 0.66 | — | — | 0.40 | 0.37 |
