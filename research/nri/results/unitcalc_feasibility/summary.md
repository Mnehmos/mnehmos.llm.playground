# Unit-calculus feasibility probe (exploratory; 12 undeclared 'neither' networks, seeds 300-311)

Probe accuracy = held-out accuracy of a linear one-hot probe (mean over networks).

| site | m(raw) | mu(label) | M(canon) | a(raw) | au(label) | A(canon) | S | M*A | F |
|---|---|---|---|---|---|---|---|---|---|
| L0:mass_val | 1.00 | 0.43 | 0.48 | 0.09 | 0.32 | 0.09 | 0.33 | 0.16 | 0.16 |
| L0:mass_unit | 0.09 | 1.00 | 0.09 | 0.10 | 0.32 | 0.10 | 0.33 | 0.17 | 0.17 |
| L0:acc_val | 0.09 | 0.34 | 0.08 | 1.00 | 0.42 | 0.48 | 0.33 | 0.16 | 0.16 |
| L0:acc_unit | 0.09 | 0.33 | 0.09 | 0.09 | 1.00 | 0.09 | 0.33 | 0.17 | 0.17 |
| L0:out_unit | 0.09 | 0.34 | 0.09 | 0.09 | 0.33 | 0.09 | 1.00 | 0.17 | 0.17 |
| L0:query | 0.10 | 0.33 | 0.09 | 0.09 | 0.34 | 0.09 | 0.32 | 0.17 | 0.17 |
| L1:mass_val | 1.00 | 0.88 | 0.72 | 0.92 | 0.91 | 0.50 | 0.85 | 0.28 | 0.26 |
| L1:mass_unit | 0.57 | 1.00 | 0.51 | 0.58 | 0.96 | 0.39 | 0.94 | 0.28 | 0.26 |
| L1:acc_val | 0.92 | 0.88 | 0.53 | 1.00 | 0.91 | 0.70 | 0.85 | 0.28 | 0.26 |
| L1:acc_unit | 0.60 | 0.95 | 0.44 | 0.57 | 1.00 | 0.49 | 0.93 | 0.28 | 0.26 |
| L1:out_unit | 0.55 | 0.93 | 0.43 | 0.54 | 0.93 | 0.39 | 1.00 | 0.27 | 0.26 |
| L1:query | 0.57 | 0.95 | 0.47 | 0.57 | 0.96 | 0.44 | 0.92 | 0.30 | 0.28 |
| L2:mass_val | 1.00 | 0.94 | 0.74 | 0.91 | 0.92 | 0.52 | 0.93 | 0.31 | 0.30 |
| L2:mass_unit | 0.59 | 1.00 | 0.52 | 0.59 | 0.95 | 0.42 | 0.96 | 0.35 | 0.32 |
| L2:acc_val | 0.91 | 0.94 | 0.55 | 1.00 | 0.93 | 0.70 | 0.93 | 0.31 | 0.30 |
| L2:acc_unit | 0.59 | 0.97 | 0.46 | 0.60 | 1.00 | 0.48 | 0.96 | 0.34 | 0.31 |
| L2:out_unit | 0.56 | 0.97 | 0.46 | 0.56 | 0.97 | 0.41 | 1.00 | 0.34 | 0.32 |
| L2:query | 0.57 | 0.98 | 0.48 | 0.57 | 0.99 | 0.43 | 0.96 | 0.41 | 0.41 |
| L3:query(post-MLP) | 0.55 | 0.95 | 0.44 | 0.55 | 0.95 | 0.41 | 0.93 | 0.49 | 0.89 |

Character spectrum of class means: share of variance at frequencies 1..6 of the multiplicative
characters (angle 2*pi*f*log2(x)/12) vs the additive characters (angle 2*pi*f*x/13).

| site / quantity | multiplicative f=1..6 | additive f=1..6 |
|---|---|---|
| L1:mass_unit|M(canon) | 0.36, 0.14, 0.09, 0.13, 0.16, 0.20 | 0.09, 0.08, 0.20, 0.07, 0.51, 0.13 |
| L2:query|M(canon) | 0.24, 0.19, 0.06, 0.16, 0.27, 0.16 | 0.15, 0.15, 0.19, 0.12, 0.28, 0.20 |
| L2:query|A(canon) | 0.31, 0.29, 0.07, 0.04, 0.26, 0.09 | 0.18, 0.22, 0.19, 0.18, 0.11, 0.19 |
| L2:query|M*A | 0.04, 0.52, 0.00, 0.01, 0.42, 0.07 | 0.21, 0.17, 0.19, 0.17, 0.17, 0.19 |
| L3:query(post-MLP)|F | 0.06, 0.51, 0.00, 0.00, 0.41, 0.05 | 0.18, 0.19, 0.19, 0.18, 0.19, 0.17 |
| W_U|F | 0.15, 0.40, 0.03, 0.05, 0.36, 0.07 | 0.18, 0.19, 0.18, 0.18, 0.18, 0.18 |

Unit trade: share of activation variance explained by canonical M vs by the raw (value, unit) pair.

| site | canonical M | raw (value, unit) |
|---|---|---|
| L1:mass_val | 0.22 | 0.53 |
| L1:mass_unit | 0.09 | 0.58 |
| L2:mass_unit | 0.11 | 0.39 |
| L2:query | 0.13 | 0.30 |
