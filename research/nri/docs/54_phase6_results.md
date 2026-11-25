# 54 — Results: Phase VI, neural compilation and mathematical construction

**Registration.** Doc 53 (commit 446bbeb), before any evaluation. Everything here is a deterministic construction
or a single-pass evaluation; no network was trained.

**Labels.** Every network is marked **constructed**, **learned**, **reconstructed** or **hybrid**.

**Data and code.**
- Data: `runs/phase6/*.json`.
- Code:
  - `sci/construct.py`: the designs;
  - `sci/construct_qualify.py`: qualification;
  - `sci/instructions.py`: the instruction system;
  - `sci/residual.py`: the residual analysis;
  - `sci/compare_hybrid.py`: comparison and hybrids;
  - `sci/compiler.py`: the general compiler.

## Plain-language summary

**Yes: a mathematical specification can be compiled directly into the weights of an ordinary network, without
training.**
- The task's arithmetic (multiplication mod 13, with unit conversion) was turned into weights for the *same small
  transformer* the learned networks use: 64 dimensions, 4 attention heads, a 256-unit hidden layer.
- The resulting network answers every one of the 9,126 possible prompts correctly, including the 1,776 the learned
  networks never saw. It handles zero, and it represents every equivalent way of writing the same mass identically.
- Its internal arithmetic is exact in the strict sense: every internal number it computes in float32 equals the
  mathematical value bit for bit. In true real-number arithmetic the attention softmax is a ~10⁻¹⁰⁷ approximation,
  and the answer is still provably correct.

**The constructed network is programmable at the level of its algorithm.** Each component sits in its own weights,
and each can be rewritten exactly, independently and reversibly:

| component | what it does |
|---|---|
| operation | ×, squared, ÷, +, − |
| unit conversion | each unit's scale |
| operand binding | which value is the mass |
| output decoding | how the result is reported |

Fifty random combinations of these edits, together with the earlier token edits, were all exact.

**A single hidden layer can do exact modular arithmetic only through the right number system.**
- Multiplication becomes addition of *integers* (discrete logarithms). The "wrap-around" is done with a small ladder
  of ReLU units: about 30–80 units instead of a 2,197-entry table.
- Adding two phase *vectors* is not adding the phases. Learned networks that carry phase vectors must approximate the
  wrap-around.

**What the construction teaches about the learned networks.**
- **The unexplained 10–17% is essential.** Removing it makes even the perfect generalisers fail on *familiar* prompts.
- **It is mostly pairwise token interactions**, made before the decoder, and different in each network:
  - in 307, mass value × acceleration value: a partial product formed in attention;
  - in 311, acceleration value × acceleration unit: conversion done early;
  - in the failures, mass value × mass unit: their lookup.
- **It is not zero handling, ReLU-boundary tuning or numerical precision.**
- **The generalisers carry the mass in a factorised, two-dimensional code**, like the constructed log design (whose
  causal dimension is also 2) and unlike the canonical design (12). But their two dimensions form a *circle*, not the
  constructed straight-line integer.
- **Generalisers and the construction are computationally equivalent** at the level of the mass variable: swapping it
  between prompts gives identical answers in all 1,761 discriminating tests.
- **Repair by splicing constructed parts into learned networks did not work.** The learned codes are not linearly
  convertible into the constructed coordinates on unseen inputs.
- **The constructed networks are exact but fragile.** 1% weight noise destroys them, while the learned networks barely
  notice.

## 54.1 Construction (doc 53 D0–D3, §53.2)

Both designs fit the existing TinyTyped shapes: d = 64, 4×16 heads, MLP 256, vocabulary 25, 8 positions. No hooks.

| design | idea | heads used | MLP units (M·A·S) | nonzero parameters |
|---|---|---|---|---|
| **C (canonical)** | layer 1: copy + 3 unit-conversion heads (P_s − I); layer 2: M-gather (tied with the output unit), A-gather, the scale-7 correction; MLP: ReLU hats on t = ℓM + ℓA + ℓS | 4 + 3 | 34 (t ∈ 0..29) | 1,291 of 69,750 |
| **L (log-lift)** | binding heads copy ℓ(value) into the unit's slot; uniform layer-2 attention sums the unwrapped logs (×8); hats on t ∈ 0..55 | 2 + 1 | 60 | 613 |

- Design C's residual layout uses 51 of 64 dimensions.
- The 4-head budget is met only by placing the fourth unit correction in layer 2, as derived in D3. The general
  compiler, which puts all corrections in layer 1, needs 5 heads for the same task (§54.6).

## 54.2 Qualification (§53.3): all criteria met

All results are exhaustive over the 9,126 prompts.

| criterion | Design C (all 6 operations) | Design L (4 monomials) |
|---|---|---|
| Q-acc familiar / unseen | 1.0 / 1.0 (7,350 / 1,776) | 1.0 / 1.0 |
| Q-zero | all 1,350 zero-operand prompts → 0; no other prompt → 0 | same |
| Q-inv | M, A, S regions **bitwise identical** within each canonical class, distinct between classes | M state equal **mod 12** only (24 distinct lift states; not bitwise invariant, as predicted) |
| Q-iit | 118,638 M interventions, 118,638 A, 27,378 S: all 1.0, discriminating subsets 1.0 (S is N/A for M±A and M/A, which do not use S) | M: 219,024 interventions (every lift state as source), 1.0 |
| Q-data | weights bitwise identical when the OOD split is deleted from the task module; the constructor reads no data | same constructor |
| **Q-exact, float32** | attention weights exactly 0, ½ or 1 in every row used; h regions, pre-activations and logits **bitwise equal to the symbolic integers**; logits exactly one-hot; margin 1 | logits exactly one-hot; layer-2 weights exactly 1/8 |
| float64 (real-arithmetic proxy) | attention deviates from 0/1 by ≤ 2.7·10⁻¹⁰⁹; logits by ≤ 2.3·10⁻¹⁰⁷; margin 1 | — |

**Distinction kept.**
- *Symbolically*, the construction is exact except for the softmax saturation, a bounded approximation of less than
  10⁻¹⁰⁰ against a margin of 1. Correctness is therefore proved in real arithmetic.
- *In float32*, every activation equals its symbolic value bitwise.
- The claim "exactly programmed" rests on Q-exact, not on accuracy.

## 54.3 Neural instruction system (§53.4)

**Data:** `instructions.json`. Every result below is exhaustive over the 9,126 prompts, with logits exactly one-hot.

| instruction (component) | parameters rewritten | result |
|---|---|---|
| operation: M·A·S, M²·A·S, M·A²·S, M+A, M−A, M/A (x/0 := 0) | `W_in`, `b_in`, `W_out` | 6/6 exact |
| unit conversion: any scale s ∈ Z₁₃* for units MU1, MU2, AU1, AU2 and MU0 (MU0 via the spare layer-2 head) | that head's OV slice | 60/60 exact |
| operand binding: swap the mass and acceleration roles (tested on M²AS, M−A, M/A) | layer-2 gather keys and the correction heads' destinations | 3/3 exact |
| output decoding: F+1, 2F, F⁻¹, −F | `W_U` rows | 4/4 exact |
| **50 random composites** (operation × 0–2 unit edits × binding × output × 0–2 Phase IV token programs) | — | **50/50 exact** |
| reversibility | — | all 16 edit types restore the weights **bitwise** |

**Independence.** Each component lives in its own parameters:
- operation → MLP;
- unit conversion → correction-head OV slices;
- binding → gather keys;
- decoding → `W_U`;
- operand identity → embeddings.

Binding and some unit edits both touch layer-2 tensors, but in disjoint head slices.

**Limits.**
- Acceleration unit 0 cannot be re-scaled in place: all 8 heads are in use, so it would need a ninth head.
- **Design L refuses M ± A.** M + A needs exp(ℓM) before addition, a second nonlinearity before the readout.
- More generally, one hidden layer computes exactly any g(Σφᵢ(Xᵢ)) whose statistic has at most about 250 values.
  Multiplicative operations need 23–41 values through the group isomorphism; a lookup would need 13³ = 2,197.
  Mixed-domain operations such as M·A + S have no small statistic. They need a second MLP (not built).

## 54.4 The unexplained residual in learned networks (§53.5)

**Data:** `residual.json`, `residual_pairs.json`.
- ρ = u − û, the part of the 256 head pre-activations not explained by the token-additive fit over the full domain.
- R²_add is 0.86 for 307 and 0.90 for 311. These differ from Phase V's 0.895/0.916 because this domain includes the
  zero prompts.

| network | R²_add | α = 0 fam / unseen | α = 0.75 | α = 1 | restore only sign-flip cells | restore pairwise (c1) fit | R²(ρ): canonical operands / zero / pairwise / product phase / L2 attention / order × token | boundary energy share |
|---|---|---|---|---|---|---|---|---|
| **307** | 0.86 | 0.49 / 0.57 | 0.89 / 0.98 | 1.00 / 1.00 | 0.56 / 0.62 | 0.63 / 0.77 | 0.00 / 0.28 / 0.76 / 0.01 / 0.41 / 0.02 | 0.39 |
| **311** | 0.90 | 0.55 / 0.35 | 0.94 / 0.94 | 1.00 / 1.00 | 0.57 / 0.40 | **0.99 / 0.98** | 0.53 / 0.12 / 0.91 / 0.00 / 0.20 / 0.02 | 0.26 |
| 300 | 0.72 | 0.33 / 0.17 | 0.94 / 0.46 | 1.00 / 0.43 | 0.37 / 0.18 | 0.88 / 0.31 | 0.00 / 0.22 / 0.90 / 0.00 / 0.61 / 0.01 | 0.49 |
| 305 | 0.68 | 0.29 / 0.21 | 0.62 / 0.46 | 1.00 / 0.72 | 0.34 / 0.26 | 0.59 / 0.40 | 0.01 / 0.08 / 0.75 / 0.00 / 0.06 / 0.22 | 0.50 |
| 302 | 0.36 | 0.25 / 0.17 | 0.87 / 0.31 | 1.00 / 0.38 | 0.58 / 0.25 | 0.83 / 0.28 | 0.56 / 0.02 / 0.89 / 0.00 / 0.15 / 0.01 | 0.75 |
| 304 | 0.47 | 0.28 / 0.14 | 0.92 / 0.27 | 1.00 / 0.27 | 0.48 / 0.20 | 0.66 / 0.12 | 0.25 / 0.02 / 0.92 / 0.00 / 0.42 / 0.00 | 0.67 |
| 310 | 0.54 | 0.26 / 0.14 | 0.80 / 0.26 | 1.00 / 0.28 | 0.44 / 0.17 | 0.96 / 0.31 | 0.73 / 0.03 / 0.97 / 0.00 / 0.27 / 0.01 | 0.68 |

**Dominant token pair** (exploratory split of the pairwise family: share of ρ, and familiar / unseen accuracy when only
that pair is restored):

| network | dominant pair | share of ρ | familiar / unseen |
|---|---|---|---|
| 307 | mass value × acceleration value | 0.41 | 0.55 / 0.58 |
| 311 | acceleration value × acceleration unit | 0.70 | 0.78 / 0.65 |
| 300 | acceleration value × output unit | 0.64 | — |
| 305 | order × value | 0.25 + 0.19 | — |
| 302 | mass value × mass unit | 0.61 | — |
| 304 | acceleration value × output unit 0.52, mass value × mass unit 0.37 | — | — |
| 310 | acceleration value × acceleration unit 0.55, mass value × mass unit 0.40 | — | — |

**Reading** (by the rule fixed in §53.5):
1. **The residual is computationally essential and belongs to the readout.** Removing it (α = 0) lowers *familiar*
   accuracy to 0.25–0.55 in every network, as much as or more than unseen accuracy. It is not a separate
   "composition" signal.
2. **It is not boundary information.** Restoring ρ only in the 6–39% of cells whose ReLU state it flips recovers
   little (307: 0.56/0.62).
3. **It is not zero handling** (≤ 0.28), **not numerical precision** (the float64 forward gives identical predictions;
   pre-activation differences ≤ 5·10⁻⁴ against an RMS of ρ of 3.7–8.2), and **not the answer's phase** (≤ 0.01).
4. **It is pairwise token interactions** (0.75–0.97 of ρ). In **311**, restoring the pairwise component alone recovers
   the network (0.99/0.98): there the residual *is* acceleration-unit conversion performed by attention. In **307**
   the largest term is the cross-operand value × value interaction (a partial product computed before the decoder),
   and attention weights explain 0.41.
5. In **failures**, a large part of ρ is the canonical code itself (0.25–0.73) and mass value × mass unit: their
   lookup.
6. Restoring ρ for the low-margin prompts restores everything. But those are 51–79% of prompts, so this only says the
   residual matters for most prompts.

**Registered prediction:**
- partly supported: the residual is essential, and pairwise terms dominate;
- not supported: its effect does not act through boundary cells;
- the attention-weight share is large only in 307, 300 and 304.

**[I] Interpretation.** The learned networks do not separate "encode additively, then decode" as cleanly as the
constructed designs. They put part of the combination into the attention layers, and the decoder's readout is tuned
to that mixture. That is why no additive program reproduced them (Phase V).

## 54.5 Constructed versus learned (§53.6)

**Data:** `compare.json`. Learned networks are measured over the full domain.

| | Design C | Design L | 307 | 311 | 300 | 305 | 302 | 304 | 310 |
|---|---|---|---|---|---|---|---|---|---|
| K3 factorisation of the M code | 0.33 | 1.00 | 1.00 | 0.98 | 0.99 | 0.97 | 0.24 | 0.70 | 0.65 |
| K2 causal dimensions of M (interchange ≥ 0.95) | **12** | **2** | **2** | **2** | never | never | never | never | never |
| K1 CKA(h, written-token features) | — | — | 0.29 | 0.27 | 0.19 | 0.22 | 0.12 | 0.18 | 0.25 |
| K1 CKA(h, canonical features) | — | — | 0.19 | 0.05 | 0.05 | 0.11 | 0.23 | 0.10 | 0.29 |
| K1 RSA(M block, Design C M) | 1 | — | 0.17 | 0.19 | 0.18 | 0.19 | **0.47** | 0.27 | 0.34 |
| K1 RSA(M block, Design L lift) | — | 1 | 0.03 | 0.03 | 0.14 | 0.16 | 0.11 | 0.10 | 0.21 |
| K4 units dominated by M / A / S | rows by design | — | 24 / 88 / 67 | 49 / 135 / 0 | 136 / 54 / 21 | 42 / 126 / 20 | 204 / 46 / 0 | 207 / 14 / 0 | 156 / 90 / 1 |
| K5 nonzero parameters | 1,291 | 613 | ≈ 69.8K | ≈ 69.8K | ≈ 69.8K | ≈ 69.8K | ≈ 69.8K | ≈ 69.8K | ≈ 69.8K |
| K6 MLP units active per prompt | 5.4% | 8.0% | 12% | 13% | 15% | 13% | 28% | 24% | 25% |
| K7 accuracy, h noise 5% / weight noise 1% | 0.54 / **0.15** | 0.22 / 0.12 | 1.00 / 1.00 | 1.00 / 1.00 | 0.89 / 0.89 | 0.95 / 0.95 | 0.88 / 0.88 | 0.86 / 0.86 | 0.86 / 0.86 |
| K9 alignment R² on unseen prompts: → C's M / → L's lifts | — | — | −0.35 / 0.92 | −0.30 / 0.49 | −0.28 / 0.65 | −0.32 / 0.68 | −0.37 / 0.70 | −0.27 / 0.84 | −0.46 / 0.68 |
| K10 M interchange: learned / agreement with constructed (1,761 discriminating) | 1.00 | — | **1.00 / 1.00** | **1.00 / 1.00** | 0.87 | 0.90 | 0.87 | 0.85 | 0.85 |

The K7 learned values for the noise-free baseline already include their own accuracy (for example 300's 0.89 is its
domain accuracy).

**Reading.**
- **Same variables, different geometry.**
  - The generalisers match **Design L** in what matters causally: the mass code is factorised (0.98–1.00) and
    two-dimensional, and interchange on it is computationally equivalent to the construction (1.00 agreement).
  - They do *not* match its geometry: RSA with the linear lift is 0.03. Their two dimensions are a **circle** (the
    phase code of Phase V), where Design L's lift is an unwrapped line. They also do not match Design C's canonical
    one-hot (12 dimensions, RSA 0.17).
  - The failures sit closest to Design C's canonical geometry (RSA 0.27–0.47), consistent with their lookup code,
    but without its invariance on unseen renderings.
- **Different decoders.** The learned decoders use many more active units and spread the operation across many units
  (K4). The constructed decoders are sparse ladders of hats.
- **Robustness is the price of exactness.** The constructed networks have margin 1, integer-coded logs and very large
  attention gains (4,000–12,000). Small perturbations break them. The learned networks keep wide margins.

## 54.6 Compiler generalisation (§53.8)

**Data:** `compiler.json`. The compiler takes the spec (prime, roles with unit scales, output scales, expression) and
emits a TinyTyped network. Width, heads and MLP size are computed from the spec. All role orders are allowed.

<!-- G_TABLE_START -->
**Interim: 33 specifications completed so far; the grid is still running and this table will be replaced by the final one.**

| p | roles | expression | verification | prompts | accuracy | logits exactly one-hot | role regions = canonical one-hot | interchange (per role × every value) | d / heads / MLP units | nonzero params |
|---|---|---|---|---|---|---|---|---|---|---|
| 13 | 2 | `mono:1,1;S` | exhaustive | 9,126 | 1.0000 | 1.0000 | True | 1.0000 (106,496) | 80 / 5 / 34 | 1,296 |
| 5 | 2 | `mono:1,1;S` | exhaustive | 1,350 | 1.0000 | 1.0000 | True | 1.0000 (13,500) | 40 / 5 / 13 | 330 |
| 5 | 2 | `mono:3,2;S` | exhaustive | 1,350 | 1.0000 | 1.0000 | True | 1.0000 (13,500) | 40 / 5 / 22 | 456 |
| 5 | 2 | `mono:1,-1` | exhaustive | 1,350 | 1.0000 | 1.0000 | True | 1.0000 (13,500) | 40 / 5 / 11 | 284 |
| 5 | 2 | `sum` | exhaustive | 1,350 | 1.0000 | 1.0000 | True | 1.0000 (13,500) | 40 / 5 / 11 | 301 |
| 5 | 2 | `diff` | exhaustive | 1,350 | 1.0000 | 1.0000 | True | 1.0000 (13,500) | 40 / 5 / 11 | 301 |
| 5 | 3 | `mono:1,1,1;S` | exhaustive | 60,750 | 1.0000 | 1.0000 | True | 1.0000 (61,440) | 56 / 7 / 17 | 515 |
| 5 | 3 | `mono:1,1,2;S` | exhaustive | 60,750 | 1.0000 | 1.0000 | True | 1.0000 (61,440) | 56 / 7 / 20 | 569 |
| 5 | 3 | `mono:1,-1,1` | exhaustive | 60,750 | 1.0000 | 1.0000 | True | 1.0000 (61,440) | 56 / 7 / 14 | 437 |
| 5 | 3 | `sum` | exhaustive | 60,750 | 1.0000 | 1.0000 | True | 1.0000 (61,440) | 56 / 7 / 15 | 476 |
| 7 | 2 | `mono:1,1;S` | exhaustive | 2,646 | 1.0000 | 1.0000 | True | 1.0000 (37,044) | 50 / 5 / 20 | 540 |
| 7 | 2 | `mono:2,3;S` | exhaustive | 2,646 | 1.0000 | 1.0000 | True | 1.0000 (37,044) | 50 / 5 / 35 | 810 |
| 7 | 2 | `mono:1,-1` | exhaustive | 2,646 | 1.0000 | 1.0000 | True | 1.0000 (37,044) | 50 / 5 / 15 | 424 |
| 7 | 2 | `sum` | exhaustive | 2,646 | 1.0000 | 1.0000 | True | 1.0000 (37,044) | 50 / 5 / 15 | 449 |
| 7 | 2 | `diff` | exhaustive | 2,646 | 1.0000 | 1.0000 | True | 1.0000 (37,044) | 50 / 5 / 15 | 449 |
| 7 | 3 | `mono:1,1,1;S` | exhaustive | 166,698 | 1.0000 | 1.0000 | True | 1.0000 (258,048) | 70 / 7 / 24 | 829 |
| 7 | 3 | `mono:2,2,1;S` | exhaustive | 166,698 | 1.0000 | 1.0000 | True | 1.0000 (258,048) | 70 / 7 / 34 | 1,069 |
| 7 | 3 | `mono:1,-1,1` | exhaustive | 166,698 | 1.0000 | 1.0000 | True | 1.0000 (258,048) | 70 / 7 / 20 | 697 |
| 7 | 3 | `sum` | exhaustive | 166,698 | 1.0000 | 1.0000 | True | 1.0000 (258,048) | 70 / 7 / 21 | 754 |
| 11 | 2 | `mono:1,1;S` | exhaustive | 6,534 | 1.0000 | 1.0000 | True | 1.0000 (90,112) | 70 / 5 / 29 | 998 |
| 11 | 2 | `mono:3,1;S` | exhaustive | 6,534 | 1.0000 | 1.0000 | True | 1.0000 (90,112) | 70 / 5 / 47 | 1,466 |
| 11 | 2 | `mono:1,-1` | exhaustive | 6,534 | 1.0000 | 1.0000 | True | 1.0000 (90,112) | 70 / 5 / 23 | 800 |
| 11 | 2 | `sum` | exhaustive | 6,534 | 1.0000 | 1.0000 | True | 1.0000 (90,112) | 70 / 5 / 23 | 841 |
| 11 | 2 | `diff` | exhaustive | 6,534 | 1.0000 | 1.0000 | True | 1.0000 (90,112) | 70 / 5 / 23 | 841 |
| 11 | 3 | `mono:1,1,1;S` | exhaustive | 646,866 | 1.0000 | 1.0000 | True | 1.0000 (1,351,680) | 98 / 7 / 39 | 1,745 |
| 11 | 3 | `mono:2,1,1;S` | exhaustive | 646,866 | 1.0000 | 1.0000 | True | 1.0000 (1,351,680) | 98 / 7 / 48 | 2,069 |
| 11 | 3 | `mono:1,-1,1` | exhaustive | 646,866 | 1.0000 | 1.0000 | True | 1.0000 (1,351,680) | 98 / 7 / 32 | 1,433 |
| 11 | 3 | `sum` | exhaustive | 646,866 | 1.0000 | 1.0000 | True | 1.0000 (1,351,680) | 98 / 7 / 33 | 1,526 |
| 13 | 2 | `mono:1,1;S` | exhaustive | 9,126 | 1.0000 | 1.0000 | True | 1.0000 (106,496) | 80 / 5 / 30 | 1,176 |
| 13 | 2 | `mono:2,3;S` | exhaustive | 9,126 | 1.0000 | 1.0000 | True | 1.0000 (106,496) | 80 / 5 / 63 | 2,166 |
| 13 | 2 | `mono:1,-1` | exhaustive | 9,126 | 1.0000 | 1.0000 | True | 1.0000 (106,496) | 80 / 5 / 27 | 1,036 |
| 13 | 2 | `sum` | exhaustive | 9,126 | 1.0000 | 1.0000 | True | 1.0000 (106,496) | 80 / 5 / 27 | 1,085 |
| 13 | 2 | `diff` | exhaustive | 9,126 | 1.0000 | 1.0000 | True | 1.0000 (106,496) | 80 / 5 / 27 | 1,085 |
<!-- G_TABLE_END -->

**Deviation 2** (logged here). The exhaustive cap was lowered from the registered 3·10⁶ to 1.2·10⁶ prompts to keep
runtime practical. Larger domains are verified on 200,000 sampled prompts, labelled "sampled".

- **Multiple declared variables.** In every 3-role network, each role region is bitwise equal to one-hot of its
  canonical value. Interchange is verified per role region, for every source value.
- **The compiler's resource law:**
  - heads = 1 + (number of non-identity unit scales), at least k;
  - width = heads × max(p + n_out, …);
  - MLP units = (range of the statistic) + 3, plus 2 if gated.

## 54.7 Hybrid repair (§53.7): negative

**Data:** `hybrid.json`. Accuracy is familiar / unseen.

| network | baseline | H1 learned encoder + constructed decoder | H2 constructed M code → learned block + learned decoder | H2 control: random grouping | H4 linear self-correction |
|---|---|---|---|---|---|
| 307 | 1.00 / 1.00 | 0.35 / 0.23 | 0.74 / 0.62 | 0.19 / 0.10 | 0.27 / 0.14 |
| 311 | 1.00 / 1.00 | 0.24 / 0.15 | 0.66 / 0.58 | 0.21 / 0.12 | 0.26 / 0.18 |
| 300 | 1.00 / 0.43 | 0.25 / 0.15 | 0.76 / 0.63 | 0.16 / 0.09 | 0.27 / 0.18 |
| 305 | 1.00 / 0.72 | 0.27 / 0.16 | 0.74 / 0.62 | 0.15 / 0.08 | 0.29 / 0.15 |
| 302 | 1.00 / 0.38 | 0.25 / 0.15 | **0.92 / 0.76** | 0.25 / 0.13 | 0.87 / 0.30 |
| 304 | 1.00 / 0.27 | 0.32 / 0.21 | 0.86 / 0.56 | 0.19 / 0.10 | 0.66 / 0.25 |
| 310 | 1.00 / 0.28 | 0.28 / 0.13 | **0.93 / 0.68** | 0.27 / 0.12 | 0.76 / 0.26 |

**H1 bypass controls** (H1 applied to networks that should not be repairable):

| control network | its own accuracy | H1 result |
|---|---|---|
| randomly initialised network | 0.07 / 0.08 | **0.42 / 0.29** (higher than any learned encoder) |
| `neither_s304` | — | 0.26 / 0.14 |
| `neither_s307` | — | 0.31 / 0.21 |

**Deviation 3** (logged before interpreting H2). The registered "wrong-code" control, a permutation of the one-hot
code, is mathematically identical to the real code under an affine fit, and it reproduced H2 exactly. It was replaced
by a random grouping of renderings into classes.

**Reading.**
- **H1 fails for every network, including the generalisers.** Scalar lift coordinates are not linearly readable from
  learned h precisely enough: the maximum error on unseen prompts is 8–17 against a tolerance of 0.5. A *random*
  network's h gives a better H1 score than any learned network, so H1 measures nothing about the learned computation.
  The prediction that H1 repairs the partial networks is **not supported**. The reason is the geometry of §54.5: the
  learned code is a circle, not a line.
- **H2 gives its best results in the failures** (unseen 0.56–0.76), and the random-grouping control collapses
  (≤ 0.13). The canonical grouping supplied by the constructed encoder is what helps. **No network meets the repair
  criterion** (unseen ≥ 0.90, familiar ≥ 0.95). The class-mean block also loses each example's context, so familiar
  accuracy falls to 0.66–0.93; this matches the Phase IV oracle placement D1.
- **H3 coincides with H1** in this architecture, because the head is only the MLP plus the readout.
- **H4** (making the learned block canonical from the learned h by a linear map) does not repair: unseen ≤ 0.30. The
  canonical code is not linearly present in the failures' h for unseen renderings, so a repair needs the constructed
  encoder (H2), not a linear edit.
- The double dissociation is supported only *directionally*: H2 helps failures most, and H1 helps nobody.

## 54.8 Answer to the central question

**Can a mathematical specification be compiled directly into the weights, creating an observable, editable and
generalisable mechanism without training?**
- **Yes**, for the family of modular expressions tested, in the existing architecture.
- **Exact and verified exhaustively.**
- **Observable:** every region and unit has a known meaning.
- **Editable:** five components can be rewritten independently and reversibly.
- **Generalisable:** perfect on renderings no learned network saw, and generated by a compiler across primes, operand
  counts, unit scales and expressions.

**What it teaches about learned networks.**
- Learned generalisers find the *same causal variables* as the construction: a factorised, two-dimensional mass code,
  computationally equivalent under interchange. They store it as a circle rather than an integer line.
- Circles cannot be added exactly by one ReLU layer (D0). So the learned networks share the combination between
  attention (pairwise token terms) and a broad, approximate readout, and that readout depends on the "unexplained"
  residual.
- **Exactness and robustness trade off.**
  - Construction buys exactness, sparsity and editability.
  - Training buys robustness and a distributed, approximate algorithm that is hard to read or edit.
- Constructed parts cannot simply be spliced into learned networks, because the two use different coordinate
  geometries (line or one-hot versus circle).
