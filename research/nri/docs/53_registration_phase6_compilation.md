# 53 — Registration: Phase VI, neural compilation and mathematical construction

This document is committed **before** any constructed network is evaluated. It records the mathematical derivations,
the designs to be built, the qualification criteria, the experiments and the predictions.

**Authorisation.** The owner's Phase VI INIT.

**Constraints.**
- No training is planned; all work is deterministic construction and single-pass evaluation.
- PR #34, the original archives and the historical results are not touched.
- The large reliability experiment stays deferred.

**Labels.** Every network in this phase is one of four kinds:

| label | meaning |
|---|---|
| **learned** | trained by gradient descent (the historical zoo) |
| **reconstructed** | a program fitted to a learned network (Phase V) |
| **constructed** | weights written from the task's mathematics, no data |
| **hybrid** | learned and constructed parts combined |

## 53.1 The task as mathematics

The task computes F = M·A·S mod 13:
- the canonical mass is M = m·s_mu, with mass-unit scales s ∈ {1, 5, 8};
- the canonical acceleration is A = a·s_au, with acceleration-unit scales s ∈ {1, 2, 7};
- the output factor is S = s_f⁻¹, with s_f ∈ {1, 6, 11}.

The prompt is `x1 u1 × x2 u2 → f =`. Which pair is the mass is given only by the unit token; the unit tokens are
role-specific. The domain has 13·3·13·3·3·2 = **9,126** prompts. The 1,776 prompts with mass unit 2 and value 9–12,
or acceleration unit 2 and value 9–12, never occurred in training. A constructed network uses no training data, so
"familiar" and "unseen" only label how the learned networks saw these prompts.

### Group structure

- Z₁₃* is cyclic of order 12, and 2 is a generator.
- The discrete logarithm ℓ: Z₁₃* → Z₁₂, with 2^ℓ(x) = x, is a group isomorphism (Z₁₃*, ·) ≅ (Z₁₂, +).
- For x, y, z ≠ 0: ℓ(xyz) = ℓ(x) + ℓ(y) + ℓ(z) mod 12. Zero has no logarithm: F = 0 exactly when m = 0 or a = 0.

### D0. Adding Fourier vectors is not adding phases

e^{iα} + e^{iβ} = 2 cos((α−β)/2) · e^{i(α+β)/2}.
- The sum's argument is (α+β)/2, defined only mod π.
- Its magnitude depends on α − β.
- So the sum of two phase *vectors* does not encode the phase α + β. Recovering e^{i(α+β)} needs a product,
  e^{iα}·e^{iβ}, which is bilinear.

**Consequence (D0′).** A one-hidden-layer ReLU readout acting on a *sum* of per-token phase vectors cannot represent
the map to e^{i(α+β)} by linear algebra alone. It can only approximate it, piece by piece, with many units. This is
consistent with Phase V: learned generalisers carry additive per-token phases, and no compact program reproduces them.

The construction avoids the problem in one of two ways:
- it carries the **unwrapped integer** ℓ (a scalar), whose addition is exact, and reduces mod 12 with ReLUs; or
- it carries **canonical one-hot codes** (per-unit permutations), so no phase arithmetic is needed before the readout.

### D1. Exact modular readout with ReLUs

Let t be an integer in [t_min, t_max]. Define r_j = ReLU(t − j) for j = t_min−1, …, t_max+1, and
hat_k = r_{k−1} − 2r_k + r_{k+1}.
- For integer t, hat_k = 1 if t = k and 0 otherwise. This is exact in real arithmetic and in floating point, since all
  quantities are small integers.
- Any function g of t is then computed exactly by the linear readout Σ_k g-indicator · hat_k, using
  (t_max − t_min + 3) hidden units.
- The modular reduction t ↦ t mod 12 followed by exponentiation, g(t) = 2^t mod 13, is one such g.

**Zero gate.** Subtract B·z from every hat's pre-activation, where z = [M = 0] + [A = 0] and B exceeds the t range.
Add two units, ReLU(z) − ReLU(z − 1) = [z ≥ 1], that write the zero class. Then the output is exactly 0 if any operand
is zero, and exactly 2^t otherwise, with logit margin 1.

### D2. Attention as exact routing

- **Uniform attention** (zero scores) gives weights of exactly 1/n. Ties among k maximal keys give exactly 1/k.
- **Hard selection:** score gaps Δ after the 1/√d_head scaling give non-selected weights ≤ e^{−Δ}.
  - With Δ ≥ 250, e^{−Δ} < 10⁻¹⁰⁸. This is below the smallest float32 subnormal (1.4·10⁻⁴⁵), so float32 softmax
    returns exactly 0 and 1.
  - In exact real arithmetic the weights are within 10⁻¹⁰⁰ of one-hot.
- **Distinction to be reported:**
  - in **float32**, every activation of the constructed network is an exact small integer or dyadic rational
    (bitwise equal to the symbolic value), which is checked;
  - in **real arithmetic**, the softmax makes the network an approximation, with error < 10⁻¹⁰⁰ on logits whose
    margin is 1, so its argmax is provably correct.

### D3. Why a per-unit transform needs attention selection

Before the head MLP, the network is additive across tokens except where softmax chooses *which* position to read.
- The canonical M = m·s_mu is a product of the value and a unit-dependent factor.
  - In the log domain it is additive, ℓ(M) = ℓ(m) + ℓ(s_mu), but unreduced: it ranges over 0–22.
  - In the value domain it is a unit-specific permutation P_s of one-hot(m).
- A single attention head applies one fixed linear map (OV) to whatever it reads. A unit-specific permutation
  therefore needs a head per non-identity scale. Each such head is selected by its query only when the unit token
  has that scale.
- **Count.** Scales {5, 8} for mass and {2, 7} for acceleration give 4 correction heads, plus 1 copy head. That is 5
  heads, but layer 1 has 4.
- **Fix.** The fourth correction (scale 7) is applied in layer 2 from the "=" position, as an additive correction
  (P_7 − I)·one-hot read from the acceleration-unit position. Layer 2 then holds M-gather, A-gather, the scale-7
  correction, and one spare head.

## 53.2 Designs to be built in the existing architecture

The architecture is TinyTyped: d = 64, 4 heads of 16 dims per layer, MLP 256, vocabulary 25, 8 positions.

### Design L (log-lift)

| stage | what it does |
|---|---|
| embedding | value tokens carry ℓ(v) (0 for v = 0) and a zero flag; unit tokens carry ℓ(scale); force-unit tokens carry ℓ(S) |
| layer 1 | the mass-unit position attends hard to the previous position and copies ℓ(m) and [m = 0] into an M slot; the same for A (2 heads, gated by unit type and a null position) |
| layer 2 | the "=" position attends uniformly and sums the slots (×8), giving ℓ̃_M = ℓ(m) + ℓ(s_mu) ∈ 0..22, z_M, ℓ̃_A, z_A, ℓ(S) |
| MLP | D1 on t = ℓ̃_M + ℓ̃_A + ℓ(S) |

- It supports exactly the **monomial family** M^a·A^b·S^c (including M/A).
- Its M representation is invariant across equivalent renderings only **mod 12** (it is a lift).
- It *cannot* compute M + A in one hidden layer. That needs exp(ℓ̃_M), a second nonlinearity applied before addition.
  This is predicted and proved in the results.

### Design C (canonical)

| stage | what it does |
|---|---|
| embedding | value tokens carry one-hot(v) (13 dims, including 0) in a shared region Q |
| layer 1 | **copy head:** every position attends to position p − 1 (rotary 2-D position code) and copies Q. **Correction heads** for mass scales 5 and 8 and acceleration scale 2: the query fires only for that unit (otherwise it attends a null position, the × token) and writes (P_s − I)·Q. After layer 1, every unit position holds one-hot(v·s). |
| layer 2 | **M-gather:** the "=" query ties the mass-unit and force-unit keys (weights exactly ½ each) and reads Q (13) plus force-unit identity (3), ×2. **A-gather:** reads Q of the acceleration-unit position. **Scale-7 correction:** (P_7 − I)·Q from the acceleration-unit position, when its scale is 7, into the A region. |
| at "=" | h holds one-hot(M), one-hot(A) and one-hot(force unit). These are **identical for equivalent renderings**. |
| MLP | any operation of the form g(φ_M(M) + φ_A(A) + φ_S(S)), with integer-valued φ, using (range + 3) + 2 units (D1) |

- Every φ is linear in a one-hot code, so ℓ, the integer value, 2ℓ, −ℓ and zero flags are all available.
- **Residual dimensions:** M/Q 13, A 13, force unit 3, unit flags 6, null 1, position 2, output 13; 51 of 64 used.
- **Parameter blocks** map to the INIT's components:

  | component | parameters |
  |---|---|
  | operand identity | embeddings |
  | unit conversion | correction-head OV matrices |
  | operand binding | copy head and gather keys |
  | operation | W_in ridge coefficients and the zero gate |
  | output decoding | W_out table g and W_U |

**Fallback.** If either design cannot be built within the existing shapes, the binding constraint is identified and
the smallest alternative (more heads or width) is built as a separate experiment.

## 53.3 Qualification (Q-criteria; all exhaustive, fixed now)

1. **Q-acc:** accuracy 1.0 on all 9,126 prompts, reported separately for familiar (7,350) and unseen (1,776) labels.
2. **Q-zero:** every prompt with m = 0 or a = 0 outputs 0, and no other prompt does.
3. **Q-inv:** the declared M region at "=" is *bitwise identical* for all renderings with the same canonical M (and
   the same for A and S). This is **Design C only**; for Design L the M state is checked to be equal mod 12.
4. **Q-iit:** for every base prompt and every one of the 13 source values of M, setting the M region to the source's
   code gives F(M_src, A, S): 9,126 × 13 = 118,638 interventions. The same for A, and for S (3 values). The
   discriminating subset (intended ≠ unchanged) is reported separately.
5. **Q-data:** no training example, label or learned parameter is read. The constructor takes only the task
   definition (P, scales, token layout). This is checked by code inspection, and by constructing with the OOD split
   undefined.
6. **Q-exact:**
   - in float32, every hidden activation and logit equals its symbolic value bitwise (integers or ½);
   - logits are exactly 1 for the answer and 0 otherwise;
   - repeated in float64.

   "Exactly programmed" is claimed only if Q-exact holds, not from accuracy alone.

## 53.4 Instruction system (internal operators)

These operations are compiled into the Design-C MLP (W_in, b_in, W_out) only; the encoder is unchanged:

| operation | t (the ridge statistic) | output g(t) |
|---|---|---|
| M·A·S | ℓM + ℓA + ℓS | 2^t, gate M, A |
| M²·A·S | 2ℓM + ℓA + ℓS | 2^t, gate M, A |
| M·A²·S | ℓM + 2ℓA + ℓS | 2^t, gate M, A |
| M + A | M + A (integer values) | t mod 13 |
| M − A | M − A | t mod 13 |
| M / A | ℓM − ℓA | 2^t, gate M, A; **M/0 := 0 and 0/A := 0, by definition** |

**Measures.**
- **Exactness:** Q-acc and Q-exact for each operation against Python modular arithmetic.
- **Component independence:** change one block only, then verify exhaustively:
  - unit conversion: reassign a unit's scale by editing its correction OV;
  - binding: swap the roles of mass and acceleration by swapping the gather keys;
  - output decoding: compose g with a permutation;
  - operation: swap the ridge.
- **Composition:**
  - operation × unit edit × output edit × the Phase IV token programs;
  - 50 random composite programs;
  - a reversibility check: the inverse edit restores the weights bitwise.
- **Design L contrast:** compile the monomials into Design L, and record why M ± A is not available.
- **Compilability bound.** The single-hidden-layer readout computes exactly the functions g(Σ φ_i(X_i)) whose
  statistic has at most about 250 distinct values.
  - Multiplicative operations need 34–56 values, through the group isomorphism. A full lookup would need
    13³ = 2,197, beyond the budget.
  - Mixed-domain operations, such as M·A + S, have no single small statistic. This is stated as an architectural
    constraint and tested in a separate two-MLP alternative only if time allows.

## 53.5 The unexplained residual in learned networks (R-experiments)

**Networks:** 307, 311, 300, 305, 302, 304, 310 over the full domain.

Let u = W_in·h + b be the head pre-activations (256 units). Let û be the token-additive fit (one-hot tokens and
order, fitted by least squares over the full domain as in Phase V), and ρ = u − û.

- **R1. Graduated removal.** Replace u by û + α·ρ for α ∈ {0, 0.25, 0.5, 0.75, 0.9, 1}. Report:
  - accuracy (familiar and unseen);
  - the mean margin of the correct logit;
  - the fraction of units whose ReLU state (on or off) flips.
- **R2. Decomposition of ρ.** The fraction of ρ's variance explained by each candidate family, fitted over the
  domain:
  - (a) operand-dependent threshold corrections: per-unit functions of canonical M, A and S separately (one-hot
    canonical values);
  - (b) zero handling: indicators [m = 0], [a = 0] and their interactions with the other tokens;
  - (c) higher order: all pairwise token interactions (one-hot ⊗ one-hot), and the product phase cos/sin(k·2π ℓ(F)/12);
  - (d) attention normalisation: the softmax weights of each layer-2 head at "=" as regressors;
  - (e) context-dependent routing: order × token interactions;
  - (f) ReLU boundary: the share of ρ's energy in (unit, prompt) cells with |û| < |ρ| (sign-relevant);
  - (g) numerics: forward pass in float64 vs float32; the size of ρ relative to float32 rounding.
- **R3. Targeted restoration.** Starting from α = 0, restore ρ only:
  - in cells where sign(û) ≠ sign(u) (activation-boundary cells);
  - for prompts whose additive-only margin is < 1;
  - in each family's fitted component (R2).

  Report how much of the accuracy each restores.
- **Reading rule (fixed now):**
  - If restoring the sign-relevant cells alone recovers ≥ 90% of the lost accuracy, the residual is *boundary
    information*.
  - If one R2 family's component recovers ≥ 90%, the residual *is* that mechanism.
  - If removal hurts familiar and unseen prompts equally, the residual is part of the readout, not of composition.
- **Prediction (stated now).** The residual is computationally essential (α = 0 lowers accuracy substantially). Its
  effect acts through ReLU boundary cells, and it is dominated by pairwise and attention-normalisation terms (c, d).
  That is, it reflects the inexactness of *vector* phase composition (D0′), which the learned networks approximate
  piecewise, rather than a separate zero or precision mechanism.

## 53.6 Constructed versus learned (K-experiments)

**Networks:** Design C and Design L against 307, 311, 300, 305, 304 and 310.

- **K1. Geometry.**
  - Linear CKA between each network's h at "=" over the full domain and (i) Design C's h, (ii) Design L's h, and
    (iii) the canonical-only and lift-only feature sets.
  - RSA of the M-block cell means (39 renderings).
- **K2. Effective causal dimensionality.**
  - The number of principal directions of the M-dependent part of h needed for interchange accuracy ≥ 0.95.
  - Constructed: 12 by construction (the one-hot affine span).
- **K3. Factorisation.** R²(value + unit) of the M-region cell means. The prediction is that Design C is *not*
  factorised (canonical) while Design L is (additive lift). Learned generalisers are factorised (Phase III).
- **K4. Operation-specific subspaces.** For Design C, the W_in rows per operation. For learned networks, the units
  whose pre-activations depend on M versus A versus S (R² per unit).
- **K5. Parameter efficiency:** nonzero parameters and the effective parameters used.
- **K6. Activation sparsity:** the fraction of MLP units active per prompt.
- **K7. Robustness.** Accuracy under isotropic noise on h, at relative sizes 0.01, 0.05, 0.1, 0.25 and 0.5 of h's
  norm, and under Gaussian weight noise.
- **K8. Editability:** Design C's component edits (§53.4) against the learned networks' Phase IV/V edit results.
- **K9. Cross-network alignment.** An affine map from learned h to constructed h, fitted on familiar prompts only.
  Report R² on familiar and unseen prompts, separately for the M region and the rest.
- **K10. Computational equivalence.**
  - Agreement of argmax over the domain.
  - Interchange equivalence: whether swapping the learned M block and swapping the constructed M region give the same
    behaviour on the same (base, source) pairs (2,048 sampled pairs, discriminating subset).
- **Prediction.** Generalisers are closer to **Design L** (an additive, factorised lift; K1 lift-feature CKA higher)
  than to Design C. Failures are closer to the canonical feature set on trained renderings. Different activations do
  not preclude equivalence, so K10 is the test of equivalence.

## 53.7 Hybrid repair (H-experiments)

**Constraints.**
- Every map is fitted on **familiar** prompts only.
- Targets are functions of the *input tokens* computed from the task definition: ℓ(value), unit scale and canonical
  codes of familiar renderings. **No held-out answer or held-out activation is used.**

| | name | construction |
|---|---|---|
| **H1** | decoder replacement (learned encoder + constructed decoder) | Read scalars ℓ̃_M, z_M, ℓ̃_A, z_A, ℓ(S) from the learned h by affine maps fitted on familiar prompts. Feed them to the Design-L MLP (D1). |
| **H2** | representation replacement (constructed encoder + learned decoder) | Replace only the learned declared M block (dims 0–15) with an affine image of Design C's one-hot(M), fitted on familiar prompts to the learned network's own block. Leave the rest of h learned. |
| **H3** | operation replacement | Keep the learned h and the learned head, but replace the head's hidden layer by the constructed MLP (D1 with the multiplicative ridge), with the H1 readouts. This is H1 restricted to the MLP. |
| **H4** | minimal compiled correction | For failures, the smallest edit to the learned network's own weights equivalent to H2: the H2 map folded into a rank-k additive update of the block. Reported by k and by the number of changed parameters. |

- **Bypass controls.** H1 is also applied to the h of (i) a randomly initialised TinyTyped and (ii) the `neither`
  networks. H2 is also applied with a randomly initialised decoder.
- **Classification.**
  - A hybrid is **genuine repair** if it meets the rescue criterion (unseen ≥ 0.90, familiar ≥ 0.95) *and* its bypass
    control does not (≤ 0.5).
  - Otherwise it is **bypass/replacement**.
  - The amount of constructed computation is reported as its parameter count.
- **Prediction: double dissociation.**
  - H1 repairs the partial networks (300, 305: correct factorised representation, interpolating decoder) and keeps
    the generalisers. It fails on failures, whose unseen-rendering representation is missing.
  - H2 repairs the failures (302, 304, 310: intact canonical-reading decoder, missing representation). It fails on
    partials and generalisers, whose decoders expect a factorised code.
  - H1's random-encoder control may succeed if token identity is linearly readable at "=". If so, H1 is labelled
    bypass for every network.

## 53.8 Compiler generalisation (G-experiments)

The constructor is generalised into a compiler. It takes a specification:
- the prime p;
- the roles, each with its unit scales;
- the output-unit scales;
- the expression, as a ridge statistic and output map.

It emits a network of the TinyTyped form. Width and heads are computed from the specification, and the
binding-constraint arithmetic is reported.

**Grid.**
- p ∈ {5, 7, 11, 13, 17, 19, 23};
- 2 or 3 roles;
- random unit scales (3 per role, seeded);
- expressions: monomials with exponents 1–3; inverse; additive (Σ X_i) and difference.

**Verification.**
- Exhaustive over the domain where it has ≤ 3·10⁶ prompts.
- Otherwise exhaustive over canonical tuples, with prompts sampled per tuple; labelled as such.
- Q-exact on every network.

**Multiple declared variables.** In the 3-role networks, each role gets its own region, and interchange is
verified per region.

## 53.9 Standards

- This registration precedes evaluation. Any construction change after a qualification failure is logged as a
  deviation, and the failed version is kept.
- Negative results are kept. No seed selection.
- The kind of each network (learned, reconstructed, constructed or hybrid) is stated with every result.
