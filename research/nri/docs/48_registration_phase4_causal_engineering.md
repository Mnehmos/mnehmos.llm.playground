# 48 — Registration: Phase IV, causal engineering of generalisable computation

Committed **before** any experiment below is run or evaluated. Authorisation: the owner's Phase IV INIT, which
approves P1–P3 (doc 46 §46.5) subject to their own registration and controls (this document). The large reliability
experiment is deferred.

**Fixed rules for all experiments:**
- No original archive is modified, and neither PR is merged.
- Historical results stand unchanged.
- Every training continuation resumes from a verified, bitwise-exact checkpoint, and is paired with a **matched
  continuation control**: the same resume point, the same data and interchange streams, no intervention. The control
  is an exact replay and must reproduce the historical trajectory bitwise; this is checked.

**Terms.** For a declared block (dims 0–15 of the last-token residual, before the MLP), the decomposition is:

- **h(x) = g + a_v + b_u + I(u,v) + e(x)**
  - u, v are the mass unit and written value;
  - g is the grand mean;
  - **a_v** is the value part and **b_u** the unit part, fitted by least squares on the (u,v) cell means of
    *trained* renderings (calibration data, never unseen renderings);
  - **I(u,v)** is the interaction: the cell mean minus g + a_v + b_u;
  - **e(x)** is the within-cell deviation, which depends on acceleration, output unit and order.
- **Per-example interaction estimate:** Î(x) = h(x) − g − a_v − b_u − ê(x).
  - ê(x) is predicted from the rest of the residual (h[16:]) by a linear map fitted on calibration data.
  - Computing Î uses only the example's own activation and calibration-fitted quantities. It is therefore a
    legitimate runtime intervention, not an oracle.
- **Oracle diagnostics** are labelled as such and never count as rescue. They use other unseen examples' activations
  or the true answers.
- **Outcomes:**
  - ID accuracy: familiar renderings;
  - **OOD-M accuracy:** mass written as unit 2 with value 9–12;
  - **discriminating unseen-source interchange:** PR #34's estimand. Eligible pairs only, where the intended,
    unchanged and wrong answers differ.
- **Panels** are fixed seeds, disjoint from the calibration seeds.

## E1 — Rescue without training (single-pass; failures 304, 310, 302; partials 300, 305; generalisers 307, 311 as references)

| id | intervention on the declared block | uses |
|---|---|---|
| R0 | none | — |
| R1 | complete interaction removal: h ← h − Î(x) | calibration fit |
| R2 | graduated suppression: h ← h − (1 − α)·Î(x), α ∈ {0.75, 0.5, 0.25} | calibration fit |
| R3 | selective suppression: remove only the projection of Î onto the network's causal subspace S (span of canonical-M class means of trained renderings), and onto the complement of S | calibration fit |
| R4 | additive reconstruction: block ← g + a_v + b_u + ê(x), discarding the network's own block entirely | calibration fit |
| R5 | cross-network transplant: replace this network's full residual h by a generaliser's (307's) residual for the same input, mapped through an affine map from 307's to this network's residual space, fitted on calibration inputs | calibration fit |
| D1 (oracle) | block ← mean block of *trained* renderings with the same canonical M (uses the true M of the unseen input) | oracle |

**Controls:**
- **Rank-matched random removal:** remove the projection of h onto a random subspace of the same rank as Î's span.
- **Magnitude-matched perturbation:** add a random vector with the per-example norm of Î.
- **Unmodified:** R0.

**Rescue criterion (pass/fail, fixed now).** OOD-M accuracy ≥ 0.90 **and** ID accuracy ≥ 0.95 **and** both controls
OOD-M ≤ R0 + 0.05.

**Representation vs decoder diagnosis (fixed now):**
- If **R4 or R1** rescues: the defect is in the representation and the decoder can compose.
- If neither rescues but **R5** (the generaliser's representation fed to this network's head) also fails: the
  decoder is defective.
- If R5 rescues while R1/R4 do not: the representation is defective in a way the additive reconstruction does not
  capture.

## E2 — Representation × decoder compatibility matrix (single-pass)

Rows are the representation source R, and columns the decoder D. The focus networks are 307, 311, 305, 300, 304 and
310.

**Procedure.** For each ordered pair, map R's full last-token residual into D's residual space with an affine map
fitted on calibration (trained) inputs. Run D's head on OOD-M inputs and on ID inputs.

**Variants:**
- **full residual:** the whole 64-dimensional residual;
- **block-only:** D's residual with only its block replaced by R's mapped block.

**Prediction (fixed now).** OOD-M is high only where R is factorised **and** D's head composes. Expected pattern:
- R ∈ {307, 311} with D ∈ {307, 311} high;
- D ∈ {304, 310} low for any R;
- R ∈ {304, 310} low for any D.

## E3 — Reverse intervention on generalisers (single-pass; 307, 311)

**Synthetic interaction.** I*(u,v) is a random matrix over cells, double-centred so it has zero value and unit main
effects, with the following properties varied:
- **magnitude** κ ∈ {0.25, 0.5, 1, 2} × the mean ‖a_v‖;
- **rank** r ∈ {1, 2, 4, 8} across the 16 block dimensions;
- **orientation:** within the causal subspace S vs orthogonal to it;
- **specificity:** applied to all cells, to the unit-2 cells only (trained and unseen), or to unseen cells only.

**Measures:** ID and OOD-M accuracy, plus the factorisation R²_add of the perturbed block.

**Prediction (fixed now).** OOD-M falls monotonically with κ for in-S orientation. ID falls less than OOD-M when the
interaction is placed on unit-2 cells. Orthogonal-to-S interactions of the same magnitude hurt less.

**Note.** A single-pass injection cannot *teach* the head to rely on the interaction. The developmental version is
P2 (E4).

## E4 — Developmental interventions (training continuations; P1–P3)

**Interaction penalty L_int.**
- On each training batch, compute the block's (unit, value) cell means for cells present in the batch, weighted by
  count.
- Fit the additive model by least squares (differentiable).
- L_int is the cell-weighted squared residual divided by the total cell-mean variance, i.e. 1 − R²_add on the batch.
- It is added as λ·L_int. With λ < 0 it *rewards* interaction (disruption).

**Continuations.** Each runs for 2,000 steps after its resume point, plus a matched no-intervention control.

| id | network(s) | resume | intervention | prediction |
|---|---|---|---|---|
| P1a | 304, 310, 302 | 2,000 (locked) | L_int, λ = 1 | R²_add → > 0.95 and OOD-M rises above 0.5 (rescue of a locked network) |
| P1b | 304, 310, 302 | 2,000 | L_int, λ = 0.1 | weaker version of P1a |
| P3a | 304, 310, 302 | 2,000 | weight decay 0.01 → 1.0 | route-specificity falls; OOD-M rises (slower than P1a) |
| P1-early | 304 at 700 and 850; 310 at 1,350 and 1,450 | inside the window | L_int, λ = 1, for 500 steps, then released | the interaction never locks; OOD-M ≥ 0.9 at the end |
| P1-path | 304 | 700, 850, 1,000, 1,500, 2,000, 4,000 | L_int, λ = 1, 2,000 steps | rescue rate declines with resume step (developmental path dependence); the step where it becomes ineffective is reported |
| P2a | 307, 311 | 850 (before the dip) | L_int, λ = −1 (reward interaction) | R²_add stays < 0.9; OOD-M < 0.5; ID ≥ 0.95 |
| P2b | 307, 311 | 850 | interchange weight × 3 | earlier saturation; R²_add lower than control; OOD-M lower |
| P2c | 307, 311 | 2,000 (consolidated) | L_int, λ = −1 | tests whether an established generaliser can be broken |
| F1 | 304 | 700 | freeze attention-2 query/key weights for 500 steps, then release | no interaction forms during the freeze; the outcome after release is reported |
| G1 | 304 | 700 | gradient projection: remove the component of each update to attention-2 q/k that increases L_int (one extra backward pass) for 500 steps | as F1 |

- **Measures** at every 100 steps: ID, OOD-M, discriminating unseen-source interchange, R²_add,
  route-specificity.
- **Controls** are reported against the matched continuation control and the historical trajectory, with the
  bitwise check on the control.
- **Rescue** counts only if OOD-M ≥ 0.90 with ID ≥ 0.95 at the end of the continuation.
- **Disruption** counts only if OOD-M ≤ 0.5 with ID ≥ 0.95.

## E5 — Permanent weight-level editing (no gradient training)

**Compilation categories.** Each edit is classified as:
- **(a) exact compilation:** a closed-form parameter change that reproduces the intended activation transformation
  for all inputs;
- **(b) approximate fitting:** a closed-form least-squares parameter change on calibration data;
- **(c) gradient retraining:** not used here.

**Exact-compilation conditions to establish:**
1. Token-level edits on **role-specific tokens** (mass units are their own tokens) compile exactly as replacements
   of embedding rows. Example: *unit conversion*, "read mass unit u as unit u′".
2. A **linear** map T on the last-token residual compiles exactly into the head:
   - W_in ← W_in T and W_U ← W_U T;
   - W_out ← T⁻¹ W_out, for invertible T;
   - an affine translation needs a logit bias, which the head lacks.
3. Edits of **shared** tokens (value tokens are shared by mass and acceleration) that should act only on the mass
   value are input-dependent. They cannot be compiled into a single linear layer at the block. Candidate
   multi-layer constructions are fitted approximately: a low-rank update of the attention-2 value/output path that
   reproduces the per-part shift for the mass contribution. They are reported as category (b).

**Edits:**
- increment (M → M + 1);
- decrement;
- doubling;
- inversion;
- constant replacement (M → c);
- unit conversion (unit u read as u′);
- identity (no change).

**Measures:**
- intended rate on trained and unseen renderings (for unseen: is the edit correct on renderings never trained);
- collateral damage: acceleration-only edits and unedited renderings;
- generalisation preservation: OOD-M on renderings where the edit is the identity;
- parameter displacement (relative Frobenius norm);
- reversibility: apply the inverse edit and compare with the original network, bitwise or by max diff;
- preservation of unrelated computation: accuracy when mass is 0, a fixed point of every multiplicative edit.

Networks: 307 (primary), 311, 300, 304 (negative control).

## E6 — Compositional editing benchmark

Built only from edits that pass E5 (intended ≥ 0.95 on trained renderings):
- sequences of 2–3 edits in all orders, including orders never tested individually;
- the composed parameter change is applied as consecutive compilations.

**Scored against the mathematically composed target**, on trained and unseen renderings. **Correct composition**
means intended ≥ 0.90 for every sequence.

## Analysis standards

Five focus networks plus 300 and 302 (both partial or failing), so no seed is selected for success. Everything here
is **exploratory** except the pass/fail criteria stated above, which are fixed now. All negative results are reported.
The interventions and their configurations are archived with manifests.
