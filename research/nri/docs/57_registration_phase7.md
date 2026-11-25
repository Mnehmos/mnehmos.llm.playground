# 57 — Registration: Phase VII, robust neural compilation and hybrid learning

Committed **before** any Phase VII experiment runs.

**Authorisation.** The owner's Phase VII INIT. Phase VI was closed first:
- docs 54 and 56, the atlas and the developmental record are complete;
- the dark-sector statements were marked provisional (doc 56 status note);
- the direct-path measurement was made reproducible.

**Standing rules.**
- Deterministic construction and existing checkpoints come first.
- New training is used **only** for the matched hybrid-learning experiments (H and K).
- Negative results are kept, and so are the original archives.
- The large reliability study stays deferred.
- PR #34 is not touched.

**Kinds of evidence.** Every result is labelled:

| label | meaning |
|---|---|
| **mathematical guarantee** | a proof or exact identity |
| **exhaustive verification** | all 9,126 prompts, or the whole finite domain |
| **sampled verification** | a stated sample |
| **observed (trained)** | measured on trained networks |

**Kinds of network.** Every network is labelled learned, reconstructed, constructed, hybrid or hybrid-trained.

## I. Corrected interpretability instruments

**Why they failed (to be confirmed).**
- **V4:** gradient × (W_E[t] − ē) uses the mean over the *whole* vocabulary as baseline. Positions whose real content
  contributes nothing still receive credit from the linear term in the baseline difference.
- **V5:** the averaged Jacobian J̄ = I + W_out·D̄·W_in replaces each ReLU gate by its mean. Ladder units then cancel
  (r_{k−1} − 2r_k + r_{k+1} with averaged gates), so the averaged readout W_U·J̄ under-weights the directions that carry
  the computation.

**Corrected instruments.**
- **I1. Interventional S-lens (finite).** For position s and read direction v:
  S_s(x) = E_{t′ ∈ type(s)} [v·h(x) − v·h(x with token s := t′)], where type(s) is the set of tokens that may occupy
  position s (values, units of the same role, or force units). Every term is exact.
- **I1b. Path-integrated S-lens.** Integrated gradients from the type-conditional mean embedding (64 steps). This is
  reported alongside I1.
- **I2. Gate-aware lens.** A_ctx = E_x[J(x)ᵀ·W_Uᵀ·W_U·J(x)], the second moment of the *exact* per-prompt Jacobians,
  replaces J̄ᵀ·W_Uᵀ·W_U·J̄. The bright sector consists of the fewest eigen-directions of A_ctx carrying 90% of h's
  readout energy; the rest is dark.
- **I3. Interventional sectors.** Directions of h are ranked by the exact finite effect of mean-ablating each direction
  alone, over 64 orthonormal directions (PCA basis). Sectors are formed from that ranking.
- **I4. Comparison of predictors.** For random subspace ablations of h (dimensions 1, 2, 4, 8, 16 and 32; 20 draws
  each), compare the actual Δlogits with three predictions: (a) the averaged-J lens, (b) the exact per-prompt J, and (c)
  the gate-aware lens. Report the median ratio of actual to predicted, and the correlation.

**Qualification (fixed now), on Designs C and L:**
- **V4′:** I1 assigns ≥ 0.95 of the mass read to the mass value and mass unit positions.
- **V5′:**
  - the bright sector lies ≥ 0.95 inside the used dimensions;
  - dark-sector ablation keeps accuracy ≥ 0.99;
  - and the ratio of actual to predicted for dark-sector ablation lies in [0.5, 2] *or* both quantities are below
    10⁻³.

Only qualified instruments are applied to learned networks.

**Re-test.** The Phase VI residual analysis (doc 56 §56.3) is repeated with the qualified lens. The question is whether
the essential residual is dark or bright.

**Prediction.** I1 and I2 qualify. Under I2 the learned residual is **bright**, so the "dark residual" was an artefact
of averaging over gates.

## R. Robust construction

**R0. Diagnosis on Design C (MAS).**
- **Noise-response curves.** σ ∈ {10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³, 0.01, 0.03, 0.1, 0.3}, 5 noise seeds each. Three noise
  models:
  - **(a)** additive noise scaled by each tensor's RMS (the Phase VI measure);
  - **(b)** multiplicative noise, w·(1 + σε), on all weights;
  - **(c)** additive noise on h, relative to ‖h‖.
- **Each component perturbed alone:** embeddings, attention-1 qkv, attention-1 o, attention-2 qkv, attention-2 o,
  W_in/b_in, W_out and W_U.
- **Single-weight dependence:** the fraction of nonzero weights whose zeroing changes any prediction.
- **Reference curves:** the learned 307 and 311 under the same protocol.

**Sources to be separated:**

| source | how it is isolated |
|---|---|
| attention saturation and gain (G1 = 4,000, H1 = 12,000) | scale relative to other entries of the same qkv tensor |
| integer-coordinate precision | hat width 1 in t |
| ReLU-ladder sensitivity | |
| decision margin | 1 |
| single-weight dependence | the fraction above |
| lack of redundancy | |

**Constructed remedies** (no training), each kept exact at σ = 0 (exhaustive) and each re-checked against the Phase VI
instruction system:
- **R1. Balanced, minimal attention gains.** The gain is split √G·√G between query and key. G is lowered to the smallest
  value that keeps argmax exact over the whole domain. Activations are then no longer bitwise symbolic, so the result is
  labelled *argmax-exact*.
- **R2. Trapezoid ladder.** Bumps with a flat top of width 1 tolerate |Δt| < 0.5 at full height and stay exact at
  integers, using 4 ReLUs per class edge.
- **R3. Redundancy.** K parallel copies of the MLP ladder, for K up to 7 within 256 units, with their logits summed. A
  wider variant (d = 128, 8 heads) duplicates the encoder heads.
- **R4. Margins.** The readout is scaled, and same-class competitors are suppressed with negative off-class writes. This
  is reported, though it is expected to be scale-invariant under noise model (a).
- **R5. Error-correcting representation.** Each variable is carried by two codes (one-hot and Fourier) that are read
  jointly by the ladder.
- **R6. Periodic code:** the Fourier coordinates of experiment F.

**Target.**
- Accuracy ≥ 0.95 at the σ where learned 307 still has ≥ 0.95, for noise model (a) at 1% and (b) at 5%.
- Exact at σ = 0.
- All Phase VI instruction-system edits still exact.

**Prediction.** The dominant source is the mixed-scale qkv tensors under RMS-relative noise, together with the unit-width
hats. R1, R2 and R3 together reach the target; R4 alone does nothing.

## G. Compositional compiler (computational graphs)

**Architecture (the smallest necessary alternative).**
- The Design-C encoder is followed by L residual MLP blocks.
- Each block holds the hat ladders of one or more graph nodes. Each node writes the **one-hot code of its intermediate
  variable** into its own residual region, so the next block can read any function of it linearly.
- A node computes g(Σᵢ φᵢ(childᵢ)) exactly, by the Phase VI rule. Nodes at the same depth share a block.
- No lookup over complete input combinations is allowed: every node's statistic is a sum of per-child terms.

**Expressions.**
- M·A + S;
- (M + A)·S;
- M² + A²: a single stage, since φ_M = M² mod 13 is per-variable;
- (M − A)/(M + A), with x/0 := 0;
- random nested expression trees of depth 1–6 over {+, −, ·, /, square}, 10 per depth, seeded.

**Verification.**
- Exhaustive over the 9,126 prompts, against Python arithmetic.
- **Per intermediate variable:**
  - observability: its region equals one-hot of the intended value, bitwise;
  - causal access: interchange of the region with every value is exact;
  - programmability: swapping one node's operation changes exactly that node's semantics.
- **Scaling:** parameter count, width (residual dimensions), depth (blocks) and MLP units against the number of nodes
  and the statistic ranges. A per-node cost model is fitted and checked.

**Prediction.**
- All expressions compile exactly.
- Depth equals the number of nonlinear stages on the longest path.
- Units per node are (range + 3), plus 2 if gated, so the size is linear in the number of nodes.

## F. Geometric (Fourier) compilation

- **Representation:** each variable is carried as (cos kθ, sin kθ), with θ = 2πℓ(X)/12, for frequencies
  k ∈ K ⊆ {1, 5, 7, 11}, together with a zero flag.
- **Unit conversion:** a fixed rotation R(kθ_s), applied by the per-unit correction heads (the analogue of P_s).
- **Combining circles:** cos(α+β) = cos α·cos β − sin α·sin β.
  - Each product xy = ((x+y)² − (x−y)²)/4.
  - The squares are computed **exactly on the finite set of values** that x ± y can take, by piecewise-linear ReLU
    interpolation through those points. This is exact on a finite set, though not on ℝ.
  - Stage 1 combines M and A; stage 2 rotates by S.
- **Readout:** logits_c = Σ_k cos(k(θ_F − θ_c)), linear in the stage-2 phases, plus a zero gate.
- **Derivation** (to be written in doc 58): why adding phase *vectors* fails (D0), and why exact products on finite
  sets succeed.

**Comparison** across scalar-log (Design L), canonical one-hot (Design C) and Fourier (F), at matched margin (logits
normalised to margin 1) and matched redundancy (K = 1):
- noise-response curves under R0's protocol;
- the role of geometry alone versus redundancy and margin (each varied separately).

**Prediction.** At matched margin and redundancy, the Fourier code is *not* markedly more robust. The learned networks'
robustness comes from redundancy and wide margins, not from circular geometry alone.

## D. The direct residual pathway

- **D1.** Decompose the final logits of 307 (and the other six networks) exactly into the parts listed below. Report
  class spread, margin share and correlation with the MLP total.

  | part | split further by |
  |---|---|
  | W_U·b_out | — |
  | the direct path W_U·h | the h components of E5: embedding, layer 1, each layer-2 head |
  | each MLP neuron's write | — |

- **D2. Coordinated relabelling F → F⁵.** Edit W_out only (refit by least squares on familiar prompts to the
  σ-relabelled full logits) against W_U permutation. This tests whether the neurons can absorb the direct path.
- **D3. Internal operation F(M⁻¹, A, S).** A coordinated least-squares refit of W_out (and of b_out), with the direct
  path left as is, on familiar prompts. Report familiar and unseen accuracy. Compare with the constructed network,
  where the internal operation is an exact MLP swap.
- **Distinction:**
  - an **output relabelling** is a permutation applied after F;
  - an **internal modification** changes the function F itself (it is not σ∘F for any σ).
- **Prediction:**
  - D2 succeeds only partially;
  - D3 fails on unseen prompts in learned networks;
  - the 307 failure (doc 56 §56.5) is explained by the direct path, because W_U permutation succeeds and the W_out-only
    edit does not.

## H. Hybrid learning (training; matched budgets)

**Task.** The dual-operation task.
- The arrow token is replaced by an operation token: **OP₁** (programmed: F = M·A·S) or **OP₂** (to be learned:
  U = M·A + S mod 13).
- The vocabulary grows by one, and the sequence length stays 8.
- The OOD split is as before: mass or acceleration unit 2 with value 9–12 never appears in training.
- U is not compilable in one MLP stage (§53.4). It is compilable in two (G), which gives condition A.

**Architecture (shared by B, C and D).**
- TinyTyped with d = 128, 8 heads of 16, MLP 512.
- The Design-C mechanism occupies residual dimensions 0–63, heads 0–3 of each layer, and MLP units 0–33. Its readout
  is gated by OP₁ through an extra zero-gate input.
- All other parameters are randomly initialised.

**Conditions.**

| | condition | what is trainable |
|---|---|---|
| **A** | entirely constructed: compositional compiler with OP₁ and OP₂ stages and a gate | no training |
| **B** | entirely trained from random initialisation | everything (same architecture and budget) |
| **B′** | entirely trained, at the parameter count of C's trainable parameters only | everything (matched trainable budget) |
| **C** | constructed components frozen by a parameter mask | the rest |
| **D** | constructed components trainable | everything, with a semantics-preservation loss: interchange on the M/A/S regions and distillation of OP₁ logits to the constructed network's, on the training data |

**Budget.**
- AdamW (lr 2·10⁻³, weight decay 0.01), batch 256, up to 6,000 steps.
- 3 seeds per condition.
- Two data regimes: all familiar prompts, and a 25% subsample.

**Measures**, every 100 steps:
- OP₁ and OP₂ accuracy, familiar and unseen (full domain);
- OP₁ exactness (fraction of logits exactly one-hot) and interchange accuracy on the M region;
- steps to criterion (OP₂ familiar ≥ 0.99);
- final unseen OP₂ accuracy;
- parameters trained.

**Predictions.**
- C and D reach the criterion faster than B and B′, and generalise OP₂ to unseen renderings, because they can read the
  constructed canonical codes.
- B generalises in a seed-dependent way, as in the historical study.
- OP₁ stays exact in C only if the learned components do not write into its regions. This interference is measured.

## K. Protecting programmed knowledge (training)

**Setting.** The H task, starting from the constructed mechanism, while OP₂ is learned. Methods:

| | method |
|---|---|
| **K0** | unprotected: all parameters trainable, no constraint (forgetting baseline) |
| **K1** | parameter mask (= C) |
| **K2** | mask, plus *residual-region protection*: learned components' writes are projected out of the constructed regions (the column space of their o and W_out is constrained to be orthogonal to the constructed dimensions) |
| **K3** | orthogonal updates: gradients of shared tensors are projected orthogonal to the constructed weights' row and column spaces |
| **K4** | semantic-preservation objective (= D) |
| **K5** | K2 + K4 |

**Measures:** OP₁ accuracy and exactness over training (forgetting), M/A/S interchange accuracy, interference (Δ OP₁
logits from the learned components), OP₂ learning speed, and final accuracy.

**Target.** OP₁ stays exact (logits one-hot, interchange 1.0, exhaustively) while OP₂ reaches its criterion.

**Prediction.** K2 or K5 meets the target. K1 alone does not, because of residual interference. K0 forgets.

## Order of execution

1. D (existing networks).
2. I (instruments), then the I re-test.
3. R (construction only).
4. G.
5. F.
6. H and K (training), each condition run only after its configuration is committed.

## Standards

- Each experiment is evaluated only after this registration is committed. Configuration changes after a failure are
  logged as deviations, and the failed versions are kept.
- No seed selection.
- Mathematical guarantees, exhaustive verification and sampled verification are labelled as such.
