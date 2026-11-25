# 58 — Results: Phase VII, robust neural compilation and hybrid learning

**Registration.** Doc 57 (commit 79f8cb1) and its addendum A (the H/K configuration), both committed before the
corresponding runs.

**Data and code.** Data: `runs/phase7/`. Code:
- `sci/direct_path.py`: D;
- `sci/instruments2.py`: I;
- `sci/robust.py`: R;
- `sci/graph_compiler.py`: G;
- `sci/fourier.py`: F;
- `sci/hybrid.py`: H and K.

**Labels.**
- Kinds of evidence:
  - **[M]** mathematical guarantee;
  - **[E]** exhaustive verification;
  - **[S]** sampled verification;
  - **[T]** observed in trained networks.
- Kinds of network: learned, constructed, hybrid-trained.

## Plain-language summary

**The central question: can known computations be compiled, unknown ones learned, and both coexist in one network
without interfering?** Yes, with one condition. Freezing the compiled part is not enough; the learned parts must also be
prevented from writing into the compiled part's working space.

**1. Hybrid learning works when the programmed part is protected properly.**
- In a network holding an exactly compiled F = M·A·S, a new operation (U = M·A + S) was learned in **600–800 steps**.
- It generalises to never-seen renderings (0.99–1.00). The programmed operation stays **exactly** intact on every
  prompt.
- The same network trained from scratch did **not** reach the criterion in 6,000 steps.
- **Freezing alone fails** (0.25 on the new operation). The learned parts write into the frozen part's dimensions and
  corrupt it. **Blocking those writes is what makes coexistence work.**

**2. Computational graphs compile exactly.**
- Every tested expression compiles exactly: M·A + S, (M + A)·S, M² + A², (M − A)/(M + A), and 60 random nested
  expressions of depth 1–6.
- Every intermediate variable is observable bit for bit, causally swappable and independently reprogrammable.
- **Cost grows linearly:** about 24 hidden units and about 500 parameters per node.

**3. The interpretability instruments were partly fixed.**
- The input-side lens now works: exact "what if this token were different" replacements.
- The averaged sector lens could not be validated, so the "dark residual" claim stays **unresolved**.
- On learned networks, exact per-prompt Jacobians predict ablations almost perfectly (ratio ≈ 1.0, rank
  correlation 0.998). Averaged Jacobians under-predict by 4–8×.

**4. Robustness remains the main gap.**
- The constructed networks are fragile because a few huge attention gains share tensors with ordinary weights. 82% of
  their nonzero weights are individually critical.
- Balancing and minimising the attention gains improves robustness about ten-fold while keeping exact answers. The
  networks are still about ten times more fragile than learned ones.
- Wider margins, flat-topped units, redundant copies and periodic (Fourier) codes did not close the gap. **Geometry
  alone does not explain the learned networks' robustness.**

**5. The residual-stream "direct path" fully explains why output-only reprogramming failed in seed 307.** The neurons
can absorb it when refitted together. Genuinely changing the computation, as opposed to relabelling answers, needs
input-side edits.

## 58.1 D: the direct residual pathway [T, E]

**Data:** `direct_path.json`.

- **Exact decomposition.** The logits decompose into the bias, the direct path (embedding, layer 1, each layer-2 head)
  and the MLP neurons, with reconstruction error ≤ 1.6·10⁻⁴.
  - The MLP carries 88–106% of the decision margin.
  - The direct path carries almost none of the margin but has a large class spread (19–86).
  - It is anti-correlated with the MLP (−0.08 to −0.57) and comes mainly from individual layer-2 heads (307: head 0
    spread 49.5).
- **Coordinated relabelling F → F⁵.** A least-squares refit of W_out and b_out on familiar prompts, with the direct path
  left in place, reproduces the W_U permutation **exactly** in all 7 networks.
  - Familiar accuracy is 1.00 and unseen accuracy equals each network's own.
  - Fit R² is 0.999–1.000.
  - So the 307 failure of doc 56 §56.5 was the direct path, which the neurons can absorb if refitted jointly.
  - The registered prediction ("only partial") is **not supported**: coordination succeeds fully.
- **Internal operation F(M⁻¹, A, S) through output weights.**
  - Learned networks: familiar 0.21–0.43, unseen 0.14–0.23.
  - Design C: 0.36 through output weights, but **exact** through the W_in operation swap (doc 54).
  - **Output-side edits relabel answers; internal changes need input-side (W_in) edits.**

## 58.2 I: corrected instruments [E on constructed networks, T on learned]

**Data:** `instruments_validation.json`, `instruments_learned.json`.

**Validation on Designs C and L.**

| instrument | criterion | Design C | Design L | verdict |
|---|---|---|---|---|
| I1 interventional S-lens | mass-pair share ≥ 0.95 (V4′) | 1.000 | 1.000 | **qualifies** |
| I1b integrated gradients, type-conditional baseline | V4′ | 1.000 | 1.000 | **qualifies** |
| I2 gate-aware sector lens | V5′ | bright inside used dims 1.00; dark ablation keeps 1.00/1.00; actual/predicted **2.25** (criterion ≤ 2) | passes | **does not qualify** (fails narrowly on Design C) |
| I3 interventional sectors | V5′ | dark ablation 0.49/0.39 | passes | **does not qualify** |

**Causes confirmed.**
- **V4:** the vocabulary-mean baseline. Type-conditional finite interventions remove the spurious credit entirely.
- **V5:** averaging over gates. The gate-aware lens fixes the *designation* (the bright sector lies inside the used
  dimensions and dark ablation is harmless), but not the *magnitude* prediction on piecewise-constant ladders.
- **I3** fails because single-direction effects do not add on a one-hot code.

**Re-test of the "dark residual" (doc 56).** No sector instrument qualified, so by the registered rule the question
stays **unresolved**. Doc 56's dark-residual statements remain provisional.

**Learned networks (qualified S-lens only).**
- The leading mass direction of the declared block draws 0.96–0.98 (307/311) and 0.90–0.98 (others) of its
  interventional credit from the mass value and mass unit positions.
- The 8 most important decoder neurons each mix the operands, on average 0.45–0.56 mass, 0.29–0.46 acceleration and
  0.06–0.21 output unit, in every network.

**I4: prediction of random-subspace ablations.**

| network | averaged J: actual/predicted (Spearman) | exact per-prompt J | gate-aware |
|---|---|---|---|
| Design C | 5.17 (0.93) | 0.23 (0.97) | 0.22 (0.98) |
| Design L | 1.41 (0.99) | 0.06 (0.75) | 0.05 (0.77) |
| learned (7 networks) | 3.7–8.4 (0.90–0.96) | **0.94–1.02 (0.998–0.999)** | **0.93–0.99 (0.993–0.998)** |

**Reading.** In learned networks, exact or gate-aware Jacobians are near-perfect predictors of finite ablations, and the
averaged lens under-predicts 4–8×. In the constructed ladders, no linear predictor is reliable. Learned decoders are
locally smooth; constructed ones are deliberately piecewise-constant.

## 58.3 R: why constructed networks are fragile, and constructed remedies [E at σ = 0; S under noise, 5 seeds]

**Data:** `robust.json`.

### Accuracy under noise

Noise models:
- **(a)** additive, σ·RMS of each tensor;
- **(b)** multiplicative;
- **(c)** additive on h.

| network | (a) 10⁻³ / 10⁻² | (b) 10⁻³ / 3·10⁻³ / 10⁻² / 3·10⁻² | (c) 10⁻² / 3·10⁻² | single weights critical | exact at σ = 0 |
|---|---|---|---|---|---|
| learned 307 | 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 | — | — |
| learned 311 | 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 | — | — |
| Design C | 0.11 / 0.09 | 0.36 / 0.22 / 0.19 / 0.19 | 1.00 / 0.74 | 0.82 | bitwise |
| R1a balanced q/k | 0.32 / 0.17 | 0.36 / 0.22 / 0.19 / 0.19 | 1.00 / 0.74 | 0.82 | bitwise |
| **R1 balanced + minimal gain (γ = 0.024)** | **0.85** / 0.24 | **1.00 / 0.93** / 0.46 / 0.24 | 1.00 / 0.74 | 0.82 | argmax (logits one-hot 0.6%) |
| R2 trapezoid ladder | 0.09 / 0.07 | 0.36 / 0.22 / 0.19 / 0.18 | 1.00 / 0.74 | 0.73 | bitwise |
| R3 redundant ladders (K = 2) | 0.09 / 0.08 | 0.36 / 0.22 / 0.19 / 0.18 | 1.00 / 0.74 | 0.67 | bitwise |
| R4 margin × 10 | 0.10 / 0.09 | 0.36 / 0.22 / 0.19 / 0.19 | 1.00 / 0.74 | 0.82 | argmax |
| R1 + R2 + R3 | 0.85 / 0.22 | 1.00 / 0.94 / 0.47 / 0.24 | 1.00 / 0.74 | 0.64 | argmax (logits one-hot 86%) |

### Component isolation (Design C)

| component perturbed alone | (a) | (b) |
|---|---|---|
| attention-1 and attention-2 qkv | 0.15 already at 10⁻⁴ (the gain-mixed tensors) | attention-2 qkv 0.97 at 10⁻⁴, 0.48 at 10⁻³ |
| embeddings | ≥ 0.96 up to 3·10⁻⁴ | 0.98 at 10⁻⁴, 0.48 at 10⁻³ |
| W_in | 1.00 up to 3·10⁻³ | 0.74 at 10⁻² |
| attention o, W_out | 1.00 up to 10⁻², or 3·10⁻² for W_out | 1.00 up to 10⁻² |
| W_U | 1.00 at every level | 1.00 at every level |

### Reading

- **Sources separated.**
  - **(i) Attention saturation and gain** is the dominant source. Gains of 4,000–12,000 share tensors with 0/1 value
    weights, so tensor-relative noise swamps the routing.
  - **(ii) Embedding precision** comes second. One-hot codes have no slack.
  - **(iii) The ReLU ladder** (W_in) matters only at about 1%.
  - **(iv) The decision margin** is irrelevant, as predicted: scaling the readout cancels under scale-relative noise.
  - **(v) Single-weight dependence** is extreme: 82% of nonzero weights are critical. Redundancy reduces this to 64%
    but does not move the noise curves, because the bottleneck is upstream of the ladder.
- **Best constructed variant (R1).** About 10× more robust than Design C under (b): 1.00 up to 10⁻³, 0.93 at 3·10⁻³.
  Answers are still exact on all 9,126 prompts, and the instruction-system edits are still exact (6/6). It remains
  about 10× less robust than learned generalisers, which hold 1.00 to 3·10⁻².
- **The registered target is not met.** The R1 + R2 + R3 prediction is **not supported** (0.47 at 1% under (b)).
- **Interpretation [I].** Exactness here is bought with *saturated, sparse routing*: hard attention through a few large
  weights. The learned networks route softly and redundantly through many moderate weights. Closing the gap probably
  needs distributed routing, meaning many heads or positions each carrying a share. That is a construction not yet
  attempted.

## 58.4 G: the compositional compiler [E]

**Data:** `graph_compiler.json`.

**Architecture.** The Design-C encoder, followed by residual MLP blocks, one per depth level. Each graph node is a
hat ladder that writes its value's one-hot code into its own residual region.

| expression | blocks | MLP units | decoder nonzero parameters | exact | intermediates observable / interchange | node swap |
|---|---|---|---|---|---|---|
| M·A + S | 2 | 52 | 1,181 | yes | M·A: bitwise / 1.00 | M·A → M + A: exact |
| (M + A)·S | 2 | 50 | 1,138 | yes | M + A: bitwise / 1.00 | → M·A: exact |
| M² + A² | 2 (**1 fused**) | 57 (27 fused) | 1,228 (762) | yes | M², A²: bitwise / 1.00 | — |
| (M − A)/(M + A) | 2 | 81 | 2,211 | yes | M − A and M + A: bitwise / 1.00 | → (M/A)/(M + A): exact |
| 60 random trees, depth 1–6 | 1–6 | 12–455 | up to 9,745 | **60/60** | **all bitwise / all 1.00** | — |

**Scaling** (fits over the 60 random trees):
- MLP units ≈ 23.8 × nodes − 2.9 (R² = 0.985);
- decoder nonzero parameters ≈ 509 × nodes − 88 (R² = 0.956);
- residual width D = 64 + 13 × nodes, exactly;
- depth = the number of nonlinear stages on the longest path. Unary per-variable functions fuse into the parent's
  statistic, as M² + A² does.

There are no lookups over input combinations: each node's statistic is a sum of per-child terms. The registered
prediction is **supported**.

## 58.5 F: geometric (Fourier) compilation [M + E for argmax; S under noise]

**Data:** `fourier.json`.

**Derivation.**
- Adding phase vectors does not add phases (D0). Products of phases are formed instead by
  xy = ((x+y)² − (x−y)²)/4.
- Each square is computed **exactly on the finite set** of values x ± y can take, by piecewise-linear ReLU
  interpolation.
- Unit conversion is a fixed rotation applied by the correction heads.
- Stage 2 rotates by S, using gated products g·z = ReLU(z − 2(1−g)) − ReLU(−z − 2(1−g)).

**Results.**
- **Exact argmax on all 9,126 prompts** for K = {1} and for K = {1, 5, 7, 11}.
- Real-arithmetic exact. In float32 the result is argmax-exact, **not bitwise**, because the code values include √3/2.
- **Margins:** exactly 1 − cos 30° = 0.134 for K = {1}, and 2.0 for K = {1, 5, 7, 11} (4 minus the largest
  off-target Ramanujan sum).
- **Size:** 174 MLP units and 880 nonzero parameters for K = {1}; 690 and 3,434 for K = 4.

**Robustness** (noise model (b): 10⁻³ / 3·10⁻³ / 10⁻²):

| network | 10⁻³ | 3·10⁻³ | 10⁻² |
|---|---|---|---|
| Design C | 0.36 | 0.22 | 0.19 |
| Design L | 1.00 | 0.91 | 0.48 |
| Fourier, K = 1 | 0.91 | 0.67 | 0.28 |
| Fourier, K = 4 | 0.84 | 0.62 | 0.37 |

- Margin-normalised variants are identical to the unnormalised ones: the margin is scale-invariant here.
- Under (a), every constructed network fails at 10⁻⁴, because it uses the same saturated attention encoder.

**Reading.** At matched routing (the same encoder), the periodic code is no more robust than the scalar log code.
**Geometry alone does not reproduce the learned networks' robustness.** The registered prediction is supported; the
dominant factor is the routing (§58.3).

## 58.6 H and K: hybrid learning and knowledge protection [T; 3 seeds per condition; full and 25% data]

**Data:** `runs/phase7/hybrid/*.json`; condition A in `A.json`.

**Condition A (entirely constructed, two stages, gated by operation):** **exact** on both operations over the full dual
domain of 2 × 9,126 prompts [E]. It has 3,501 nonzero parameters.

**Full data, final step 6,000 (mean of 3 seeds):**

| condition | trainable parameters | steps to OP₂ criterion | OP₁ familiar / unseen | OP₂ familiar / unseen | OP₁ sub-circuit exact / silent on OP₂ | OP₁ M-interchange |
|---|---|---|---|---|---|---|
| B (entirely trained) | 269,952 | none in 6,000 | 0.85 / 0.82 | 0.82 / 0.75 | — | 0.05 |
| B′ (trained, matched to C's trainable count) | 237,720 | none | 0.78 / 0.70 | 0.78 / 0.71 | — | 0.05 |
| C = K1 (constructed frozen) | 191,516 | none | 0.43 / 0.36 | 0.25 / 0.16 | 0.00 / 0.00 | 0.06 |
| K3 (frozen; writes into constructed dims frozen at random init) | 191,516 | 1,500–5,600 | 0.94 / 0.80 | 0.93 / 0.79 | 0.00 / 0.00 | 0.27 |
| K0 (constructed, unprotected) | 269,952 | 4,500 (1 of 3 seeds) | 0.65 / 0.56 | 0.67 / 0.60 | 0.00 / 0.00 | 0.05 |
| D = K4 (semantic-preservation loss only) | 269,952 | none | 0.32 / 0.24 | 0.35 / 0.28 | 0.00 / 0.00 | 0.16 |
| **K2** (frozen + learned writes into constructed dims held at 0) | 191,516 | **600–800** | **1.00 / 0.99** | **1.00 / 0.99** | **1.00 / 1.00** | 0.66 |
| **K5** (K2 + semantic-preservation loss) | 191,516 | **600** | **1.00 / 1.00** | **1.00 / 0.995** | **1.00 / 1.00** | **1.00** |

**25% data:** HYBRID_Q25

### Reading

1. **Known computations can be installed permanently while unknown ones are learned**, provided the learned components
   cannot write into the installed mechanism's residual subspace (K2, K5). Then:
   - the programmed operation's sub-circuit stays **exactly** one-hot on every OP₁ prompt and exactly silent on every
     OP₂ prompt [E at every evaluation];
   - the new operation is learned about 10× faster than any other condition reaches even 0.9;
   - it generalises to unseen renderings, by reading the installed canonical codes.
2. **Parameter masks alone are insufficient, even harmful.**
   - C does worse than training from scratch, and so does D.
   - Randomly initialised learned heads write into the constructed regions from step 0, which destroys the sub-circuit
     (exactness 0.00). The learned parts then have to compensate.
   - Freezing those writes at their random values (K3) is better (0.93) but still not exact.
   - **Interference through the shared residual stream, not drift of the protected weights, is what destroys programmed
     knowledge.** The registered prediction (K1 fails, K2/K5 meet the target, K0 forgets) is **supported**.
3. **A semantic-preservation loss alone (D) does not protect;** combined with structural protection (K5), it adds causal
   faithfulness. The end-to-end M-region interchange is 1.00 in K5 against 0.66 in K2.
4. **Constructed mechanisms reduce training steps** (at least 7.5× to criterion against B, which never reaches it) and
   **trainable parameters** (191,516 against 269,952), and they confer unseen-rendering generalisation.

## 58.7 Answer to the central question

**Can we build networks whose known computations are compiled, whose unknown computations are learned, and whose
internal components remain observable, robust and independently programmable?**

- **Compiled and learned can coexist.** With residual-subspace protection, an exactly compiled operation stays exact
  while a new one is learned faster, with fewer trainable parameters and better generalisation than from scratch. [T]
- **Observable and independently programmable.** In compiled components, and through arbitrary computational graphs,
  every intermediate variable is observable, swappable and reprogrammable. [E]
- **Robust: not yet.** Constructed mechanisms are about 10× more fragile to weight noise than learned ones, because
  exact computation is currently bought with saturated, sparse routing. Balanced minimal-gain attention closes about 10×
  but not the rest. Margins, redundancy in the readout and periodic geometry do not help. [S]

**Which parts of learning can be replaced by engineering:**
- the representation of known variables (canonical codes);
- unit conversion;
- arithmetic on known variables;
- composition through intermediate variables.

**What learning still contributes:**
- robustness through distributed, soft routing;
- the unknown operations themselves.
