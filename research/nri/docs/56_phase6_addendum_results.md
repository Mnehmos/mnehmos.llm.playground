# 56 — Results: Phase VI addendum, bidirectional neuron-level interpretability

**Registration.** Doc 55 (commit f0cb990), committed before any measurement. Everything is single-pass analysis; no
training.

**Data:** `runs/phase6/atlas/`:
- `validation.json`;
- `mass_only_s*.json`: per-neuron atlases, 256 neurons × 7 networks;
- `direct_path.json`;
- `development.json`: 5 networks every 20 steps over 600–2,000, and 7 networks every 100 steps over 0–6,000.

**Code:** `sci/neuron_lens.py`.

**Sources.**
- The neuron-lens repository was read: README, `docs/design.md`, `docs/findings.md`.
- The two Anthropic posts were **not** read. Their hosts are blocked by this environment's network policy, and no URLs
  were supplied. Only the owner's summary of the output-centric principle cᵢ = aᵢ·wᵢ is used.

## Plain-language summary

1. **First we checked the instruments on networks whose answers we know** (the constructed networks of doc 54).
   - The output-side measures passed: what each neuron writes to the answer, and the neuron-by-neuron accounting.
   - Two borrowed tools failed:
     - **the input-side lens (S-lens):** it gave credit to the wrong tokens;
     - **the averaged "dark/bright" lens:** it called the constructed network's real computation "dark", invisible.

     In the constructed network, ablating that "dark" part destroys accuracy at 180–1,200 times what the lens
     predicts. The reason is known there: averaging over ReLU on/off states cancels a staircase of ReLU units.
   - This reproduces the neuron-lens "dark sector" effect in a network where we know exactly why it happens.
2. **The unexplained residual in the learned decoders is dark in exactly this sense.**
   - 99% of its energy lies in directions the averaged lens says do not matter.
   - Restoring only that "dark" part recovers the networks (307: 0.87/0.95; 311: 0.97/0.93). The "bright" part
     recovers nothing.
   - Variance-based and lens-based analyses both miss it for the same reason. The computation lives in *which
     neurons switch on*, not in their average effect.
3. **Each important neuron writes one clean mathematical pattern.** It is a cosine over the discrete logarithm of the
   answer, at the same frequency for nearly every important neuron. Their inputs are a different matter: they come from
   written tokens, not from canonical quantities. The output side is simple; the input side carries the combinatorics.
4. **Native neurons are not the best basis.** The top 8 principal directions of the activations carry about twice as
   much of the decision (0.50–0.59) as the top 8 neurons (0.24–0.27). A random rotation is worst.
5. **Output-weight programming fails, for a structural reason.**
   - A large part of each answer comes directly from the residual stream into the readout, bypassing the MLP neurons.
   - The neurons partly cancel it (correlation −0.3 to −0.6).
   - Rewriting only the neurons' output weights therefore cannot relabel the answer (0.37–0.65). Rewriting the readout
     itself does (1.00 in the generalisers).
6. **Development.** In *every* network that leaves the plateau, during the escape:
   - the important neurons' writes snap to a single clean frequency (purity about 0.4 → 0.99);
   - their input shifts from the embeddings to the second attention layer.

   This happens in failures too (304 does it first). **What separates generalisers is what happens afterwards.** Their
   decoder input becomes *more additive* again (the residual shrinks to 11–13% of variance) and their contributions
   re-concentrate. Failures keep a large non-additive residual (43–64%) and recruit ever more neurons (105–144).

## 56.1 Validation on constructed networks (§55.2)

| test | Design C | Design L | verdict |
|---|---|---|---|
| V0 exact write additivity (E1) | max error 0.0 | 0.0 | **pass** |
| V1 no residual in Design L | — | R²_add = 1.000 | **pass** |
| V2 Design C's residual = unit conversion | R²_add = 0.957; mass value × unit + acceleration value × unit pairs explain 1.000 of the residual (0.68 + 0.32); without the residual 0.27/0.14; restoring the two pairs 1.00/1.00 | — | **pass** |
| V3 hat structure | each unit writes 1–3 classes; up to **7** units carry margin per prompt | 13 | **support passes; the count criterion was mis-specified.** It omitted the competitor class's hats: 7 = 3 for the answer + 3 for the competitor + 1 zero unit |
| V4 S-lens attribution of the mass read to the mass tokens | 0.50 | 0.25 | **fail** |
| V5 bright sector of h within the used dimensions | 95% of the bright direction lies outside the used dimensions; ablating the "dark" sector gives 0.07/0.08 at 183× the lens prediction | 1,157× | **fail** |

**Consequences** (rule of doc 55 §55.2):
- The S-lens token attribution and the averaged-lens sector designation are **not used to interpret learned
  networks**.
- The contribution atlas, the exact decompositions and the exact per-prompt Jacobians are used.

**Why they fail.**
- **V4:** gradient × (embedding − mean embedding) credits positions whose actual content contributes nothing; the
  mean-embedding baseline is not neutral.
- **V5:** the averaged Jacobian replaces each ReLU gate by its mean activity. The hat combinations
  r_{k−1} − 2r_k + r_{k+1} then cancel, so the averaged lens assigns near-zero gain to the directions that carry the
  computation.

This is a **mechanistic explanation of the dark-sector effect** reported by neuron-lens (300–5,700× in Qwen3-8B),
shown here in a network with a known algorithm. The exact per-prompt Jacobian over-predicts instead: 1.07 against an
actual 0.15 in Design C, because ReLU states flip under large ablations.

## 56.2 The contribution atlas of learned networks

All values are over the full domain.

| | 307 | 311 | 300 | 305 | 302 | 304 | 310 |
|---|---|---|---|---|---|---|---|
| accuracy familiar / unseen | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.43 | 1.00 / 0.72 | 1.00 / 0.38 | 1.00 / 0.27 | 1.00 / 0.28 |
| neurons carrying 90% of \|margin\| | 62 | 56 | 69 | 67 | 141 | 105 | 122 |
| top 64: pure single-frequency write (purity ≥ 0.8) | 0.97 | 0.98 | 1.00 | 0.95 | 0.80 | 0.98 | 1.00 |
| top 64: dominant write frequency k | 1 (63) | 5 (64) | 1 (64) | 1 (64) | 1 (37), 2 (27) | 1 (64) | 5 (64) |
| top 64: input explained by ≥ 2 of the canonical M, A, S | 0.02 | 0.00 | 0.00 | 0.00 | 0.23 | 0.02 | **0.78** |

**Functional polysemanticity (N2).**
- The important neurons are **monosemantic on the output side**: one discrete-log cosine at the network's frequency.
- Inputs are tied to written tokens rather than canonical variables; their R² with canonical M, A, S is low. The
  exception is 310, whose neurons mix the canonical M and A and so compute on a canonical code.
- So neurons driven by different operands *do* perform one consistent mathematical function, in the sense of the
  owner's question. The polysemanticity sits on the input side.

## 56.3 Contribution-weighted residual (N3) and the dark residual (N4)

| | 307 | 311 | 300 | 305 | 302 | 304 | 310 |
|---|---|---|---|---|---|---|---|
| without the residual (α = 0) | 0.49 / 0.57 | 0.55 / 0.35 | 0.33 / 0.17 | 0.29 / 0.21 | 0.25 / 0.17 | 0.28 / 0.14 | 0.26 / 0.14 |
| neurons for 90%: variance / output-weighted / Jacobian / actual margin | 71 / 81 / 59 / 62 | 122 / 109 / 74 / 70 | 125 / 116 / 90 / 77 | 79 / 80 / 69 / 70 | 160 / 152 / 120 / 116 | 139 / 130 / 111 / 99 | 146 / 141 / 115 / 112 |
| neurons for 90% of the single-neuron accuracy effect (restore / remove) | 28 / 12 | 39 / 31 | 44 / 52 | 35 / 23 | 86 / 67 | 65 / 63 | 78 / 60 |
| restore residual in the top 16 by variance | **0.82 / 0.84** | 0.59 / 0.45 | 0.47 / 0.22 | 0.56 / 0.46 | 0.37 / 0.21 | 0.34 / 0.17 | 0.38 / 0.20 |
| restore residual in the top 16 by actual margin | 0.58 / 0.74 | 0.63 / 0.58 | 0.47 / 0.19 | 0.39 / 0.33 | 0.34 / 0.23 | 0.34 / 0.15 | 0.32 / 0.18 |
| Spearman: variance ~ actual margin / variance ~ single-neuron accuracy gain | 0.81 / 0.18 | 0.81 / 0.33 | 0.84 / 0.25 | 0.77 / 0.30 | 0.70 / 0.50 | 0.84 / 0.24 | 0.71 / 0.51 |
| residual energy in lens-dark directions of u | 0.992 | 0.996 | 0.992 | 0.992 | 0.994 | 0.997 | 0.997 |
| α = 0 plus dark part / plus bright part | **0.87 / 0.95** vs 0.49 / 0.58 | **0.97 / 0.93** vs 0.59 / 0.41 | 0.70 / 0.44 vs 0.39 / 0.15 | 0.71 / 0.50 vs 0.27 / 0.20 | 0.87 / 0.36 vs 0.27 / 0.18 | 0.96 / 0.26 vs 0.28 / 0.13 | 0.91 / 0.30 vs 0.26 / 0.16 |
| ablate the dark sector of h: accuracy / actual ÷ lens prediction | 0.22 / 0.17, 30× | 0.20 / 0.14, 49× | 0.16 / 0.12, 82× | 0.13 / 0.11, 35× | 0.23 / 0.15, 16× | 0.24 / 0.13, 45× | 0.24 / 0.15, 28× |

**Reading.**
1. **The essential residual is a lens-dark subspace** (the owner's hypothesis is supported):
   - more than 99% of its energy lies in directions the averaged lens rates near-zero;
   - restoring that dark part alone recovers most or all of the decision;
   - the bright part recovers nothing.

   By the validation (§56.1), "dark" here means "carried by *which* ReLUs switch on", not "unimportant".
2. **Raw variance is not the worst guide** (prediction not supported).
   - Variance, output-weighted, Jacobian and actual-margin orderings agree closely (Spearman 0.70–0.84).
   - Variance even gives the best top-16 restoration in 307. But no ordering predicts a single neuron's accuracy effect
     well (0.18–0.51).
   - The residual is *collective*: no small set of neurons holds it. Even 64 neurons restored by any ordering leave
     failures far from full accuracy.
3. **A causal minority still exists** in generalisers: 12 neurons account for 90% of the single-neuron accuracy loss in
   307. It is more diffuse in failures (60–67).
4. The averaged lens under-predicts the effect of dark ablation by 16–82×. The exact per-prompt Jacobian over-predicts
   it by 0.9–2.1×, because of ReLU flips. Neither replaces the actual ablation.

## 56.4 Native versus rotated basis (N5)

Share of the total margin contribution, and keep-only accuracy (familiar / unseen), for the top 8 directions:

| | native neurons | PCA of activations | random rotation |
|---|---|---|---|
| 307 | 0.27, 0.28 / 0.18 | **0.59, 0.58 / 0.50** | 0.11 |
| 311 | 0.26, 0.27 / 0.22 | **0.53, 0.64 / 0.52** | 0.09 |
| 300 | 0.24 | 0.50 | 0.08 |
| 305 | 0.24 | 0.51 | 0.09 |
| 302 | 0.18 | 0.24 | 0.08 |
| 304 | 0.17 | 0.48 | 0.08 |
| 310 | 0.15 | 0.40 | 0.09 |

- The native basis beats a random rotation (2–3×), so neurons are privileged relative to random directions.
- But a *distributed* basis aligned with the activation geometry is about twice as compact and twice as faithful.
- At the level of explanation, these decoders are better described by a low-dimensional activation subspace than by
  individual neurons. This is consistent with the phase-circle code of doc 54, in which each neuron samples the circle
  at a different phase.

## 56.5 Neuron-level programming through output weights (N6)

| | 307 | 311 | 300 | 305 | 302 | 304 | 310 |
|---|---|---|---|---|---|---|---|
| F → F⁵ by rewriting **all** 256 neuron writes (familiar) | 0.46 | 0.44 | 0.48 | 0.38 | 0.74 | 0.75 | 0.64 |
| same, by permuting the readout W_U | **1.00** | **1.00** | 0.89 | 0.95 | 0.88 | 0.86 | 0.86 |
| direct path (h → W_U) class spread / MLP class spread | 67 / 153 | 53 / 127 | 56 / 164 | 86 / 153 | 22 / 74 | 19 / 152 | 29 / 164 |
| correlation of direct path with MLP logits | −0.48 | −0.43 | −0.39 | −0.57 | −0.32 | −0.08 | −0.27 |
| M → M⁻¹ by refitting the writes of the M-dominated neurons (familiar / unseen) | 0.22 / 0.11 (24 neurons) | 0.18 / 0.12 (49) | 0.19 / 0.08 | 0.21 / 0.10 | 0.65 / 0.20 | 0.71 / 0.17 | 0.20 / 0.10 |

**Deviation note.**
- The registered N6(a) ordered neurons by importance and reported the number needed. Even all 256 are insufficient.
- The cause was found after the run. The residual stream's direct path to the readout carries large class-dependent
  logits that the MLP partly cancels. The direct-path margin averages −0.8 to +1.2, so it was invisible to the
  margin-based atlas.
- This is reported as a finding, not fixed: an output-weight edit must include the residual stream's own "write" (W_U)
  to act on the whole computation.

**Reading.**
- Operand-specific computation is **not localised** in M-dominated neurons: refitting their writes cannot implement
  M⁻¹.
- In the constructed networks the same edits are exact, because their readout dimensions are clean (doc 54).

## 56.6 Development through the critical window

**Data:** `development.json`. The table gives step: accuracy familiar / unseen; neurons for 90% of margin;
median purity of the top-32 writes; residual variance share; share of the top-32's input from layer 2.

| | 307 (gen.) | 311 (gen.) | 305 (partial) | 304 (fail) | 310 (fail) |
|---|---|---|---|---|---|
| 600 | 0.24 / 0.15; 64; 0.41; 0.09; 0.34 | 0.27 / 0.15; 83; 0.42; 0.19; 0.36 | 0.23 / 0.15; 75; 0.36; 0.07; 0.46 | 0.25 / 0.16; 88; 0.44; 0.18; 0.30 | 0.24 / 0.15; 90; 0.37; 0.06; 0.31 |
| purity ≥ 0.9 first reached | **900** | **900** | **1,200** | **800** | **1,600** |
| peak residual share during escape | 0.44 (900) | 0.45 (900) | 0.42 | 0.52 | 0.43 |
| 2,000 | 1.00 / 0.99; 64; 0.99; **0.13**; 0.74 | 0.99 / 0.95; 59; 0.99; **0.11**; 0.72 | 1.00 / 0.82; 66; 1.00; 0.34; 0.68 | 1.00 / 0.27; 105; 0.99; **0.52**; 0.77 | 1.00 / 0.27; 109; 1.00; **0.43**; 0.74 |

**At 100-step resolution, steps 0–6,000:**
- **300 (partial):** purity 1.00 from step 1,200; residual share 0.25–0.28; 66–74 neurons.
- **302 (fail):** frequency **k = 2** from step 1,200 (folding 12 phases onto 6), later mixing k = 1 and k = 2;
  residual share rises to 0.64; 144 neurons by step 6,000.

**Reading** (doc 55 §55.4).
1. **The escape is marked in every network**, generaliser or not, by two transitions:
   - **output purification:** the important neurons converge on a single discrete-log frequency;
   - **connectivity reorganisation:** input moves from the embeddings to the layer-2 heads (0.3 → 0.7–0.8).

   The registered prediction that only generalisers acquire pure writes is **not supported**. 304 purifies first,
   at step 800.
2. **Reusable computation corresponds to what happens to the *input* side after the escape.**
   - In generalisers, the decoder's input becomes more token-additive again: the residual share falls from 0.44 to
     0.11–0.13. The contribution concentrates (n90 back to about 60), and the network runs without the residual at
     about half accuracy.
   - In failures, the non-additive input persists or grows (0.43–0.64), more neurons are recruited (105–144), and
     accuracy without the residual stays at chance-plus (about 0.28 / 0.14).
   - The partial networks fall in between (0.28–0.34).
3. This matches the factorisation story (docs 46 and 49) from the decoder's side. Generalisers keep or restore an
   additive (factorised) input to a single-frequency readout. Failures build a pairwise (lookup) input to the same
   kind of readout. **The readout is shared. What differs is whether its input is composable.**
4. **Previously hidden subspace.** The residual that differs between the groups is precisely the averaged-lens-dark
   component (§56.3). It would be missed by variance-ranked PCA of the decoder input or by averaged lenses. It is seen
   by exact restoration and by the additive-versus-pairwise decomposition.

## 56.7 Distinctions kept

| kind | measured by | main result |
|---|---|---|
| **mathematical predictability** | R² of token-additive and pairwise fits | 86–90% additive in generalisers |
| **interpretability** | each neuron's stated function | single-frequency discrete-log writer; token-level inputs |
| **causal importance** | restoration and ablation | the dark residual is essential; output-only edits fail because of the direct path; operand edits are not neuron-local |

Each of the three ranks neurons differently (§56.3), so none substitutes for the others.

## 56.8 Implications

**For neuron-level methods in large models** (relating to neuron-lens and, by the owner's account, the Anthropic
posts):
- Output-side reads (what a neuron writes) were reliable and interpretable here.
- Averaged input-side reads and averaged dark-sector designations failed validation on networks with known algorithms.
  Where computation is carried by which units switch on, averaging hides it.
- Exact, context-specific Jacobians and actual ablations are required to decide what matters.
- A residual-stream "direct write" can carry a large, class-specific signal that neurons cancel. A neuron-only account
  of a decision can then be incomplete even when every neuron is understood.
