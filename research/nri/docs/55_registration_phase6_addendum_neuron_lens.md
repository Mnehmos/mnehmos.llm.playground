# 55 — Registration: Phase VI addendum, bidirectional neuron-level interpretability

Committed **before** any measurement below is run.

**Authorisation.** The owner's Phase VI addendum, which asks for:
- output-centric neuron analysis;
- the neuron-lens methodology;
- a bidirectional Concept Formation Atlas;
- contribution-weighted residual analysis;
- developmental tracking;
- validation on constructed networks first.

All work is single-pass analysis of existing or constructed networks. There is no training.

## 55.0 Sources and what was verified

- **neuron-lens** (github.com/stdlogits/neuron-lens; README, `docs/design.md`, `docs/findings.md`). **Read.** The
  definitions below are taken from those files, adapted to our architecture.
  - **R-lens:** the average Jacobian from a layer to the final residual, under RelP rules, read through W_U.
  - **S-lens:** the transpose direction, the average Jacobian from token embeddings to the layer, read through W_E
    as (W_E[v] − ē).
  - **Per-firing exact linearisation:** a firing's contribution is the rank-one map x → w(c·x).
  - **Coordinate sectors:** top-11% coordinates by readout share (R) and by token share (S); core c = R∩S,
    r = R∖S, s = S∖R, d = the rest.
  - **Dark sector:** for the readout matrix's eigen-directions ordered by gain, the bright sector is the fewest that
    carry 90% of the lens energy of a layer's writes; the dark sector is the rest. Ablation replaces a sector by its
    held-out mean.
  - The repository reports that dark-sector ablation costs 300–5,700× what the lens predicts.
- **Anthropic, "Returning to the Neuron Basis for Interpretability" and "Using Neurons for Safety Applications"
  (owner-cited, November 2026).** **Not read.**
  - alignment.anthropic.com and transformer-circuits.pub are not reachable from this environment's network policy,
    and no URLs were supplied.
  - They are used only for the owner-stated principle that a neuron's computational contribution is its activation
    times its output weight vector, cᵢ(x) = aᵢ(x)·wᵢ, together with the owner's summary.
  - They will not be quoted or characterised beyond that until they are read.

## 55.1 Adaptation to our networks: exact identities

**Decoder.** h ∈ ℝ⁶⁴ (last-token residual) → u = W_in·h + b → a = ReLU(u) (256 neurons) → r = h + W_out·a + b_out →
logits = W_U·r.

- **E1. Exact additivity of neuron writes.** logits(x) = W_U(h + b_out) + Σᵢ aᵢ(x)·qᵢ, with qᵢ = W_U·W_out[:, i] ∈ ℝ¹³.
  - Each neuron's write to the logits is exactly rank one, and context only scales it.
  - Zero-ablating neuron i changes the logits by exactly −aᵢ(x)·qᵢ, so at this layer the output-weighted contribution,
    the Jacobian prediction and the ablation effect coincide on the logits.
  - They differ only after the argmax (accuracy). This identity is checked numerically as instrument validation V0.
- **E2. Margin contribution.** mᵢ(x) = aᵢ(x)·(qᵢ[y] − qᵢ[y*]), where y is the correct class and y* the strongest
  competitor of the intact network.
- **E3. Exact context Jacobian of the residual.** J(x) = ∂r/∂h = I + W_out·D(x)·W_in, with D(x) = diag(u(x) > 0).
  - This is exact everywhere except on measure-zero ReLU boundaries, and it is piecewise constant.
  - RelP's rules (norm and gating linearisation) reduce to true gradients here: there are no norms or gates.
  - So the **R-lens equals the J-lens**: R̄ = W_U·J̄, with J̄ = the mean of J(x) over the 9,126 prompts.
- **E4. Exact S-lens.** For each position s, K_s(x) = ∂h/∂e(s), the Jacobian of the last-token residual with respect
  to the embedding at position s, by autograd through both attention layers.
  - The S-read of a direction v for token t at position s is v·K_s·(W_E[t] − ē).
  - This is computed both per prompt (exact) and averaged per position (the lens).
- **E5. Exact upstream decomposition.**
  - h = x₀[last] + Σ_heads (layer-1 head output at last) + Σ_heads (layer-2 head output at last), exactly, with no
    norms. Note that layer-1 outputs at other positions reach h only through layer 2.
  - Hence uᵢ = W_in[i]·h + bᵢ splits exactly into embedding, layer-1 direct, and per-head layer-2 contributions.
    This is the "connectivity at firings" measure.

## 55.2 Instrument validation on constructed networks (run first; interpretation is gated on it)

Ground truth comes from Design C and Design L (doc 54):

| test | requirement |
|---|---|
| **V0** | E1 holds to float32 rounding on all networks (max abs error < 10⁻⁴ of the logit scale) |
| **V1** (Design L) | the token-additive fit of u is exact (R² = 1, since the hats read sums of per-token logs), so the residual is 0 |
| **V2** (Design C) | the token-additive residual is nonzero, and the pairwise decomposition attributes it to mass value × mass unit and acceleration value × acceleration unit (the unit conversion), with ≥ 0.95 of residual variance; removing it destroys accuracy, and restoring only those pairs restores 1.00 |
| **V3** (Design C) | the contribution atlas recovers the hat structure: each unit's qᵢ is supported on the classes g(k) of its hat, and the units whose margin contribution is nonzero for a prompt are exactly the ≤ 3 hats adjacent to that prompt's t, plus the zero units |
| **V4** | the S-lens attributes Design C's M-region units to the value and unit tokens of the mass pair only, at the correct positions |
| **V5** (sectors) | in Design C the bright sector of h is contained in the used dimensions (M/A/FU/flags); unused dimensions are dark and ablating them changes nothing |

A measure that fails its validation test is reported, but it is not used to interpret learned networks.

## 55.3 Measurements on learned networks

**Networks:** 307, 311 (generalisers); 300, 305 (partial); 302, 304, 310 (failures). All 9,126 prompts.

- **N1. Contribution atlas (bidirectional, per neuron).**
  - *Output side:* qᵢ; its Fourier purity over the discrete-log classes (the fraction of qᵢ's energy at its best
    frequency k of cos/sin(2πk·ℓ(c)/12)); mean |mᵢ| on familiar and unseen prompts; the fraction of prompts on which
    the neuron is active.
  - *Input side:* the R² of uᵢ explained by each variable (canonical M, A, S; written tokens; order); its S-lens
    token reads per position; its exact connectivity per E5.
  - *Causal:* the single-neuron zero-ablation and mean-ablation Δaccuracy and Δmargin, familiar and unseen.
  - *Developmental:* §55.5.
- **N2. Functional polysemanticity.**
  - Input polysemanticity: the number of variables with R² ≥ 0.2 for uᵢ.
  - Output monosemanticity: Fourier purity ≥ 0.8.
  - Report the joint distribution. The question is whether neurons driven by different operands write one consistent
    mathematical function (a single-frequency phase pattern).
- **N3. Contribution-weighted residual analysis.** Split each neuron's activation change Δaᵢ = ReLU(ûᵢ + ρᵢ) − ReLU(ûᵢ),
  where û is the token-additive fit and ρ the residual (doc 54 §54.4). Rank neurons and compare the four orderings:

  | measure | definition |
  |---|---|
  | (i) raw variance | var(ρᵢ) |
  | (ii) output-weighted | E\|ρᵢ\|·‖qᵢ‖ |
  | (iii) Jacobian-predicted margin effect | E[1(uᵢ > 0)·ρᵢ·(qᵢ[y] − qᵢ[y*])] |
  | (iv) actual causal effect | restore ρ for neuron i alone from α = 0, and remove it alone from α = 1, measuring Δmargin and Δaccuracy |

  - **Outputs:**
    - Spearman correlations between the four orderings;
    - the number of neurons needed for 90% of each measure's total;
    - the accuracy recovered by restoring ρ in the top-k neurons by each ordering, for k ∈ {4, 8, 16, 32, 64}.
  - **Prediction:** (ii)–(iv) agree with each other more than with (i); a minority of neurons (≤ 32) carries ≥ 90% of
    the causal effect; ranking by variance is the least efficient.
- **N4. Sectors and the dark subspace (adapted).**
  - For h: per-coordinate readout share (with A = J̄ᵀ·W_Uᵀ·W_U·J̄) and token share (from the S-lens).
  - Sectors c, r, s, d use the top 7 of 64 coordinates (11%), in the native basis and in a random rotation (control).
  - Dark sector of h: from the eigen-directions of W_U·J̄, ordered by gain; bright = the fewest carrying 90% of the
    lens energy of h.
  - Ablation: mean-replace a sector (means from familiar prompts) and report Δaccuracy and Δmargin against the
    lens-predicted change. The same is done for the decoder pre-activation residual ρ: project ρ onto the dark and
    bright sectors of W_in-space and ablate each.
  - **Question:** is the essential residual concentrated in directions that the averaged lens calls dark, but that
    matter causally (the neuron-lens dark-sector phenomenon)? Exact per-context Jacobians are compared with the
    averaged lens throughout: the Jacobian prediction error of the averaged versus the per-prompt J.
- **N5. Native versus rotated basis.** The top-k native neurons against the top-k directions of PCA on the
  post-ReLU activations (and of a random rotation of activation space), for k = 1–64, measuring:
  - the share of total margin contribution captured;
  - the faithfulness of top-k ablation (accuracy after removing everything else, and after removing only them).
- **N6. Neuron-level programming through output weights.** Keep W_in and b (the activation mechanism) and rewrite only
  W_out columns.
  - **(a)** Output operator F → F⁵ (5 is invertible mod 12) applied per neuron, as qᵢ' = qᵢ∘σ. Measure the number of
    neurons whose writes must be rewritten, ordered by causal importance, to reach ≥ 95% on the new target, familiar
    and unseen.
  - **(b)** Operand-selective edit: rewrite only the neurons whose input is dominated by M so that their write acts as
    if M were replaced by M⁻¹, with qᵢ' fitted by least squares on familiar prompts to the target F(M⁻¹, A, S),
    using those neurons' activations only. Measure the familiar and unseen accuracy on the edited function. This
    tests whether operand-specific computation is localised in neurons.
  - The same procedures are run on Design C as a control (expected: (a) exact with all writing units; (b)
    impossible, since Design C's hats read the summed statistic, not M alone, and this is reported as such).

## 55.4 Developmental tracking through the critical window

**Checkpoints.**
- 307, 311, 305, 304 and 310: every 20 steps over steps 600–2,000 (10-step window replay, `/home/user/swap_window`).
- All 7 networks: every 100 steps over steps 0–6,000 (`/home/user/swap_recon/zoo`).

**Per checkpoint:**
- the number of neurons carrying 90% of |margin contribution| (contribution concentration);
- the median Fourier purity of the 32 most important neurons' writes;
- the readout share of the declared block (dims 0–15) in the R-lens;
- the bright-sector size of h;
- the dark-ablation Δaccuracy;
- the residual ρ's share of causal margin effect against its share of variance;
- connectivity: the fraction of important neurons' input driven by layer-2 heads against embedding or layer 1 (E5).

**Analysis.**
- Change points of each series, against the escape from the shared plateau and the factorisation dip
  (docs 45 and 46).
- **Question:** does the emergence of reusable computation (307 and 311 around steps 900–1,400) coincide with an
  identifiable change in output purity, contribution concentration, connectivity or dark-sector size that does not
  occur in 304 and 310?
- **Prediction:** generalisers acquire single-frequency (high-purity) writes from an M-and-A-mixed input population
  during their escape. Failures acquire low-purity, lookup-like writes driven by mass value × mass unit.

## 55.5 Standards

- Validation (§55.2) runs before learned-network interpretation.
- Three things are kept distinct:
  - **mathematical predictability:** variance explained by a model;
  - **interpretability:** a stated function per neuron, such as purity or input variables;
  - **causal computational importance:** ablation or restoration effects.
- Negative results and failed validations are kept. There is no seed selection, and exploratory refinements are
  labelled.
- The atlas is written to `runs/phase6/atlas/*.json`, one file per network, with a summary in doc 56.
