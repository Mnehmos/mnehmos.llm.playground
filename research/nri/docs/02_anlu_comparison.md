# 2. ANLU against its specification, the literature, and the experiments

*Revised 2026-11-22 against the original documents, read in full:*

- *ANLU: Typed Token Embeddings as Structural Priors for Language Model Pre-Training. Foundation
  Specification (Vario / Mnehmos Research, Feb 2026)*
- *ANLU & Pedagogy as Training Alternatives v3.0*

*Companion papers (Typed Loops, Three Scaling Axes, Learned Heuristics) are reviewed in doc 07.*

*An earlier version of this document was written before the specification was available, and
assumed types are supplied at inference. That assumption was wrong. The specification says the
opposite, and doc 08 tests the specified version.*

## 2.1 What the specification claims

| Spec item | Content |
|---|---|
| **Claim 1, the bootstrapping tax** | Compute spent discovering functional structure that could be declared up front is wasted capacity. The specification argues this tax "may grow with scale". |
| **Component 1, whole-word tokenization** | Whole words, compound units (m/s²), numbers and multi-word domain terms each become one token; numbers and units are separate tokens; polysemy is kept through context-dependent typing. |
| **Component 2, typed embeddings** (Eq. 1) | `e_i = c_i + t_{τ_i} + p_i`, with 15–38 type ids. A soft prior that "pre-separates the representational space into functional regions from step zero". |
| **Training/inference asymmetry** | "Type vectors are present during training. At inference, they are not required. The model has internalized the geometric structure." Pedagogy v3.0 gives the mechanism: "an auxiliary type prediction objective". |
| **Type schema** | Six layers (closed class, open class, rhetorical, quantitative, domain, cognitive/MCP). Higher layers override lower; resolution takes the highest active layer (Def. 2). Types are contextual (Principle 2). |
| **Labeling** | A teacher LLM labels the data, checked by inter-annotator agreement (κ > 0.8). Pedagogy v3.0 adds self-typing from dictionaries, rule-based typing for textbooks, and self-annotation later. |
| **Curriculum ("crescendo")** | Dictionaries → textbooks → literature → internet, with phase transitions gated on probes. |
| **Hypothesis 1, scaling exponents** | Typed models have larger data and parameter scaling exponents, so the gap widens with scale. The argument is by analogy to SE(3) equivariance. |
| **Claim 2, self-knowledge** | In a typed space, recall, novel composition and structural incoherence have distinct geometric signatures. |
| **Claims by strength** | Strong: faster convergence, more knowledge per parameter, better structured reasoning, better calibration, benefits compounding with scale. Moderate: types pre-organize the embedding space and reduce steps. Conservative: the changes alter information per token and geometry. |
| **Measurements 1–8** | Convergence; knowledge storage; reasoning; OOD generalization; representation geometry; **type retention at inference**; confidence–accuracy correlation; scaling exponents. |
| **Risks 1–7** | Wrong types; over-specification; open scaling question; polysemy; teacher bias; vocabulary size; curriculum may not matter. |

## 2.2 Each claim against the literature and this repository's experiments

Evidence tags: **[lit]** published work (doc 01 / doc 07); **[06]** first prototype, types at
inference; **[08]** the specified version, untyped inference; **[R6]** interchange intervention
training (doc 09). All experiments are toy-scale: 69K parameters, modular arithmetic with
*contextual* roles, 5 seeds per arm.

| Spec claim | Literature | This repository | Status |
|---|---|---|---|
| Bootstrapping tax / faster convergence (moderate claim) | **[lit]** Factored input features help in low-resource NMT (Sennrich & Haddow 2016). Allen-Zhu & Li 3.3: domain labels raise knowledge capacity. | **[06]** 40–64% faster *with types at inference*, but the `canonical` control (same role information by position, no type vectors) is equally fast. **[08]** No speed-up when inference is untyped (500–610 vs 500 steps). | **Not supported in the specified form.** The speed-up is an information effect that exists only while types are supplied. |
| Types not required at inference (asymmetry) | **[lit]** Syntactic Scaffolds (Swayamdipta et al. 2018) shows a training-only auxiliary structure loss works. | **[08]** Without a mechanism, removing types drops accuracy to 0.14 (chance ≈ 0.09). With 50% type dropout, 1.00. | **Holds only with an explicit mechanism.** The spec should name one (dropout and/or an auxiliary loss). |
| Types pre-structure geometry "from step zero" (Component 2) | **[lit]** Learned arithmetic geometry (Fourier features, helices) emerges *without* types. | **[06]** Input-side geometry does not survive the first attention layer; typed models are slightly *less* additive (R² 0.53 vs 0.63). **[08]** Typed-input dropout gives a small, n.s. gain. | **Weak / not supported** as an input-side mechanism. |
| Internalization via an auxiliary type-prediction objective (Pedagogy v3.0) | **[lit]** Syntactic Scaffolds; low-layer auxiliary POS supervision (Søgaard & Goldberg 2016). | **[08]** Operator localization 0.43 → 0.70; edit success ×2.7; three matched auxiliary-target controls give nothing. **[R1]** A blinded white-box interpreter made 9/10 verified causal claims on such a network (doc 10). | **Supported, specifically**, as an *interpretability / editability* effect, not an efficiency effect. |
| Structure enforced inside the network | **[lit]** IIT (Geiger et al., ICML 2022). | **[R6]** doc 09: every variable perfectly localizable by blind tools (DAS 1.00); op-swap edit 0.96 vs 0.22; held-out operand-shift edit 0.80 vs 0.04; no accuracy cost. **[R1]** doc 10: LLM-written edit programs, white- or black-box, reach 0.94–0.96 on IIT networks they never saw, vs 0.41–0.57 on baseline networks. | **Strongly supported at toy scale.** The most effective form of "structural prior" found here. |
| Hypothesis 1, scaling exponents | **[lit]** Equivariance changes exponents in force fields (Ngo & Ravanbakhsh, ICLR 2026; the spec attributes this to "Batatia"). Exponent figures appear in arXiv 2509.21811. Functional types are not group symmetries; dimensional analysis is (doc 07 §7.4). | Not testable at toy scale. | **Open.** Fix the citation; recast around unit-rescaling equivariance. |
| Claim 2, self-knowledge (three regions) | No direct evidence. Input-side types do not constrain *generated* content. | Not tested. | **Open.** Needs output-side type prediction to be testable (doc 07 §7.4). |
| Component 1, whole-word tokenization | **[lit]** Tokenization affects arithmetic and numeracy (well documented). | Not tested: the toy has atomic tokens by construction. | **Open.** It is the Q3/Q4 column of the spec's own 2×2. |
| Curriculum crescendo | **[lit]** Mixed results for curricula in LM pre-training. | Not tested. | **Open.** |
| Risk 1, wrong types hurt | — | **[06]** Shuffled types: no gain, no harm with types at inference. **[08]** Shuffled role targets as an auxiliary loss: no gain. | Partly tested: meaningless types are *inert* here, not harmful. |

## 2.3 The spec's measurement list: what has been run

| # | Spec measurement | Status here |
|---|---|---|
| 1 | Convergence speed | ✓ toy ([06], [08]) |
| 2 | Knowledge storage (bits/parameter) | ✗ |
| 3 | Reasoning tasks | partial: the toy *is* a compositional reasoning task, solved perfectly by every arm |
| 4 | OOD generalization | ✗ |
| 5 | Representation geometry | ✓ CKA, probes, additivity, plus *causal* measures (IIA, DAS) that the spec does not list |
| 6 | **Type retention at inference** | ✓ [08]: retained vs. removed, with and without dropout or an auxiliary loss |
| 7 | Confidence–accuracy correlation | ✗ |
| 8 | Scaling exponents | ✗ (needs ≥ 3 model sizes; deliberately deferred) |

## 2.4 The spec's 2×2 and where these experiments sit

The spec crosses tokenization (BPE vs ANLU) with typing (untyped vs typed). These experiments sit
entirely in the **typing** column, with atomic tokens throughout. The spec's tests rest on three
discriminations: "Q2 ≈ Q4 ≫ Q1 ≈ Q3 → typing does all the work", and so on. They are underpowered
unless each typed cell is split into **type input** vs **type target** arms. In [08] that
distinction is where the entire effect lies.

**Recommended:** expand the spec's 2×2 to a 2×3:

| tokenization | untyped | typed input (+ dropout) | typed auxiliary target |
|---|---|---|---|
| BPE | Q1 | Q2a | Q2b |
| ANLU | Q3 | Q4a | Q4b |

Evaluate *every* cell untyped, as the asymmetry requires.

## 2.5 Revised claim the evidence supports

> **ANLU-R.** Contextual functional types are most valuable as a **training-time auxiliary
> target**. Supervising the network to compute them internally yields representations whose
> variables are more causally localizable and more editable by closed-form weight programs, at
> zero inference cost and with no accuracy loss. As an **input** feature they mainly supply
> information. That speeds learning only while the information is supplied, and does not by
> itself organize internal computation.

This keeps ANLU's ontology (the six-layer schema, contextual typing, the teacher pipeline) and moves
its mechanism from the embedding sum to the loss. It also resolves the companion papers'
dependence on types at inference: the auxiliary type head supplies **predicted** types (doc 07 §7.3).
