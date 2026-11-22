# 7. Review of the Mnehmos research program (ANLU, Typed Loops, Three Scaling Axes, Learned Heuristics, Pedagogy)

*Reviewed 2026-11-22. Documents reviewed in full:*

- *ANLU Foundation Specification (Feb 2026)*
- *Typed Loops*
- *Width, Depth, Structure: Three Scaling Axes*
- *Learned Heuristics as Adaptive Computation*
- *ANLU & Pedagogy as Training Alternatives v3.0*

*The aim is to strengthen the program: what holds up, what needs fixing before anyone else reads it, and
what the experiments in this repository say about it.*

## 7.1 Summary judgment

The program has one coherent idea: **declare structure the learner would otherwise have to discover**
("push ambiguity out of the learner and into the design"). It applies that idea at the token level
(ANLU), the compute-allocation level (Typed Loops, Heuristics) and the scaling-theory level (Three
Axes). The experimental designs are mostly falsifiable and honestly hedged ("it does not claim to
work"). That is better practice than much of the November-2026 WSP literature.

Four issues would hurt it with reviewers, in order of severity:

1. **A citation error carries the central scaling argument** (§7.2).
2. **The SE(3)-equivariance analogy does not hold as stated**, because functional types are not
   symmetries. A version that *does* hold is available: dimensional analysis (§7.4).
3. **Three internal contradictions**: whether the benefit grows or shrinks with scale; whether
   types are needed at inference; and how type removal works mechanically (§7.3). The
   experiments here show the third is not a detail: types removed without preparation leave the
   model at chance accuracy (§7.6).
4. **Missing direct prior art for "typed training, untyped inference"**: Syntactic Scaffolds (2018)
   and low-layer auxiliary supervision (2016) (§7.5).

## 7.2 Citation verification

| As cited in the papers | What it actually is | Status |
|---|---|---|
| "Batatia et al. (2025), *Scaling Laws and Symmetry: Evidence from Neural Force Fields*, arXiv:2510.09768" | **Khang Ngo & Siamak Ravanbakhsh**, ICLR 2026 (McGill/Mila). Batatia (MACE) is only cited inside it. | **Wrong authors.** The paper exists and the quoted sentence ("…we should not leave it to the model to discover fundamental inductive biases such as symmetry, especially as we scale…") is verbatim from their abstract (verified on the authors' GitHub README). |
| "EquiformerV2 achieved ~5× larger dataset scaling exponents and ~3× larger parameter scaling exponents than unconstrained Transformers" | These numbers match a **different paper**, *Scaling Laws for Neural Material Models* (arXiv 2509.21811): data exponent −0.242 vs −0.052 (≈4.7×), model-size exponent −0.383 vs −0.120 (≈3.2×). | **Misattributed** (search-level confirmation). Ngo & Ravanbakhsh report architecture-dependent exponents and better scaling for higher-order equivariance, but not these figures. Cite both papers. |
| "The gap does not close with scale. It accelerates." | A larger power-law exponent does mean the absolute gap widens as data grows. "Accelerates" is your interpretation. | Acceptable if phrased as "has a larger exponent". |
| Ouro, Zhu et al. 2025, arXiv:2510.25741 | Correct. Looping leaves storage at ~2 bits/parameter and improves manipulation. Reported 1.4B/2.6B models match up to 12B. | ✓ |
| Allen-Zhu & Li Part 3.3: "prepending domain labels significantly increases knowledge capacity" | Correct (arXiv 2404.05405, ICLR 2025). | ✓ **The strongest existing evidence for ANLU's storage claim.** Make it more prominent. |
| Hill, *The Bittersweet Lesson* | Dated 2023 in the Foundation spec and 2024 in Three Axes. | Make consistent. |
| Yue et al. 2025 (RLVR, arXiv:2504.13837) | Not re-verified this session. | Check venue and award before citing. |

**Action:** fix the scaling citation in all four papers. As written, a reviewer who checks
arXiv:2510.09768 will find different authors and no "5×/3×", and will distrust the rest.

## 7.3 Internal contradictions

1. **Does the advantage grow or shrink with scale?**
   - The Foundation spec and Typed Loops say the gap widens with scale.
   - Pedagogy v3.0, risk 3, says "The benefit likely scales inversely with model size."
   - Three Axes argues the widening thesis, but includes a section titled *"Structure Benefits Scale
     Inversely with Model Size"* (Pielka; BabyLM) as supporting evidence.

   **Fix:** state these as **competing, pre-registered predictions** (H-widen vs H-narrow) that the
   125M/350M/1.4B arms decide between. That turns a contradiction into the program's most
   falsifiable test.
2. **Are types needed at inference?**
   - Foundation spec and Pedagogy: no, "not a runtime dependency".
   - Typed Loops' type-conditioned exit gate (Eq. 2), type-aware prior π_τ (Eq. 4) and
     type-stratified KV cache: all need τ_i **per token at inference**.
   - Learned Heuristics, Level 1: "known at tokenization time, before any transformer computation".

   **A consistent resolution:** at inference, use the model's **own predicted types** from an
   auxiliary type head, not teacher labels. That makes Levels 1–3 self-contained and costs one
   linear layer. The experiment below tests whether such a head can be trained cheaply.
3. **How does "removing types at inference" work mechanically?**
   - The Foundation spec says the model "has internalized the geometric structure", but gives no
     mechanism. With `e = c + t + p`, deleting `t` shifts every input off the training distribution.
   - Pedagogy names one: an auxiliary type-prediction objective.
   - Pedagogy also predicts Q2 (BPE + types) needs types at inference while Q4 (ANLU + types) does
     not. Both use the same additive mechanism, so tokenization cannot cause that difference.

   **Specify the mechanism**: type dropout, an auxiliary type loss, or both. §7.6 measures each.

## 7.4 Conceptual issues

**Functional types are not symmetries, but dimensional analysis is.** Equivariance helps in Ngo &
Ravanbakhsh because the target function provably commutes with a group action: rotate the molecule
and the forces rotate. "Numbers behave differently than nouns" is a *category distinction*. No group
acts on text so that next-token prediction is invariant or equivariant under it. As written, the
SE(3) analogy will be read as rhetorical.

There is, however, a genuine symmetry inside ANLU's own type system. **Physical laws are invariant
under change of units** (the scaling group behind the Buckingham Π theorem). A model told which
tokens are NUMBER, UNIT and DIMEN can be made *equivariant to unit rescaling*, either by unit-rescaling
data augmentation or by an equivariant number–unit encoding. That is a real group action with a
real equivariance target, exactly the setting where Ngo & Ravanbakhsh's result applies. Recommendation:
**make dimensional equivariance the scaling-law claim**, and keep the broader type system as a
category prior (an efficiency claim).

**The "three regions" claim needs an output-side mechanism.** Typed embeddings act on *inputs*.
Nothing in `e = c + t + p` makes a *generated* "force equals seven purple" geometrically expensive,
because the type labels are never predicted, and so never constrained, on the output side. To make
Region 3 detectable, the model must **predict the types of what it generates** (a type head on the
output) and be penalized for type and dimensional inconsistency. The prototype adds evidence here:
input-side type geometry did not survive the first attention layer (doc 06 §6.4), so geometry seeded
at the embedding layer is unlikely to constrain outputs.

**"Do no harm" guarantees** (Heuristics; Typed Loops: "none can make the model worse than baseline
by construction"):
- At ĉ = 0 the *forward function* reduces exactly to the base architecture. That gives a clean
  ablation without retraining, which is valuable.
- It does **not** imply the trained model is never worse than a baseline trained *without* the
  module. The base weights were co-trained with the heuristic path and its gradients.
- With σ-gating initialized at −3, the gate's gradient is small but not zero.

Restate it as "exact ablation by construction", not "cannot be worse than baseline".

**Rhetorical overreach to trim before submission:**
- "NLS proves… reasoning + internet = correct answers to any factual question". One programming
  result does not establish open-domain factual reliability.
- The DNUM section. It is explicitly called "not merely analogical… structural", but no formal map
  (shared equations or invariants) is given. It reads as metaphor, and reviewers will discount the
  surrounding technical material because of it. Move it to a separate essay.

**Compute claims:**
- The $40 / 125M-parameter / 10–50M-token plan is plausible for a convergence-speed comparison.
  But 50M tokens is about 2% of Chinchilla-optimal for 125M parameters, so that regime cannot speak
  to scaling exponents.
- The 2×2×2 factorial with 5–10 seeds (which Three Axes correctly says interactions need) at
  125M/350M is a different budget, roughly two orders of magnitude more.

State both budgets.

## 7.5 Prior art to cite and position against

| Work | Why it matters to the program |
|---|---|
| **Swayamdipta, Thomson, Lee, Zettlemoyer, Dyer, Smith, *Syntactic Scaffolds for Semantic Structures*, EMNLP 2018** | Uses linguistic structure **during training only, through an auxiliary loss, at zero inference cost**, with gains of +3.6 F1 on FrameNet SRL. This is the published precedent for "typed training, untyped inference". ANLU must say what it adds: contextual functional types, an embedding-level prior, a curriculum, and the scaling claim. |
| **Søgaard & Goldberg, ACL 2016**, low-level tasks supervised at lower layers | POS supervision works best at **lower** layers, which is precedent for where an auxiliary type head should attach. Our design follows it (layer-1 residual). |
| Sennrich & Haddow, WMT 2016, *Linguistic Input Features Improve NMT*; factored LMs (Bilmes & Kirchhoff 2003) | Additive or concatenated type embeddings at the input: the direct ancestor of `e = c + t + p`. The Foundation spec cites Sennrich's BPE paper but not this one. |
| Strubell et al., EMNLP 2018, *LISA* (linguistically-informed self-attention) | Supervises attention heads with syntactic structure. Prior art for Heuristics Level 2 (linguistically informed Q/K biasing). |
| Gao et al. 2025 (OpenAI), weight-sparse transformers | The strongest evidence that a *training-time* prior buys interpretability. The baseline any ANLU interpretability claim must beat. |

## 7.6 What the experiments in this repository say

Two experiments, both toy-scale (68–69K parameters, modular arithmetic with contextual roles,
5 seeds per arm, all CPU). Details are in docs 06 and 08.

| Program claim | Experiment | Result |
|---|---|---|
| Types speed up learning (Foundation H1; Pedagogy "skip the ontology phase") | doc 06, types at inference | **Yes, 40–64% faster**, but a no-types `canonical` control that delivers the same role information by position is equally fast. The gain is the **information**. |
| | doc 08, untyped inference | **No speed-up** (500–610 vs 500 steps). |
| Types are not a runtime dependency | doc 08 | **Only with a mechanism.** Trained without dropout or auxiliary loss, removing types drops accuracy to 0.14 (chance ≈ 0.09). With 50% type dropout, accuracy is 1.00. |
| Input type geometry pre-structures internal representations | doc 06 + doc 08 | **Not supported.** Input-side geometry does not survive the first attention layer. Typed models are slightly *less* additive. Typed-input dropout gives a small, n.s. gain. |
| Types internalized through an **auxiliary type-prediction objective** (Pedagogy v3.0) | doc 08 | **Supported, and specific.** Causal localization of the operator rises 0.43 → 0.70 and weight-edit success rises ~2.7×. Shuffled, lexical and same-lookup contextual auxiliary targets give no gain. |
| Three regions / self-knowledge; scaling exponents; loops; heuristics | — | Not testable at this scale. The design implications are in §7.3 and §7.4. |

**Net:** the evidence favors a narrower and more defensible version of ANLU.

- **Not supported here:** "type vectors in the input embedding make training faster and
  representations better."
- **Supported here:** "functional types used as a training-time auxiliary target make the model's
  internal computation more localizable and more programmable, at zero inference cost."

That reframing also resolves contradiction 2 in §7.3. The same auxiliary type head that improves
internal structure can supply **predicted types at inference** to the Typed-Loops exit gate and to
the Heuristics Level-1 classifier, with no teacher labels at runtime.

## 7.7 Recommended changes to the $40 experiment

1. **Add type-dropout and auxiliary-type-loss arms** to the 2×2, and evaluate *every* typed arm
   with types removed. Without them, the "not a runtime dependency" claim is untested.
2. **Add the information control.** Supply the same role information without type geometry, as
   this repository's `canonical` arm did, so any gain can be attributed to the geometry or to the
   information.
3. **Add a shuffled-type control** (wrong types, same marginals). This tests the spec's own risk 1,
   "wrong types hurt".
4. **Pre-register** H-widen vs H-narrow (§7.3), and kill criteria for each.
5. **Measure internalization directly.** Probe the hidden states of untyped inputs for role and
   type, and test causal role localization (the NRI `linear_rep` / `necessity` claims). Don't rely
   on downstream accuracy alone.
6. **Make dimensional equivariance a separate, headline arm** (§7.4). It is the one place the
   symmetry argument genuinely applies.
