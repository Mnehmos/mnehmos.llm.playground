# 1. Research landscape: LLM-driven reverse engineering and weight-space programming

*Compiled 2026-11-22. Scope: what has actually been shown, what is claimed, and what is speculation.*

## 0. How each source was verified (read this first)

This environment could not reach `arxiv.org`, `rikiyat.github.io`, `honnibal.dev` or most blogs:
the network proxy refuses them. Every source below therefore carries a verification tag. A tag
says how far I checked a source. It does not say whether the source is right.

| Tag | Meaning |
|---|---|
| **[P]** | **Primary material read directly**: the authors' own repository README, released logs or result files (fetched from `raw.githubusercontent.com`). |
| **[C]** | **Catalog source-check**: summarized from the *Awesome-Weight-Space-Programming* evidence matrix (`docs/wsp-evidence.md`, updated 2026-11-20). Its maintainers state that they inspected each paper but did not reproduce it. I did not read these papers myself. |
| **[S]** | **Search-level**: existence, title, authors, date and abstract claims confirmed through web-search results (often several independent pages). Full text not read. |
| **[K]** | **Established pre-2026 literature** from background knowledge, with arXiv IDs. Not re-fetched in this session; check before citing. |

**Source type:** almost every 2026 WSP item is a **preprint, technical blog or GitHub report**. Only
these are confirmed as peer-reviewed or accepted: HyVE (EMNLP 2026 Findings) [S], *Weight-sparse
transformers* (ICML 2026) [S], mHC (ICML 2026) [S], the Platonic Representation Hypothesis (ICML 2024)
[S], Hase et al. (NeurIPS 2023) [S], and Qwen Gated Attention (NeurIPS 2025 best paper) [S].
**Treat every WSP number below as a claim reported by its authors.**

---

## 1. Weight-space programming (WSP): the November 2026 wave

WSP went from one blog post (2 Nov 2026) to more than 34 preprints in about three weeks. That pace
leaves little time for review, and most results come from a single run. The useful question is
which claims survived **controls** and **independent replications**.

### 1.1 *Sparks of Weight-Space Programming* (MIT CSAIL) [P logs, S post]

- **Who/when:** Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab; MIT CSAIL technical blog,
  2 Nov 2026. Not peer reviewed. Logs: `github.com/RikiyaT/lm_understand-logs` (CC BY 4.0 data, MIT scripts).
- **Model:** OpenAI `gpt-6-astra` through the Codex CLI. The logs README states that built-in tools
  were off and that all 1,125 turns were audited as tool-free **[P]**.
- **Eight experiments [P]:**

  | # | Task | Tools |
  |---|---|---|
  | 1 | which Atari game a policy was trained on | none |
  | 2 | what shape a diffusion model draws | none, no forward pass |
  | 3 | which language a language model learned, from middle blocks | Python sandbox, no forward pass |
  | 4 | one-shot weight edit fixes a failing car network | none |
  | 5 | write MinAtar policies from scratch | none |
  | 6 | write a 1.5M-weight Go network | sandbox; 0/10/25/50 passes |
  | 7 | which LM is sycophantic | sandbox, then an attention tool |
  | 8 | improve Qwen3.5-0.8B at math | sandbox; 10/25/50 passes |

- **Headline claim [S]:** an edit to 0.016% of Qwen3.5-0.8B's weights raised MATH-500 from 7.9% to 48.1%.
- **Key distinction [P]:** experiments 1, 2, 4 and 5 are genuinely tool-free weight reading or writing
  on small networks. Experiment 8, the headline result, is **an agent with a code sandbox and
  practice passes on the task**, a verifier-guided search loop. Honnibal's critique (3 Nov 2026) [S]
  makes the same point: only experiments 1, 2 and 4 read raw weights, on networks of 17K–172K weights.

### 1.2 Independent audits of the Qwen edit: behavioral success, causal story revised

These are the clearest examples in the field of **behavioral success without the claimed causal
explanation**, which is the distinction your brief asks for.

- **EditLens** (Zhang, Shang, Li; UBC / Vector Institute; Nov 2026 technical report) **[P results file]**:
  - *What the edit is:* a rank-one change `w' = w − α u(uᵀw)`, α = 6, on 64 `down_proj` columns
    in each of layers 15 and 19.
  - *What it does:* the edit does **not** improve reasoning. It turns "go back and re-check" neurons
    into an **anti-copy switch**. Path patching puts **91%** of the effect through layer 16–23 attention
    (induction attention drops by about a third); the edited sites' direct path carries 3%.
  - *Thinking off beats it:* under the auditors' protocol (16k cap), the unedited model with
    thinking turned off scores higher (MATH-500 0.510 vs 0.312 for the edited model with thinking on).
  - *Astra's localization was partly right:* its 64 neurons overlap the true top-64 by 44 (L15) and
    36 (L19), against a chance level of about 1.
  - *Side effects:* they are mostly the same knob as the fix.
- **Same Fix, Different Weights** (2611.20953) [C]: a pre-registered Claude replication got
  6.7% → 58.1% on MATH-500, but through a *different* mechanism (output-layer logit and
  effective-temperature changes).
- **Just Sort It!** (2611.21432) [C]: the LP-derived anti-loop edit can be replaced by equal
  coefficients on eight neurons (67.0% vs 67.4%). This suggests a near-constant directional bias,
  not a finely programmed computation.

**Takeaway.** Behavioral success **replicated**. The causal story **did not**: several different
mechanisms give the same behavior. Any WSP claim needs a mechanism-level check.

### 1.3 *WeightCode* (arXiv 2611.25476) [P README + repo]

- **Who/when:** Bo Li, Taoyong Cui, Bob Zhang, Mengran Li, Chengyang Zhang, Yang Zhang, Pheng Ann Heng,
  Wanli Ouyang, Zhenfei Yin, Yingcheng Wu, Ling Yang; arXiv, Nov 2026; MIT license; `github.com/Gen-Verse/WeightCode`.
- **Method [P]:** a frozen LLM computes `W′ = p(W, I)`: a Python *weight program* that reads the
  weights `W` and task knowledge `I` and returns `{tensor: delta}`.
  - The editor **can run the network and measure activations** in a NumPy sandbox.
  - Each session has **8 checks** against a labeled split that `I` does not contain.
  - A fixed rule selects the program: the best score among candidates that lose ≤ 0.5% of clean check images.
  - Edit bounds: ≤ 5% of weights, ≤ 0.5 × tensor norm.
- **Results [P]:**
  - SCAN "jump": 0% → 99.8% with **257 of 803,975 weights changed**; old commands stay at 99.6%.
  - BabyAI: 0% → 98.4%.
  - ImageNet ResNet-18 noise robustness from 64 + 64 labeled images: net gains of +10.47 / +17.62 pt,
    against +8.67 for the best of 8 fine-tuning methods.
- **Mechanism check [P]:** removing the installed noise gates drops the net gain from +31.75 to +10.05 pt.
- **Transfer [P]:** rerunning the *program* on 5 independently trained networks keeps **83–110%** of
  the gain; **copying the weight delta gives −0.10 pt**. This is the strongest published evidence
  that portable "neural programs" should be programs that **re-derive** host-specific edits, not
  coordinate-level deltas. Our prototype tests the same distinction (§6 of the results).
- **Caveats [C]:** these are tested SCAN/BabyAI/ImageNet models; the paper does not claim
  architecture-independent transfer. It is gradient-free, but it is **not search-free**: there is a
  verifier loop with 8 checks.

### 1.4 *Training Vision Models for LLM-Driven Weight-Space Programming*: Edit-Ready Training (arXiv 2611.26408) [S]

- **Claim [S]:** five LLM editors edit CNNs and ViTs on relabeling and error-correction tasks
  (Waterbirds, CIFAR-100, PACS, Entity-30), and the networks are trained four ways: ERM, label
  smoothing, gradient-based editable training (ENN; Sinitsin et al. 2020), and **Edit-Ready Training (ERT)**.
- **What ERT does [S]:** it "simulates head weight edits, computes the errors of the edited network,
  and backpropagates them through the edit into all weights". The edit-loss weight λₑ ∈ {0.3, 1, 3}.
- **Result claimed [S]:** ERT gives the highest mean score for GPT-6 Astra and Claude Opus 5.5.
- **Not verified:** the author list (search results disagree; one names Timo Ropinski), the
  per-dataset numbers, and the limitations.
- **Relevance:** this is the **direct prior art for your Experiment E** (`L_edit`). Two differences
  to note: (a) ERT simulates **head (last-layer) edits**, not internal-circuit edits; (b) it optimizes
  editability by *LLM editors*, not *human-legibility* of the mechanism. Internal-circuit
  editability and structure priors appear to be open.

### 1.5 Controls and negative results that decide how to interpret WSP

| Work | Finding | Tag |
|---|---|---|
| *Computation, not weight access* (2611.07847) | LLM auditors of clinical networks (105–38,401 weights) recover planted relations in 39/39 audits **with computation tools**, and **black-box access performs as well as white-box**. The auditors' advantage comes from running computations, not from reading weights. | [C] |
| *Can an LLM actually reason about weights?* (2611.08070) | Claude Opus 5.5 reading ~125K-parameter CNNs: local computations are accurate, but there are multiple wrong dataset/semantic identifications, including failures under label permutation. | [C] |
| *Orbit Audits* (2611.05798) | Weight readers must be invariant to function-preserving permutations and ReLU rescalings; separates consistency from correctness. | [C] |
| *Replicating & Extending WSP* (2611.21449) | Astra's networks were decompiled into Python matching all 49,189 tested states. On *new* games Astra recovers board layout but mostly misses mechanics. Audits flag poisoned models but **never identify the trigger**. Three-day sprint; contamination possible. | [C] |
| *Rule-to-Network Compilation* (2611.16981) | Written networks are 93% zeros. Python programs beat written networks (97–100% vs ≤ 90%). **Written weights do not survive PPO fine-tuning** (0.50 → 0.01). Writers read simulator source, so the comparison is about information access. | [C] |
| *YOLO26 editing* (2611.17789) | Under matched budgets, LLM-guided edits do not beat random guidance (AP +0.004, CI −0.026 to +0.034). | [C] |
| *Mixed Signals* EEG/ECG (2611.13758) | No reliable task reading or improvement of healthy networks; repair works only with feedback. | [C] |
| *WSP-Scope* (2611.09382) | Qwen3.5 0.8B/9B: 6 of 16 editing trajectories kept the original weights; outcomes vary widely. | [C] |
| *WeightBench* (2611.18553) | 42 tasks, networks from 2K to 7B parameters: Opus 5.5 macro 0.535; GPT-6.1-sol 0.568 (on 38 tasks). | [C] |
| *iWSP* (2611.18114) | Agents build transparent transformers for 13 algorithmic tasks (26/26 runs reach 100%) with validation feedback; the neuro evaluators fit ridge readouts. | [C] |
| *Weight Oracles* (2610.07334; NeurIPS 2026 NeuralArtifacts workshop spotlight) | LLMs reading weights for safety audits; predates *Sparks*. | catalog metadata only |

### 1.6 What WSP has and has not demonstrated

**Demonstrated** (multiple sources, with controls):

1. Frontier LLMs can **write** small networks for **known, compilable rules**: games, Boolean and
   algorithmic functions, PDE solvers. The networks resemble compiled programs (sparse, hand-structured)
   more than trained networks.
2. With **execution tools and feedback**, LLM agents can find **small, bounded weight edits** that
   repair specific behaviors (loop-breaking, lexical grounding, noise gating) in models up to ~1B
   parameters, and sometimes install narrow new behaviors (SCAN "jump").
3. **Re-derivable edit programs** transfer across independently trained hosts (WeightCode); copied
   deltas do not.

**Not demonstrated, or contradicted:**

1. "Native" reading of *trained* weights at scale. Most reading successes are on toy or constructed
   networks, are matched by black-box access, or fail under label permutation and on new tasks.
2. That an edit's stated **explanation is its causal mechanism**. Shown false for the flagship Qwen
   edit (EditLens) and shown non-unique (Same Fix, Different Weights).
3. **Installation of genuinely new competence** in large models: math gains were decoding/stopping repair.
4. That LLM guidance beats **matched random or evolutionary search** in general (YOLO26: no).
5. Durability: written weights get overwritten by later gradient training.

---

## 2. LLM-assisted mechanistic interpretability

- **HyVE**, *Can Language Model Agents be Helpful Circuit Explainers in Mechanistic Interpretability?*
  (arXiv 2606.24026; **EMNLP 2026 Findings**; Ayan Antik Khan, Harsh Kohli, Yuekun Yao, Huan Sun, Ziyu Yao) [S]
  - *Method:* an agent runs an *observe → hypothesize → validate (intervene) → explain* loop for each
    component; refuted hypotheses are fed back in.
  - *Benchmark:* **AgenticInterpBench**, 84 semi-synthetic transformer circuits with 163
    component-level annotations; four LM backbones.
  - *Findings:* explanations are useful, no backbone is uniformly best, and **"reliable validation
    remains the main bottleneck."**
  - *Relevance:* HyVE is the closest existing NRI-like workflow. It produces *natural-language*
    explanations. The NRI makes the validation step a first-class executable contract.
- **Earlier work [K]:** Bills et al. 2023 (OpenAI) on LLM neuron explanations, scored by simulation;
  MAIA (Shaham et al., ICML 2024, arXiv 2404.14394), an automated interpretability agent with
  experiment tools; Shi et al. 2024, *Hypothesis Testing the Circuit Hypothesis in LLMs*
  (arXiv 2410.13032) [S], which gives formal tests of faithfulness, localization and minimality.

## 3. Mechanistic toolkit: what works, and the known failure modes

- **Activation patching / causal tracing [K]:** Meng et al. 2022 (ROME, arXiv 2202.05262); Zhang &
  Nanda, ICLR 2024 (arXiv 2309.16042); Heimersheim & Nanda 2024 (arXiv 2404.15255). EditLens adds a
  practical warning: attribution patching correlates only ρ ≈ 0.62–0.67 with exact patching, so use
  it only to screen [P].
- **Causal abstraction / interchange interventions [K]:** Geiger et al. 2021 (arXiv 2106.02997);
  Distributed Alignment Search, Geiger et al. 2023 (arXiv 2303.02536); the `pyvene` library (Wu et al.
  2024, arXiv 2403.07809). **Interchange-intervention accuracy (IIA) is the metric our prototype uses,
  because it directly tests a causal claim of the form "variable V lives in subspace S".**
- **Circuit discovery [K]:** ACDC (Conmy et al., NeurIPS 2023, arXiv 2304.14997).
- **Sparse autoencoders:** Bricken et al. 2023 and Cunningham et al. 2023 (arXiv 2309.08600) [K].
  Strong negative evidence follows:
  - Google DeepMind's mech-interp team (Mar 2025) found SAEs underperform linear probes on OOD
    harmful-intent detection and **deprioritized fundamental SAE research** [S].
  - *Sanity Checks for SAEs* (arXiv 2602.14111) reports random baselines matching trained SAEs on
    interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72) and causal editing (0.73 vs 0.72) [S].
- **Transcoders / attribution graphs [K]:** Dunefsky, Chlenski & Nanda, NeurIPS 2024 (arXiv 2406.11944);
  Anthropic's *Circuit Tracing* methods and *On the Biology of a LLM* (transformer-circuits.pub, Mar 2025).
  These replace MLPs with sparse cross-layer transcoders to get attribution graphs; the stated
  limitation is reconstruction error ("dark matter").
- **Weight-sparse transformers** (Gao, Rajaram, Coxon, Govande, Baker, Mossing; OpenAI; arXiv
  2511.13653; **ICML 2026**) [S]: training mostly-zero weights gives task circuits about **16×
  smaller** than in dense models of matched loss. A follow-up (arXiv 2607.02964) reports 17–35% of
  nonzero weights individually interpretable, against 5–9% in dense models [S]. **This is the
  strongest existing evidence that a training-time structural prior buys interpretability**, and the
  main baseline any ANLU-style "interpretability prior" must beat.
- **Targeted Parameter Decomposition** (arXiv 2607.13047; ICML 2026 MI workshop) [catalog]:
  recovers weight-space mechanisms for targeted rewiring. Relevant to NRI edit programs.

## 4. Model editing

- **ROME** (Meng et al., NeurIPS 2022, arXiv 2202.05262) and **MEMIT** (ICLR 2023, arXiv 2210.07229)
  [K]: closed-form rank-one / multi-layer key-value edits to MLPs. Our prototype's edit program is
  structurally ROME-like.
- **Hase, Bansal, Kim, Ghandeharioun**, *Does Localization Inform Editing?* (NeurIPS 2023 spotlight,
  arXiv 2301.04213) [S]: Causal Tracing localization has **near-zero correlation with edit success**
  for ROME, MEMIT and fine-tuning. **This is the central caution for the NRI: "where a variable is
  readable" and "where an edit works" are different questions and must be tested separately.**
- **Editable training** [K]: Editable Neural Networks (Sinitsin et al., ICLR 2020); MEND (Mitchell et
  al., arXiv 2110.11309). These are the gradient-based ancestors of ERT.

## 5. Representation geometry, alignment and similarity

- **Linear representation hypothesis** (Park, Choe, Veitch, ICML 2024, arXiv 2311.03658) [K]:
  formalizes concepts as directions and the role of a causal inner product. This is the precondition
  for closed-form edits like ours.
- **Platonic representation hypothesis** (Huh, Cheung, Wang, Isola; ICML 2024 oral position paper;
  arXiv 2405.07987) [S]: representations converge across models and modalities as scale grows,
  measured with mutual nearest-neighbor kernel alignment. It is about *kernels* (similarity
  structure), not shared coordinates, which is why copied deltas fail and alignment is needed.
- **Similarity and alignment tools [K]:** CKA (Kornblith et al., ICML 2019, arXiv 1905.00414); model
  stitching (Bansal, Nakkiran, Barak 2021, arXiv 2106.07682); Git Re-Basin permutation alignment
  (Ainsworth et al., ICLR 2023, arXiv 2209.04836).
- **Superposition** (Elhage et al. 2022, *Toy Models of Superposition*) [K]: why neuron-basis
  readability is not guaranteed.
- **Learned arithmetic geometry [K]:** models already discover structured number representations
  without type priors, including Fourier features in grokked modular addition (Nanda et al., ICLR 2023,
  arXiv 2301.05217), in pretrained LLMs (Zhou et al., NeurIPS 2024, arXiv 2406.03445), and a helix
  for addition (Kantamneni & Tegmark 2025, arXiv 2502.00873). **This bears directly on ANLU's NUMBER type** (see doc 02).

## 6. Efficient training and architecture (DeepSeek, Moonshot/Kimi, Z.ai/GLM, Qwen, MiniMax)

| Work | Verified claim | Relevance to ANLU / NRI |
|---|---|---|
| **Muon** (K. Jordan, Dec 2024 blog) [K]; **Muon is Scalable for LLM Training** (Moonshot AI + UCLA, arXiv 2502.16982) [S] | Orthogonalized (Newton-Schulz) momentum updates for matrix parameters; token-efficiency gains at scale | Changes the spectral geometry of updates. Whether Muon-trained weights are **more or less editable / readable** is untested: a cheap, open question (roadmap R5). |
| **Kimi K2**, MuonClip (arXiv 2507.20534) [S] | Muon + QK-Clip; 15.5T-token pretraining with no loss spike | Same as above. |
| **GLM-4.5 / GLM-5** using Muon [S, secondary sources only] | Reported in PyTorch/NVIDIA posts; not checked against Z.ai's own report | Do not cite without checking the GLM technical report. |
| **DeepSeek-V3** (arXiv 2412.19437) [K] | MLA, fine-grained MoE, multi-token prediction | MoE routing is itself a learned "type" assignment: a natural place to look for typed computation. |
| **mHC: Manifold-Constrained Hyper-Connections** (DeepSeek-AI, arXiv 2512.24880; ICML 2026) [S] | Projects residual-mixing matrices onto the Birkhoff polytope (doubly stochastic) via Sinkhorn-Knopp; reported 6.7% overhead at n = 4 (v1); adopted in DeepSeek V4 per a secondary source | A **geometric constraint on the residual stream** imposed during training. It is the nearest industrial analogue to "structural priors on representation geometry", though its motivation is stability, not interpretability. |
| **Gated Attention** (Qwen, NeurIPS 2025 best paper) [S] | Head-specific sigmoid gate after SDPA; removes attention sinks; used in Qwen3-Next | Attention sinks complicate attribution. Sink-free attention may be easier to patch. Untested. |
| **MiniMax-M1** lightning (linear) attention (arXiv 2506.13585) [K]; **Kimi Linear** [not verified] | Hybrid linear-attention long-context models | Linear-attention state is a matrix memory: a different object for weight-space or state-space editing. |

**Honest summary:** none of these labs' published work tests **typed input embeddings** or trains for
interpretability. Their relevance to this programme is indirect: they show that geometric
constraints (orthogonalized updates, doubly-stochastic residual mixing, gating) are cheap to impose
at scale, and their effect on *editability* is unmeasured.

## 7. Interpretable-by-construction networks, and ANLU's direct prior art

- **Compiled transformers [K]:** RASP (Weiss, Goldberg, Yahav, ICML 2021, arXiv 2106.06981); **Tracr**
  (Lindner et al., NeurIPS 2023, arXiv 2301.05062) compiles RASP programs into transformer weights;
  **Learning Transformer Programs** (Friedman, Wettig, Chen, NeurIPS 2023, arXiv 2306.01128) trains
  transformers constrained to be decompilable into Python. *An NRI is the inverse direction of
  Tracr: weights → program.*
- **Training for modularity and legibility [K]:** BIMT (Liu, Gan, Tegmark 2023, arXiv 2305.08746);
  Concept Bottleneck Models (Koh et al., ICML 2020, arXiv 2007.04612); Codebook Features (Tamkin et al.
  2023, arXiv 2310.17230).
- **Typed / factored input embeddings, the direct ancestors of `e = c + t + p` [K]:**
  - Factored language models (Bilmes & Kirchhoff, NAACL 2003).
  - *Linguistic Input Features Improve NMT* (Sennrich & Haddow, WMT 2016, arXiv 1606.02892):
    POS, lemma and dependency labels as added or concatenated embeddings.
  - BERT segment embeddings.
  - Numeric-value encodings such as xVal (Golkar et al. 2023, arXiv 2310.02989).

  **ANLU's additive typed embedding is not new as an architecture.** What could be new is (a)
  *contextual functional types* used as a pre-training curriculum prior, and (b) the claim that such
  types organize *internal* geometry for interpretability and editing. Only (b) is tested here.

---

## 8. The research gaps that matter most

1. **No WSP study varies how the target network was *trained* in order to test whether
   interpretability or editability can be designed in.** The exception is ERT, which is limited to
   head edits and LLM success rates. Weight-sparse transformers vary training, but measure circuit
   size, not editability or transfer.
2. **Reading versus computing is unresolved.** Black-box access matches white-box access in
   audits. A credible claim that a network is *more interpretable to an LLM* must show the advantage
   survives when the LLM can also run experiments, or must measure it with causal metrics that do
   not depend on an LLM at all.
3. **Explanation ≠ mechanism.** Flagship edits work for reasons other than the ones stated. No WSP
   output format makes explanations *executable and refutable*. HyVE names validation as the
   bottleneck. → This is the case for the **NRI**.
4. **Transfer has been shown only for re-derivable programs** (WeightCode). Nobody has asked whether
   *training priors* make representations reproducible enough that **coordinate-level deltas**, or
   alignment-mapped deltas, transfer.
5. **Information-access confounds** in typed or structured training: a model given role labels
   solves an easier problem. No study separates "the prior supplies information" from "the prior
   shapes the geometry". Our `canonical` control is designed for exactly this.
