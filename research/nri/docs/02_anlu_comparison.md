# 2. ANLU against the published record and the prototype

*Caveat:* the original specification (*ANLU: Typed Token Embeddings as Structural Priors for
Language Model Pre-Training*, Feb 2026) is not in this repository, and this session could not
access it. This analysis uses the summary in the research brief: `e_i = c_i + t_{τ_i} + p_i`,
contextual functional types (NUMBER, UNIT, QUANTITY, OPERATION, DIMENSION, grammatical roles,
domain types), and a staged curriculum. Corrections are welcome. Point me to the spec and I will
revise this.

## 2.1 ANLU decomposed into separately testable hypotheses

| ID | Hypothesis | Type of claim |
|---|---|---|
| **H1** efficiency | Type priors reduce the compute spent rediscovering functional relations (faster learning) | learning dynamics |
| **H2** geometry | Type priors organize internal representations: role and variable information sits in predictable, low-rank, separable subspaces | representation |
| **H3** interpretability | Networks trained this way are easier to reverse engineer (for an LLM or an algorithm), *even without being told the types* | interpretation |
| **H4** editability | Targeted edits are more successful and have less collateral damage | intervention |
| **H5** reproducibility | Independently trained networks share more representation structure, so edits transfer more easily | transfer |
| **H6** curriculum | A dictionary → textbook → unstructured curriculum improves pre-training | curriculum |

These are logically independent: H1 can hold while H2–H5 fail. That is exactly the pattern the
prototype finds in its regime (doc 06).

## 2.2 Where published results support, contradict, or fail to address ANLU

### Support

- **Factored and typed input features help when data is scarce.** Linguistic input features in NMT
  (Sennrich & Haddow 2016) and factored LMs (Bilmes & Kirchhoff 2003) gave consistent but modest
  gains, mostly in low-resource settings. This **supports H1** in the low-data regime that ANLU's
  curriculum starts in.
- **Training-time structural priors can buy interpretability.** Weight-sparse transformers (Gao et
  al. 2025) give circuits about 16× smaller; Learning Transformer Programs gives decompilable
  models; BIMT gives modular ones. These **support H3 in principle**, but for *architectural or
  regularization* priors, not input-type priors.
- **Training for editability works.** ERT (2611.26408) reports better LLM edit success on vision
  heads. This **supports H4 in principle** for an *explicit* edit objective.

### Contradict or complicate

- **Models already discover the geometry ANLU would impose on content.** Fourier features for
  modular arithmetic (Nanda et al. 2023), Fourier features in pretrained LLMs (Zhou et al. 2024),
  and a helix for numbers (Kantamneni & Tegmark 2025) all emerge *without* type supervision. A
  NUMBER type is therefore unlikely to add much **content** geometry. Its potential value is in
  **role** information (which number is the operand, which is the unit), and that information has
  to come from a typer.
- **Convergence with scale** (Platonic representation hypothesis). If representations converge as
  scale grows, the marginal benefit of hand-designed priors shrinks with scale, which is the
  "bitter lesson" pattern. ANLU's best case is small models and small data, which is also the
  cheapest place to test it.
- **The information confound.** Contextual types are produced by a typer, which is computation
  done *outside* the network. A typed network solves an easier problem. Any H1 gain must be charged
  the typer's cost, and any H2–H5 gain must be shown to come from geometry rather than from the
  information the types supply. No ANLU-adjacent publication separates the two; our `canonical`
  control does.
- **Localization ≠ editability** (Hase et al. 2023). Even if types make roles *readable*, that does
  not imply edits at those locations work. H2 and H4 must be tested separately.
- **Imposed structure can be fragile.** Written or structured weights do not survive later
  gradient training (*Rule-to-Network Compilation*: 0.50 → 0.01 after PPO). This is a warning for
  any prior that is not maintained by the loss.

### Not addressed anywhere we could find

- **H2–H5 for typed input embeddings** specifically: no published study.
- **Contextual (role) types as interpretability scaffolding:** no published study.
- **H6:** the curriculum literature (Bengio et al. 2009 and successors) reports mixed results for
  LM pre-training. Nothing tests ANLU's specific staged curriculum.

## 2.3 What the prototype adds (doc 06)

| hypothesis | verdict on the toy | key evidence |
|---|---|---|
| **H1 efficiency** | **supported, but it's the information, not the geometry** | Typed reaches 95% in 310 vs 520 steps (light regime) and 380 vs 1,063 (strong regime), p = 0.008. **Canonical** (same role information by position, no type vectors) is as fast or faster (270 / 340). Shuffled types give nothing. The orthogonal-subspace prior adds nothing over plain typing. |
| **H2 geometry** | **not supported; slightly contradicted** | Variables are decodable but not causally separable in any condition (DAS ≤ 0.50 vs 1.00 on the validity control). Role information makes the site *less* additive (R² 0.53 typed / 0.55 canonical vs 0.63 none) and operand information more redundant. |
| **H3 interpretability** | **not supported** | No head carries a variable; routing purity is ≈ 0.5 everywhere. Blinded LLM interpreters recovered the full rule on both typed and untyped nets (n = 2). The typed net's interpreter decoded the unlabeled type channel *itself*: types are trivially interpretable as *inputs*, which is not the same as making *internals* interpretable. |
| **H4 editability** | **not supported** | Rank-one edit success 11–32% everywhere, typed ≤ none. The agent-written edit gap disappears on cross-evaluation (it's interpreter strategy plus own-host fit). |
| **H5 reproducibility** | **not supported** | CKA across seeds is not higher for typed. Copied ΔW fails in all conditions. |
| H6 curriculum | not tested | — |

**The central mechanism:** a geometric prior applied at the **input** does not survive the first
mixing operation. Attention combines typed embeddings into op-specific interaction features, so the
neat `c ⊕ t` factorization is gone by the site where the computation happens. Giving the network
role information *accelerates* that mixing.

## 2.4 Revised hypothesis worth keeping

The data reject "typed input embeddings ⇒ interpretable, editable internals". They leave open, and
point toward, a sharper version:

> **ANLU-R:** Functional types are valuable as **(a) an efficiency prior** (they supply role
> information the network would otherwise compute) and **(b) a shared vocabulary for interpretation**
> (the NRI's `ftype`). Internal interpretability and editability, however, require the type
> structure to be **enforced at the internal sites where computation happens**, through the training
> objective (an interchange-intervention / IIT-style `L_struct` and an internal `L_edit`). Seeding it
> at the embedding layer is not enough.

This is testable cheaply (roadmap R6). It also connects ANLU to the strongest existing evidence that
training-time structure buys interpretability (weight-sparse transformers; ERT). Two falsifiers:

1. If an IIT-style `L_struct` that pins *typed* variables to fixed subspaces gives no better held-out
   edit success than the same loss with *shuffled* type assignments, then "typedness" adds nothing
   beyond "some enforced factorization".
2. If any interpretability gain costs more than the efficiency gain in task performance or learning
   speed, the prior is not worth it at the scales ANLU targets.

**Also to measure in R2 (not yet tested):** with a *noisy, learned* typer, does H1 survive once the
typer's errors and cost are charged? Here the oracle typer was perfect and free.
