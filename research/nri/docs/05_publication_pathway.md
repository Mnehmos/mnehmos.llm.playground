# 5. Publication pathway

## 5.1 What would merely replicate existing work (avoid as a headline)

| Result | Already shown by |
|---|---|
| An LLM writes a working small network for a known rule | *Sparks* (exp. 5), *Rule-to-Network Compilation*, *Physics2Weight*, iWSP |
| An LLM agent with execution tools repairs a behavior through a small edit | *Sparks* exp. 8, WeightCode, WSP-Scope, π0.5 debugging |
| Rerunning an edit *program* transfers; copying ΔW does not | WeightCode (83–110% vs −0.10 pt) |
| Training for head-level editability helps LLM editors | ERT (2611.26408) |
| A sparsity prior yields smaller, more interpretable circuits | Gao et al. 2025 (ICML 2026) |
| LLM agents explain circuits with an intervention loop | HyVE (EMNLP 2026 Findings) |
| Factored or typed input embeddings | Sennrich & Haddow 2016; factored LMs (2003); BERT segments |

## 5.2 Candidate original contributions, strongest first

1. **NRI: executable interpretations, and a benchmark that scores interpreters by intervention.**
   HyVE names validation as the bottleneck, and the WSP literature shows explanations that are
   behaviorally right but causally wrong. No published format makes an interpreter's claims
   machine-checkable (bindings + output functions + intervention predicates + edit programs).
   - *Deliverable:* schema + executor + a zoo with ground truth across training priors + scored
     LLM runs with a black-box arm.
   - *Novelty risk:* moderate (HyVE, AgenticInterpBench, causal-abstraction tooling such as pyvene
     are adjacent). The differentiator is the executable-claim contract and the variable-recall /
     edit-score metrics.
2. **A causal test of whether input-type priors create internal structure, with an information
   control.** The `canonical` arm separates "the prior supplies information" from "the prior
   shapes the geometry". To our knowledge no WSP or interpretability paper makes this separation.
   A clean **negative** result ("typed embeddings speed learning but do not localize variables;
   the effect is the information, not the geometry") is publishable at a workshop, and is more
   useful to the field than an uncontrolled positive one.
3. **Internal Edit-Ready Training (R6).** ERT edits heads. Training for editability of *internal*
   variables, evaluated on *held-out* edit types and with LLM editors, is an open, direct extension.
4. **Optimizer geometry → editability (R5, Muon).** Cheap, and to our knowledge unmeasured.
5. **Coordinate portability as a function of training prior (R4).** WeightCode shows that
   programs transfer and copied deltas don't. Whether any prior makes *aligned* deltas transfer
   (Procrustes / Re-Basin) is open.

## 5.3 What a result must include to count

- **Pre-registration** (OSF or a dated commit in this repo) of hypotheses, metrics and kill criteria.
- **≥ 10 seeds per arm** for the primary metrics. Report effect sizes with bootstrap CIs, not
  only p-values, and correct for multiple comparisons on the primary family.
- **Instrument validity:** positive and negative synthetic controls (already in `validity_check.py`).
- **Three separate successes** reported separately: behavioral, causal (an NRI `linear_rep` /
  `necessity` pass at the claimed site), and transfer.
- **Black-box control** for every LLM-interpreter claim.
- **Released artifacts:** checkpoints with sha256, configs, NRI documents (raw and executed), every
  prompt and transcript (as *Sparks* and WeightCode did).

## 5.4 Venues and timing (as of Nov 2026; verify each deadline)

| Target | Fit | Notes |
|---|---|---|
| Mechanistic-interpretability workshops at ICML / NeurIPS 2027 | contribution 2 (toy, controlled) | Fast feedback; non-archival options |
| ICLR 2027 workshops (spring) | 1 or 2 | Same |
| **TMLR** | 1 + 2 + R1 | Rolling submissions; judges correctness, not hype. Good for an independent researcher |
| NeurIPS 2027 Datasets & Benchmarks | 1 (NRI benchmark + zoo) | Needs breadth: several task families and interpreter families |
| arXiv + LessWrong / Alignment Forum write-up | all | Most WSP discussion currently happens there. First-time arXiv submitters in cs.LG may need an endorsement |

## 5.5 Practical notes for a solo, low-budget researcher

- Everything through R5 runs on a laptop CPU. R1 costs only API calls; cap tool calls per run.
- Report negative results prominently. The WSP field currently over-rewards headline numbers,
  and careful controls are the scarce contribution.
- Keep ANLU's *curriculum* claims (R7) separate from its *geometry / interpretability* claims.
  They need different evidence, and the second can be settled much more cheaply.
