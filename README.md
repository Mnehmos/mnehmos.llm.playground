# Mnehmos LLM Playground

**Research into interpretable, editable and reusable neural computation.**

This is an independent, CPU-first experimental research program. We train small neural networks from scratch, examine how mathematical concepts develop inside them, test whether those concepts actually control computation, and investigate whether learned knowledge can be preserved and reused instead of relearned.

Our long-term goal is a hybrid mathematical intelligence that can acquire, organize, compose and verify reusable computational knowledge. It is not a general-purpose LLM release or a claim that general mathematical intelligence has already been achieved.

## Start here

| I want to… | Read |
| --- | --- |
| Understand the research and its current status | [Research map](research/README.md) |
| See the results, baselines and limitations | [Evidence ledger](research/RESULTS.md) |
| Run a small neural mathematical demonstration | [H01 inspection demo](research/hybrid_math/README.md) |
| Reproduce the original neural representation experiments | [NRI documentation and commands](research/nri/README.md) |
| Locate experiments on other branches and archived work | [Artifact registry](research/ARTIFACTS.md) |
| Collaborate or reproduce a result | [Reproduction guide](research/REPRODUCIBILITY.md) |
| Understand what remains unsolved | [Research roadmap](research/ROADMAP.md) |

## What we have found

- **Causal interpretability:** in controlled modular arithmetic, declaring a variable improved variable-specific editability across 12 matched seeds. Accurate decoding does not by itself establish causal use. [Role-swap results](research/nri/docs/15_role_swap_results.md).
- **Generalization is distinct from editability:** a registered 70-network dimensional-reasoning study supported a limited editability benefit but did not support its unseen-unit generalization hypothesis. [Study results](research/nri/docs/12_dimensional_results.md).
- **Programmed and learned computation:** later branch experiments investigate exact compiled operations, protected computational interfaces and learning new operations while preserving earlier ones. These are results under review in [PR #36](https://github.com/Mnehmos/mnehmos.llm.playground/pull/36).
- **Cumulative discovery:** archived G05 experiments show that retaining representations can reduce the observations required for subsequent structured mathematical tasks. The strongest savings involve conventional exact algebra and sparse recovery; they are not demonstrated general neural superiority. [Evidence and archive status](research/RESULTS.md).
- **Computational minimality:** the archived H01 hybrid uses 13 learned phase coordinates and a 183-parameter method selector. Its broader mathematics is supplied executable software, not 196 weights doing unrestricted mathematics. The checked-in inspection demo exposes the actual frozen weights and narrow inference path.

## Try the small demo

From the repository root, in a Python environment with NumPy and PyTorch installed:

```sh
python research/hybrid_math/h01_smoke.py
```

The demo verifies the 13-parameter digit adder on all 169 input pairs and uses the frozen 183-parameter router to rank seven supplied function families. Exact fitting and evaluation are conventional Python. The full archived mathematical runtime is a separate artifact.

## Research principles

We separate training accuracy, unseen generalization, causal control, representation integrity, and preservation. We keep negative results, compare against strong classical baselines, and account for external programs, observations, search and verification rather than neural parameter counts alone.

The public repository is a research workspace. Not every reported experiment is merged or released. The research map explicitly labels branch-only and archive-only evidence; a result summary is not a substitute for its source, checkpoints or independent reproduction.

## Collaboration

Reproductions, counterexamples, stronger baselines and implementation reviews are welcome. Start with the [reproduction guide](research/REPRODUCIBILITY.md) and open an [issue](https://github.com/Mnehmos/mnehmos.llm.playground/issues) with the exact experiment, revision and environment.

No repository-wide license has been selected at this revision. Public visibility does not itself grant an open-source license; contact the maintainer about reuse and publication.

*Research index updated 27 November 2026. Historical results and source branches are preserved.*
