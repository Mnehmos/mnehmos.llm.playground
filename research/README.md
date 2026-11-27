# Research map

The program moves from observing neural mechanisms to acquiring and organizing reusable mathematical knowledge. The tracks below are an index, not a claim that all implementations have been imported into the default branch.

## Research areas

| Area | Central question | Location and availability |
| --- | --- | --- |
| NRI / typed priors | Can a network's representations be decoded, edited and causally tested? | [NRI](nri/README.md), source on default branch |
| Neural development / Birth of a Concept | When do representations form, become usable and generalize? | [PR #36](https://github.com/Mnehmos/mnehmos.llm.playground/pull/36), [pinned handoff](https://github.com/Mnehmos/mnehmos.llm.playground/blob/5958e2d9e22b234f94353e386bf1c4d0a734efc6/research/nri/docs/50_handoff.md); draft/branch evidence |
| Causal authority and retention | Can installed computation retain control during continued training? | [Paper Three branch](https://github.com/Mnehmos/mnehmos.llm.playground/tree/d34aa61cb5826b5287380ad96f33af96ad219bf9), separate research branch; publication approval not implied |
| Matrix and CPU work (Tracks A/B) | Can training or structured execution become cheaper? | [Finite matrix benchmark](nri/matmul_benchmark/RESULTS.md), [CPU-01](https://github.com/Mnehmos/mnehmos.llm.playground/blob/5958e2d9e22b234f94353e386bf1c4d0a734efc6/research/nri/docs/61_track_b_cpu01.md); no general GEMM replacement claim |
| Track C | Can programmable attention and decoder construction be made robust? | Separate research archive; integration and independent qualification pending |
| G03 / G04 | How do partial abstractions, modular primitives and compositional planning develop? | Archive-only, see [evidence ledger](RESULTS.md) |
| G05 | Can acquired coordinates reduce the cost of future mathematical discovery? | Archive-only, see [evidence ledger](RESULTS.md) and [artifact registry](ARTIFACTS.md) |
| Parallel / Gap Closure | Can an optimizer predict and control developmental outcomes economically? | Archive-only, see [evidence ledger](RESULTS.md) |
| Track L | Can small networks learn and discover through formal mathematical language? | Archive-only preliminary Lean-style expression pilot; actual Lean checks pending |
| H01 | What is the smallest useful hybrid for supplied mathematical tasks? | [Hybrid inspection demo](hybrid_math/README.md) in this documentation change; full archive separate |

## Reading order

1. Begin with the [NRI specification](nri/docs/03_nri_spec.md) and [original prototype results](nri/docs/06_prototype_results.md).
2. Read the [registered dimensional study](nri/docs/12_dimensional_results.md) and [role-swap results](nri/docs/15_role_swap_results.md) for the difference between decoding, causal editability and generalization.
3. For later work, follow the [pinned Phase VIII report](https://github.com/Mnehmos/mnehmos.llm.playground/blob/5958e2d9e22b234f94353e386bf1c4d0a734efc6/research/nri/docs/60_phase8_results.md) and [reproducibility report](https://github.com/Mnehmos/mnehmos.llm.playground/blob/5958e2d9e22b234f94353e386bf1c4d0a734efc6/research/nri/docs/61_track_b_cpu01.md) rather than treating the unmerged draft as default-branch code.
4. Read [RESULTS.md](RESULTS.md) for later archive findings and limitations, then [ARTIFACTS.md](ARTIFACTS.md) for release status and archive identities.

## Status vocabulary

- **Default-branch source:** implementation and reports are present in the existing main lineage.
- **Branch evidence:** source exists at the linked revision but has not necessarily been merged.
- **Archive-only:** a separate research package exists; the repository summary is not a complete reproduction package.
- **Proposed:** a research objective, not a demonstrated result.

These categories describe availability, not scientific quality. Registered tests, exploratory follow-ups, exact replays and independent replications are different evidentiary levels.

Do not move or rewrite historical numbered reports to fit this taxonomy. New work should receive its own directory and a registered protocol; this index links the tracks together.
