# Evidence ledger

This ledger summarizes bounded experiments. A correct neural prediction, an exact compiled algorithm, and a hybrid solver using supplied code are not interchangeable claims.

## Source available on main

| Study | Reported outcome | Qualification |
| --- | --- | --- |
| Typed training / untyped inference | Auxiliary type targets improved editability; type removal without preparation hurt accuracy. | No untyped-inference learning-speed benefit established. [Doc 08](nri/docs/08_untyped_inference_results.md) |
| Dimensional reasoning, 70 networks | Held-out edit success 0.36 vs 0.17 for full causal specification vs none. | Registered OOD hypothesis unsupported; 0.61 vs 0.80. Failed seeds exhibited sign ambiguity. [Doc 12](nri/docs/12_dimensional_results.md) |
| Role swap, 36 networks | Primary specificity contrast +0.374, 12/12 matched seeds positive. | Copying actual patterns worked much better than fitted edits; edit construction remained a bottleneck. [Doc 15](nri/docs/15_role_swap_results.md) |
| Unit-aware editing | Improved route-aware and permanent editing. | Existing-weight permanent version involves substantial refitting. [Docs 17](nri/docs/17_editor_results.md), [19](nri/docs/19_permanent_edit_results.md) |
| Matrix benchmark | No tested finite alternative beat native PyTorch in the recorded CPU comparison. | The exponent-bound proof-to-code effort did not yield a practical qualifying finite certificate. [Results](nri/matmul_benchmark/RESULTS.md) |

## Branch evidence (review pending)

The [Phase IV–VIII draft](https://github.com/Mnehmos/mnehmos.llm.playground/blob/5958e2d9e22b234f94353e386bf1c4d0a734efc6/research/nri/docs/60_phase8_results.md) and [PR #36](https://github.com/Mnehmos/mnehmos.llm.playground/pull/36) describe compiled arithmetic, protected interfaces and progressive acquisition. Selected protected hybrids met the registered competency criterion 3/3 versus 1/3 for matched scratch controls; the reported 12–47× training-FLOP-per-success comparison counts failures. This is a specific finite task, not a general scaling law. Canonical representation access, not simply installed arithmetic, was central to the advantage. Some data-efficiency, extraction and robustness criteria failed. See the full report for definitions and accounting.

The [CPU reproducibility report](https://github.com/Mnehmos/mnehmos.llm.playground/blob/5958e2d9e22b234f94353e386bf1c4d0a734efc6/research/nri/docs/61_track_b_cpu01.md) distinguishes bitwise historical replay from stability under rounding-level changes. Historical generalizers are sensitive to such changes. Exact replay does not establish population-level reproducibility.

The [causal authority branch](https://github.com/Mnehmos/mnehmos.llm.playground/tree/d34aa61cb5826b5287380ad96f33af96ad219bf9) investigates maintaining control of installed computations under continued optimization. Read the branch and [PR #30](https://github.com/Mnehmos/mnehmos.llm.playground/pull/30) before drawing broader conclusions; manuscript approval and independent replication are separate.

## Archive-only research

These are summaries of the retained experimental reports. They are not claims that all code/checkpoints are present on main. Exact archive names and available SHA-256 hashes appear in [ARTIFACTS.md](ARTIFACTS.md).

### G04 — modular computation and planning

- Session 02: 587 trainable parameters for a modular finite-field FMA operator versus 3,393 monolithic; perfect unseen results 3/5 versus 1/5, and 5/5 with registered permutation completion. The quotient decomposition, supervision and bijection constraint were supplied. The 390-bit compiled table count excluded parser and execution program.
- Session 05: 629-parameter arithmetic library and explicit positional/runtime algorithms. A 13-parameter ordering primitive generalized across eight tested initializations. Exact signed arithmetic was tested through 256 bits; unbounded correctness is an algorithmic claim, not a neural-only benchmark.
- Session 06: two 7,609-parameter compositional controllers generated 3,000/3,000 exact programs on restricted mathematical expressions through depth five. Grouping, operator vocabulary and local templates were supplied. Neural induction from numerical demonstrations remained near zero on the original broader panels. The deterministic grammar parser is a necessary baseline.

### G05 — cumulative representations and residual discovery

- S06: the graded persistent library saved 77.85% of representation observations versus direct scratch (68.76% of total acquisition calls in its supported stream), not optimizer updates.
- S07: at 64 variables, 22 supported sparse/reuse tasks required 4,480 discovery calls under learned residual ranking or classical OMP, versus 236,720 for full interpolation and 69,600 for prior-library full fallback. This is 98.11% and 93.56% fewer discovery calls respectively. Supplied polynomial/context structure and conventional recovery account for most of the advantage; learned ranking does not consistently beat OMP.
- S08: two 433-parameter stopping policies used 163,734 and 163,288 calls across 72 test trajectories, versus 167,876 for a fixed exploration cap. This modest learned advantage concerns counted observations in replayed trajectories, not whole-system wall time. Exact fiber witnesses distinguish an insufficient representation from an insufficient head in the tested families. Exact observation recycling and modular certificates are conventional algorithms.

### Developmental optimization / Gap Closure

- Study05: natural task-only learners and reversible restrictions produced useful counterexamples, but the validation-selected policy was fixed random restriction. Learned intervention selection was not promoted.
- Gap01: growing versus frozen libraries under the same recovery method isolated a cumulative-learning benefit; at 64 variables discovery calls 6,520 to 4,480. Frozen neural developmental policies did not reliably beat AdamW or simple controls on fresh seeds. Reversible neural recovery improved exact polynomial recovery from 33.33% to 76.83% on a supplied dictionary, below OMP's 98%.
- Gap02: an invertible basis change destroyed practical acquisition via conditioning; exact canonicalization recovered the original outcomes. A classical adaptive factor method recovered 96/96 supported product-of-linear-form tasks, with 91.27% fewer discovery observations than dense interpolation at 64 variables. Factorization family was supplied. Ordinary AdamW with the additional speculative update budget beat every tested shorter intervention and even the retrospective oracle on four new states.

### Track L — preliminary typed language

Two 3,749-parameter factorized controllers achieved 400/400 exact typed AST matches per group through depth five, using supplied grouping and type context. Combined with the preserved 629 weights, the smallest tested combination has 4,378 neural parameters. Numerical induction: 32/104 known-family cases, 0/68 withheld-family cases; finite symbolic enumeration solved all. **Lean was not installed; no elaboration, kernel-checked proofs or unrestricted Lean competence are established.**

### H01 — hybrid mathematical minimality

The [inspection demo](hybrid_math/README.md) uses the actual frozen 13 phase parameters and 183 router weights from the archived prototype. Full H01 research results:

| Measure | Reported result |
| --- | --- |
| Raw neural storage | 196 FP32 values, 784 bytes |
| Arithmetic panel | 368/368 cases, 46 supplied task types; zero-neural arithmetic baseline also passes |
| Family recovery | 1,400/1,400 supported functions within 3 solver attempts, 0/300 control acceptances |
| First solver | Neural 1,373/1,400; heuristic 1,374/1,400 |
| Total supported solver attempts | Neural 1,427; heuristic 1,452; fixed order 5,600 |
| OOD first solver | Neural 676/700; heuristic 700/700 |

The full standalone package is 137,565 bytes of file payload before ZIP compression (plus dependencies). The original 13 coordinates were structure-initialized from earlier inferred geometry. The selector learns ranking over seven supplied families, not the families themselves. Its other seed was weaker; no robust neural-over-heuristic, runtime, formal-proof or minimum-size claim is established. The new inspection demo is not the full standalone runtime.

## Open qualification

Reliable native unseen generalization across seeds; autonomous representation/decomposition discovery; calibration and cost-aware intervention control; discovery outside supplied families; actual Lean kernel qualification; independent external reproductions. See [roadmap](ROADMAP.md).
