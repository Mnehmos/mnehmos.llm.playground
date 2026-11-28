# G05 Session 15 — Developmental controllability: participation is not plasticity

## Decision

Concentrate the next program on **identifiable, transferable, and controllable computational distinctions**. Do not increase observer/controller capacity, add more graph-search grammars, or promote a native repair based on local causal participation or short-horizon training improvement alone. A useful observer must predict a beneficial *weight-space intervention* on an independently observed mathematical relation, not merely reproduce an activation-space effect.

## Review of our research library

- G05 S08 established the equal-coordinate collision test: one witness proves insufficiency, but lack of a witness does not prove sufficiency. S11–12 showed that perfect ordinary outputs can mask deficient intermediate factors, and a supplied factorized readout can acquire the missing unit-invariant computation. These are positive representational/readout controls, not native repair.
- G05 S13–14 found jointly consequential components and nonmonotone coalitions, but the learned policy's lower query count did not translate into faster CPU execution. S14's native target is donor prediction, not necessarily mathematical truth.
- Agent A, CIE-37 (cumulative computational graphs): adaptive DAG plus complementary coordinate qualified 36/36 on the bounded follow-up; raw search was inexpressive under its supplied depth/constant constraints. Repair the expressibility-matched baseline and charge full library acquisition/search/verification costs before claiming learned efficiency. H02's zero-weight catalogs also establish a strong deterministic comparison.
- Agent B, CIE-36 and CIE-38: matching reset displacement norms while retaining Adam history recovered 81.0% and 95.97% respectively of the reported mean reset advantage in their respective synthetic populations; not an additive causal decomposition. CIE-38's 3x LR beat full reset at 200 steps, and the learned predictor reduced to a schedule. Anchor-null projection reduced drift but impaired acquisition and cost more. Do not equate a reset with removal of harmful history or finite anchor retention with computation preservation.
- Agent C, CIE-28 R4: retained-history release reached 100% ordinary train exactness but only 27.12% train / 20.92% held-out independent donor recombination exactness. Conditional means and exact AA/AB/BA/BB decomposition reveal partner-dependent computation; auxiliary interchange supervision improves the intended pathway but is not autonomous discovery. Adaptive guards did not consistently beat fixed controls. A useful protected constraint must preserve the mathematical invariant while permitting migration of its implementation.

These are reported exploratory findings in the Library, not independently replayed Agent A/B/C trajectories. S15 uses the five exact G05 S14 parent checkpoint identities.

## Relevant literature and what it changes

Focused reading of arXiv abstracts and available indexed descriptions; not a systematic review or novelty claim.

1. Hase et al., [Does Localization Inform Editing?](https://arxiv.org/abs/2301.04213): causal localization in factual LMs need not identify the best parameter edit location. Direct motivation for the matched native head pilot; our mathematical developmental setting is different.
2. Geiger et al., [Causal Abstraction](https://arxiv.org/abs/2301.04709), [Inducing Causal Structure](https://arxiv.org/abs/2112.00826), and [Distributed Alignment Search](https://arxiv.org/abs/2303.02536): assess the faithfulness of high-level variables through interventions, allow distributed/non-axis-aligned coordinates, and distinguish supplied interchange targets from discovering the variables.
3. Kramár et al., [AtP*](https://arxiv.org/abs/2403.00745), and Heimersheim & Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255): attribution approximations have failure modes; use cheap proposals and exact verification, and distinguish effects of intervention choice from necessity/sufficiency.
4. Nanda et al., [Progress measures for grokking](https://arxiv.org/abs/2301.05217), and Varma et al., [Explaining grokking through circuit efficiency](https://arxiv.org/abs/2309.02390): mechanistic progress can be continuous despite discontinuous accuracy; partial/competing circuits warrant a mechanistic trajectory score rather than loss alone.
5. Jacot et al., [Neural Tangent Kernel](https://arxiv.org/abs/1806.07572), and He & Su, [Local Elasticity](https://arxiv.org/abs/1910.06943): functional response to parameter updates offers a different object from an activation patch; measure empirical finite-model Jacobians and cross-context transfer without assuming the infinite-width limit applies.
6. Farajtabar et al., [Orthogonal Gradient Descent](https://arxiv.org/abs/1910.07104), and Yu et al., [Gradient Surgery](https://arxiv.org/abs/2001.06782): preservation and conflict projections are established baselines. S15 uses a much simpler scalar-loss projection; it is not OGD or a full PCGrad reproduction.
7. Lee et al., [Surgical Fine-Tuning](https://arxiv.org/abs/2210.11466): the relevant parameter subset depends on the kind of distribution shift. Compare architectural components and coordinated encoder/consumer directions, not merely a ranking of heads.
8. Chen et al., [Learning to Optimize: A Primer and A Benchmark](https://arxiv.org/abs/2103.12828): distribution-specific learned optimizer performance makes held-family and cost-matched deterministic controls essential.
9. Schölkopf et al., [Towards Causal Representation Learning](https://arxiv.org/abs/2102.11107), Cranmer et al., [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287), and Palmarini et al., [Bayesian Program Learning by Decompiling Amortized Knowledge](https://arxiv.org/abs/2306.07856): causal variable discovery and amortized reusable program libraries have prior art. Focus on the developmental acquisition of missing distinctions and whether the amortized cost beats equally expressive controls, rather than claiming novelty for reuse itself.
10. Xu, [Low-Dimensional and Transversely Curved Optimization Dynamics in Grokking](https://arxiv.org/abs/2602.16746), is a recent preprint motivating trajectory-subspace/curvature controls. No direct replication or endorsement of its proposed mechanism is claimed.

## Executed pilot A — localization vs plasticity

Five exact step-6000 parents checked against S14 SHA locks. One competent parent (34001_exact) had no sign errors and was excluded by registered eligibility; the four remaining parents include two intentionally quotient-trained, one spontaneous quotient and one exact-objective but sign-folded model. Native head groups of QKV rows/bias and output projection columns, 4,144 weights each. No original parameter/state overwritten. No originally withheld combinations used. Retrospective familiar-grid sampling of known error and correct examples, separate train and exam groups. 64 matched 20-step branches, 1,280 normalized SGD updates, displacement .04 per step; target-only and first-order protection of mean incumbent-correct CE. Only selected weights modified. Patch scores measured on same-answer train-only donors; 7/512 fallback donors matched output only instead of both canonical quantities. No historical Adam continuation.

Mean across 32 branches per arm at step20:

| Measure | Base | Target only | Protected |
|---|---:|---:|---:|
| Previously incorrect examples, exact | 0% | 32.01% | 40.82% |
| Previously correct examples, exact | 100% | 53.49% | 63.72% |
| Balanced exact | 50% | 42.75% | 52.27% |
| Balanced quotient accuracy | 100% | 72.33% | 92.24% |
| Mean reserved sign-loss improvement (higher better) | 0 | -0.00581 | -0.00408 |

Mean per-parent Spearman correlation between patch hit rate and actual reserved sign-loss improvement at step20: -0.119 target and -0.141 protected. Largest initial gradient norm: -0.113 and -0.024. Patch mean margin: +0.089 and +0.048. Mean best-of-eight oracle gains were small (0.0121 target, 0.0065 protected) and not deployable. No robust repair.

## Executed pilot B — paired distinction rather than flipping outputs

After A, fixed a fresh familiar-only paired panel per parent (64 train, 64 calibration, 64 exam opposite-mass groups); excluded all A-sampled prompt groups and their opposite partners, including donor groups and both presentation orders. Opposite outputs, same other fields, one originally correct and one wrong. Three objectives: balanced sign CE; paired margin-difference loss; paired difference with first-order quotient-loss protection. Same head groups and fixed step displacement. 96 branches, 1,920 updates. Calibration gradient-response score measured at initial state and tested on separate exam. All branch/selection results retained, no exam-based action.

| Measure (means over 32 branches per arm) | Base | Balanced | Pair | Pair + protect |
|---|---:|---:|---:|---:|
| Exact | 50% | 45.43% | 45.24% | 46.83% |
| Both members exact | 0% | 5.27% | 4.00% | 4.74% |
| Quotient | 100% | 92.31% | 91.19% | 94.73% |
| Mean pair distinction margin | 0.00010 | -0.00040 | -0.00093 | -0.00054 |
| Mean absolute common bias | 0.03256 | 0.06547 | 0.12930 | 0.12977 |

No step20 branch exceeds 53.125% exact; maximum both-member exactness 18.75%. 27/96 branches improve balanced sign loss (12/32, 6/32, 9/32); none qualifies as a robust repair. At 20 steps, calibration gradient-response rank mean Spearman +0.089/-0.179/-0.107, by arm. The improvement in some originally wrong prompts is not evidence of acquiring an independent sign coordinate.

## Local response audit — identify the immediate failure

Exploratory post-hoc float64 audit on the fixed B panels and exact parent weights, never using the exam gradient to choose a training action. For each of 32 head-parent combinations, normalized negative train balanced sign-loss direction, evaluated displacements 1e-4, 1e-3, 1e-2, 0.04. Within-panel prediction is negative gradient dot displacement. Across-panel prediction uses a different calibration gradient.

| Step norm | Calibration gradient vs actual calibration, mean rank rho | Calibration gradient vs actual exam, rho | Exam sign agreement using calibration gradient |
|---:|---:|---:|---:|
| 0.0001 | 1.000 | -0.065 | 43.75% |
| 0.001 | 1.000 | -0.065 | 46.88% |
| 0.01 | 0.982 | -0.119 | 46.88% |
| 0.04 | 0.685 | -0.190 | 40.63% |

Mean cosine train/calibration -0.0724; train/exam +0.0474; calibration/exam -0.0858. At 1e-4, all 32 train directions improve their own loss, only 14 improve the exam loss. The actual same-panel calibration changes also rank exam changes poorly (rho -0.065 at 1e-4). Even exact local response on one context panel is insufficient for transfer here. Larger radius adds local approximation error, but cannot explain the small-radius cross-context failure. No conclusion that all transferable directions are absent; restricted head groups, particular objective, small contextual samples and mature quotient state are important limitations.

## Next razor-edge experiment: distinguish existence, estimation and reachability

Use an independently defined missing mathematical distinction D(theta) and preserved invariant K(theta). For an admissible parameter subspace S, calculate empirical J_D and J_K on acquisition observations. Solve a trust-region residual problem: minimize ||J_D delta - r||^2 + lambda ||J_K delta||^2 with delta in S and ||delta|| <= epsilon. Include a strict null-space version (when feasible), ordinary balanced loss, full Jacobian least squares, normalized SGD and random subspaces with the same free dimensions. This is a conventional local reachability diagnostic, not a new learning algorithm or proof of global impossibility. Compare achieved residual reduction to projected rank/singular values and to independent context transfer. Keep first-order predictions, finite observed changes and later developmental outcomes separate.

Registered future work should test three falsifiable alternatives: (1) a transferable local direction exists but is poorly estimated: increase independent acquisition contexts and use train-only nested sample curves and uncertainty; (2) the chosen parameter group lacks a usable direction: compare encoder-only, decoder-only, coordinated groups, and a supplied constructive readout as positive control; (3) successful movement requires a finite representation/consumer reorganization: matched temporary constraints, release and recompute tangent diagnostics at early/mid/late checkpoints. Reject an explanation if the corresponding control fails. Negative local rank results only describe the chosen state, subspace, metric and tolerance.

- Agent A: build compact, independently verified contrast/measurement resources and equally expressive raw and library search baselines. Charge acquisition and verification; assess observation efficiency vs real runtime separately. Prioritize missing independent distinctions over expanded residual expressions.
- Agent B: measure actual functional update response and cross-context gradient transfer under original optimizer states, displacement-matched retained-history/reset/LR controls, then finite-horizon regret and cost. Use fixed baselines; no larger controller until a held-family signal beats them.
- Agent C: test whether temporary separation of operand representations creates transferable causal contrasts after removal. Distinguish preservation of existing mathematical invariants from preservation of the mechanism implementing them. Matched parameter/compute budgets; compare output-only, contrast supervision, canonical labels and sham interventions, including native donor recombinations, then sealed exact OOD.

Shared promotion gate: predeclare fresh task/seed families and sealed splits, 99.5% exact and independent multi-site recombination thresholds on learnable controls, report failure/retention/migration/compute and fixed-baseline performance. No claiming general mathematics from a supplied readout, no choosing a gauge on the final exam, no declaring a static causal circuit a repair policy. Thresholds and future experiment are proposed, not measured here.

## Reproduction and audit

Extract S12 core archive at `../g05_s15_parents/nri_G05_session12/` relative to this directory, retaining its structure. Only `sources/nri_G05_session09/*` and its nested native source/checkpoints plus `PARENT_MANIFEST.sha256` needed. Run in a clean copy of this directory: `python pilot.py`, `python pair.py`, `python local.py`, `python audit.py` (Python, PyTorch CPU, NumPy, SciPy, pandas; one thread). `REGISTRATION.md`, `PAIR_REGISTRATION.md`, `LOCAL_REGISTRATION.md` document successive exploratory registrations. Every historical parent checked; 64 full final states and 288 absolute selected-slice states reloaded and rescored (max discrepancy 4.44e-16 from CSV serialization); not full training replay. 3,200 actual gradient updates across 160 branches. The per-branch training loops with evaluations/checkpoint serialization took 12.70 + 17.79 elapsed seconds summed, excluding setup, separate local audit, archive, and orchestration. Do not interpret as end-to-end speed. Source and outputs preserve negative results, no new learned controller, no native qualification, no blind original withheld evaluation. Scope is four sign-folded mature models, chosen familiar-error panels, equal norm but not equal actual runtime, scalar first-order protection, short fixed horizon, no joint all-head or decoder training; not a proof against all native learning or evidence of missing information. The review is focused, not exhaustive. Parent hashes, saved prompts and branch states remain in archive; exact original weights are available from S12.
