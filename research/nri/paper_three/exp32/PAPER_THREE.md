# Preserving Causal Control in Programmed Neural Networks
## When correct installed computations lose control of the answer

**Author-review preprint.** Author names, affiliations, acknowledgments, order, and submission decisions are reserved for owner review.

### Abstract

An explicitly programmed computation can remain correct while optimization removes its control over a network's answers. In a preregistered 12-seed study, frozen converters retained perfect isolated accuracy but the full model followed only 3.4% of held-out quantity interventions despite 99.8% practiced accuracy. Readout suppression restored exact control, but the learned pathway reappeared within a few updates even with reset optimizer state. The same dissociation occurred after installation into a trained host with reserved capacity. A new matched study separated confidence gradients from shared-task spillover. Routing ordinary conversion gradients away from the competing path did not prevent displacement when another task trained the same readout. Soft counterfactual margins reached 98.6% held authority while a flagged task reached 92.6%. Eight of twelve seeds passed every prespecified criterion. A five-state noncommutative automaton replicated a smaller dissociation without the binary parity ambiguity. We distinguish mathematical guarantees, empirically qualified protection, and durable recovery.

### Introduction

An installed algorithm and a learned alternative can both contribute to the same output. Ordinary accuracy answers whether the sum works, while a surgical change to the algorithm's internal state asks whether the final computation actually respects that state. The distinction matters when weight-space interventions install interpretable computation in networks that will continue learning. Parameter retention as studied in continual learning, direct neural compilation, and intervention-based identification offer complementary tools. Our testbed explicitly exposes the two paths so these tools can separate preserved execution from loss of causal control. The experimental claim is limited to finite tasks and the shared additive path studied here.

### 1. Program integrity and causal authority

Behavioral correctness is full-model accuracy on untouched inputs. Program integrity is the accuracy of the installed computation in isolation. Causal authority is full-model agreement with the specified counterfactual after replacing only the declared program representation with a legal donor's representation. Repairability denotes immediate recovery after changing a model, while durability denotes subsequent retention during continued training. Plasticity is acquisition of a separate task. Compositional reuse requires transfer to unfamiliar combinations of known operands. High performance on any one of these does not establish the others.

*Figure 1. Interchanging a legal program state changes the programmed branch while the learned path continues to see the original written input. For parity, the prefix and suffix transitions take the place of encoder and decoder.* The figure is in the compiled manuscript and reproduction archive.

### 2. Related work and novelty boundary

Tracr uses compiled transformer programs as mechanistic ground truth (Lindner et al., 2023). Omlin and Giles (1996) establish direct recurrent-weight encodings of finite-state automata and their stability under forward execution. Our parity study uses this well-established construction idea and separately measures how a learned residual affects the finite-state program’s role under optimization. WeightCode installs new computations into trained networks, checks their causal roles, and tests retention through successive weight programs (Li et al., 2026). That comparison concerns additional executable installations. Experiment 30 instead tracks the same installed operation under continued gradient updates. Residual bypass is established in concept-bottleneck work, including Residual Stochastic Concept Bottleneck Models (Marcou et al., 2026), which estimate concept–residual dependencies and condition the residual when intervening. Laguna et al. (2024) train existing black boxes for concept intervenability, and Geiger et al. (2021, 2022) formalize and train interchange-based causal correspondences. Koh et al. (2020) formalize human-intervenable concept bottlenecks. Work on shortcut learning, continual-learning retention and competing circuits motivates separating outputs from the paths supporting them (Geirhos et al., 2020, Kirkpatrick et al., 2017, Merrill et al., 2023). Gradient routing can localize learning signals (Cloud et al., 2024). Marconato et al. (2023) characterize reasoning shortcuts as unintended optima of a neuro-symbolic objective. Our comparison keeps the programmed computation fixed in selected arms and measures the learned contribution separately at the output. Logistic optimization can keep increasing confidence on separable data (Soudry et al., 2018). Their max-margin setting is not an AdamW theorem about this hybrid transformer. Decoupled decay and accumulated optimizer state motivate separating raw gradients from effective updates (Loshchilov and Hutter, 2019). Our narrower contribution is the prospective separation of program integrity, competing-path authority, targeted recovery and subsequent learning in the same known computation. Priority across all prior work is not asserted.


### 2.1 Formal model and objective

Write the full logits as $z(x,q)=P(q)+R(x)$ on the declared domain. Ordinary inference sets $q=Q(x)$. Interchange substitutes $q=Q(x')$ while holding the input $x$ to the competing branch fixed. Let $y$ denote the mathematical target for the intervention. For rival $j\ne y$, define the program margin $p_j=P_y-P_j$, the competing margin $c_j=R_j-R_y$, and the effective margin $p_j-c_j$. The intended output is a unique winner exactly when the smallest effective margin is positive. We count argmax ties by the implementation rule in measurement and exclude near-zero margins from the algebra verification.

For attenuation factor $\alpha\in[0,1]$, intersect $p_j-\alpha c_j>0$ over all rivals. Competitors with positive $c_j$ impose upper bounds $\alpha<p_j/c_j$. Negative $c_j$ impose lower bounds, and zero $c_j$ require $p_j>0$. On exact isolated inputs all $p_j>0$, so the supremum retained fraction is the minimum upper ratio clamped at one. A strict equality can tie. Componentwise bounds of $|R_k|\le0.4$ guarantee a final margin of at least $0.2$ when the legal one-hot program margin is one. Isolation sets $c_j=0$. Soft training leaves inference unrestricted.

For natural cross-entropy the residual gradient is $\partial L/\partial R_j=\mathrm{softmax}(z)_j-\mathbb{1}_{j=y}$. Correct classification at a finite logit margin does not set that gradient to zero. For $K$ classes with winning $m$ and all other logits zero, its norm is $\sqrt{K(K-1)}/(e^m+K-1)$. At $K=13$, increasing $m$ from one to five attenuates the initial gradient but does not zero it. The hinge $[0.5-\min_{j\ne y}(z_y-z_j)]_+$ has zero derivative when the margin is strictly above 0.5. The natural hinge can reactivate after other tasks change the branch. The causal hinge applies the same comparison to a valid transplant with the correct counterfactual target.

Integrity, correctness and authority are the expectations of the corresponding success indicators over their stated domains. Displacement is the combination of retained isolated performance and reduced full-model authority, diagnosed against competing-branch suppression. Repairability requires immediate improvement by at least 15 points to at least 95%. Durability additionally requires subsequent thresholds at fixed updates. Plasticity is measured on specified other tasks. Reuse requires accuracy on new combinations plus selective causal effects. These operation-specific measures are not a general taxonomy of all learning.

### 3. Exact programs can lose causal authority

For written value x and source and destination unit scales s,d, the exact converter produces q = xs mod 13 and output q/d mod 13. Unit-indexed permutation matrices implement encoding and decoding. The shared learned trunk in parallel can add conversion logits. Its input remains unchanged during an interchange at q. A legal counterfactual changes the converter's one-hot quantity only. The winning programmed logit has margin 1. A residual bounded to ±0.4 can reduce a pairwise margin by at most 0.8, leaving a guaranteed margin of at least 0.2. Isolation independently guarantees the same decisions throughout this mathematical domain.

The sealed endpoint used nine distinct written base prompts and 216 discriminating intervention pairs with four distinct intended, unchanged, raw-number, and donor-answer targets. The unit of replication is the seed. A separate process on the same machine generated, committed, and held the final assignment until the terminal review. This was process-separated custody in a shared environment. An external researcher did not administer it. Twelve fresh matched seeds each received six conditions, 600 AdamW updates, and seven audited checkpoints.

| Condition | Practiced accuracy | Final ordinary | Final authority | Isolated authority |
| --- | ---: | ---: | ---: | ---: |
| A Trainable program, unrestricted | 100.0% | 37.0% | 8.9% | 100% |
| B Frozen program, unrestricted | 99.8% | 7.4% | 3.4% | 100% |
| C Frozen, bounded | 100% | 100% | 100% | 100% |
| D Frozen, counterfactual supervision | 99.8% | 23.1% | 10.4% | 100% |
| E Isolated | 100% | 100% | 100% | 100% |
| F Frozen, matched ordinary supervision | 99.9% | 9.3% | 3.5% | 100% |

The four preregistered paired mean effects in percentage points were E−B +96.6 [96.3,97.0], C−B +96.6 [96.3,97.0], D−F +6.9 [4.2,9.9], and B−A −5.5 [−6.2,−4.8]. Intervals are 95% paired bootstrap. The exact two-sided seed-level sign-flip p-values were Holm-adjusted to .00195 for all four contrasts. This direction for B−A is specific to the tested design. Every B run first crossed the sustained-loss threshold between updates 20 and 50 on development examples. The timing grid does not identify an exact update.

Removing only the residual readout restored perfect authority in all 48 unrestricted models at the endpoint. Scaling the block to one eighth restored perfect authority in all 12 D runs, while B averaged 16.5% and F averaged 17.6%. A averaged 97.2%, so easy recovery did not require causal supervision. This attenuation changes 87.5% of the edited block norm. Causal supervision improved authority and facilitated immediate repair relative to its matched extra-supervision control. It fell far below practical preservation. All arms reached 99.8–100% on held forms of the routing task. Held canonical force pairs scored 13.9–15.0%, and force swaps were weak, so the study does not establish compositional reuse.

*Figure 2. Experiment 30 shows the visible authority trajectory, sealed endpoint and immediate residual attenuation. Thin lines and points show the seed-level comparisons.* The figure is in the compiled manuscript and reproduction archive.

### 4. Competing logits and controlled replay

Experiment 29's retrospective replay identified ordinary conversion loss as sufficient for the original decrease in authority from initialization. Force loss alone left the zero-initialized competing conversion readout inactive. After displacement, existing optimizer state could move conversion readout rows even with both fresh losses zeroed. This motivates a retained-versus-reset optimizer contrast on actual continuation copies. Local gradients are sensitivity measurements on controlled inputs and do not reconstruct a unique historical contribution for each token. The Experiment 32 retrospective replay reconstructs 5,208 panels from 1,302 conversion checkpoints across Experiment 30, repair, and trained-host trajectories, plus all 84 Experiment 29 checkpoints on their archived swap panel. Experiment 30 frozen-only program margin remained one. On visible held swaps, mean seed-median full margin changed from 1.00 to -6.41 and -12.77 at 0, 100 and 600. Checkpoint logits reproduce non-tied argmax outcomes at all four tested attenuation strengths. The endpoint thresholds reproduce the recorded Experiment 30 attenuation scores. This contemporaneous check is algebraic verification. Archived early points often lack full raw tensors, so no past logit trajectories were synthesized for those points.


### 5. Durability of targeted repair

We branched 96 continuation copies from 12 prespecified displaced Experiment 30 endpoints, covering three seeds and source conditions A, B, D and F. The table reports means across the same 12 endpoints per repair. Reused Experiment 30 questions make the result exploratory. Fit-based repair used original training pairs only and changed the conversion readout. Durability requires crossing the immediate recovery threshold and retaining at least 95% at all three measured continuation endpoints.

| Repair or control | Immediate | +100 | +300 | +600 | Durable / 12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zeroed readout | 100.0% | 7.8% | 6.9% | 6.5% | 0/12 |
| Zeroed plus reset | 100.0% | 7.6% | 8.0% | 8.1% | 0/12 |
| Retain one eighth | 56.4% | 7.5% | 6.8% | 7.3% | 0/12 |
| 10% block budget | 6.4% | 7.3% | 6.6% | 6.9% | 0/12 |
| Causal readout fit | 38.1% | 7.2% | 7.1% | 6.9% | 0/12 |
| Causal fit + soft penalty | 38.1% | 34.5% | 34.1% | 36.1% | 3/12 |
| Untouched | 6.1% | 7.4% | 6.5% | 7.6% | 0/12 |
| Untouched plus reset | 6.1% | 7.1% | 6.9% | 8.3% | 0/12 |

The zeroed-readout mean fell to 48.9% after five updates with retained AdamW state and 43.3% with reset. Both recovered perfectly at step zero and fell below 8% by 100. In a separate one-step replay with no fresh gradient, retained moments alone moved the zeroed rows by a mean Frobenius norm of 0.00315, while reset left them exactly zero. The intervention demonstrates that stored moments can restart the readout. The continuation experiment shows that displacement still returns without those moments.

The 10% block-budgeted fit changed the weights by mean absolute Frobenius norm 0.110, or 0.277% of the full parameter norm. Retaining one eighth changed the conversion block by 87.5% and the full model by 2.424%, with mean absolute norm 0.963. Zeroing changed the block by 100% (mean absolute 1.100 and 2.770% global). Small global ratios reflect the block's size. They do not make these edits small within the edited block. The centered-residual loss improved recovery relative to continuation without it, but its mean at 600 was 36.1% and only 3 of 12 met strict durability, all from trainable-program source A. It operated without a mathematical guarantee. We did not test a fully crossed pairing of every repair with every protection objective.

### 6. Installing a program into a trained host

All three hosts reached 100% held written-form routing before installation. The installation assigned the exact encoder and decoder into zero reserved tensors, leaving the preinstallation and post-installation host logits bitwise identical. All promoted forks began exact on ordinary, isolated, causal, and equivalent-quantity tests. The six continuation arms shared the starting host for each seed. The final independent conversion partition had 288 eligible pairs and 12 written bases. Its identities were determined prospectively and it was scored at the terminal boundary. This design tests a prepared host with reserved architectural capacity. It does not test arbitrary edits into an unprepared network.

| Condition | Final swap success | Final ordinary conversion | Held host routing |
| --- | ---: | ---: | ---: |
| A | 13.3% | 83.3% | 100.0% |
| B | 4.2% | 16.7% | 99.7% |
| C | 100.0% | 100.0% | 100.0% |
| D | 13.9% | 33.3% | 99.8% |
| E | 100.0% | 100.0% | 100.0% |
| F | 5.9% | 5.6% | 96.1% |

Release of gradients after installation reproduced the integrity–authority gap in all three frozen-unrestricted forks. B fell from 100% to 66.4% at the first update on the visible panel and retained 4.2% on final swaps while its isolated converter was perfect. The different starting trunk, seeds and held partitions prevent a clean claim that pretraining itself caused the faster onset. The bounded and isolated references preserved exact authority, and the ordinary host task stayed highly accurate.

### 7. Recurrent finite-state task

The parity implementation uses two fixed-size weight matrices, identity for zero and state flip for one, applied successively across eight input tokens. At token four a state interchange replaces the prefix state while preserving the base suffix. The competing transformer still reads all original bits. Across the independent terminal split of 48 sequences and 3,856 discriminating pairs, the frozen-unrestricted recurrent program stayed 100% correct in isolation. Full ordinary accuracy was 22.9%. Raw swap success was 88.9%, but an unpatched output already happened to equal the swapped target on 77.0% of these binary trials. A retrospective correct-base control gave 52.5% swap success. This controls for the binary collision and shows partial displacement in this arm, without treating the unconditional score as evidence of reliable state use.

Trainable, bounded and isolated parity arms retained perfect ordinary and interventional accuracy. In contrast to the conversion task, trainable programs did not lose authority. The result supports replication of the frozen-only failure in a second computational structure under these conditions, while preserving task-dependent and arm-dependent outcomes. These are three matched seeds, not thousands of independent replications.

*Figure 3. Visible or reused authority trajectories across the three Experiment 31 studies. Averages include all runs. Parity curves show the registered unconditional binary metric and require the unpatched and correct-base controls for interpretation.* The figure is in the compiled manuscript and reproduction archive.

### 8. Scope of the durability and installation studies

The strongest evidence is the preregistered 12-seed Experiment 30 comparison. Experiment 31 uses small descriptive replications and an explicitly reused durability panel. The constructed operations are exact in finite mathematical domains and share an additive learned competitor by design. Performance on these tasks does not establish behavior for general programs or growing networks. Routing plasticity had a ceiling, and poor force performance leaves compositional reuse unresolved. Counterfactual supervision alone was insufficient under the tested objectives.

The registered Experiment 30 code is at commit [`5eceb6cf`](https://github.com/Mnehmos/mnehmos.llm.playground/commit/5eceb6cfade4c2d196b6b5e81e71c7a38cf95592). The three Experiment 31 protocols and source digests are at commit [`f274ecc3`](https://github.com/Mnehmos/mnehmos.llm.playground/commit/f274ecc3553df5915cfbe882b7a396c140404905). Experiment 31 contains 891 full checkpoints, including three host pretraining stages, with model and optimizer states, RNG, objective-separated gradients, signed occurrence traces, per-step effective optimizer updates, tensor changes, audit receipts and hashes. Temporal cosine comparisons use fixed probe activations. They do not reconstruct exact concept identity or historical token-to-weight causation. Data are indexed by seed, source arm and repair condition, with archive and checkpoint checksums. See the accompanying README and per-study registration for launch and review commands.


### 9. Objective interventions separate confidence and spillover

Experiment 32 runs 12 matched seeds per task for 300 AdamW updates, then opens a prospectively committed final partition. The objective task uses exact frozen unit-indexed maps of shape 3 by 13 by 13 for encoding and decoding. The additive transformer has two layers, width 16, two heads and MLP width 32. Its residual readout starts at zero. AdamW uses learning rate .002, betas .9 and .999, epsilon $10^{-8}$, and weight decay .01. Each update draws 40 natural cases, 40 extra bases and 40 donors. In the objective task, a distinct flag asks for the written operand instead of converted quantity. This outside-domain task trains the same residual output rows. All treatments also train a simple binary routing readout. Its held labels can be imbalanced, so it is not a demanding retention or transfer test. Prior host retention was evaluated independently in Experiment 31.

Natural CE is the unrestricted baseline. The natural hinge stops rewarding confidence above margin .5, scale multiplies program logits by five, and routing detaches the natural loss from the residual while leaving other learning signals. The soft objective adds four times a hinge on valid counterfactuals. The matched ordinary objective uses the identical extra weight, batch, passes and hinge on ordinary targets. Original interchange CE is reported in the previous experiment and all two-seed pilot outputs. It was not repeated at full scale. The pilot established baseline learnability of the flagged task and was separate from confirmation.

Conversion has 468 written prompts. Aliases for each semantic triple stay together during partitioning. The final panel contains 52 written bases representing 13 triples, 1,365 eligible draws and 1,119 unique intervention pairs. A separate local process created and committed hidden-development and final assignments, retained them in memory, and released the scores only after all reviews. It was not externally administered. Reported intervals use matched seed bootstrap, and exact two-sided sign flips use Holm correction within each family.

| Treatment | Authority | Ordinary | Flagged | Pass / 12 |
| --- | ---: | ---: | ---: | ---: |
| Natural CE | 7.6% | 23.6% | 92.8% | 0/12 |
| Natural margin satisfied | 84.5% | 85.9% | 93.3% | 3/12 |
| Program scale 5 | 90.6% | 90.7% | 91.7% | 3/12 |
| Natural gradient stopped | 8.7% | 11.7% | 96.0% | 0/12 |
| Matched ordinary hinge | 11.4% | 46.8% | 88.8% | 0/12 |
| Soft counterfactual hinge | 98.6% | 99.8% | 92.6% | 8/12 |

| Paired contrast | Mean (points) | 95% bootstrap interval | Holm p |
| --- | ---: | ---: | ---: |
| satisfied-ce | +76.9 | [+66.6, +86.2] | 0.00195 |
| scale-ce | +83.0 | [+76.6, +88.7] | 0.00195 |
| routed-ce | +1.1 | [-1.6, +3.4] | 0.44043 |
| soft-ordinary | +87.2 | [+84.4, +89.7] | 0.00195 |

Stopping the natural conversion gradient alone did not prevent the outside task from training a competing predictor that carried over into natural input. This is controlled evidence that the natural confidence gradient is not necessary in this setting. Experiment 29's loss replay showed that natural confidence learning alone can be sufficient in its original implementation. The natural hinge substantially improves authority yet does not satisfy the full threshold in nine seeds, and scaling supplies both initial logit headroom and an attenuated gradient. Neither isolates confidence from all other side effects. All routed natural readout gradients are exactly zero by design. Shared flagged and routing objectives and actual AdamW changes are archived separately.

Soft training materially changes authority while still learning the flagged task. Eight seeds pass every registered practical criterion, against zero with matched ordinary supervision. The continuation window here ends at 300 updates and includes the protective objective throughout. This tests preservation while learning, without implying all seeds pass, that the objective can be removed safely, or that the outside-domain task requires composition. The frozen-scale and natural hinge arms each meet the joint rule in three seeds. Cross-objective comparisons should therefore report authority and useful learning separately.

*Figure 4. Effective margin, sampled natural gradient and visible authority across objective treatments. Curves average 12 seeds, with pointwise seed bootstrap bands. The gradient floor is a plotting convention for exactly zero.* The figure is in the compiled manuscript and reproduction archive.

The prespecified slope extrapolation from steps 20 and 50 catches 23 of 42 eventual below-.8 objective trajectories, with one false alarm among 30 nonfailures. The step-50 authority benchmark gives identical confusion counts. This forecasting rule does not add detectable operational value on these data. Neither rule identifies an exact future displacement update. The automaton panel has no below-.8 final cases and cannot establish forecast sensitivity.

### 10. A five-state independent computation

The programmed automaton applies four 5 by 5 transition matrices to a one-hot state across six input symbols, with a controlled state transplant after three tokens. The matrices implement noncommuting transitions and one non-injective transition. Integer computation exhaustively agrees with the parameter program for 20,480 start-state and word combinations. A distinct input-dependent suffix executes after the transplant. A competing transformer still sees the unaltered original word. The held allocation contains 384 final sequences, 246 with eligible donors, 2,632 eligible draws and 2,604 unique pairs. Distinct oracle targets exclude unchanged truth, donor truth and reversed suffix alternatives from the intended class.

| Transition treatment | Full ordinary | Authority | Isolated | Correct-base swaps |
| --- | ---: | ---: | ---: | ---: |
| frozen | 96.2% | 94.6% | 100.0% | 94.5% |
| trainable | 100.0% | 100.0% | 99.9% | 100.0% |
| bounded | 100.0% | 100.0% | 100.0% | 100.0% |
| isolated | 100.0% | 100.0% | 100.0% | 100.0% |
| soft | 100.0% | 100.0% | 100.0% | 100.0% |

Frozen-only authority is 94.6% and isolated accuracy stays exact. An unpatched model predicts the intended counterfactual on 2.9%, so binary collisions do not explain the effect. All four registered contrasts improve authority by 5.4 points, paired 95% interval [3.7, 7.2], Holm p=.00195. In this task, trainable transition weights can increase effective margin while retaining near-exact isolated accuracy. Some frozen-only runs still pass the practical threshold. Soft transfer matches bounded/isolated authority empirically under the tested soft objective, which has no architectural guarantee and no matched extra supervision comparator in this secondary task. These results generalize the dissociation to a distinct computation while showing its severity varies substantially by task.

*Figure 5. Final authority versus learned flagged-task accuracy (dots are seeds), visible five-state trajectories (pointwise seed intervals), and reconstructed frozen-converter margins from the prior initialization and post-training installation studies.* The figure is in the compiled manuscript and reproduction archive.

### Discussion

The controlled evidence supports a specific failure mode of additive architectures. A program may continue computing exactly as specified even as gradients build a second route that defeats the intended response to an intervention. An ordinary training objective treats any computation that increases the target logit as beneficial. For correct predictions it can still supply a nonzero gradient. A different loss updating shared output rows can independently create a competing pathway, which explains why removing only the natural conversion gradient is insufficient in the new setting. AdamW state can move a repaired readout with zero fresh gradient, yet the earlier reset controls show it is not required for fast renewed displacement. The intervention and replay results identify sufficient routes under controlled protocols, rather than universal causes of trained behavior.

The difference between integrity and authority is distinct from the difference between repairability and durability. An explicit margin bound protects the stipulated legal domain by proof, and the reported reference tests verify its implementation. The soft loss demonstrably helps without hard restriction and permits a useful alternate written-value task on the same readout. Its partial seed-level qualification, finite horizon, and lack of a protected-task composition test leave open broader claims about continued plasticity. Primary studies used seeds as the independent unit and finite task suites as evaluation domains. Generalization to unconstrained large models, arbitrary installation, or naturally emerging features was not tested.

The earlier Experiment 30 final domain used nine base prompts. Experiment 32 expands written coverage but groups aliases to avoid inflated novelty. The flagged task is deliberately simple. Parity's conditional check remains retrospective, durability reuses the earlier final data, and the post-training host reserves capacity by construction. These limitations distinguish controlled existence results from estimates of prevalence. The goal of this study is to measure failure and protection within the explicit architecture, not to prescribe a universal optimization rule.

### Reproducibility and audit record

Experiment 32's prospective protocol is pinned at [`7a7fc0ef`](https://github.com/Mnehmos/mnehmos.llm.playground/commit/7a7fc0ef17927663312971b6ea6636f3ae89c4e8). Retain 528 registered checkpoints, 42 formal qualification checkpoints, source hashes, optimizer and RNG states, per-step telemetry and objective-separated gradients. Every 100-step boundary includes an authored cohort review tied to individually authenticated measurement cards. Token traces include embeddings, attention, MLP, programmed states and logits. These are sampled measurements, not exhaustive mechanistic explanations. Early diagnostic snapshots in the new study record margin and causal metrics at 1, 5, 10, 20 and 50. Statistical scripts, exact per-seed scores and fit-free inequality checks accompany the manuscript. Prior registered trajectories and archived manuscripts are preserved.

### References

- Geiger et al. (2021). [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997).
- Geiger et al. (2022). [Inducing Causal Structure for Interpretable Neural Networks](https://proceedings.mlr.press/v162/geiger22a.html).
- Omlin and Giles (1996). [Stable encoding of large finite-state automata in recurrent neural networks](https://pubmed.ncbi.nlm.nih.gov/8624958/).
- Lindner et al. (2023). [Tracr](https://arxiv.org/abs/2301.05062).
- Laguna et al. (2024). [Beyond Concept Bottleneck Models](https://arxiv.org/abs/2401.13544).
- Marcou et al. (2026). [Residual Stochastic Concept Bottleneck Models](https://arxiv.org/abs/2610.23421).
- Li et al. (2026). [WeightCode](https://arxiv.org/abs/2611.25476).
- Geirhos et al. (2020). [Shortcut Learning](https://arxiv.org/abs/2004.07780).
- Kirkpatrick et al. (2017). [Overcoming Catastrophic Forgetting in Neural Networks](https://doi.org/10.1073/pnas.1611835114).
- Merrill et al. (2023). [A Tale of Two Circuits](https://arxiv.org/abs/2303.11873).
- Cloud et al. (2024). [Gradient Routing: Masking Gradients to Localize Computation in Neural Networks](https://arxiv.org/abs/2410.04332).
- Koh et al. (2020). [Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a.html).
- Loshchilov and Hutter (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
- Soudry et al. (2018). [The Implicit Bias of Gradient Descent on Separable Data](https://www.jmlr.org/papers/v19/18-188.html).

- Marconato et al. (2023). [Not All Neuro-Symbolic Concepts Are Created Equal](https://doi.org/10.52202/075280-3170).
