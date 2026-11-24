# Experiment 30 · Preserving causal control in programmed neural networks

**Prospective draft.** The six-arm qualification uses public development examples and two fresh pilot seeds. No confirmatory seed has been run or sealed outcome scored. This draft specifies the intended study and the remaining conditions for a registration that can be locked.

## Question and outcomes

The registered comparison will test whether an exact installed computation retains control when other parts of the network continue learning. The independent unit is a matched seed, not an intervention pair. Report for each seed and time point:

- Program integrity, the accuracy of the installed encoder–decoder evaluated in isolation.
- Behavior, ordinary conversion accuracy of the full model.
- Causal authority, the fraction of discriminating canonical swaps whose full-model answer matches the specified counterfactual, preserving the base's written input and requested destination on the residual pathway.
- Causal displacement, a drop in causal authority while isolated ordinary and isolated swap accuracy remain at least 95%, followed by recovery under residual suppression. Interventional evidence supports displacement only where the competing route is localized by this lesion.
- Plasticity, learned routing on withheld written forms. The routing task returns one written operand selected by the parity of the destination unit. Report fresh force pairs separately as the composition endpoint. A routing ceiling limits sensitivity to small plasticity costs, so publish early learning trajectories as well as endpoints.

For base $b=(x,s,d)$ and donor $j=(x',s',d')$, canonical quantity is $q=x smod13$ and the target is $q_j d^{-1}mod13$. At the program interface replace only the encoder value. Enumerate pairs for which the intended answer, unpatched truth, raw-number alternative and donor answer are pairwise distinct. Wrong-variable outcomes apply to the two-operand force intervention, where both possible swaps and the corresponding original outcome must differ. Run identity, equal-quantity, unrelated-site, and randomized donor controls. Report both unconditional authority and authority on the subset with correct unpatched behavior. Do not let conditional accuracy change the registered denominator.

For onset use the first sampled boundary below 80% full authority with both isolated scores at least 95% that remains below 80% at the next sampled point. Report interval censoring between observations. If no run qualifies, report no observed sustained onset. Use 0, 1, 5, 10, 20, 50, and then fixed 100-step boundaries. Time-integrated authority is a trapezoidal average over those fixed times. Recovery is full authority reaching at least 95% and improving by at least 15 points relative to the untouched endpoint. Also report continuous outcomes so cases near thresholds stay visible.

## Architecture and arms

The modulo-13 converter is two trainable-shape tensors `encoder[3,13,13]` and `decoder[3,13,13]`. Both are initialized to unit-indexed permutation matrices. The explicit input route selects unit and written value from the task. In parallel the token embedding and two attention layers feed an MLP and shared `residual.W_U.weight[39,32]`. Conversion uses columns of the prediction corresponding to rows 0–12 of the weight matrix. Force uses rows 13–25 plus a learned MLP reading both canonical operands. Routing uses rows 26–38. The converter has gradients from conversion and from the canonical force head where unfrozen. Force and routing can also update the shared transformer, though their direct gradients to the conversion readout rows are zero. Preserve task-specific local gradients and distinguish them from AdamW updates.

| Arm | Maps | Conversion residual | Extra objective |
| --- | --- | --- | --- |
| A | Trainable | Unrestricted | No |
| B | Frozen | Unrestricted | No |
| C | Frozen | Bounded | No |
| D | Frozen | Unrestricted | Cross-input canonical-interchange supervision |
| E | Frozen | Disabled | No |
| F | Frozen | Unrestricted | Same-sized extra ordinary conversion batch |

Use matched model initialization and task streams. Preserve the same allocated tensor shapes. A has 1,014 extra trainable map parameters. E has zero conversion loss gradient to the residual. D and F have equal extra batch size and loss weight, with the same base and donor draws. D detaches neither competing path nor chosen target during the ordinary update. No post-hoc tuning of D from sealed results. In this additive implementation, causal supervision can succeed by reducing residual dominance. Success would establish that an unrestricted learned path can be made safer by training in this architecture. It would not, by itself, show that a useful alternative conversion computation survived.

The exact program has a winning logit of 1 and all other logits zero, including on any allowed one-hot canonical transplant. If each additive residual coordinate is bounded in absolute value by $\alpha$, the pairwise winning margin is at least $1-2\alpha$. C uses $0.4\tanh(z)$, so the lower bound is $0.2$ throughout the one-hot domain. This guarantee concerns predicted conversion classes, with no guarantee about trainability, calibration, other tasks, soft interventions, or training with unfrozen maps. E guarantees the same decisions through isolation.

## Confirmatory scale and inference

Candidate configuration uses 12 fresh matched seeds, 72 total runs with F, 600 updates, and AdamW LR .002. Candidate seed list is 830701, 830703, 830709, 830713, 830719, 830729, 830731, 830737, 830741, 830743, 830747, 830753. Preserve all valid runs, including failed learners. Only exclude an unrecoverable infrastructure/data/hash failure under a preannounced rule, identifying the seed and reporting a complete-pair analysis with sensitivity bounds. Immediately stop on a failed positive control or audit-gate validation.

Primary endpoint is final sealed, unconditional causal authority. Report paired absolute effect sizes, percentile 95% paired-bootstrap intervals using 10,000 seed resamples and a fixed analysis RNG seed, and two-sided exact paired sign-flip tests of the mean difference, with Holm correction across four contrasts. Those contrasts are E–B, C–B, D–F and B–A. The first two estimate gaps from guaranteed references. D–B is secondary. Quantify B's drop from step zero to test whether freezing alone is sufficient. Report magnitude and frequency of losing at least 20 points with integrity intact. For H4 distinguish statistical improvement over F from practically high preservation under the preregistered 95% authority threshold. Use the unconditioned denominator as primary and publish all per-seed differences. Support for plasticity requires at least 80% accuracy on the held routing task. Report the paired gap versus B and its interval, with a five-point tolerance as the candidate practical margin. No claim of noninferiority from a nonsignificant difference.

## Partition custody and lock gates

Before launch, commit exact training and visible development split definitions and input-matched sampling rules. A separate scorer must provision supervisor-hidden development and final example manifests, using secret random salts for splitting at the base-question level. Commit the manifests' digests before training. Never expose held base questions through ordinary or counterfactual training, traces, instrument construction or supervisor-visible audits. The deterministic task oracle can be known. Have the evaluator report the number of distinct eligible base prompts and distinct pairs per partition before training, then use the same training-pool donors across arms. Report dependence among multiple pairs from a base by retaining seed-level inference. The finite task warrants disclosing all exact split counts and any sparsity of crossed cases. Require a sufficient discriminating count in each class and confirm that C/E score exactly under all valid patch controls before proceeding.

The draft pilot uses a public holdout. Before locking this draft, finish the hidden-split runner and force controls, test full restoration and reviewer enforcement, freeze code and manifests, and obtain the stage budget decision. The next two tasks and post-installation follow-up are separately scoped. Do not represent the present pilot as having hidden evaluation or as confirmation.

## Mechanism, repairs and follow-up

Replay conversion loss on/off crossed with force loss on/off from identical program, RNG and AdamW moments. Always advance matching batch streams. In the no-new-signal condition, distinguish zero gradients with AdamW state active from skipping the optimizer step. For attribution of residual readout drift, separately zero moments and remove decoupled decay on checkpoint clones. Track margin and direction of displacement together. Population comparisons test preservation. Replay tests loss-specific mechanisms.

For displaced endpoints, apply converter restoration, residual ablation, gradual residual attenuation, a conversion-independent readout edit with matched norm, a budgeted edit with a preregistered relative parameter cap, and a compute-matched causal repair phase. Preserve original archives. Evaluate immediate repair and after 100 further ordinary updates, reporting all task outcomes and relative norms. Study subsets and compute budgets require a separate lock.

A post-installation test will first train matched hosts on the routing task with dormant converter capacity, install parameters after the host reaches the frozen qualification threshold, verify exact installed behavior plus host retention, and start matched continuation arms. This tests installation into a trained host architecture with reserved slots. It will not establish unprepared arbitrary-network installation. A second task family should be an explicitly compiled state transition (for example, a parity tracker with an intervenable intermediate state) to test more than a change of modulus. Either can be a marked manuscript limitation if its budget is unavailable.
