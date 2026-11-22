# 13. Plan for the next registration (draft: NOT a registration)

*This lists the design changes required before the next pre-registered study. The registration
itself will be a separate, timestamped document committed before any run, as doc 11 was.*

## 13.1 Edits must identify which variable changed (reviewer's proposal)

The registered edit in doc 11 (M→2M) cannot distinguish an edit of M from an edit of A, or from
a direct rescaling of the output, because F = M·A·S gives 2F in every case. The next registration
therefore:

1. **Uses only variable-discriminating edits.** For each edit of variable V, there must be a
   *wrong-variable* counterpart W whose target differs from V's on a non-trivial set of inputs.
   Candidates on the dimensional task:
   - additive: M→M+1 vs A→A+1 (outputs differ by A·S vs M·S, whenever M ≠ A);
   - inversion: M→M⁻¹ vs A→A⁻¹ (outputs differ whenever A ≠ ±M).
2. **Scores on the discriminating set only**, where intended, wrong-variable and unchanged
   targets are pairwise distinct. It classifies every output as intended, wrong, unchanged or
   other.
3. **Primary edit metric = specificity** = P(intended) − P(wrong), with P(intended) also reported.
4. **Requires a declared/undeclared dissociation** wherever the design allows one: a
   partial-specification arm that declares M only must show M specificity > A specificity, paired
   by seed. That is the sharpest prediction of "declaring a variable makes *that* variable
   programmable".

## 13.2 Separate localization from edit expressibility

The post-hoc analysis (doc 12 §12.8) showed that an edit can fail because the *edit family*
cannot express the change in the network's code (affine class-mean moves on a multi-cluster code
for +1), even when the variable is perfectly localized. The next registration therefore:

1. **Pre-specifies a positive control:** the same edit applied in the *designed* block of a
   network whose interchange accuracy is ≥ 0.95. If this control fails, the result is recorded as
   "edit family inadequate", and the localization claim is evaluated separately.
2. **Validates every edit family on the synthetic disentangled and entangled hosts** before
   registration. This was done for +1: disentangled 1.00, entangled 0.00.
3. **Considers a non-affine edit family:** for example, a learned-free lookup edit that maps each
   class's empirical cluster centers (per unit route) to the target class's, so that a
   multi-cluster code can be edited.

## 13.3 Make interchange training reliable before testing it

Doc 12 found that half of the IIT seeds fall into a sign-ambiguity trap. Before registering:

1. Use **λ warm-up** (0 → λ over the first third of training) and **best-of-k restarts** selected
   on a held-out validation split, using task accuracy only, never the test metrics.
2. Choose the budget rule from pilots on **≥ 5 seeds**, requiring ≥ 80% to converge (ID accuracy
   ≥ 0.95, designed-block IIA ≥ 0.9). A single seed is not enough.
3. Optionally add a **within-class compactness term** on declared blocks, so that blocks encode
   the variable and not the route that produced it. Doc 12 §12.8 found 62% of the within-class
   variance was route information.

## 13.4 Candidate primary hypotheses (to be finalized at registration)

| | hypothesis | metric |
|---|---|---|
| H1 | Declared variables are specifically editable | M-edit specificity, `iit_full` vs `none` |
| H2 | Declared vs undeclared dissociation | within `iit_partial`: M specificity − A specificity (paired) |
| H3 | Generalization to held-out unit–value combinations | OOD accuracy, `iit_full` (with structure enforced at the value–unit binding site) vs `none` |

Holm correction applies across H1–H3. Use ≥ 10 seeds per arm, and report non-converged seeds in
the primary analysis.
