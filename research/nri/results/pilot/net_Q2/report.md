# net_Q2: reverse-engineering report (blinded)

## What the network computes
**Modular arithmetic mod 11 with three operators.**

- Tokens 0–10 are digits. Tokens 11, 12 and 13 are operators: 11 = `+`, 12 = `-`, 13 = `*`. Token 14 is a constant query token at position 6.
- Every input has exactly one operator token, at position 1–4. Call the token just before it `a` and the token just after it `b`. The other tokens are random distractor digits.
- `ch2` labels the roles: 1 = a, 3 = op, 2 = b, 4 = query, 0 = filler. It gives no information beyond the token positions, so all bindings are written over `t` alone.
- **Output = (a op b) mod 11.** This matches the network's argmax on 20000/20000 inputs (100%), and holds at 100% for each operator separately.

## How I found it
1. I tabulated per-position token and `ch2` statistics. `ch2` always contains the consecutive pattern 1,3,2, and the token at the "3" slot is always in {11,12,13}, exactly once per row. There are 11 output classes and digits 0–10, which suggested mod-11 arithmetic.
2. I tested `a+b`, `a-b`, `a*b` and `b-a` (mod 11) for each operator. Each operator matched one hypothesis at 1.000 and the others at chance (about 0.09).
3. Representation at `resid_pre_mlp`:
   - The operator is linearly decodable at 100%, from a rank-2 subspace.
   - b is linearly decodable at 99% and a at 97% with full-rank or rank-10 probes.
   - The result itself is only about 70% decodable linearly. The MLP computes it.
   - h is a strongly non-additive function of (a, b): one-hot(a) + one-hot(b) explains R² ≈ 0.64, while the full (a, b) table explains 0.99.
4. Necessity depends heavily on how the subspace is chosen:
   - Discriminative subspaces (LDA or reduced-rank regression) can be mean-ablated with no loss in accuracy. The information is redundant and also carried nonlinearly.
   - High-variance class-mean subspaces do matter. Ablating the rank-10 class-mean span drops accuracy to about 0.45 for a or b, and to about 0.19 for the result.
   - I therefore only claim the robust statement: ablating the rank-2 operator subspace leaves accuracy at 0.95 or above. This held under all three subspace methods (0.96–1.0).

## Edit (+ → −) with W_in only
- The target is the counterfactual pre-activations `W h_cf + b`, where `h_cf` is `resid_pre_mlp` with token 11 replaced by 12. Non-scope rows keep their own pre-activations.
- A plain affine least-squares remap reaches only about 45–60% success. The reason is that h_+(a,b) → h_−(a,b) is not affine: the `+` representation holds features tailored to a+b.
- Other target choices did not help: op `+` with (a, −b), op `−` with (a+c, b+c), or picking a target per row by EM.
- What worked is a **ReLU-aware, per-unit weighted least-squares fit** (edit_program.py). Every step is a linear solve, with no gradient descent:
  - Where a unit should be active, its pre-activation must match the target. Where it should be inactive, it only has to stay at or below 0 (alternating projection with target = min(prediction, 0)).
  - Each row and unit pair is weighted by how much that unit's error moves the logit margin (rms of `U[y,j]-U[k,j]` with `U = W_U W_out`, cubed). Scope rows get weight 5.
  - Six boosting rounds double the weight of rows that are still misclassified, capped at 16x.
- Held-out results (fit on 14k rows, tested on the other 6k): **success ≈ 0.92–0.93, preservation ≈ 0.98–0.99**. The claim thresholds are set conservatively at 0.85 and 0.96.
- With exact counterfactual pre-activations the ceiling is 0.989, because the direct residual path still carries the original operator. The remaining gap comes from the limited expressiveness of a W_in-only edit.
- It takes about 90 s on 4 CPU cores.

## Confidence
- Task identification ((a op b) mod 11, op = +, −, ×): very high, since it reproduces the network exactly on all 20000 inputs.
- linear_rep claims: moderately high. The thresholds were chosen to hold under LDA, reduced-rank-regression and class-mean-span probes.
- Necessity claim: moderate, because it depends on what the executor's ablation metric means.
- Edit: success should be about 0.9 and preservation about 0.98, but these could vary with the sampler.
