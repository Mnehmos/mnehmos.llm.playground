# net_J6 — black-box interpretation report

## What the network computes
Every input has 7 tokens: `t6 = 14` (end/query marker). Exactly one of `t1..t4` is an operator token
(11, 12 or 13). All the other tokens are digits 0..10. The output class (0..10) is

    a = digit immediately LEFT of the operator,  b = digit immediately RIGHT of it
    11 -> (a + b) mod 11      12 -> (a - b) mod 11      13 -> (a * b) mod 11

The other digits and the operator's position (1..4) are distractors and have no effect on the output.

## Evidence (all from `predict`)
- **Input structure:** each of the 20000 inputs in `inputs.pt` has exactly one operator in positions 1..4, spread
  evenly across them. `t0` and `t5` are always digits, and `t6` is always 14.
- **Which tokens determine the output:** the prediction is a deterministic function of
  (left neighbour, right neighbour, operator) on 100% of rows. Every pair of fixed positions plus the
  operator leaves 18–35% of rows unexplained (only ~35% when the pair happens to straddle the operator).
- **Operator tables:** the 11×11 table for each operator matches `a+b`, `a-b` and `a*b` mod 11 exactly
  (121/121 cells each).
- **Accuracy:** the `M_modarith` output_fn matches the network on 20000/20000 inputs. The smallest top-class
  probability is 0.943 and the mean is 0.9998.
- **Invariance tests:** I resampled all non-adjacent digits on 3000 rows and the prediction never changed. I moved
  the (a, op, b) triple to a random other position with fresh distractors on 3000 rows, and again the
  prediction never changed.
- **Edit target:** on the 6600 '+' rows, the network's own output after replacing 11 with 12 equals
  `(a-b) mod 11` 100% of the time, so `edit_fn` is exactly the counterfactual behaviour.

## Edit program (`edit_program.py`)
The goal is to make 11 behave like 12, which turns addition into subtraction. The program works in closed
form, with no gradient descent:
1. It samples inputs and makes counterfactual copies with 11 replaced by 12.
2. It reads `h` for both versions.
3. The target change to the MLP pre-activation is `W_in (h_cf - h)` on '+' rows and 0 on every other row.
4. It solves one weighted ridge least-squares problem for `[dW | db]` acting on `[h; 1]`.
5. It tries several SVD-rank truncations and scales of that solution. On a held-out sample it scores each
   one with `host.model.head_from_resid`, measuring success against the network's own counterfactual output
   and preservation on non-'+' rows, and keeps the best.

The edit only touches `W_in`. The residual copy of `h`, which still encodes '+', also reaches the head
directly, so success depends on the MLP dominating the readout. The scale search partly compensates for this.
I tested the program on stand-in transformers with the same shapes, never on the real network.
Results:
(i) A random, untrained stand-in: 90% success, 99.7% preservation, rank 1.
(ii) A trained stand-in (post-LayerNorm encoder layers with their own MLPs, 96.6% accurate): success only
about 35%. There, h_cf is a poorly linear function of h (R² 0.57).
(iii) A trained stand-in that matches the described architecture (2 attention-only layers, then a residual
MLP, 100% accurate): 66% success and 99.1% preservation (held-out), relative norm about 1.9.

On (iii) the ceiling is high: with W_in unchanged but the MLP fed the exact counterfactual pre-activations,
success reaches 98%. So the bottleneck is that an edit to W_in can only be an affine function of h, and the
operator interacts non-additively with the operand features. It is not the residual path.
For that reason the edit claim states success >= 0.6 and preserve >= 0.97. If the real network adds the
operator to the residual stream more cleanly, the rank-1 'op-swap' part of the solution would give a
much higher success.

## Representation claims (hypotheses, not verified by me)
I had black-box access only, so I could not test any `linear_rep` or `necessity` claim. They are stated as
plausible hypotheses: the operator is linearly coded at the final position (rank 2), and each operand
has a class-mean subspace (rank 10) at `resid_pre_mlp` that the MLP reads.
The operand claims carry modest thresholds, because a mod-p network may store a and b in entangled
Fourier features, and then swapping a's class-mean subspace might not move the output cleanly.

## Confidence
- Behavioural description: very high (exact on all 20000 inputs plus the invariance tests).
- Edit claim: moderate. The method worked on stand-ins, but the real network's geometry is unknown.
- linear_rep / necessity claims: low to moderate (unverifiable black-box).
