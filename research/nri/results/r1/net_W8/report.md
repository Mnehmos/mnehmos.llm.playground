# net_W8: what it does

## What the network computes
Input: 7 tokens `[n0, x1, x2, x3, x4, n5, 14]`. Numbers are 0..10, token 14 is a fixed end/query token,
and `ch2` is always 0. Exactly one of positions 1-4 holds an **operator** token:
11 = addition, 12 = subtraction, 13 = multiplication, **all mod 11**. The other slots hold random numbers.

**Output (11 classes) = (token left of operator) OP (token right of operator) mod 11.**
The other numbers are distractors. This rule matches the network's argmax on **20000/20000** inputs
(100% in each of the 12 operator x position cells).

## How it is implemented (evidence)
- Attention patterns. In L1, the tokens near the operator read the operator. In L2, heads 0 and 1 make the
  last position attend about 0.7 to the operator's position. Heads 2 and 3 split their attention between
  the operator's two neighbours (about 0.45/0.55 to pos-1/pos+1), wherever the operator is.
- So at `resid_pre_mlp` the operands a and b are written into **nearly the same subspace**: the class-mean
  subspaces of a and b have principal cosines from 1.0 down to about 0.4. They are separated only by small
  weight asymmetries and content-dependent softmax effects. Linear probes from h reach op 100%, a 83%,
  b 88% and the result only 37%. The direct path W_U h alone gets 18%, so the MLP does the arithmetic.
  No MLP units are specialised to one operator: the same units serve +, - and *.
- Interchange interventions (swapping the rank-r class-mean subspace):
  - joint (op,a,b) code, rank 20: **1.00**. Rank 14: 0.97. So the whole task state is in about 20 dimensions.
  - op, rank 2: 0.70.
  - a and b, any rank up to 10, swapped separately: about 0.13. They cannot be swapped on their own
    because they share a subspace.
  - result alone, rank 10: 0.43. (op, result), rank 20: 0.99.
  - operator position: about 0.98. This holds only because the output does not depend on position.
- Mean-ablation (necessity):
  - (op,a,b) rank 20: accuracy drops to 0.10.
  - a, rank 10: 0.31. b, rank 10: 0.27.
  - op, rank 2: 0.62.
  - position: 1.0, so position is not needed.

## Edit (token 11 -> behave as 12, i.e. + becomes -)
`edit_program.py` works in closed form, with no gradient descent:
1. It samples inputs and makes copies with every 11 replaced by 12.
2. Target MLP pre-activations are `W_in h' + b` (from the 12-version) for inputs that contain 11, and the
   original pre-activations for all other inputs.
3. It solves a ridge-regularised least-squares problem (in-scope rows weighted 3x) for the affine change
   `[dW | db]` acting on `[h, 1]`.

On held-out data: success about **0.78**, preservation about **0.99**. dW has rank about 48, and
||dW||/||W_in|| is about 4. With more ridge, the norm falls to 0.3-1.8 and success to 0.72-0.76.
Success has a hard ceiling, which I checked:
- Even the best linear map from h(+) to h(-), fitted on in-scope rows only, gets just 0.78.
- The difference h(-) - h(+) is mostly a non-additive function of (a,b): 91% of its variance is explained
  by (a,b) jointly, but only 38% by a and b separately.
- An edit that only touches W_in is affine in h, so it cannot reproduce that difference exactly.
- I also tried ReLU-aware iterative reweighting and reassigning targets to other inputs with the same
  result. Neither helped.

I claim success >= 0.75 and preservation >= 0.98.

## Confidence
- Task rule: very high. It matches the network exactly on every input.
- Joint-subspace linear_rep and the necessity claims: high (my own executor reproduced them).
- op rank-2 linear_rep at >= 0.6: moderate (measured 0.70).
- Edit claim: moderate. It meets my own thresholds, but it falls short of the ideal of about 0.9 success.
