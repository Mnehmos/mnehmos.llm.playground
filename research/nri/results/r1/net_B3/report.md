# net_B3: black-box interpretation report

## What the network computes
Each input is 7 tokens: six slots plus the end token 14 at position 6. Exactly one slot in positions 1..4 holds an
**operator** token (11, 12 or 13). Every other slot holds a **number** 0..10. The output (classes 0..10) is

    y = (a OP b) mod 11,   a = t[k-1], b = t[k+1], k = position of the operator
    11 -> addition, 12 -> subtraction (a - b), 13 -> multiplication

The other three numbers are distractors and have no effect on the output.

## Evidence
- Distribution: the 20000 inputs have exactly 4 operator-position patterns (`nonnnn`, `nnonnn`, `nnnonn`, `nnnnon`), each about 25%.
- Purity search: for each (operator position, operator type) I checked every pair and every single number position.
  The pair (k-1, k+1) determines the output with purity 1.000 in all 12 cases. The next best pair reaches about 0.31.
- The recovered 11x11 tables are exactly +, - (non-commutative, a-b) and * mod 11.
- `M_main` matches the network on 20000/20000 provided inputs and on 5000/5000 fresh random inputs I generated.
  All predictions have probability about 1.0.
- If I replace 11 with 12, the network outputs (a-b) mod 11 on 100% of rows. So the edit target `edit_fn` is exact.

## nri.json
- Variables: V_op, V_a, V_b, V_oppos and V_result.
- Mechanisms: M_main reads op, a and b. M_result is the identity on V_result.
- Behavior claims: threshold 0.99. I am highly confident in these.
- linear_rep / necessity claims at `resid_pre_mlp` are **untested hypotheses**, because I had no activation access.
  - V_op linear_rep (rank 2, threshold 0.5): on a toy stand-in I trained, this interchange scored only 0.41.
  - Necessity for V_a and V_b (rank 10, threshold 0.5): the stand-in gave about 0.31. Treat these as low-to-moderate confidence.

## Edit (edit_program.py): "treat + as -"
The edit is computed in closed form: per-neuron least squares, with no gradient descent.
1. Sample inputs. Compute h = site(x), plus two counterfactual activations:
   - (i) x with 11 -> 12
   - (ii) x keeping '+' but with b -> (-b) mod 11, since a-b = a+(-b)
2. Fit a per-neuron linear change of W_in and b_in on the features [h; 1]:
   - In-scope pre-activations should match the counterfactual pre-activations.
   - Out-of-scope pre-activations should stay the same.
   - Where the target is negative, the fit uses a ReLU-aware squared hinge, solved by active-set least squares.
3. Several variants are scored on held-out samples through `head_from_resid`. The program returns the variant with the best success that keeps preservation at or above 0.97.

Limitation: a W_in-only edit can only apply a change that is linear in h. To switch the operator only on '+' inputs, the network needs a change that depends jointly on the operator and on a and b. That joint change is linearly achievable only if the operator is encoded roughly additively in h.
- On my toy stand-in (2 MultiheadAttention layers + MLP), the representation is strongly entangled. The mean shift explains only 5% of the counterfactual activation change. The edit reached only about 0.24 success at 0.965 preservation.
- How well the edit works on the real network is therefore uncertain. The claim predicate is success >= 0.5 and preserve >= 0.95.

## Confidence
- Function / behavior: very high (>99%).
- Representation claims and edit success: low to moderate.
