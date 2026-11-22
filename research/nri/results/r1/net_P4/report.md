# net_P4: what it computes

## Function
Inputs are 7 tokens (`ch2` is always 0). Positions 0 and 5 are digits 0..10, position 6 is always 14
(query), and exactly one of positions 1..4 holds an operator token 11/12/13; the rest are digits.
The network outputs (11 classes)

    (a OP b) mod 11,   a = token before the operator, b = token after it,
    OP: 11 = +, 12 = -, 13 = *

This matches the network's argmax on **20000/20000** inputs, and it holds at every operator
position (1..4) for all three operators. The other digits are distractors.

## How I found it
1. Input statistics: there is exactly one operator per row, at position 1..4. The output is near
   uniform over 0..10.
2. I guessed from a few examples (8,+,1 -> 9; 7,*,8 -> 1; 9,-,7 -> 2) and then checked all
   20000 rows: 100% match.
3. Attention patterns at the last position: in layer 2, heads 2 and 3 split their attention about
   50/50 between positions p-1 and p+1 (a and b). Heads 0 and 1 attend to the operator position
   (about 0.66) plus its neighbours. So attention gathers (a, OP, b), and the MLP does the
   arithmetic.
4. Structure of the residual at `resid_pre_mlp`:
   - About 98% of its variance is explained by the full (a, b, OP) interaction. Only about 67% is
     explained by additive a + b + OP terms.
   - Feeding the MLP an additive reconstruction gives only 27-39% accuracy. Feeding it the class
     means of (a, b, OP) gives 100%. So attention already builds nonlinear operand interactions.
5. Causal tests, run my own way (interchange on the class-mean subspace, random pairs):
   - Swapping the a, b or OP subspace does **not** carry the variable over (11-42%), even between
     inputs with the same operator. The operand code depends on the operator, so I make no
     linear_rep claim for these.
   - Swapping the joint (a, b, OP) subspace works: 0.84 at rank 10, 0.995 at rank 14, 1.0 at
     rank 18. I claim it at rank 18.
   - Mean-ablation accuracy: (a, b, OP) rank 18 gives 0.094; result rank 10 gives 0.13; a rank 10
     gives 0.31; b rank 10 gives 0.33.

## Edit (token 11 -> 12, i.e. '+' becomes '-'): the edit largely fails
`edit_program.py` fits a closed-form weighted ridge least-squares change to `W_in`. It targets the
pre-activations of the counterfactual input with 11 replaced by 12.

With the default arguments (ridge = 1000): success about 0.11, preserve about 0.99, relative norm
1.2. A success of 0.11 is barely above the trivial rate: when b = 0, a+b already equals a-b, which
covers about 9% of in-scope inputs.

Why a plain W_in edit cannot do this well here:
- h('+') is almost symmetric in (a, b). Swapping a and b moves h by about 5, against a
  within-class spread of about 3. Only about 50% of a-b can be decoded linearly from it.
- Only about 20% of h('-') - h('+') is a constant vector; the rest depends on (a, b).
- 212 of the 256 neurons are active for all three operators, so there are no '+'-only neurons to
  rewire.

Other things I tried:
- Fitting only on '+' inputs, ignoring preservation: about 56-60% success at best, for several
  target families: (a,b,-), (a,-b,+) and (-b,a,+).
- Trading off with a higher weight on in-scope rows: 0.24 success at 0.82 preserve, up to 0.50
  success at 0.46 preserve.
- A ReLU-aware active-set least squares: no better.
- A linear logit correction gated to '+' inputs through the 24 dead neurons. '+' is linearly
  separable in h, and the upper bound for this approach is about 34%. It is still infeasible: the
  dead neurons' centred logit vectors lie in an open half-space, so they have no positive
  null-space combination.

The edit claim is therefore stated weakly (success >= 0.08, preserve >= 0.97).

## Confidence
- High (>0.95) that the network computes (a OP b) mod 11 as described.
- High that the joint (a, b, OP) code sits in an approximately 18-dimensional causal subspace,
  and for the necessity claims.
- Low for the edit: it does not achieve the requested behaviour.
