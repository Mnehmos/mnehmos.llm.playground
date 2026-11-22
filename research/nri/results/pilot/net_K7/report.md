# net_K7: what the network computes

*(Written by the blinded interpreter agent. Its file tool refused to write this file, so the
orchestrator saved the text from its final message verbatim.)*

## Summary
Single-operator modular arithmetic mod 11.
- Tokens 0-10 are digits, 11/12/13 are the operators + / - / *, and 14 is an end marker at position 6. ch2 is always 0.
- Each input has exactly one operator, at position 1-4.
- Output = (a OP b) mod 11, where a and b are the digits immediately left and right of the operator. Other digits are distractors.
- Matches the network's argmax on 20000/20000 inputs.

## How I found it
1. **Token statistics by position.** Operators occur exactly once per row, only at positions 1-4.
2. **Hand checks, then a full test.** Examples: 10*6=60≡5, 8+1=9, 7-8≡10. The rule then scored 100% on all inputs.
3. **Probes at resid_pre_mlp** (class-mean subspace of rank r, held-out rows):
   - Operator: rank 2 gives 0.985-0.99.
   - Operands over all ops: about 0.57 at rank 10. Within plus inputs alone: 0.95-0.97 with a full one-hot readout.
   - Result: about 0.40. Operator position: about 0.46.
   - Means over (op,a,b) cells explain 98% of the variance in h; an additive model explains only about 65%.
   - A 2-D Fourier analysis over (a,b) shows large pure-a and pure-b terms, plus op-specific interaction terms: (k,k), i.e. a+b features, for plus and (k,-k), i.e. a-b features, for minus.
4. **Necessity.**
   - Removing the result's rank-10 class-mean subspace leaves accuracy at 0.17 (resample 0.11, zero 0.15). A random rank-10 subspace leaves 0.96.
   - Removing the a or b subspaces leaves about 0.41. Removing the operator's rank-2 subspace leaves about 0.85, because the operator is also carried by the operand features.
   - Caveat: a subspace built from least-squares probe weights has no effect (1.0).
5. **MLP.**
   - Neurons are mostly shared between plus and minus; about 50 neurons favour multiply.
   - The direct path alone gets 0.33.
   - Feeding the MLP the counterfactual h (token 11 replaced by 12) gives the minus answer 97.7% of the time.

## Edit (11 behaves as 12, W_in only)
**Method:**
1. Target preactivations: W·h_cf + b for inputs containing 11; unchanged for all other inputs.
2. Per neuron, solve ridge least squares, refitting a few times. Rows where the target neuron is inactive only need to stay below -margin.

**Result:** success about 0.37, preservation about 0.97 (minus 0.94, multiply 0.997).

**Why it is limited:**
- The plus activations lie in the span of the minus and multiply activations, so there is no plus-only direction.
- a-b can't be read out linearly within plus inputs (0.49, against 0.86 for a+b).
- With no preservation constraint at all, the best is about 0.68 success.
- Other settings only trade one for the other: 0.44/0.92 and 0.57/0.80.

Claim thresholds: success >= 0.3, preserve >= 0.95.

## Confidence
- Task identification: very high.
- Operator linear code: high.
- Necessity of the result subspace: moderate (depends on how the subspace is chosen).
- Edit: partial, as stated.
