# Opus participant instructions

You have a trained, small arithmetic network. One of two quantities, mass (M) or acceleration (A), received special training at a set of 16 coordinates in a 64-number internal vector. Find which quantity it was, predict what changing an internal pattern will do, then make two lasting edits to the model. A failed prediction is a valid result.

## Your workspace

Use this packet and an empty working directory with Python, CPU PyTorch and safetensors installed. Work offline. Use only the supplied calibration rows as examples. You may run them through the model, inspect the weights, take gradients, fit an edit, and swap patterns between calibration rows. Keep all experiments within this session. Tell the owner if you had access to another case, training logs, private files, prior experiment conversations, or extra examples. This is a white-box test with examples.

Unzip the packet and run `python participant.py` in that directory to check it. For experiments:

```python
from copy import deepcopy
from participant import load_case, save_edit

original, rows, info = load_case()
fitted = deepcopy(original)
# Modify fitted, keeping original available for comparison.
# Once the prediction is sealed, export each edit from its own fresh copy.
# save_edit("M.safetensors", original, fitted, info, "M")
```

`rows` contains `tokens`, `M`, `A`, `S` and `force`. All arithmetic is modulo 13. The answer is M × A × S. Call the model with zero type IDs. The internal vector in question is the return value of `resid_pre_mlp(tokens, zero_type_ids)`. Use `head_from_resid(h)` to turn a modified vector into answer scores. The predicted answer is the index of the largest score.

## First return a prediction

Send the owner one file, `claim.json`, before constructing the submitted edits. Include these four top-level fields:

| Field | Value |
| --- | --- |
| `checkpoint_sha256` | Copy the string from `info["checkpoint_sha256"]`. |
| `declared_guess` | Your best guess, `"M"` or `"A"`. |
| `M` | An object containing `coordinates` (16 distinct integers from 0 through 63) and `predicted_swap_intended_rate` (a number from 0 to 1). |
| `A` | The same two fields for acceleration. The two coordinate lists may overlap. |

Here is what the swap prediction means. Start with one question and a donor question. They use the same output unit and word order. Copy the donor's values into just your 16 named coordinates of the original question's vector. For M, predict how often the model will now answer as if only the donor's mass had been substituted. Predict the analogous rate for A. The examiner uses only pairs where keeping everything, changing M, changing A, and copying both all give different answers.

Wait for the owner to acknowledge receipt of the claim. You get a receipt, with no accuracy feedback. Use the same analysis session for your two edits.

## Then return two edits

Produce `M.safetensors` and `A.safetensors` with `save_edit`. Start each from a separate copy of the original checkpoint. For M, change the network so it gives the answer for the inverse of M while keeping A and S fixed. For A, do the reverse. The inverse is the multiplicative inverse modulo 13, and zero maps to zero. The scorer distinguishes the intended answer, the answer for editing the other quantity, the original answer, and any other answer.

The examiner loads your replacement tensors into the original architecture and runs ordinary inference. Only parameters already present in the model can change. A per-question lookup, added neurons, inference hook, or external helper cannot travel with a submission. Send a short note stating what information and tools you used and how long you worked. Report any exposure before the owner scores the case.
