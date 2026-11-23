# Blind checkpoint handoff

The participant receives one ordinary trained network and a calibration set. The custodian keeps the training condition, seed, internal coordinate map, withheld questions, and scoring workspace. This pilot asks whether an editor can identify a deliberately trained variable, predict what a swap will do, and write a lasting change to the original weights.

## The prepared pilot

Give an isolated Opus session only [participant.zip](pilot/participant.zip). Its instructions are inside. Save the [public receipt](pilot/receipt.json) before collecting responses. It commits to both participant packets and both salted answer keys. The custodian holds a private archive with `receipt.json`, `first/` and `second/`. Each directory contains a participant packet and a private key. Compare its receipt with the public copy before grading.

Use an empty filesystem or container with Python, CPU PyTorch and safetensors already installed. Give the editor no linked repository, credentials, shared volumes, examiner access, network, saved memory, or conversation history. Start a separate fresh session and filesystem for each case. Use the same model version, prompt, tools, work budget, and feedback policy. Record those settings and any unintended exposure. An earlier Opus session that trained the model can write the public training recipe. The custodian runs that recipe privately for a new draw. Starting a new chat attached to the old checkout would still expose that checkout.

| When | What crosses the boundary |
| --- | --- |
| Start | One participant ZIP goes to one fresh session. |
| Before building the final edit | Opus returns a prediction. The custodian seals it and returns only the receipt hash. |
| After the prediction | Opus returns two safetensors patches and an access note. The custodian seals the files and records the note separately. |
| Second session | The custodian gives only `second/participant.zip` to a fresh, isolated session. Neither session sees the other's work. |
| After both submissions | The custodian runs the scorer on the locked files and compares the reports. |

The public task and the model are inspectable. The finite task could be enumerated by a participant who ignores the example limit. An offline workspace hides the actual split and answer key. Enforcing a hard example budget would require a query-controlled interface instead of releasing the weights. Treat a reported violation as a protocol deviation.

## Custodian commands

Run from `research/nri/` with Python 3.12, CPU PyTorch 2.5.1, NumPy, and safetensors (tested with 0.8.0). Extract the private archive in an owner-only directory outside any checkout mounted by a participant. In these commands, `$vault` is that directory. The current pilot is already issued. For a *new* pilot, use `python -m blind.examiner issue --vault /new/private/path`. This trains once with the doc 14 warm-up recipe, keeping the random seed, arm and logs private. Choose study size and stopping rules before switching from a pilot to a confirmatory study.

```sh
python -m blind.examiner lock --case "$vault/first" --claim /returns/first/claim.json
python -m blind.examiner submit --case "$vault/first" --mass /returns/first/M.safetensors --acceleration /returns/first/A.safetensors
# Repeat the two commands for second, using its own session's returns.
python -m blind.examiner grade --vault "$vault"
```

`lock` and `submit` return a SHA-256 receipt. Identical retries are accepted. Different submissions cannot replace a receipt. The grading command waits for both sealed returns and stores the aggregate scores privately in `report.json`. Record the receipt and time outside the working folder when they arrive. The hash binds the file contents. The person holding the keys remains responsible for access, order and timing. The scorer only reads replacement tensors and reconstructs the model in trusted code.

## What the comparison measures

- The custodian trains one fresh network once, keeping the first random draw including weak learners. Two independent reorderings of its 64 internal coordinates preserve predictions. The issuer checks all 9,126 distinct inputs, including held-out unit combinations, for identical top answers and numerically close scores. The twins are two views of one network and count as one training run.
- All unit spellings and word orders of a given canonical (M, A) pair stay together. From the 169 pairs, 84 are public calibration, 42 are privately scored, and 43 remain unused. Fitting and grading use only in-distribution unit/value combinations. These questions are held out from the editor's calibration. The base trainer samples from the original training distribution.
- The examiner scores the declared-role guess, the claimed coordinates' causal swap rate, the gap between the predicted and actual rate, and overlap with the designed coordinates for the declared role. It maps both sessions' coordinate guesses back into the same layout and reports how many agree. Other coordinates can also be causal. The known training block provides a comparison. We also report a random site and a Python site finder using the same calibration information as the participant.
- Permanent edits are scored on the same eligible rows for both variables, using the doc 14 inversion rule. The report includes intended, wrong-variable, unchanged and other answers, specificity (intended minus wrong), and agreement with the original model where the operation should do nothing. It includes the original model, a Python affine edit, and that affine edit given the true site of the declared role. A strong edited-output score alone cannot establish how the editor achieved it.
- This is a white-box pilot with examples. A population claim needs new independently trained networks. A claim about reading raw weights needs an additional weight-only condition with controlled access.
