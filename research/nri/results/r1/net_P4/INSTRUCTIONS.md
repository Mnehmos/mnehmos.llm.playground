# Task: reverse-engineer an unknown neural network

You are given a small trained network and nothing about what it was trained to do.

Files here:
- `arch.py`      architecture; `from arch import load_net; net = load_net()`
- `weights.pt`   the trained weights
- `inputs.pt`    a dict with `tokens` [N, 7] and `ch2` [N, 7] (long tensors):
                 20000 inputs drawn from the network's training distribution. `ch2` is a second
                 input channel; its meaning is not given (it may be constant).
- `nri-0.1.schema.json` and `nri_example_shape.json`: the output format.

Rules:
1. Work only inside this directory and your scratch subdirectory `work/`. Do NOT read any other
   files on this machine: no repositories, no parent directories, no answer keys. This is a
   blinded evaluation, and reading outside this directory invalidates it.
2. You may write and run Python (torch is installed) to query the network, read its weights and
   activations, and run interventions.
3. Deliverables, written into this directory:
   a. `nri.json`: an NRI v0.1 document (see the schema and the example shape). It must contain:
      - `variables`: every intermediate quantity you believe the network computes, each with a
        `binding` (a Python lambda over the token list `t` (list of ints, length 7) that
        returns an int) and an `ftype`.
      - `mechanisms`: at least one with an `output_fn` (lambda over dict `v` of variable values ->
        predicted output class).
      - `claims`: `behavior`, `linear_rep` (site `resid_pre_mlp`, give a `rank`) and `necessity`
        claims you believe are true. NOTE: `linear_rep` is a CAUSAL claim, tested by interchange
        intervention: the executor swaps the rank-r class-mean subspace of the variable between two
        inputs and checks that the output follows output_fn with that variable taken from the
        source input. Linear decodability alone does not pass it. Each will be executed and verified or refuted automatically,
        so only claim what you are willing to have tested.
   b. `edit_program.py` with `def program(host, doc, **args)` returning `(dW_in, db_in)`, the
      additive changes to `W_in.weight` [256, 64] and `W_in.bias` [256], so that:
        **wherever token 11 occurs in the input, the network outputs what it would output if
        that token were 12 instead; all other inputs keep their outputs.**
      `host.sampler(n, generator)` returns `(tokens, ch2)` from the input distribution,
      `host.site("resid_pre_mlp", tokens, ch2)` returns the [n, 64] activations, and `host.model`
      is the network. Compute the edit with code, not by gradient descent. Also add a claim of kind
      `edit` to `nri.json` that references `edit_program.py` (entry `program`), with a `scope`
      lambda and an `edit_fn` lambda written over your variables.
   c. `report.md`: what you think the network does, how you found out, and your confidence.
4. Budget: try to finish within about 60 tool calls.
