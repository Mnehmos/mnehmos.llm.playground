# Task: reverse-engineer an unknown neural network (BLACK-BOX access)

You are given query access to a small trained network and nothing about what it was trained to do.
You CANNOT see its weights or activations. You may only query it with inputs and observe outputs.

Files here:
- `query.py`     `from query import predict; predict(list_of_token_lists)` returns predicted classes
                 and output probabilities. Each input is a list of 7 ints (token ids 0..14).
                 Batches of thousands of rows are fine.
- `inputs.pt`    a dict with `tokens` [N, 7] (long tensor): 20000 inputs drawn from the
                 network's training distribution. (`ch2` is a second input channel; here it is
                 always 0 and you can ignore it.)
- `nri-0.1.schema.json` and `nri_example_shape.json`: the output format.
- Architecture (all you are told): a small transformer (2 attention layers, d_model = 64), then an
  MLP with residual connection. h = the residual stream at the final position before the MLP
  (64-dim, site "resid_pre_mlp"); the MLP first layer is W_in: [256, 64] weight and [256] bias.

Rules:
1. Work only inside this directory and its scratch subdirectory `work/`. Do NOT read any other files
   on this machine, and do NOT open or import what `query.py` calls (the hidden network): no
   repositories, no parent directories, no answer keys. This is a blinded evaluation, and reading
   outside this directory invalidates it.
2. You may write and run Python (torch is installed), but only `predict` touches the network.
3. Deliverables, written into this directory:
   a. `nri.json`: an NRI v0.1 document. It must contain `variables` (each with a `binding`, a Python
      lambda over the token list `t` returning an int, and an `ftype`), `mechanisms` (at least one
      with an `output_fn`: a lambda over dict `v` of variable values giving the predicted output
      class), and `claims`. Every claim is executed against the real network and verified or
      refuted automatically. You may include `linear_rep` / `necessity` claims about the site
      `resid_pre_mlp` as hypotheses. NOTE: `linear_rep` is a CAUSAL claim, tested by interchange
      intervention: the executor swaps the rank-r class-mean subspace of the variable between two
      inputs, and the output must follow output_fn with that variable taken from the source input.
   b. `edit_program.py` with `def program(host, doc, **args)` returning `(dW_in, db_in)`, the
      additive changes to `W_in.weight` [256, 64] and `W_in.bias` [256], so that:
        **wherever token 11 occurs in the input, the network outputs what it would output if
        that token were 12 instead; all other inputs keep their outputs.**
      You cannot run this against the real network. The evaluator will run it once, and at that
      time your program may use: `host.sampler(n, generator)` -> `(tokens, ch2)` from the input
      distribution; `host.site("resid_pre_mlp", tokens, ch2)` -> [n, 64] activations;
      `host.model.W_in.weight` / `.bias` and `host.model.head_from_resid(h, W_in, b_in)` -> logits.
      Compute the edit with code, not gradient descent. Add a claim of kind `edit` to `nri.json`
      referencing `edit_program.py` (entry `program`), with `scope` and `edit_fn` lambdas over your
      variables.
   c. `report.md`: what you think the network does, how you found out, and your confidence.
4. Budget: about 60 tool calls.
