# 45 — Registration: high-resolution exact replay of the critical window (steps 600–2,000)

Written and committed **before** this replay runs. It is part of the authorised reconstruction work (doc 41 §41.6),
in response to the owner's request to examine the critical window at finer resolution. It is **not** a new
training intervention:
- no setting, seed, data stream or update rule changes;
- the replay resumes from the verified, bitwise-exact step-600 checkpoints of the reconstruction.

**Relation to PR #34.** PR #34 is ChatGPT Work's independent crossing of initialisation, ordinary data and
interchange data. It also saves 10-step snapshots in this window. This replay does **not** cross any stream. It
re-observes only the five historical trajectories, with additional gradient decomposition. Its tensors are needed
here because PR #34's binary artefacts are not available in this session. Neither study modifies the other.

## Networks

`mass_only`, seeds 307 and 311 (generalise), 304 and 310 (fail), and 305 (partial).

## Procedure

1. **Resume** from `/home/user/swap_recon/zoo/mass_only_s<seed>/ckpt/step_00600`. That checkpoint holds the model,
   AdamW state, OneCycle state, both generator states, and torch and Python RNG. Run the original arithmetic
   (`run_swap.train_one`, aaa67c6) to step 2,000.
2. **Save every 10 steps:** `model.safetensors` and `resume.pt`. That is 141 states per network, 705 in total.
3. **Record per step (read-only):**
   - the gradient of the **ordinary task loss** and of the **interchange (IIT) loss** separately, per parameter
     tensor: norms, and the cosine between them;
   - the combined-gradient norm and the actual update norm per tensor;
   - the IIT weight λ(step);
   - both loss values.
4. **How the separate gradients are obtained.** They come from `torch.autograd.grad` on the same graph, with the
   graph retained, before the ordinary combined backward pass. They are never written into `.grad`.
5. **Qualification (pass/fail, fixed now).** Each network's step-1,000 and step-2,000 states must be bitwise equal
   to the 100-step reconstruction's checkpoints. If either is not, the replay is labelled FAILED, its measurements
   are not used, and the cause is reported.

## Analyses (exploratory; computed offline on the saved states)

For each 10-step state, at the declared-block site *h* (last token, before the MLP) and at the layer-1 value and
unit tokens:

- **Behaviour:** ID and OOD accuracy, and declared-variable OOD accuracy.
- **Causal carriage of unseen inputs:** OOD-source block interchange (validated in doc 41R).
- **Unit invariance:** for pairs that write the same canonical mass with different (unit, value) renderings
  ("aliases"):
  - the distance between their block activations;
  - whether their predictions agree.
- **Representation geometry:** the route-specificity index, and how the OOD-unit renderings sit relative to the
  in-distribution renderings of the same canonical value.
- **Attention:** layer-1 and layer-2 patterns. In particular, how much the value token attends to its own unit
  token, which is the "binding" route.
- **Weights:** per-tensor change since step 600, and the largest coordinate changes.
- **Gradients:**
  - which loss (ordinary vs IIT) dominates each tensor's gradient;
  - when the two gradients conflict (cosine < 0) or cooperate;
  - how the OOD-unit embedding rows are updated (they receive gradient only from trained (unit, value)
    combinations).

**Comparisons.** The earliest 10-step state at which each measure separates the generalisers {307, 311} from the
failures {304, 310} is reported for:
- three consecutive states with a group-mean difference ≥ 0.20, for bounded measures;
- the same with a pooled-SD-scaled difference ≥ 2, for unbounded measures.

Seed 305 is reported individually.

**Interpretive limit.** Two networks per group. This is a description of these trajectories, not a population
estimate and not a causal demonstration. Causal tests are written as proposals (doc 46).

## Resources

- **Compute:** 5 × 1,400 steps, about 25 CPU-minutes with gradient decomposition.
- **Disk:** about 705 × 1.1 MB ≈ 0.8 GB, at `/home/user/swap_window`.
- **Preservation:** archived and manifested as in doc 41 §41.6.
