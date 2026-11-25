# 42 — A generalization qualification standard, and a proposed prospective experiment (NOT run)

Phase IV of the Birth of a Concept continuation. §42.1 is a standard for claiming that a small network has learned a
*reusable computation*. It is derived from what failed and what held in docs 37–41. §42.2 is a **proposal** for a
prospective, registered experiment. It requires a separate registration commit and **owner approval** before any
training runs. Nothing in §42.2 has been run.

## 42.1 Generalization qualification standard (GQS)

A claim that network N "learned a reusable computation for variable V" qualifies only if all of the following hold.

### G-behaviour

- On combinations **never trained** (held out at split time, fixed by hash), accuracy exceeds the best constant
  answer by ≥ 0.2 at **two consecutive** audits.
- The final claim uses a **reserved** set evaluated exactly once.
- Report per-network values and the fraction of networks qualifying; a condition mean alone is not enough.
  - Reason: the factorial's B wd 1.0 condition showed that heterogeneous held-out sets can hide combination-specific
    partial solutions (doc 38).

### G-causal

- At a site whose instrument passed planted-mechanism validation (doc 41: P1–P5, including the
  trained-only-reader case P3), an intervention whose **implied** combination was never trained must move the
  output to the predicted answer.
  - Factorial study: switch rate ≥ 0.5 in the *implied-held-out* stratum.
  - Role-swap study: interchange accuracy ≥ 0.5 with a never-trained source.
- The effect must be reported against:
  - the no-intervention baseline;
  - a dimension-matched random-subspace control (≤ 0.10);
  - a same-value donor control (the canonical code predicts no change).

### G-not-probe

- Probe decodability (linear or centroid) fitted on trained inputs is **not** evidence about never-trained inputs,
  in either direction.
- Basis (doc 41R): probes read 0.00 on OOD inputs for networks whose block causally carries those inputs at
  0.99–1.00.

### G-replay

- The network's training must be replay-qualified, EXACT (bitwise) or FUNCTIONAL (prediction-identical), with the
  label carried into every claim.
- Claims about *when* something happens require complete checkpoints at the stated resolution (here every 100
  steps).

### G-population

- "Reliably produces" means a stated fraction of **independent seeds** qualifies under G-behaviour and G-causal,
  with a seed-level confidence interval (Clopper–Pearson).
- A condition is called *reliable* only if the lower 95% bound is ≥ 0.8.
- Seeds are never excluded.

### G-dependency

- Longitudinal claims are made on per-network summaries, never on pooled transitions.

## 42.2 Proposed prospective experiment (requires registration and owner approval)

**Status: proposal only. Not registered, not run.** Its final form depends on the Phase III results now being
produced (doc 41R). It will be revised and committed as a registration before any training, after owner approval.

### Question

In the declared-variable task (doc 14), identical hyperparameters give perfect generalisers (307, 311) and complete
failures (304, 310) of the declared variable. Two questions:

- **(a)** Is the outcome fixed by the **initialisation**, by the **data order**, or by their interaction?
- **(b)** Can a training intervention suggested by the factorial (strong decoupled weight decay) make the good
  outcome **reliable** (GQS G-population)?

### Design A — initialisation × data-order crossing

- `mass_only` only, original hyperparameters.
- Initialisation seed i ∈ {304, 307, 310, 311}, base-data seed d ∈ {304, 307, 310, 311}, IIT-source seed tied to d.
- 16 networks; the 4 diagonal cells are the historical networks, already replay-qualified, so 12 new networks.
- **Prediction if initialisation determines the outcome:** row-wise generalisation (i ∈ {307, 311} generalise for
  every d).
- **If data order determines it:** column-wise.
- **Analysis:** a two-way table of declared-variable OOD accuracy and G-causal, with exact permutation tests over
  rows and columns.
- **Compute:** about 2.5 CPU-hours.

### Design B — reliability under strong decay

- `mass_only`, `acc_only` and `neither`.
- Weight decay ∈ {0.01 (historical), 0.1, 1.0}.
- 12 fresh seeds (400–411): 108 networks, of which 36 at wd 0.01 serve as a replication of the historical
  distribution.
- **Primary:** the fraction of seeds qualifying under the GQS for the declared variable at wd 1.0 vs 0.01 (exact
  test on paired seeds), and the lower Clopper–Pearson bound.
- **Compute:** about 20 CPU-hours.
- **Risk:** strong decay may prevent the IIT objective from fitting (convergence is reported as a secondary
  outcome; no seed is excluded).

### What would change the program

- If B shows wd 1.0 makes the GQS outcome reliable (lower bound ≥ 0.8) in this second task as well, the factorial's
  interaction generalises across tasks and architectures. That is a recommended default.
- If not, the factorial result is task-specific.
