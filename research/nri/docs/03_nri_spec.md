# 3. Neural Representation Interface (NRI), v0.1 specification

## 3.1 Design goal

An NRI document is a **typed, executable intermediate representation** of what a network computes.
It sits between raw weights and conventional programs:

```
weights ──inspect──▶ hypotheses ──encode──▶ NRI document ──execute──▶ verified / refuted claims
                                                │
                                                └──edit programs──▶ ΔW ──execute──▶ success / collateral
```

The literature review shows that natural-language explanations of networks, whether from LLM
agents (HyVE) or WSP authors (Sparks), are routinely **behaviorally right and causally wrong**
(EditLens; Same Fix, Different Weights; Hase et al. 2023). The NRI therefore has one non-negotiable
rule:

> **Every statement in an NRI is a predicate that an executor checks against the network. Status
> fields are written by the executor, never by the author.**

This makes an LLM interpreter's output *refutable* and lets interpretations be scored
automatically against ground truth. It is the missing piece behind the "validation bottleneck"
that HyVE reports.

## 3.2 Object model

| Object | Purpose | Executable part |
|---|---|---|
| `model` | identity: `weights_sha256`, sites with dimensions | hash check; site extractors |
| `variables` | the computational quantities the network tracks (operand, operator, role, …), each with an **ANLU-style functional type** (`NUMBER`, `OPERATION`, `ROLE`, …) | `binding`: code that computes the variable's value from the input |
| `mechanisms` | functions over variables that a component computes | `output_fn`: code giving the predicted output from variable values |
| `claims` | falsifiable statements linking variables, sites and mechanisms to the network | `kind` + `predicate`, run by the executor |
| `edits` (claims of kind `edit`) | a weight program, the intended scope and the intended new behavior | `program` (path + entry point), `scope`, `edit_fn` |

### Claim kinds in v0.1 (implemented in `nri/nri_exec.py`)

| kind | Test the executor runs | What a pass licenses |
|---|---|---|
| `behavior` | `model(x) == output_fn(vars(x))` on ≥ threshold of samples | "the network computes this function" (input-output only) |
| `linear_rep` | **interchange intervention**: replace the rank-r class-mean subspace of V at the site with that of a source input; the output must become `output_fn(vars with V from source)` (IIA ≥ threshold) | "V is causally represented in this subspace, and downstream computation reads it as V" |
| `necessity` | mean-ablate that subspace; behavior accuracy must drop to ≤ threshold | "the computation depends on this subspace" |
| `edit` | run the weight program, which returns ΔW computed by code from the host's own statistics; measure success on in-scope inputs (against `edit_fn`) and preservation out of scope; report rank and relative norm | the three separate successes in your brief: behavioral (success), causal (the program edits only the claimed subspace), and transfer (rerun on another host) |

**Planned for v0.2:** `path` (path patching between components), `invariance` (claims must survive
the function-preserving permutations and rescalings of *Orbit Audits*), `composition` (two edits
applied in either order), and `portability` (rerun on a symmetry-transformed or independently
trained host).

## 3.3 Where ANLU fits: types on variables, not just tokens

ANLU puts functional types on *input tokens*: `e_i = c_i + t_{τ_i} + p_i`. The NRI puts the same
vocabulary of functional types on **internal variables**. That gives the bridge a testable form:

- If an ANLU-typed network "carries its types inside", then for every input type τ there should be
  an NRI variable with `ftype = τ` whose `linear_rep` claim passes at small rank, **whether or not
  the interpreter was given the type labels**.
- The typed vocabulary then becomes a shared language between the training prior and the
  interpretation. A failure is just as informative: if role variables are *not* recoverable, the
  types were used as scaffolding and then discarded.

## 3.4 Worked example (an excerpt of `results/nri/typed_s0.nri.json`, produced by the prototype)

```json
{
  "nri_version": "0.1",
  "model": {"id": "typed_s0", "weights_sha256": "…", "sites": [{"id": "resid_pre_mlp", "dim": 64}]},
  "author": {"kind": "reference", "blinded": false},
  "variables": [
    {"id": "V_op", "ftype": "OPERATION", "binding": "lambda t: t[[i for i, x in enumerate(t) if 11 <= x <= 13][0]] - 11"},
    {"id": "V_a",  "ftype": "NUMBER",    "binding": "lambda t: t[[i for i, x in enumerate(t) if 11 <= x <= 13][0] - 1]"}
  ],
  "mechanisms": [{"id": "M_arith", "site_in": "resid_pre_mlp",
                  "output_fn": "lambda v: (v['V_a'] + v['V_b']) % 11 if v['V_op'] == 0 else …"}],
  "claims": [
    {"id": "C_behavior", "kind": "behavior", "mechanism": "M_arith",
     "predicate": [">=", 0.99], "status": "verified", "measured": 1.0},
    {"id": "C_rep_V_op", "kind": "linear_rep", "variable": "V_op", "rank": 2,
     "predicate": [">=", 0.9], "status": "refuted", "measured": 0.397},
    {"id": "C_nec_V_a", "kind": "necessity", "variable": "V_a", "rank": 10,
     "predicate": ["<=", 0.5], "status": "verified", "measured": 0.389},
    {"id": "C_edit_add_to_sub", "kind": "edit",
     "program": {"path": "programs/op_swap.py", "entry": "program", "args": {"var": "V_op", "x": 0, "y": 1}},
     "scope": "lambda v: v['V_op'] == 0",
     "predicate": {"success": [">=", 0.9], "preserve": [">=", 0.97]},
     "status": "refuted", "measured": {"success": 0.140, "preserve": 1.0, "rel_norm": 0.177, "rank": 1}}
  ]
}
```

These are the actual executor verdicts for this model. The network computes the task perfectly, the
operand subspace is *necessary*, but the operator is **not** causally carried by its class-mean
subspace, and the rank-one edit fails. This is the honest, machine-checked description the NRI
exists to produce. See `docs/06_prototype_results.md`.

## 3.5 Scoring an interpreter, human or LLM, with the NRI

Given a blinded interpreter's NRI document for a network with a known answer key:

| Score | Definition |
|---|---|
| **Verified-claim rate** | fraction of the interpreter's claims the executor verifies |
| **Variable recall** | fraction of ground-truth variables for which the interpreter wrote a binding that agrees with the true value on ≥ 99% of inputs (up to relabeling) |
| **Causal precision** | among the interpreter's `linear_rep` claims, the fraction at small rank (≤ number of classes − 1) that pass |
| **Edit score** | success − collateral on a held-out edit goal stated only in behavioral terms |
| **Cost** | tool calls, tokens, wall-clock time |

This removes the "LLM judge grades LLM explanation" circularity. Claims are scored by intervention,
and only *variable recall* needs the answer key.

## 3.6 Design decisions and their reasons

1. **Bindings are code over inputs.** Variables must be checkable without trusting the author's
   label. A variable named "carry bit" means nothing until its binding agrees with the network's
   causal structure.
2. **The executor finds subspaces itself.** Authors claim *that* V is represented at rank r. The
   executor derives the subspace from data. This stops authors from passing overfit, hand-picked
   directions; to use a specific direction, file it as an `edit` program.
3. **Edits are programs, not deltas.** WeightCode found that copying a numeric ΔW to another host
   gives −0.10 pt, while rerunning the program keeps 83–110% of the gain. The NRI stores the
   *procedure*, so an edit is portable by construction, and coordinate-level portability becomes a
   separate, measurable property.
4. **Executable strings are a security surface.** v0.1 evaluates lambdas in a restricted namespace
   and runs program files directly. That is acceptable for toy models in a sandbox. A production
   version needs WeightCode-style sandboxing (bubblewrap, no network) and hash-pinned programs.
5. **The schema is JSON** (`schema/nri-0.1.schema.json`), so any LLM can emit it and any tool can
   validate it.
