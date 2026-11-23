# 26. Registration: LLM-in-the-loop supervised training with bespoke per-checkpoint edits (exploratory follow-up to Issue #22)

*Committed before the loop network is advanced past initialization. Owner-directed follow-up to
docs 24–25: instead of one fixed edit family, the research agent (an LLM) inspects the network at
every 100-step checkpoint and **designs bespoke read-out and edit methods**. It may edit **any
weights**, but every edit must pass a **fixed deterministic gate** (§26.4) before it is installed.
Exploratory: one network, no statistics.*

## 26.1 Question

Can an agent that watches a network form its concepts every 100 steps, and writes its own
measurement and editing methods as it goes, steer the network toward a mass representation that is
**more persistently editable**, without losing accuracy or held-out generalization, where the fixed
linear upstream edit of doc 24 never passed its checks (doc 25)?

## 26.2 Design (fixed)

| item | value |
|---|---|
| network | **one** new trajectory, `runs/zoo_llm_loop/`: seed 700, doc-14 `mass_only` recipe, same initialization and batch streams as the Issue #22 pilot (`run_correction.new_state` / `train`, unchanged) |
| matched control | the pilot's **control** trajectory (`runs/zoo_atlas/control`), already trained and sealed-evaluated. The loop network reproduces it bitwise until the first installed edit; `run_loop.py advance` checks the state hash at every checkpoint and refuses to diverge without an installed edit |
| cadence | checkpoints at 100, 200, …, 6,000. At each: train 100 steps → save model, full training state and activation record → print the observation packet (the doc-24 atlas card, the pilot control's values at the same step, and extra measures) → the agent studies it and may write an edit module → the gate decides → install or not → continue |
| what may change | any parameter, by an additive delta computed by the agent's module (closed form, least squares, constrained optimization on FIT, …). The optimizer state is never modified (stale moments, declared) |
| journal | `results/llm_loop/journal.md`: for every checkpoint, what the agent looked at, what it concluded, whether it proposed an edit and why. Every edit module is kept in `loop_edits/` and every gate verdict in `runs/zoo_llm_loop/gate_log.jsonl`, including rejections |

## 26.3 Data (frozen; the doc-24 partition)

The agent's modules and the gate see only **FIT, VAL, OBS and OOD_DEV** (`run_loop.ctx`). **FINAL
and OOD_FINAL** are touched once, at step 6,000, by `run_loop.py final`, which refuses to run twice.

## 26.4 The fixed gate (`run_loop.run_gate`)

Each edit module declares **one target** from a fixed menu. The edit must deliver at least that
gain on VAL:

| target | measure (VAL unless stated) | minimum gain |
|---|---|---|
| `edit_inv` | pooled inversion edit at the block, intended (fit FIT) | +0.01 |
| `interaction` | block mass × route interaction share decreases, and `edit_inv` does not fall | 0.02 |
| `unit_transfer` | mass unit-transfer probe ratio at the block (FIT → VAL) | +0.03 |
| `ood_dev` | OOD_DEV accuracy (held-out values 9, 10) | +0.01 |
| `iia` | block IIA(M) | +0.01 |

**Safety checks (all edits):**

| check | criterion |
|---|---|
| S1 | VAL accuracy ≥ before − 0.005 |
| S2 | ≥ 99% of previously correct VAL answers stay correct |
| S3 | OOD_DEV accuracy ≥ before − 0.01 |
| S4 | block IIA(M) ≥ before − 0.01 (once IIA ≥ 0.5) |
| S5 | route probe at the block and mass-unit probe at h each ≥ before − 0.02 (unit information kept) |
| S6 | pooled-edit wrong-variable rate ≤ before + 0.02 |
| S7 | finite; ≤ 50% relative change of any tensor; ≤ 10% of the whole-model norm per edit; ≤ 50% cumulative |
| S8 | the target gain exceeds the mean gain of 3 random deltas with the same per-tensor norms (seed 9,900) by at least half the target threshold |

The gate evaluates on a deep copy; a failing edit leaves model and optimizer untouched.

**Held out from every target:** the add-one edit and OOD_FINAL. They are the final tests of
generalization.

## 26.5 Final evaluation and outcome (vs the pilot control, sealed, step 6,000)

Same sealed measurements as doc 24 §24.8, on FINAL and OOD_FINAL. **Outcome:**

| outcome | criterion |
|---|---|
| **improved** | inversion edit Δ ≥ +0.05 **and** held-out add-one edit Δ ≥ +0.03, **and** ΔID ≥ −0.005, ΔOOD_FINAL ≥ −0.02, Δ block IIA ≥ −0.01 |
| **traded off** | editability improves as above, but a safety margin fails |
| **no effect** | inversion Δ < 0.05 and add-one Δ < 0.03 |
| **harmed** | anything else |
| **not tested** | no edit installed |

The doubling edit is dropped; it is not variable-discriminating (doc 25 §25.6).

## 26.6 Disclosures and non-claims

- **The agent knows this network's future.** The control's whole trajectory and its sealed final
  scores are already known (doc 25), and the loop network is bitwise that control until the first
  edit. Edits are designed with that knowledge; only FINAL and OOD_FINAL *items* stay unseen. This is
  a feasibility demonstration of agent-supervised training, not a blind test.
- **Gate thresholds are reused from doc 24**, and the target menu was written before advancing.
- **Exploratory:** one network, one seed; no population claims.
- **An agent may label a neuron or direction only provisionally.** A label is supported only by the
  causal tests the gate and the atlas run.
