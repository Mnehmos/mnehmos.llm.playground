# 27. LLM-in-the-loop precursor (docs/26): findings from steps 0–2,000 (exploratory, stopped)

*Registration: doc 26. Journal: `results/llm_loop/journal.md`. Gate log:
`runs/zoo_llm_loop/gate_log.jsonl`. The network is resumable at step 2,000.*

**The run was stopped at step 2,000 by the research team's decision.** The owner then issued
spec v1.0 ("Adaptive Concept Lineage and Causal Supervision"). It supersedes doc 26 in two ways:
- **Edit execution.** Edits must be declarative compositions of safe primitives. Doc 26 executed
  agent-written Python modules.
- **Instrumentation.** The spec adds gradient lineage and persistent concept records.

Doc 26's registered outcome (step 6,000) therefore does not exist; this document reports what the
first 2,000 steps showed. Those findings shaped the spec's frozen rules (doc 28).

## 27.1 What happened

| step | event |
|---|---|
| 0–1,000 | bitwise identical to the pilot control (seed 700). Four gate submissions (700; 1,000 ×2; none at 800 / 900): all rejected, by S2 (≤ 99% of correct answers kept) and/or the target rule |
| 1,100 | **installed**: interaction-penalty edit (`organize_edit`), unit-transfer +0.034, interaction 0.53 → 0.43, 1.35% of model norm |
| 1,192 | **training diverged** (gradient norm 5 → 151; accuracy 0.99 → 0.21 by 1,200) |
| 1,300 | full recovery (accuracy 0.999). **The control's organization re-formed** (interaction 0.61 vs 0.60; unit 1 split again) |
| 1,500 | **installed**: editability edit (`rewire_edit` with off-manifold KL and a frozen route probe), editability 0.073 → 0.114, OOD_DEV unchanged, route information 0.74 → 0.95 |
| 1,800 / 1,900 | **installed**: same method, editability 0.100 → 0.167 → 0.200, OOD_DEV −0.002 / −0.001, keep 1.000 |
| 1,960–1,990 | **training diverged again** (accuracy 0.74 at 2,000) |

## 27.2 Findings

1. **Mechanism (bespoke head inspection).**
   - Layer-2 heads 0 and 2 copy mass from the value token into the block.
   - Head 2 starts first, with a shared code; head 0 is recruited from step 900 and is **unit-split
     from the start**.
   - The split already exists in the layer-1 value-token residual. Canonical mass is linearly
     decodable there (κ 0.96), so a unit-invariant writer is linearly possible.
2. **Upstream reorganization alone breaks behaviour; joint reorganization is cheap.**
   - Re-fitting the heads' value weights to a unit-invariant code cut the interaction 0.47 → 0.20–0.30,
     but accuracy dropped by up to 0.53: the readout reads unit 1's pattern.
   - Optimizing a **differentiable interaction penalty** jointly over attention and MLP weights,
     with KL to the original predictions, cut the interaction by 0.10–0.58 at 1.4–5% of model norm.
     The closed-form `attn2.o` edit of doc 24 needed 4–13× the rows' norm.
3. **The editability / generalization / unit-information frontier.** At step 1,500 (OBS
   pre-checks):

   | variant | editability | OOD_DEV | route info |
   |---|---|---|---|
   | training for translation-editability | +0.41 | **−0.13** | — |
   | + off-manifold KL | +0.23 | −0.05 | **−0.12** |
   | + off-manifold KL + a frozen route probe, 0.4× | **+0.07** | **0.00** | **+0.19** |

   Without the guards, editability is bought with generalization and with unit information; with
   them, a small gain is free.
4. **Timing.** The interaction can be removed almost for free around steps 1,000–1,100. Later (1,200–1,500
   on the control's snapshots) the same removal costs 0.04–0.12 OOD_DEV.
5. **Edits during training can destabilize the optimizer, and static checks cannot see it.**
   - Both edit episodes were followed, about 60–90 steps later, by an Adam loss spike that escalated
     into divergence.
   - The control has spikes in the same windows (gradient norm ≈ 11 at 1,165–1,183) but recovers.
   - Static Hessian sharpness did not predict the collapses. At 1,100 both networks were above
     Adam's edge (preconditioned 1,361 vs 2/lr = 1,022), and the edit *lowered* it. At 1,800 / 1,900
     the edited network was far below the edge (4.9–126).
   - **This is why doc 28 adds a deterministic 100-step lookahead** on a copy of the full training
     state, which would have caught the 1,192 divergence.
6. **The control's organization is an attractor.** After collapse and recovery, the network rebuilt
   the same unit-split organization.
7. **Method hygiene.** Tuning against the gate's VAL split on the control's future snapshots was
   disclosed. Afterwards, tuning used OBS only, with ≤ 2 gate submissions per checkpoint. OBS and VAL
   disagreed by ~1.5 points on keep-correct at step 1,000.

## 27.3 Not claimed

No final-network claim (the run stopped at step 2,000). One network, one seed, whose future (the
pilot control) the agent knew.
