# Supervisor notes (LLM interpretation; proposals only)

These notes are **proposals and interpretations** written by the LLM supervisor from
`audit_cards.jsonl` and `summary.md`. They executed nothing. Every decision in this pilot was made
by the registered deterministic checks (V1–V5) in `run_correction.py`.

## Interpretations

1. **Three regimes, fixed by step 1,500.** All three networks localize mass causally (block
   interchange ≥ 0.9) by step 1,500, but organize it differently:
   - **seed 600:** canonical-like. The pooled editor works; route variance is unused.
   - **seed 601:** intermediate.
   - **seed 602:** route-dependent. Route variance is needed (ID 0.85 without it) and the pooled
     editor fails.

   No network changed regime between steps 1,500 and 6,000.
2. **The readout canonicalization is safe exactly when D1 says route variance is unused.** D1 is
   a cheap pre-check (one forward pass) that predicted every V3/V4 outcome in this pilot.
3. **Continued training erodes a readout correction rather than adopting it upstream** (seed 601).
   The upstream route share is unchanged (0.845 vs 0.840). The behavioural gain decays from +0.081
   to +0.055 while the weight difference persists (about 0.49). The rest of the network adapted
   around the change.

## Proposed candidates (NOT executed in this pilot; each needs its own registration)

- **A. Earlier, repeated intervention.** Steps 750, 1,000 and 1,250, inside the formation window,
  with the same validation rule plus a D1 gate.
- **B. Unit-aware readout correction for route-dependent networks.** A small added or folded
  table in the doc 17/19 style, instead of canonicalization.
- **C. Upstream correction.** Edit `attn2.o` / `attn1.o` so the block activations themselves move
  toward a target organization. Test whether training keeps it.
- **D. A sham branch** (equal-norm random block edit) for seed 601, to separate "targeted
  correction" from "any perturbation of `W_in` columns 0–15".
