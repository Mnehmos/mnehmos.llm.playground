# 61 — Track B (CPU-01A/B/C/F): training-speed qualification and numerical sensitivity of the role-swap trainer

**Status.** Committed with the owner's approval (25 Nov 2026). The work ran outside the Phase VIII registration, as an
owner-assigned engineering track. Historical documents are not altered; the implications are recorded in ledger item
71 and in the manuscript (§Reproducibility tiers).

**Artifacts.**
- Code: `trackb/tools/`. The tools take `--repo` pointing at a checkout of `af17da2`; this file's own branch also works,
  because the five source files are unchanged.
- Raw results: `trackb/results/` (JSON, logs, profiles, the sweep).
- The nine full-run `model.pt` files: the six compiled fast-path runs, the two uncompiled runs and the original-path
  replay.
- Every other checkpoint was not committed. They are listed in `trackb/results/SHA256SUMS_all_checkpoints.txt`.

**Reproduction.**

    cd trackb/tools
    python fast_swap.py --repo <checkout> --zoo OUT --group mass_only --seed 307              # original path
    python fast_swap.py --repo <checkout> --zoo OUT --group mass_only --seed 307 --fast       # compiled fast path
    python qualify.py BASE_TRACE CAND_TRACE...                                                # tiers from trainbench --trace
    python sweep_pop.py OUTDIR                                                                # CPU-01A sweep

## Reproducibility tiers used here

| tier | definition | status for the historical role-swap zoo |
|---|---|---|
| **exact historical reconstruction** | same code, same library, same machine type, same thread count gives bit-identical weights | **holds** (mass_only s307 6,000-step replay = historical `model.pt`) |
| **numerical stability** | rounding-level changes (library version, thread count, algebraically equal kernels) leave the scientific outcome unchanged | **fails** for mass_only s307/s311 OOD and for some ID convergence outcomes; holds for the neither arm in the seeds tested |
| **general (population) reproducibility** | arm-level outcome distributions are preserved under such changes across many seeds | **untested**: it requires a preregistered multi-seed run (see Not done / open) |


**Date:** 2026-11-25

**Source:** `af17da246aa47e81570ccfc74ff5bd2b75b8a090`. The sources were exported with `git archive`. All 5 blob SHAs match TRACK_B.md.

**Runtime:** Python 3.13.18, PyTorch 2.15.0 (MKL 2024.2, oneDNN 3.13.3, AVX-512), 4 CPUs (affinity 4), no cgroup CPU quota, 16 GiB RAM.

**Isolation:**
- No repository edits and no writes to the historical zoo. All outputs are in fresh directories.
- The historical models under `runs/zoo_swap` were only read.

## Headline

1. **The original path is exactly reproducible here.** Replaying `mass_only` s307 for 6,000 steps reproduces the historical `model.pt` **bit for bit**.
2. **4 single-threaded workers is the best population configuration.**
   - Population: 4×1 gives 54.1 aggregate updates/s, versus 33.1 for 1×4. That is 1.63× more models per hour.
   - Single-run latency: 4 threads cut ms/update from 71 to 30 (2.4×). **But multi-threading changes the numerics:** 1×4 is not bitwise equal to 1×1.
3. **Bit-exact optimisations give almost nothing on the training loop.**
   - foreach AdamW is bitwise identical but within noise: +2% at 1×1, +2% at 4×1.
   - The data path is under 3% of a step, so CPU-01D cannot move end-to-end time (this agrees with ChatGPT's fixed-batch result).
4. **The numeric-tier fast path gives 1.45× population throughput.** The fast path is layer-2 last query with split Q, manual softmax, `torch.compile` of loss forward+backward, and foreach AdamW.
   - 4×1: 78.6 vs 54.1 updates/s. That is 47 vs 32.5 estimated models/hour.
   - Uncompiled: 1.22×.
5. **Critical finding: the numeric tier is NOT scientifically equivalent for this task.** Rounding-level differences (gradients equal to about 1e-7 at the same weights) change per-seed outcomes in mass_only and acc_only:
   - the exceptional OOD of s307/s311 (0.998/0.999) disappears;
   - in one case the network never leaves chance.

   So numeric-tier speedups **cannot** be used for historical replication. They are admissible only for new, preregistered, population-level experiments.

## CPU-01B — profile (d = 64, batch 512, 1 thread, mass_only)

- **Step time:** about 71–81 ms per update.
  - Two encodes: about 36 ms.
  - Backward: about 39 ms.
  - Head/loss: 2 ms.
  - AdamW: 1.4 ms.
  - Data (sample + build + counterfactual): about 1.9 ms.
  - Periodic `D.acc` (every 250 steps): about 0.8 ms amortised.
- **Neither arm:** 38 ms/update.

Op level, the largest single cost is **`aten::_softmax` over rows of length 8**:
- 2.6 ms per call, 15% of self CPU;
- plus 6% in `_softmax_backward_data`.

PyTorch's CPU softmax vectorises along the reduced dimension, which is only 8 here. A manual `exp(x - max) / sum` is 5.6× faster forward.

The next largest costs:
- `bmm` on (2048 × 8 × 16) blocks: 345 µs per call. This is already the fastest option; einsum, broadcast and contiguous variants were all slower.
- `cat` from the qkv `unbind` backward.

## CPU-01C/F — candidates and qualification

The 30-update trace, seed 307, used the warmup schedule and the 6,000-step OneCycle horizon. Data digests and RNG were identical for every candidate.

| candidate | tier (mass_only) | tier (neither) | max param diff after 30 updates |
|---|---|---|---|
| foreach AdamW | **exact** | **exact** | 0 |
| fused AdamW | numeric | numeric | 2.0e-5 |
| batched base/source | numeric | exact (no source) | 2.2e-5 |
| last-query layer 2 | numeric | numeric | 2.4e-5 |
| SDPA | numeric | numeric | 2.2e-5 / 3.6e-5 |
| manual softmax | numeric | numeric | 1.8e-5 |
| manual + last (+ split Q, + batch) | numeric | numeric | 1.7e-5 to 1.9e-5 |
| compiled manual + split | numeric | — | same-weight gradients within 1e-7 through 1,250 steps |

The baseline itself differs by 2.5e-5 across PyTorch 2.8 and 2.15 (ChatGPT vs Claude runtime), so cross-runtime replication is also numeric-tier only.

**Compiled path determinism:** the compiled path is deterministic run to run, and also across a fresh Inductor cache.

**Fast path compile cost:**
- cold: 16–34 s on the first step;
- warm on-disk Inductor cache: 1.5 s (mass_only).

**Rejected candidates:**
- vmap batching of several models in one process: slower per model;
- skipping deterministic-mode NaN fill: no gain;
- `torch.compile` of the encoder only: no gain;
- SDPA: no better than manual softmax.

**Forward+backward ranking.** This was measured in-process with 7 interleaved repetitions under background load. Speed-up vs baseline:

| variant | mass_only | neither |
|---|---|---|
| last | 1.30 | 1.24 |
| manual+split | 1.51 | 1.25 |
| sdpa+split+batch | 1.40 | 1.38 |
| **manual+split, compiled** | **1.75** | — |

## CPU-01A — workers × threads (clean machine, 300-step mass_only prefixes, 3 interleaved trials)

| path | W×T | aggregate updates/s (range) | estimated models/h (6000 steps) | ms/update per run | peak RSS MiB |
|---|---|---|---|---|---|
| original | 1×1 | 14.0 (13.8–14.2) | 8.4 | 71.3 | 856 |
| original | 1×2 | 21.1 (20.7–21.5) | 12.7 | 47.3 | 857 |
| original | 1×4 | 33.1 (31.0–34.2) | 19.8 | 30.2 | 857 |
| original | 2×1 | 27.3 (26.7–27.3) | 16.4 | 73.3 | 856 |
| original | 2×2 | 40.3 (38.9–41.7) | 24.2 | 49.6 | 868 |
| original | **4×1** | **54.1 (53.5–54.9)** | **32.5** | 73.9 | 856 |
| exact (foreach) | 1×1 | 14.4 (14.4–14.7) | 8.6 | 69.6 | 862 |
| exact (foreach) | 4×1 | 55.1 (54.7–57.0) | 33.1 | 72.6 | 862 |
| numeric uncompiled | 1×1 | 17.2 (16.2–17.6) | 10.3 | 58.3 | 854 |
| numeric uncompiled | 4×1 | 65.9 (65.7–69.0) | 39.5 | 60.7 | 854 |
| numeric compiled | 1×1 | 20.4 (19.7–20.8) | 12.3 | 49.0 | 899 |
| numeric compiled | 4×1 | **78.6 (75.2–84.4)** | **47.1** | 51.0 | 900 |

- Steady-state rates exclude the first step (which includes compilation).
- Models/h is extrapolated from the prefix. It excludes the final ID/OOD evaluation and checkpoint writes.
- No failures in 36 configuration-trials.

## Full 6,000-step runs vs the historical zoo (warmup schedule, original eval cadence)

| run | historical ID / OOD | fast compiled ID / OOD | fast uncompiled ID / OOD |
|---|---|---|---|
| mass_only 307 | 1.000 / **0.998** | **0.246 / 0.145** (never leaves chance) | 1.000 / 0.444 |
| mass_only 311 | 1.000 / **0.999** | 1.000 / 0.278 | 1.000 / 0.792 |
| acc_only 307 | 1.000 / 0.775 | 1.000 / 0.729 | — |
| acc_only 311 | 1.000 / 0.253 | **0.546** / 0.263 | — |
| neither 307 | 1.000 / 0.942 | 1.000 / 0.933 | — |
| neither 311 | 1.000 / 0.942 | 1.000 / 0.923 | — |
| original-path replay, mass_only 307 | — | **bitwise identical to historical model.pt** | |

**Diagnosis of mass_only 307 compiled.** I compared compiled against eager gradients at the same weights inside the real training loop, at steps 1, 250, 500, 750, 1000 and 1250. The maximum difference was ≤ 8.7e-8, with identical gradient norms. So compilation computes the correct gradients, and the run diverges through accumulated rounding.

**Context from the historical zoo (12 seeds per arm):**
- mass_only: ID converged in 7/12. OOD was 0.26–0.71 except s307/s311, the only two near 1.0.
- acc_only: 9/12.
- neither: 11/12, OOD 0.48–0.97.

**Implications:**
- The s307/s311 mass_only OOD outcomes are reproducible bit for bit but are knife-edge. Under rounding-level perturbation they fall back into the arm's ordinary distribution.
- The neither arm is stable.
- **Implication for earlier analyses:** claims that depend on particular seeds (the critical-window mechanism, 307/311 vs 304/310) describe rare trajectories of a chaotic process, not robust seed properties. Arm-level conclusions should rest on population statistics. This is flagged for owner review; no historical document was changed.

## Recommendations

1. **Historical replays and acceptance:** original path, 4 workers × 1 thread.
   - foreach AdamW is safe (exact) but worth only about 2%.
   - Do not use multi-threading, which is not bitwise.
2. **New preregistered experiments:** the compiled fast path with 4×1 gives 1.45× more models per CPU-hour. Conditions:
   - fix the variant before outcomes;
   - compare arms by distributions over many seeds;
   - report the trainer variant as part of the condition;
   - warm the Inductor cache once per shape.
3. **Single-run latency** (when one result is needed fast): 1×4 threads cuts 71 → 30 ms/update. It is numeric-tier, so the same caveat applies.
4. **Analysis/DAS:** adopt ChatGPT's frozen activation cache (CPU-01E, exact, about 1.9× on the frozen analysis).

## Not done / open

- CPU-01G (width 128/256) was left to ChatGPT's ladder.
- Frozen-scorer (DAS/edit) comparison of the fast-path models was not run; ID/OOD already shows non-equivalence.
- A seed-population test of whether the fast path shifts arm-level distributions (rather than just reshuffling seeds) needs a preregistered run of ≥ 12 seeds × 3 arms. About 108 model-trainings at 4×1 would take roughly 2.3 h with the fast path versus 3.3 h with the original.
- Checkpoint resume is not qualified for `run_swap` (the historical script saves only at the end).

## Package contents (in this repository)

- `trackb/tools/`: `cand.py` (candidate encoders), `fast_swap.py` (trainer; the original loop is the default and the fast path is opt-in), `trainbench.py`, `qualify.py`, `rank.py`, `profile_swap.py`, `sweep_pop.py`.
- `trackb/results/`: raw JSON for every trial, profiles, traces, full-run configs and logs, and the sweep.
