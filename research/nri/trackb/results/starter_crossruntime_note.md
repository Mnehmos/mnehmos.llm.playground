# Track B: cross-runtime starter replication (Claude Code)

- **Commit:** af17da246aa47e81570ccfc74ff5bd2b75b8a090. The sources were exported with `git archive` to a scratch directory. No repository or zoo writes were made.
- **Source check:** all 5 blob SHAs match TRACK_B.md.
- **Runtime:** Python 3.13.18, PyTorch 2.15.0, 4 CPUs (affinity 4), no cgroup CPU quota, no effective memory limit.
- **Contention:** 3 other single-threaded training workers were running. These runs used nice 10 and 1 thread.

## Command

Starter `bench_swap.py --trace --steps 30 --warmup 5 --schedule warmup --seed 307 --evaluate`, for variants `baseline` and `last`, with group mass_only.

## Results

| Check | Result |
|---|---|
| Batch digest and base/source/global RNG hashes: baseline vs last here | match |
| Batch digest and RNG hashes: Claude runtime vs ChatGPT runtime (both variants) | match |
| Loss max difference, last vs baseline (this runtime) | 4.77e-7 |
| Parameter max difference, last vs baseline (this runtime) | 2.36e-5 |
| Parameter max difference, **baseline vs baseline**, Claude vs ChatGPT runtime | 2.53e-5 (not bitwise) |
| Parameter max difference, last vs last, cross-runtime | 2.22e-5 |
| Loss max difference, cross-runtime (each variant) | 4.77e-7 |
| Frozen ID / OOD accuracy at step 30, baseline and last | 0.17249 / 0.07788 (equal) |

## Reading

The data stream is reproduced bitwise across runtimes.

The baseline itself is **not** bitwise across PyTorch 2.8 and 2.15. Its parameter drift after 30 updates is the same order as the last-query deviation (about 2e-5). So the last-query candidate sits inside the numeric tier, at the same size as ordinary cross-runtime rounding. It cannot be exact-tier.

This does not qualify a full 6,000-step scientific equivalence. Timings from these runs are trace-mode under contention, and are deliberately not reported as performance.
