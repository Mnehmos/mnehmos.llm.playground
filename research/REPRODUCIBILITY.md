# Reproduction and contribution guide

The aim is to make each scientific claim executable, scoped and refutable.

## Original NRI code

Use an isolated environment and a fresh checkout. The historical [NRI README](nri/README.md) contains the exact experiment-specific entry points and its security guidance.

```sh
python -m venv .venv
# Activate the environment for your platform.
python -m pip install torch matplotlib pytest
cd research/nri
python -m pytest -q tests/
```

These commands come from the existing NRI documentation. The documentation migration did not rerun the NRI suite. Record your Python, PyTorch, BLAS, CPU, thread settings and exact Git revision. Do not assume that using a newer dependency produces identical training. Some historical jobs are lengthy; inspect the protocol before running them.

The [CPU-01 report](https://github.com/Mnehmos/mnehmos.llm.playground/blob/5958e2d9e22b234f94353e386bf1c4d0a734efc6/research/nri/docs/61_track_b_cpu01.md) explains why an exact historical replay is different from numerical stability and replication across seeds.

## H01 inspection demo

```sh
python -m pip install numpy torch
python research/hybrid_math/h01_smoke.py
```

The small inspection demo was locally exercised using Python 3.13.5, NumPy 2.3.5 and PyTorch 2.10.0+cpu, one Torch thread. It verifies the weight hashes, all digit additions, and an example of router-guided exact fitting. The full H01 arithmetic library and original training/exam data are separate archives. See [its scope](hybrid_math/README.md).

## Evidence requirements for new work

1. Register the hypothesis, controls, seeds, partitions, budget and acceptance rules before evaluation.
2. Record exact parent checkpoint identities; preserve model, optimizer, RNG and scheduler where applicable. Never reconstruct a missing historical checkpoint and call it identical.
3. Separate training/development, calibration and final qualification. Once examined, a holdout becomes diagnostic data for subsequent tuned experiments.
4. Preserve every scheduled seed, failed run, interruption and correction. Continue the established 100-update inspection cadence when relevant.
5. Report neural capacity, supplied mathematical priors, explicit program/library storage, observations, search expansions, wall time and total training/inference compute separately.
6. Compare against matched ordinary optimization, fixed heuristics and strong structure-aware classical methods. Distinguish a hindsight oracle from a deployable policy.
7. Distinguish behavioral correctness, causal control, recovery, durability and unseen generalization. Exact reconstruction and compilation require their own correspondence tests.
8. For Lean, require real pinned Lean elaboration and kernel verification, plus a dependency audit. Do not treat a simulated verifier or a generated source file as a proved theorem.

Treat generated NRI documents and executable programs as untrusted. The [NRI executor guidance](nri/README.md#security-nri-documents-are-untrusted-input) distinguishes restricted expressions from explicitly authorized Python edit programs. Run archived training and experimental code in a fresh isolated copy, not in the preserved evidence directory.

## Suggested submission layout

New track/session directories should include a README, registration, source, result tables, environment manifest, checkpoint manifest and provenance. Keep large artifacts out of ordinary Git where practical, but preserve stable access and checksums. Link the new result in the evidence ledger and mark its availability accurately.

If you find a counterexample or reproduction discrepancy, report the experiment ID, exact command, source revision, environment, artifact hash, expected output and observed output. Negative results are valuable.
