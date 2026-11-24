# Experiment 31 follow-ups

[Durability](REPAIR_REGISTRATION.md) · [Trained host](HOST_REGISTRATION.md) · [Finite-state replication](PARITY_REGISTRATION.md)

Run with CPU PyTorch and with `research/nri`, `research/nri/paper_three`, and this directory on `PYTHONPATH`. Before starting, copy the locked `experiment31_lock.json` to `lock.json` in the common output root. Durability reads the verified Experiment 30 endpoint paths (see `OLD` in the runner) and never writes there.

For `repair`, `prehost`, `host`, and `parity`, invoke `python runner.py STUDY STEP OUTPUT_ROOT/STUDY`. Run from step 0 in increments of 100, through 200 for prehost and 600 otherwise. At each boundary inspect `python review.py STUDY STEP OUTPUT_ROOT/STUDY` and the per-network measurements, and author a review with `--record` and your assessment on stdin. The next interval checks the submitted hashes. Create the full prehost run including its terminal review before running host step 0. After the terminal review, use `final` in place of step for host/parity. Repair evaluations are reused from Experiment 30 and therefore exploratory. Checkpoints provide raw state and controlled probe layers and gradients in addition to each card's summary.
