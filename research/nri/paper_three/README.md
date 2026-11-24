# Preserving causal control in programmed neural networks

- [Qualification results and Experiment 29 re-audit](QUALIFICATION.md)
- [Prospective registration draft](REGISTRATION_DRAFT.md)
- [Primary-source review](LITERATURE.md)

Run from `research/nri` with PyTorch and `PYTHONPATH=.`. Operator reviews are separate, human-readable JSON receipts with one assessment and a note per seed and arm. Checkpoints and receipts live outside Git.

```sh
python paper_three/audit29.py /path/to/extracted/experiment29 > audit29.json
python paper_three/replay29_losses.py /path/to/extracted/experiment29 > replay29.json
PYTHONPATH=. python paper_three/pilot.py 0 /path/to/empty/run_dir
# After inspecting the actual measurements, use the audit evidence to author the receipt.
PYTHONPATH=. python paper_three/review_pilot.py 0 /path/to/run_dir < audit_notes.txt
PYTHONPATH=. python paper_three/pilot.py 100 /path/to/run_dir
# Repeat through 600, including its terminal review.
PYTHONPATH=. python paper_three/check_pilot.py /path/to/run_dir
PYTHONPATH=. python paper_three/repair_pilot.py /path/to/run_dir > repairs.json
python paper_three/plot_pilot.py /path/to/run_dir authority.svg
```

Notes contain a cohort synthesis followed by one evidence line per run sorted by seed and arm. The final evaluator and custody protocol are pending. No files from earlier repository experiments were changed by this branch.
