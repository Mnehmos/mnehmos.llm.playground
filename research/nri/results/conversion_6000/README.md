# 6,000-update conversion trajectory

Read [the study report](../../docs/34_conversion_6000_trajectory.md) for the results and limitations. The CSV and plot cover all 61 checkpoints. Each audit directory contains the supervisor review, the lesson and source provenance, and the evidence manifest. The final checkpoint directory contains the review, measurements, state metadata and manifest. The full checkpoint download includes the model, optimizer, random state, interval log and signed token traces.

The complete trajectory archives are distributed separately because they contain over a gigabyte of uncompressed tensors and logs. The manifest files provide the original hashes for checking those archives.

The diagnostic scripts use the extracted experiment directory. Run them from the NRI root with that root on PYTHONPATH. For example, to repeat the final held-out breakdown, run `PYTHONPATH=. python results/conversion_6000/diagnostics/heldout_strata.py /path/to/conversion-run/step_006000`.
