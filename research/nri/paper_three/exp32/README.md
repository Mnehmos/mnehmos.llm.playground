# Experiment 32 and Paper Three

The final two experiments, qualification, archived margin replay, and revised manuscript are separated from the earlier studies. The prospective lock is at 7a7fc0ef17927663312971b6ea6636f3ae89c4e8.

Set PYTHONPATH to research/nri, research/nri/paper_three, and this directory. The code uses CPU PyTorch plus NumPy and Matplotlib for analysis. Extract the experiment checkpoint shards into a single run root. Each shard carries identical public, released and review metadata for its task. The statistical unit is the seed.

- Recover the earlier checkpoint archives before running `python margins.py INPUT_ARCHIVES MARGIN_OUTPUT`. For the earlier Experiment 29 panel, `python replay29_margins.py EXTRACTED_EXP29` writes CSV to stdout.
- Run `python verify.py REGISTERED_ROOT` to authenticate the 528 registered checkpoints and independently rescore representative held endpoints.
- Run `python analyze.py REGISTERED_ROOT` for all paired analyses and `python figures.py REGISTERED_ROOT MARGIN_OUTPUT/margins.csv FIGURE_OUTPUT` for the empirical plot panels.
- To recreate the actual training streams use the released public manifests and the registered code, starting at 0 then 100, 200 and 300. Run `experiment.py study obj REGISTERED_ROOT/obj STEP` or the equivalent auto command. The root lock binds all training sources. At each step `review.py study TASK RUN_DIR STEP` shows the measurements. The next interval requires a separately authored, hash-bound review recorded with `--record`. Do not use already released final scores to change treatment behavior.
- A fresh replication can use `broker.py NEW_ROOT` to create a new prospective partition. That is not the original sealed study. Its administrative process releases only train and visible indices until both terminal audits pass and a score request arrives. Preserve new locks and commitments before training.
- The reproduction archive includes manuscript.tex, its graphics and a compiled PDF. Build with XeLaTeX twice in a clean directory, or regenerate TeX using `render_manuscript.py MANUSCRIPT.md manuscript.tex`. Citations already link to primary sources in the standalone TeX. The complete BibTeX metadata accompanies it. Author fields await owner review.

Report and result tables are indexed by registered cohort. All paths and checksums for the prospective checkpoints are supplied in the evidence bundle. No earlier checkpoints were edited.
