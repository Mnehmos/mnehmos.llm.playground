# Why two identical training recipes can generalize differently

This directory follows the modular role-swap study. The question is which parts of the randomized training history change how a network handles unfamiliar written values in familiar fictional units. The inputs are a model initialization, a stream of ordinary physics questions, and an independently drawn stream of questions that donate their mass representation. The task, architecture and training arithmetic are held fixed.

## Sources and scope

The historical training source and endpoints are pinned at `aaa67c6b2fc00adf1ce3c9919759f2554f22e499`. The existing `run_swap.py`, `run_dimensional.py`, `nri/model.py`, `nri/dimtask.py`, and `nri/train.py` have the same Git blobs on this branch. The reconstruction manifests and measurements used here are pinned at `672e94f42616741ae6a65af2648f05b83a5bd651` on PR 33. PR 32 was read at `d171f2f3bea315b15ee75a14833964cf5cb441e4`. These are read-only references. Fetch the PR 33 object locally to run archival comparisons.

Docs 11–13 motivate separate behavior and manipulation tests. Doubling either operand doubles the answer. Docs 14–15 address that ambiguity by declaring one variable, swapping the declaration, and evaluating edits whose possible answers differ. Docs 16–19 show how much depends on how an editor writes the replacement. The editor's reserved questions and the network's held-out inputs are different evaluation domains. Docs 35–40 concern a different network and conversion task. Meaningful intermediate targets combined with strong weight decay improved its reserved performance. Docs 41–44 show how a decoder fitted only to familiar inputs can miss a code that the network demonstrably uses on unfamiliar inputs. They also document reconstruction and limit interpretation of the historical decision window to the measured networks. The present design keeps the original role-swap weight decay fixed.

## Reproduction

Use Linux x86-64, one CPU thread per worker, PyTorch `2.15.0+cu132` for CPU execution, NumPy and safetensors. Python 3.13.5 is used for the independent trainer. Historical reconstruction recorded Python 3.13.18. Use a separate Python with Matplotlib for the figures if desired. The original five committed endpoint files are in `../runs/zoo_swap`. Commands assume the working directory is `research/nri/seed_cross`, and `ROOT` is an absolute directory outside the repository for the binary artifacts.

```sh
git fetch origin refs/pull/33/head:refs/remotes/origin/review-pr33
python checks.py
python observe.py /tmp/independent-archived-endpoints.json
python history.py show
python history.py bind --root "$ROOT" --reviews results/historical_reviews.jsonl
```

Run the binder in its own process. An archival scientific assessment is transferable only to exactly matching historical tensor bytes. The independently measured local card is recorded alongside it. Use `python train.py "$ROOT" 307 307 307 --wait` and likewise for 311, then 305, 304 and 310. The scorer records familiar accuracy, ordinary unfamiliar accuracy, block replacement with a genuinely unfamiliar donor, and a separately fitted change to the weights. Its independently programmed integer oracle covers all 9,126 written questions. The immutable checkpoints use PyTorch pickle for full optimizer and RNG restoration. Load these only from a verified trusted archive.

```sh
python qualify.py "$ROOT" /tmp/qualification.json 307 311 305 304 310
python checks.py --resume "$ROOT" --output /tmp/restart.json
python driver.py "$ROOT" /tmp/qualification.json --workers 6
```

Each new cohort pauses at the 100-step boundary. `cohort.py "$ROOT" show` prints the actual observed measurements. After a scientific review, `cohort.py "$ROOT" sign --step STEP --assessment TEXT --metric METRIC --range LOWER UPPER --prediction TEXT` binds the assessment and testable next prediction to each named card. Restrict receipts to individual models or subgroups with `--runs`. Model-specific structured observations and the outcome of the previous prediction are stored in each receipt. Detailed native-resolution measurements for steps 600–2,000 accompany the 100-step reviews.

```sh
python analyse.py "$ROOT" results
python site_controls.py "$ROOT" results/site_controls.json
```

A saved state contains all model parameters, AdamW moments, the learning-rate scheduler, the global CPU and Python random states, and the independent base and donor generator states. Its manifest records the file checksum and the checksum of the canonical safetensors serialization of the model alone. Run identifiers always list initialization, ordinary seed, then donor seed. Every checkpoint is immutable once committed. A restarted writer reuses its last complete manifest entry and trims only an uncommitted telemetry suffix.
