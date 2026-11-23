# 30 — Registration: LLM Toolsmith, adaptive semantic curriculum, post-training feature programming

Issue #25. Committed **before** the study pretrain (seed 2501) and before any comparative training. Everything
below is frozen by the commit that adds this file; deviations are logged in doc 31 as deviations.

## 30.1 Question

From one common pretrained student, does a supervisor (the LLM in this session) that inspects the student every
100 steps, builds validated instruments when existing ones cannot answer its question, and chooses the next lesson
from evidence (arm **B, adaptive**) reach a better semantic/compositional end state than a fixed, pre-registered
curriculum over the *same pools, steps, batch size, optimizer and schedule* (arm **A, fixed**)?

This is a **single matched pair** (n = 1 seed). It can falsify "adaptive selection is obviously better here"; it
cannot establish a general adaptive advantage. Any B−A difference is reported with example-level bootstrap CIs and
explicitly as a one-pair result.

## 30.2 Student, language and truth contract (code is authoritative)

* Toy semantics (`sem/lang.py`): fictional units, F = M·A·S **mod 13**. This is a modular-arithmetic toy and is
  **not** real-world unit arithmetic. Tasks: force, convert, equiv (with hard negatives), inverse, sense
  (polysemy of *force/mass/work*: physical vs other sense; other-sense sentences also act as distractors).
* Canonical ASTs are rendered to surface text by phrasing families; an **independent validator** (`L.validate`,
  own lexicons, no access to the generator's choices) must recover the AST and the answer from the tokens alone,
  otherwise the example is quarantined. Validator negative controls are tested (`tests/test_sem.py`).
* Student (`sem/model.py`): 3 attention layers + readout MLP, d = 64, 130,255 parameters (129,280 without the
  training-only role head). Whole-word lexicon tokenizer (136 tokens). **No BPE-vs-ANLU comparison is made.**
  The ANLU-style role signal is an auxiliary training target only (doc 08 "aux_only"); **no gold types at
  inference**.
* Architecture map: `results/sem/student_architecture_map.json`, version `b98012f25582578b`, generated from the
  source and a runtime trace; `toolsmith.archmap.revalidate()` must pass before tools are run.

## 30.3 Splits (frozen: `runs/zoo_sem/eval/manifest.json`, sha256 `1ec746277e9b3d5bcf911ef4352d9b79520ca08a69b23062bf69f39480a70fc9`)

Five independent axes: phrasing family (pretrain / train / dev / sealed), sense context (train / dev / sealed),
tuple hash (train 60% / dev 15% / sealed 25%), numeric hold-out (unit 2 with values 9–10 = dev, 11–12 = sealed,
excluded from training both as inputs and as answers), and **crossed** sets (new family × new context × held-out
value). 16 dev sets are on disk; 14 sealed sets are **never written to disk**: they are regenerated in memory only
by `run_sem.py final`, checked against the frozen content hashes, evaluated once, and the lock
(`sem.data.SEALED_LOCK`) blocks every sealed request elsewhere. OOV rates of new-family sets are recorded in the
manifest and will be reported (they are high for some sets: e.g. dev equiv_newfam 0.40).

## 30.4 Toolsmith rules (supervisor-authored instruments)

* A tool = `toolsmith/instruments/<id>/{tool.py, manifest.json, test_tool.py}`. The manifest must state purpose,
  hypothesis, input/output schema, read scope, measurement definitions, kind, author and the checkpoint at which it
  was created.
* **Static check** before execution: import allowlist (math/json/collections/itertools/statistics/random/
  functools/dataclasses/torch/numpy), no exec/eval/compile/open/getattr/__import__, no private/dunder attributes,
  no attribute or string mentioning the sealed area/lock.
* **Sandbox**: separate `python -I` process, minimal environment (no secrets, proxies blanked), CPU and address-
  space rlimits, own session, temporary working directory; audit hook blocks sockets, subprocess/exec/spawn/fork,
  ctypes, `pickle.find_class`, deletions/renames, writes outside the output directory (and `/tmp`), reads outside
  the allowlisted roots and any read of the sealed area. Inside the child the sealed lock is on and sealed
  families, contexts and evaluation specs do not exist. Tools receive **copies** of `model.safetensors` only; the
  runner hashes the source checkpoints before and after (**nonmutation**).
* **Validation gate**: a tool must define `controls(ctx)` — synthetic or known-answer cases with planted ground
  truth — and all must pass in the sandbox before `run` is allowed for that exact source hash. Every validation
  and run is appended to `results/sem/tool_registry.jsonl` with source hash, git commit, args, checkpoint hashes,
  runtime and evidence file. Runs on checkpoints older than the tool are flagged `retrospective`.
* Tools may not request oracle answers for sealed items, and may not read sealed data indirectly.
* Tool CPU time is recorded separately from training compute and never counted as student training.

## 30.5 Common pretraining and stabilisation trigger

Seed 2501. Force task on the three pretrain families only (`PRETRAIN_LESSON`), batch 128, AdamW lr 2e-3 (200-step
warm-up, then constant), IIT on h[0:16] (λ 3, warm-up over 1,000 steps), aux role 0.5. Observation every 100 steps.
Stop at the first checkpoint ≥ 1,000 where, over the last 3 checkpoints: `dev/force_id_pretrainfam` ≥ 0.95 with
range ≤ 0.02, `dev/force_ood_value` slope ≤ 0.01/100 steps, CKA(h, previous) ≥ 0.98; cap 5,000 steps. The trigger
is evaluated only on dev sets. The saved state at the trigger is the **common checkpoint**; it is kept untouched as
the frozen non-training reference and the rollback point.

## 30.6 Continuations (matched)

* Both arms load the common checkpoint, then: fresh AdamW, lr cosine 1e-3 → 1e-4 with 100-step warm-up, 1,500
  steps, batch 128, identical data RNG seed (2502). `cont-init` asserts identical state hashes for A and B.
* **A (fixed)**: `FIXED_CURRICULUM` in `run_sem.py` — 15 blocks of 100 steps, three stages (force/convert/equiv/
  sense → + inverse → uniform, increasing distractor and hard-negative rates), replay 0.3, style 0.3/0.3, pair
  rate 0.1, objectives aux 0.5 / IIT 3 / consistency 0.2.
* **B (adaptive)**: before every 100-step block the supervisor files a review with all of: what changed, what the
  student knows, what fails, explanation, competing explanation, falsifiable prediction for the next block,
  tool needed, next action, why this checkpoint, outcome of the previous prediction. `advance` refuses to run
  without the review and a lesson. Lessons are validated by `D.validate_lesson`: training families/contexts only,
  replay ∈ [0.2, 0.8], objectives within aux [0,1], IIT [0,3], consistency [0,1], pair rate ≤ 0.5, optional unit
  focus. **B may not author new phrasing families in the confirmatory comparison** (that would be a data-diversity
  advantage, not a selection advantage).
* Matched: steps (1,500), examples (128/step), optimizer, schedule, replay floor. Tokens differ with sentence
  lengths and are reported per arm. Supervisor reading time and tool CPU time are recorded in a separate ledger.
* B's supervisor sees dev sets and all A/B checkpoints. Selecting lessons on dev feedback is the intended
  adaptive mechanism; dev is therefore **not** an unbiased estimate for B. Only the sealed sets are.

## 30.7 Final evaluation (once) and endpoints

`run_sem.py final` refuses to run twice (writes `results/sem/final_sealed.json`). Networks: common reference,
A_fixed, B_adaptive.

* **Primary (confirmatory, one pair)**: mean sealed accuracy over the eight *new-skill/generalisation* sets
  `convert_newfam, convert_ood_value, equiv_trainfam, equiv_newfam, inverse_newfam, inverse_ood_value,
  sense_newframe_newctx, sense_trainframe_newctx`; B − A with a paired example-level bootstrap 95% CI (2,000
  resamples, seed 0). "B better" is claimed only if the CI excludes 0 **and** the retention guard holds for B.
* **Retention guard**: `sealed/force_retention_pretrainfam` must not fall more than 0.05 below the common reference.
* **Secondary (each reported, not combined)**: force newfam / distractor / ood_value / crossed (compositional
  generalisation); paraphrase invariance on sealed families (all-families-agree rate, within/between cosine); sense
  discrimination; block IIA on sealed families; editability — inversion and add-one reported **separately**;
  binding probe ID vs OOD; compute ledger.
* Exploratory: everything from instruments built during the run, lineage observations, review predictions'
  hit rate. Reported in a separate section.

## 30.8 Post-training feature programming (exploratory, after the sealed evaluation)

On the final A/B checkpoints (copies), add a **new capability** not trained before — a new operation over existing
concepts (e.g. "difference": (canonical mass − canonical acceleration) mod 13, introduced by a new question frame)
— by three bounded methods: (i) a new output head on frozen trunk, (ii) a low-rank adapter / added readout neurons
with the original weights frozen, (iii) a bounded modification of existing weights (norm-limited update of a
declared parameter subset). Then add a **second** capability and test coexistence. For every method: new-skill
accuracy (on fresh in-memory held-out sets generated from non-sealed tuple and numeric splits; the sealed data is
not reused), preservation of all previous dev tasks (drop ≤ 0.02 target), interference between the two new
capabilities, parameter count added, and an exact rollback check (restored hash = pre-edit hash). Every persistent
change is made on a copy with a state backup, validated deterministically, and rolled back if the preservation
check fails. Results are exploratory.

## 30.9 Stop rules and budgets

* Pilot stops early only on a code defect (logged; restart from the common checkpoint for **both** arms).
* Up to 6 new instruments during the run; each must pass its controls before its outputs inform a lesson.
* No edits to weights during the A/B comparison (lessons only); weight-level programming happens only in §30.8.
* Previous registered experiments (docs 01–29, their runs and results) are not modified.

## 30.10 Pre-registration status of the smoke run

A non-study smoke (seed 999, scratch directory, not committed) exercised freeze/pretrain/trigger and the first
instrument (T001). Its observations informed the trigger's choice of `dev/force_id_pretrainfam` and the T001 design;
they are labelled non-study in the registry and evidence cards.
