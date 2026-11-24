# 31 — Results: LLM Toolsmith, adaptive semantic curriculum, post-training feature programming

Issue #25; registration: doc 30 (commit 921a8d6, before any study training). One matched pair, seed 2501.
Confirmatory = §31.4 (sealed primary endpoint). Everything else is exploratory and labelled so.

## 31.1 Verdict

* **No adaptive advantage.** On the registered primary endpoint (8 sealed new-skill/generalisation sets) the
  LLM-directed curriculum B scored **0.242** vs the fixed curriculum A **0.265**: B − A = **−0.023**, paired
  example-bootstrap 95% CI **[−0.035, −0.011]**; retention guard held for both. By the registered rule A is better.
  Caveat (§31.5): that CI resamples examples of fixed models and ignores training-state noise; within the study,
  single dev sets of this kind moved by ~0.1 per block with no lesson change, so the CI is anti-conservative for a
  one-pair comparison. The defensible reading is **"B did not beat A; if anything A is slightly better."**
* **The Toolsmith worked as an instrument factory.** Six supervisor-authored instruments were built, validated on
  planted-truth controls in a sandbox and replayed on checkpoint copies (37 executions, nonmutation verified in all
  37). They produced the most solid findings of the study (§31.6), including two reversals of my own conclusions.
* **Adaptive selection on dev feedback overfit dev.** B's lessons were chosen with the dev sets in view; B led A on
  both dev sense sets at the end (+0.04, +0.01) but trailed on both sealed sense sets (−0.04, −0.07).
* **The student mostly did not learn the new tasks in either arm.** After 1,500 continuation steps, convert and
  inverse are at chance even in-distribution; equiv and sense are prior-driven. Lessons moved output priors, not
  skills (tested directly at step 700, §31.3).
* **Post-training feature programming**: §31.7.

## 31.2 What was built (all in `research/nri/`)

| Component | Files | Tested by |
|---|---|---|
| Semantic toy language: 5 tasks, 33 phrasing families, 18 polysemy contexts, canonical ASTs, renderer, **independent validator** (truth contract, quarantine) | `sem/lang.py` | `tests/test_sem.py` (negative controls: corrupted AST, unbound number, cue-less sense) |
| Five split axes + crossed sets; dev on disk, sealed **never on disk** (in-memory, hash-frozen), `SEALED_LOCK` | `sem/data.py`, `runs/zoo_sem/eval/manifest.json` | sealed-lock tests on every entry point |
| Student (130,255 params), trainer, safe persistence (safetensors + JSON, `weights_only=True`), matched A/B driver with mandatory reviews and bounded lessons | `sem/model.py`, `sem/train.py`, `run_sem.py` | bit-exact save/restore, identical A/B starts, rejected lesson leaves state untouched, no pickle loads |
| Architecture map from source + runtime trace, versioned, `revalidate()` | `toolsmith/archmap.py` → `results/sem/student_architecture_map.json` | drift test |
| Toolsmith: static check, sandbox child (`python -I`, rlimits, minimal env, audit hook), controls gate, provenance registry, evidence cards | `toolsmith/runner.py`, `toolsmith/sandbox_child.py`, `results/sem/tool_registry.jsonl`, `results/sem/evidence/` | 9 static-rejection tests, 8 audit-hook blocks each asserted to fail **by policy** (socket, subprocess, write outside, read outside, pickle, sealed request, sealed family, sealed spec), nonmutation/provenance test |
| Instruments T001–T006 | `toolsmith/instruments/*` | each: planted-truth controls in the sandbox + unit tests |
| Post-training feature programming (3 methods, rollback) | `sem/program.py` | rollback hash checks in every run |

Test suite: 49 Issue-#25 tests; whole repository suite 214 passed at registration.

## 31.3 Continuation: reviews, predictions, outcomes (exploratory)

Every 100 steps the supervisor filed a review (`runs/zoo_sem/reviews/rNNNN.json`) with a falsifiable prediction
before the lesson ran. Scorecard (15 reviews): **2 confirmed, 5 partly confirmed, 8 falsified**.

| Step | Hypothesis tested by B's lesson | Outcome |
|---|---|---|
| 0 | H-lex: style-heavy force + twin consistency binds word numbers/abbreviations (ID ≥ 0.85) | **Falsified** — both arms *lost* canonical force (0.99 → 0.63, T003) |
| 100 | H-interf: replay 0.5 protects canonical force (≥ 0.90); A will not recover (vs H-shock) | Partly: B 0.77 (falsified), A stayed 0.65 → H-shock falsified |
| 200 | Concentrating binary tasks forms answer types (equiv ≥ 0.45) | **Falsified** — answer types formed in A too; not caused by the lesson |
| 300 | aux_role at max improves value-unit binding (probe ≥ 0.60) | **Falsified** — probe flat at 0.47 |
| 400 | Cross-frame twins + max consistency → frame-invariant sense | **Falsified** — in-distribution 0.93 (memorised), new contexts *more* phys-biased (80%); equiv routing broke |
| 500 | Restore equiv routing (≥ 0.45, yes/no ≥ 85%) | **Confirmed** (0.46, 97%) — but only recovered what step 400 cost |
| 600 | Convert is readout-limited (T005) → convert dose (≥ 0.20) | **Falsified** — 0.115; A equal without the dose |
| 700 | H-prior: on A's own lesson B converges to A | Partly: 15/16 sets within 0.057; new-frame sense showed hysteresis |
| 800 | Sense dose restores new-frame type recognition; H-slot mechanism | Partly: accuracy restored (0.50); mechanism falsified (T006) |
| 900 | Sense holds at lower dose; equiv new families ≥ 0.45 | Partly: sense held (0.46); equiv falsified |
| 1000 | H-iit-tax: IIT dominates the loss (4.4/6.2) and crowds out learning | **Falsified** — no change at IIT 1 (CKA 0.999) |
| 1100 | Hold; equiv gap prior-driven | Partly (new-frame 0.398 vs ≥ 0.40) |
| 1200 | H-distract: distractor exposure closes the distractor gap | **Falsified** |
| 1300 | Frozen regime: B within 0.06 of its previous values | Partly: B−A bounded, but B's own sets moved ≤ 0.12 |
| 1400 | Final: dev mean B−A in [−0.05, +0.03]; sealed CI includes 0 | Dev confirmed (+0.001); sealed **falsified** (A better) |

The most useful single experiment was the step-700 **diagnostic block**: giving B exactly A's lesson made 15/16 dev
sets converge within one block. Together with CKA(h, prev) ≥ 0.98 from step 300 on, this shows that in this regime
the curriculum acted mainly on **output priors** (answer-type and class biases), not on representations.
I also explicitly declined one available lever: biasing equiv toward the dev "yes" prior (61% of gold) would have
raised dev and sealed numbers with no skill.

Matching (ledger): 1,500 steps × 128 examples per arm; tokens A 2.81 M / B 2.80 M; training CPU A 207 s / B 198 s;
observation 187 s; supervisor tool executions on study checkpoints 69 s wall (27 runs) — recorded separately and
never used as student training.

## 31.4 Sealed final evaluation (run once; confirmatory)

| Sealed set | common | A fixed | B adaptive | B − A |
|---|---|---|---|---|
| force_retention_pretrainfam (guard) | 0.539 | 0.581 | 0.544 | −0.036 |
| force_id_trainfam | 0.279 | 0.354 | 0.349 | −0.005 |
| force_newfam | 0.190 | 0.258 | 0.250 | −0.008 |
| force_ood_value | 0.188 | 0.227 | 0.224 | −0.003 |
| force_distractor | 0.206 | 0.250 | 0.188 | −0.062 |
| force_crossed (compositional) | 0.128 | 0.169 | 0.159 | −0.010 |
| **convert_newfam** | 0.078 | 0.083 | 0.065 | −0.018 |
| **convert_ood_value** | 0.050 | 0.050 | 0.065 | +0.016 |
| **equiv_trainfam** | 0.000 | 0.583 | 0.544 | −0.039 |
| **equiv_newfam** | 0.000 | 0.398 | 0.380 | −0.018 |
| **inverse_newfam** | 0.120 | 0.125 | 0.125 | +0.000 |
| **inverse_ood_value** | 0.109 | 0.088 | 0.078 | −0.010 |
| **sense_newframe_newctx** | 0.000 | 0.318 | 0.276 | −0.042 |
| **sense_trainframe_newctx** | 0.000 | 0.471 | 0.398 | −0.073 |
| **Primary mean (bold sets)** | — | **0.265** | **0.242** | **−0.023 [−0.035, −0.011]** |

Secondary (sealed families/sets): block IIA cross-family common 0.145 / A 0.152 / B 0.128; pooled edits —
inversion intended 0.058 / 0.064 / 0.093, add-one 0.072 / 0.065 / 0.075 (reported separately as registered; all
near chance); binding probe ID 0.466 / 0.461 / 0.466, OOD 0.221 / 0.227 / 0.224; paraphrase all-families-agree
0.00 / 0.11 / 0.11, within-AST cosine 0.57 / 0.56 / 0.57 vs between −0.005. Equiv and sense at 0.000 for the
common reference reflect never-trained answer types. Numbers near chance (1/13 = 0.077 for 13-way, ~0.5 binary)
should be read as "not learned".

## 31.5 Deviations, defects and limitations (read before the numbers)

1. **Stabilisation trigger was unattainable (registration defect).** `dev/force_id_pretrainfam` contains 48% word-
   number/abbreviated-unit items that pretraining never shows (it uses canonical style only), so ID ≥ 0.95 could not
   be reached; canonical-style ID was 0.995. The registered cap (5,000) was applied as written; the common checkpoint
   is therefore ~2,000 steps past practical stabilisation. Both arms start from it, so the comparison is unaffected.
   The seed-999 smoke had fired the trigger because it ran against the eval set *before* its re-freeze — a
   smoke/study mismatch I did not catch before registering.
2. **Registered card metric `observe.pooled_edit` is contaminated** for this student: it fits on a styled pool
   whose word/abbreviation items have unbound block codes (T002). Kept unchanged (registered); T001 v3 gives the
   uncontaminated reading (§31.6).
3. **Harness change after registration**: `Ctx.dev_sets()` added at step 100 (tools may not import pathlib); the
   architecture map version changed accordingly (`b98012f2…` → `9d515845…`); no effect on training or evaluation.
4. **Single pair, n = 1 seed**; the example bootstrap understates between-run variability. Block-to-block dev
   fluctuation of prior-driven sets was ~0.1 even at CKA ≥ 0.99.
5. **Small, slow student**: 130 K parameters, 1,500 continuation steps; numeric composition beyond the pretrained
   force task was not learned by either arm. The comparison is therefore between two curricula that mainly shaped
   priors. Whole-word tokenizer: new frames/contexts carry `<unk>` cue words (manifest OOV up to 0.40); this limits
   the sense and new-family sets for both arms. **No BPE-vs-ANLU comparison was run.**
6. B's lessons were restricted to the frozen training pools (registered); B authored no new phrasing families.
7. Modular-arithmetic toy semantics (mod 13, fictional units) — not real-world unit arithmetic.

## 31.6 Toolsmith: instruments and what they found (exploratory)

| Tool | Built when / why | Validation (planted truth) | Finding |
|---|---|---|---|
| T001 edit decomposer (v1→v3) | Smoke: block IIA 1.0 but pooled edit 0.02 | 5 synthetic mechanisms (additive, route-rotated, scrambled, ring, random) recovered | v2 "undetermined" was an artefact (see T002); v3 with canonical pool: copy 1.00, per-route affine 0.72–0.75, pooled 0.10–0.34 → **class-separable but curved code (H2)**; inversion more linear than add-one |
| T002 swap reconciler | To explain IIA 1.0 vs exemplar copy 0.55 | self-swap, full-h swap, empty block, strata bookkeeping | Canonical-style sources swap at **1.00**, word-number sources **0.07–0.08**: unseen surface forms write unbound block codes. Replicated on the study common checkpoint (0.99 vs 0.19) |
| T003 stratified accuracy | Step 100: both arms regressed | strata reweighting, constant-predictor oracle | The regression was a real loss of canonical force (0.99 → 0.63), identical in A and B |
| T004 generalisation gap (v1→v2) | Step 200: new tasks ~0 on dev | training-pool-only check, oracle 1.0, constant predictor | Not a generalisation gap: **answer-type collapse** in-distribution (equiv answered "0"); v2 dev histograms showed B reading the new sense frame as numeric and equiv answering from frame-conditional priors |
| T005 decodability ladder | Step 600: why is convert at chance? | planted linear code, random features, planted route-rotated circle (kNN must decode, linear must under-read) | Convert ingredients present at the query position (v 0.98, t 0.91 linear) and an answer probe reaches ~0.5 vs the model's 0.21–0.25 → **readout-limited, not input-limited**; canonical mass in the block is not linearly decodable (0.39 linear vs 0.62 kNN) — linear probes under-read this student. **Six failed validations are logged**: the controls caught an arbitrary threshold, an ill-posed planted code, a decoder flaw (per-dimension standardisation inflating noise) and an ambiguous gold case before any result was used |
| T006 cue occlusion | Step 800: B lost new-frame sense type | planted causal token 0.88 vs irrelevant 0.0 | Falsified my "physics-word" hypothesis; the type cue is "is" (drop 0.47–0.56); "quantity" pushes B toward numeric answers on the new frame |

Every instrument ran in the sandbox on checkpoint copies with source hash, git commit, args and before/after
checkpoint hashes in `results/sem/tool_registry.jsonl`; `run` refuses any tool without a passing validation of its
current source.

## 31.7 Post-training feature programming (exploratory, after the sealed evaluation)

{{PROGRAM}}

## 31.8 Executable follow-up specification

1. **Multi-seed replication** (≥ 5 common checkpoints × A/B) with a run-level CI; reuse `run_sem.py` unchanged except
   `SEED`/`CONT_SEED`.
2. **Fix the trigger**: compute it on canonical-style items, or include styled forms in pretraining.
3. **Held-out supervisor dev**: split dev into a supervisor-visible half and a hidden half to measure
   dev-selection overfitting directly (this study's sense reversal suggests it is substantial).
4. **A student that can learn the tasks** (larger d, longer continuation) so that curricula act on skills rather
   than priors; then repeat the comparison.
5. **Positional-slot test** for T006's "quantity" effect with a new non-training frame — outside the confirmatory
   arm.

## 31.9 Reproduce

```
python run_sem.py freeze && python run_sem.py pretrain && python run_sem.py cont-init
# per block: python run_sem.py review runs/zoo_sem/reviews/rNNNN.json; python run_sem.py lesson runs/zoo_sem/reviews/lNNNN.json; python run_sem.py advance
python run_sem.py final            # refuses a second run
python -m toolsmith.runner validate|run <tool_id> name=ckpt_dir ... [--args JSON] [--evidence NAME]
python -m sem.program runs/zoo_sem/<arm>/step_1500 out.json --pair easy|hard
python -m pytest -q tests/test_sem.py
```
