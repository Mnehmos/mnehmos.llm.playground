# NRI: Neural Representation Interface and typed-prior interpretability experiments

An isolated research directory. It is self-contained and touches nothing else in the repo.

**Question:** can neural networks be made more understandable and programmable by designing their
representation geometry, specifically ANLU-style typed token embeddings `e = c + t_τ + p`, and can
an LLM's interpretation of a network be made *executable and refutable*?

## Deliverables

| # | Document | Contents |
|---|---|---|
| 1 | [`docs/01_literature_review.md`](docs/01_literature_review.md) | Sourced review of WSP (Sparks, WeightCode, ERT, and 30+ Nov-2026 follow-ups and audits), LLM-assisted interpretability (HyVE), the mech-interp toolkit, model editing, representation geometry, and Muon/mHC/gated attention. Every source is tagged by how far it was verified. |
| 2 | [`docs/02_anlu_comparison.md`](docs/02_anlu_comparison.md) | ANLU split into hypotheses H1–H6; where the literature supports, contradicts or is silent; the revised hypothesis |
| 3 | [`docs/03_nri_spec.md`](docs/03_nri_spec.md) + [`schema/nri-0.1.schema.json`](schema/nri-0.1.schema.json) | The NRI: a typed, executable interpretation format; every claim is checked by intervention |
| 4 | [`docs/04_roadmap.md`](docs/04_roadmap.md) | Prioritized experiments R0–R7 with kill criteria |
| 5 | [`docs/06_prototype_results.md`](docs/06_prototype_results.md) | **The prototype**: 50 trained networks, causal metrics, a transfer test, instrument-validity controls, and a blinded LLM-interpreter pilot |
| 6 | [`docs/05_publication_pathway.md`](docs/05_publication_pathway.md) | What would be original rather than a replication, and where to publish it |
| 7 | [`docs/07_review_of_research_program.md`](docs/07_review_of_research_program.md) | Review of the five ANLU-program papers: citation fixes (the scaling-law citation is misattributed), internal contradictions, the symmetry argument, missing prior art (Syntactic Scaffolds 2018), recommended design changes |
| 8 | [`docs/08_untyped_inference_results.md`](docs/08_untyped_inference_results.md) | **ANLU as specified**: typed training, untyped inference, 40 networks. No speed-up; removal needs a mechanism; **types as an auxiliary target make internals ~2.7× more editable**, specifically |
| 9 | [`docs/09_r6_interchange_training.md`](docs/09_r6_interchange_training.md) | **R6**: interchange intervention training at the computation site → every variable perfectly localizable (DAS 1.00), op-swap edit 0.96, **held-out operand-shift edit 0.80 vs 0.04**, no accuracy cost |
| 10 | [`docs/10_r1_white_vs_black_box.md`](docs/10_r1_white_vs_black_box.md) | **R1**: blinded LLM interpreters, white-box vs black-box × baseline vs role-supervised networks, plus cross-evaluation of their edit programs on unseen networks (incl. IIT) |
| 11 | [`docs/11_preregistration.md`](docs/11_preregistration.md) | **Pre-registration** (committed before the run) of the dimensional-reasoning study: hypotheses, Holm-corrected primary family, decision rules |
| 12 | [`docs/12_dimensional_results.md`](docs/12_dimensional_results.md) | **Results**: P1 (editability) supported but small (0.36 vs 0.17); P2 (OOD unit generalization) not supported; IIT unstable (sign-ambiguity trap); partial specs give the best edits; compute-matched control null |
| 13 | [`docs/13_next_registration_plan.md`](docs/13_next_registration_plan.md) | Plan (not a registration) for the next study: variable-discriminating edits with a specificity metric, a designed-block positive control, reliable IIT training |
| 14 | [`docs/14_registration_role_swap.md`](docs/14_registration_role_swap.md) | **Registration** (committed before the run) of the role-swap study |
| 15 | [`docs/15_role_swap_results.md`](docs/15_role_swap_results.md) | **Role-swap result**: the edit advantage follows the declared variable in 12/12 seeds (primary Δ +0.37, p = 0.0005); copying real patterns works (0.8–0.9) but the fitted edit doesn't (0.2–0.3), so edit construction is the bottleneck |
| 16 | [`docs/16_registration_editors.md`](docs/16_registration_editors.md) | **Registration** (committed before scoring the fresh sample) of the editor comparison on the fixed role-swap networks: pooled vs route-aware editors in three tiers (weight edit / stored table / per-question), copy as benchmark |
| 17 | [`docs/17_editor_results.md`](docs/17_editor_results.md) | **Editor result**: the same value leaves a different pattern per unit; a unit-aware editor reaches 0.76 / 0.89 (copy 0.80 / 0.91; current editor 0.18 / 0.31), 0.97–0.98 on converged networks, with no rise in wrong-variable changes; but only as a per-question lookup. A single linear weight edit gains just +0.05–0.07 |
| 18 | [`docs/18_registration_permanent_edit.md`](docs/18_registration_permanent_edit.md) | **Registration** (committed before any editor touched the new networks): can the unit-aware edit be installed once (added layer) or within the existing weights (distillation)? New networks (seeds 400–411), 30% of questions reserved from the editors |
| 19 | [`docs/19_permanent_edit_results.md`](docs/19_permanent_edit_results.md) | **Permanent-edit result**: on new networks and reserved questions, the unit-aware edit installed once, as an added ~70-unit layer or by re-fitting the existing post-site weights, matches the stored table (0.90 mass / 0.87 acceleration vs 0.27 / 0.34 for the linear edit), with fewer wrong-variable changes; the existing-weights version is a large re-fit (35–59% weight change) |
| 20 | [`docs/20_design_unit_calculus.md`](docs/20_design_unit_calculus.md) | **Design** (not a registration): map every quantity in the computation (values, labels, label meanings, converted quantities, the calculator) and test whether label↔value relationships correspond to calculations. Feasibility probe: the calculator multiplies in discrete-log space (≈ 0.93 multiplicative-character concentration; additive flat) |
| 21 | [`docs/21_unit_calculus_pilot_findings.md`](docs/21_unit_calculus_pilot_findings.md) | **Pilot findings** (exploratory, n = 1–2): the 6-label design does not learn (4 labels does, unevenly); calculation type shows in answer geometry (scaled → discrete-log circles after the MLP; offsets → value-order circles already in attention); no converted quantity is ever a separable variable: value fuses early, the label stays separable through layer 1; the declared control failed |
| 22 | [`docs/22_registration_checkpoint_correction.md`](docs/22_registration_checkpoint_correction.md) | **Registration** (Issue #20; committed before training): checkpoint-guided correction pilot. 3 fresh `mass_only` networks, checkpoints 0/600/1,500/3,000/4,500/6,000, bitwise-matched control vs downstream-readout-correction branches from step 3,000 |
| 23 | [`docs/23_checkpoint_correction_results.md`](docs/23_checkpoint_correction_results.md) | **Checkpoint-correction result** (exploratory, n = 3): mass representations form between steps 600 and 1,500 in three distinct regimes (canonical-like / intermediate / route-dependent); the registered readout correction passed in 1 of 3 networks and met the 'improved' criterion narrowly (+0.055), but the same fold applied after training did as well (0.722 vs 0.705); no upstream reorganization; canonicalization destroys the calculation in the route-dependent network |
| 24 | [`docs/24_registration_atlas_continuous_edit.md`](docs/24_registration_atlas_continuous_edit.md) | **Registration** (Issue #22; committed before training): two-trajectory concept-formation atlas (61 checkpoints each, every 100 steps) and a continuous, validated **upstream** edit loop on `attn2.o` rows 0–15 that shrinks the mass × route interaction while keeping unit information; one control and one supervised network from the same seed |
| 25 | [`docs/25_atlas_continuous_edit_results.md`](docs/25_atlas_continuous_edit_results.md) | **Atlas result** (exploratory, one matched pair): mass becomes decodable at step 400, causal at 1,000, verified at 1,200, inside the declared block; layer 1 fuses unit into the value token; a discrete-log circle forms just before causal use; unit routes **split** at formation (no coalescence); the registered upstream edit was rejected at all 51 eligible checkpoints (E1), so the two networks stayed bitwise identical and the central claim is **not tested**; full interaction removal needs 4–13× the producer's norm and destroys accuracy |
| 26 | [`docs/26_registration_llm_in_the_loop.md`](docs/26_registration_llm_in_the_loop.md) | **Registration** (owner-directed follow-up to #22; committed before the loop network advanced): the research agent inspects one network every 100 steps and designs bespoke read-out and edit methods for any weights; every edit passes a fixed deterministic gate; the matched control is the pilot's control (bitwise identical until the first edit) |

## Layout

```
nri/            package: tasks.py (typed task, 5 conditions), model.py (LayerNorm-free transformer),
                train.py (deterministic), analysis.py (IIA, DAS, head patching, role purity,
                edit programs, transfer), nri_exec.py (NRI executor), safe_expr.py (sandboxed
                expression evaluator, no eval), reference_nri.py
programs/       NRI weight programs (op_swap.py)
schema/         NRI JSON schema
pilot/          blinded-LLM-interpreter packaging and scoring
tests/          unit tests (determinism, exact rank-one-fold equivalence, executor, safe
                evaluator vs Python on every stored NRI document, rejection of unsafe input)
runs/zoo_*/     checkpoints + config + training log for every model (never overwritten)
results/        metrics.json, summary.md, executed NRI documents, figures, validity.json, pilot/
```

## Reproduce (CPU only)

```bash
pip install torch matplotlib pytest
python -m pytest -q tests/                                    # 140 tests, ~2 min
python validity_check.py                                      # instrument positive/negative controls
python run_experiment.py --seeds 5 --steps 3000 --wd 0.01     # ~15-30 min on 4 cores
python run_experiment.py --seeds 5 --steps 3000 --wd 1
python run_untyped_inference.py --seeds 5 --steps 3000   # doc 08, ~1 CPU-hour
python run_iit.py --seeds 5 --steps 3000                 # doc 09 (R6), ~1 CPU-hour
python run_dimensional.py --seeds 10 --steps 6000        # docs 11-12 (pre-registered), ~3 CPU-hours
python edit_specificity.py                               # doc 12 §12.8 (post hoc), ~15 min
python run_swap.py train --schedule warmup --seeds 300 301 302 303 304 305 306 307 308 309 310 311   # docs 14-15
python run_swap.py analyse --seeds 300 301 302 303 304 305 306 307 308 309 310 311
python run_swap.py report  --seeds 300 301 302 303 304 305 306 307 308 309 310 311
python run_editors.py --split test && python run_editors.py --report   # docs 16-17 (fixed networks)
python train_perm.py && python run_perm.py analyse && python run_perm.py report   # docs 18-19
python run_correction.py run && python run_correction.py report                   # docs 22-23 (Issue #20)
python run_atlas.py train && python run_atlas.py analyse && python run_atlas.py final && python run_atlas.py report   # docs 24-25 (Issue #22)
python make_figures.py
```

Training is bitwise deterministic (a test enforces it). Existing checkpoints are reused, not retrained.

## Security: NRI documents are untrusted input

NRI documents are written by interpreters, including LLMs, so the executor treats them as untrusted:
- **Expressions** (`binding`, `output_fn`, `scope`, `edit_fn`) are never passed to `eval`.
  `nri/safe_expr.py` interprets a small whitelisted subset of Python:
  - arithmetic, comparisons and conditionals;
  - indexing, list comprehensions and lambdas;
  - a few builtins: `range`, `len`, `min`, `max`, `abs`, `sum`, `any`, `all`, `int`, `list` and `enumerate`.

  There is no attribute access and there are no imports. Loops, lengths, nesting depth and
  integer sizes are bounded.
- **Edit programs** are Python files, which means running code. `execute(..., allow_programs=True)`
  and the pilot scripts' `--allow-programs` flag are required to run them. Use them only for
  reviewed programs or inside an isolated environment. Without the opt-in, an edit claim is
  recorded as an error, never as evidence.

