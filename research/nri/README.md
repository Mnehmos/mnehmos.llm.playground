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

## Layout

```
nri/            package: tasks.py (typed task, 5 conditions), model.py (LayerNorm-free transformer),
                train.py (deterministic), analysis.py (IIA, DAS, head patching, role purity,
                edit programs, transfer), nri_exec.py (NRI executor), reference_nri.py
programs/       NRI weight programs (op_swap.py)
schema/         NRI JSON schema
pilot/          blinded-LLM-interpreter packaging and scoring
tests/          unit tests (determinism, exact rank-one-fold equivalence, executor)
runs/zoo_*/     checkpoints + config + training log for every model (never overwritten)
results/        metrics.json, summary.md, executed NRI documents, figures, validity.json, pilot/
```

## Reproduce (CPU only)

```bash
pip install torch matplotlib pytest
python -m pytest -q tests/                                    # 7 tests, ~15 s
python validity_check.py                                      # instrument positive/negative controls
python run_experiment.py --seeds 5 --steps 3000 --wd 0.01     # ~15-30 min on 4 cores
python run_experiment.py --seeds 5 --steps 3000 --wd 1
python run_untyped_inference.py --seeds 5 --steps 3000   # doc 08, ~1 CPU-hour
python make_figures.py
```

Training is bitwise deterministic (a test enforces it). Existing checkpoints are reused, not retrained.
