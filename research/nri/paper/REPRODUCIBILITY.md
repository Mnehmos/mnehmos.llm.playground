# Reproducibility package — Birth of a Concept

Everything needed to regenerate every number in the preprint, from code in this repository plus the checkpoint
archives (held outside Git).

## Environment

| item | value |
|---|---|
| Python | 3.13.18 |
| torch | 2.15.0+cu132, CPU only |
| execution | `torch.use_deterministic_algorithms(True)`, one thread per training worker |
| OS / architecture | Linux x86_64 (cloud container) |
| other packages | `safetensors`, `numpy`, `matplotlib` (figures); TeX Live 2023 (paper) |

Bitwise reproduction has been verified in this environment:
- factorial run `wd1.0_C_s2004`, regenerated from Git for 300 steps (doc 40 / PRESERVATION.md);
- historical role-swap networks 304, 305, 307, 310 and 311, final weights bitwise equal to the 2026-11-22 originals
  (doc 41R; `runs/swap_recon/qualification.json`).

Bitwise identity on other CPUs or BLAS builds is **not** guaranteed. Functional identity (identical predictions)
is the fallback criterion (doc 41 §41.6).

## Data held outside Git

| set | location while this session lives | integrity record in Git |
|---|---|---|
| 96 factorial run archives (5,856 checkpoints, 15.5 GB) | `/home/user/nri_archives/*.tar.xz`; delivered to the owner | `runs/concept_factorial/SHA256SUMS`, per-run `manifest.json`, `PRESERVATION.md` |
| 36 role-swap reconstructions (2,196 checkpoints, ≈ 2.8 GB) | `/home/user/swap_archives/*.tar.xz`; delivered to the owner (Phase V) | `runs/swap_recon/<run>/manifest.json`, `runs/swap_recon/archives/SHA256SUMS_reconstruction`, `PRESERVATION.md` |
| 10-step critical-window replay (5 networks, steps 600–2,000) | `/home/user/swap_window_archives/`; delivered to the owner (Phase V) | `runs/swap_recon/archives/SHA256SUMS_window`, `PRESERVATION.md` |
| 46 training-intervention runs (Phase IV) | results only, in Git; checkpoints are not kept (each run regenerates bitwise from an archived checkpoint) | `runs/phase4/dev_intervene/*.json` |
| 36 historical role-swap endpoints | `runs/zoo_swap/` (in Git, 11 MB) | commit `aaa67c6` |

**Owner-side verification of the delivered archives (factorial, reconstruction and window) is pending.** Checksums and manifests prove
integrity; they do not preserve the tensors. Until a durable copy is verified, the container copies are the only
copies.

## Regeneration commands

| result | command | output |
|---|---|---|
| factorial training (do not re-run without justification) | `python -m sci.factorial` (doc 37) | 96 runs |
| registered analysis | `python -m sci.factorial_analysis` | `runs/concept_factorial/analysis.json` |
| independent verification | `python -m sci.verify_factorial` | `verification.json` |
| held-out causal test (doc 39) | `python -m sci.heldout_causal` | `heldout_causal.json` |
| instrument validation (planted P1–P5) | `python -m sci.causal_instruments validate` | `runs/instruments/planted_validation.json` |
| instruments on all 96 networks (I1 every 100 steps; I1–I7 every 500) | `python -m sci.causal_instruments apply 3` | `runs/instruments/factorial/*.json` |
| H-A … H-E mechanism | `python -m sci.mechanism_sswd` | `runs/instruments/mechanism_sswd.json` |
| historical replay (Q1) | the original `run_swap.train_one` from `git archive aaa67c6` | compare with `python -m sci.swap_recon compare` |
| reconstruction (Q2–Q4) | `python -m sci.swap_recon train <zoo> group:seed ...` | 61 checkpoints per run |
| role-swap audits | `python -m sci.swap_audit <zoo>` | `<run>/audits.jsonl` |
| cohort reviews | `python -m sci.swap_cohort <zoo> runs/swap_recon/reviews` | 61 reviews |
| Phase III/IV analysis | `python -m sci.swap_analysis <zoo>` | `runs/swap_recon/analysis.json` |
| critical-window replay (doc 45) | `python -m sci.window_replay <zoo> <out> 304 305 307 310 311` | 141 states per network, bitwise-qualified |
| window factorisation and battery | `python -m sci.window_analysis <root> <out>` | `runs/swap_recon/window_*.json` |
| Phase IV rescue without training (E1, R1–R7L, D1) | `python -m sci.rescue` (plus `completion_suite`, `labelled_class_completion`, `angle_completion`) | `runs/phase4/{rescue,completion,labelled_completion,angle_completion}.json` |
| compatibility and decoder trace (E2, E5) | `python -m sci.rescue` (compat), `python -m sci.decoder_trace` | `runs/phase4/{compat,decoder_trace}.json` |
| exact programming (E6) | `python -m sci.compile_edit` | `runs/phase4/programs.json`, `programmability_population.json` |
| training interventions (E4, 46 runs) | `python -m sci.dev_intervene runs/phase4/dev_intervene --workers 4` | `runs/phase4/dev_intervene/*.json`; controls checked bitwise against the reconstruction |
| head atlas | `python -m sci.atlas_heads` | `runs/phase4/atlas_heads.json` |
| Phase V D1, D2, D3, D5 | `python -m sci.phase5_drivers {d1,d1_exact,d2,d3,d5}` | `runs/phase5/{decoder_profiles_*,decoder_birth,tolerance,transplant}.json` |
| Phase V D4 algorithm | `python -c "import sci.algorithm as A; A.main(); A.main_preact(); A.main_program(); A.main_additive(); A.main_pair()"` | `runs/phase5/{algorithm,preact_phases,program_validation,additive_program,pair_program}.json` |
| Phase V D7 output operators | `python -m sci.operator_edit` | `runs/phase5/operator_edit.json` |
| Phase VI construction and qualification | `python -m sci.construct_qualify` | `runs/phase6/qualification.json` |
| Phase VI instruction system | `python -m sci.instructions` | `runs/phase6/instructions.json` |
| Phase VI residual (R1–R3, per-pair split) | `python -m sci.residual`; `python -m sci.residual pairs` | `runs/phase6/residual*.json` |
| Phase VI comparison and hybrids | `python -m sci.compare_hybrid` | `runs/phase6/{compare,hybrid}.json` |
| Phase VI compiler grid | `python -m sci.compiler` | `runs/phase6/compiler.json` |
| Phase VI addendum: instrument validation, neuron atlas, development, direct path | `python -m sci.neuron_lens validate`; `... learned`; `... develop`; `... direct` | `runs/phase6/atlas/*.json` |
| Phase VII D, I, R, G, F | `python -m sci.direct_path`; `python -m sci.instruments2 {validate,learned}`; `python -m sci.robust` (+ `combo`); `python -m sci.graph_compiler`; `python -m sci.fourier` | `runs/phase7/*.json` |
| Phase VII H/K (training, 48 runs; resumable) | `python -m sci.hybrid A`; `python -m sci.hybrid train --workers 3 [--only C1,C2]` | `runs/phase7/hybrid/*.json` |
| Phase VIII A1 (bitwise reproduction of Phase VII) | `python -m sci.p8_audit_repro run`; `python -m sci.p8_audit_repro compare` | `runs/phase8/audit/` |
| Phase VIII A2/A4 accounting | `python -m sci.p8_cost` | `runs/phase8/accounting.json` |
| Phase VIII S, A3, P2, E (training; resumable, ≤ 3 workers) | `python -m sci.p8 run {S,A3,P,E} --workers 3` | `runs/phase8/runs/*.json` |
| Phase VIII P1/P3 | `python -m sci.p8_p1` | `runs/phase8/p1_guarantee.json` |
| Phase VIII G2, R2 | `python -m sci.graph2`; `python -m sci.routing2` | `runs/phase8/graph2.json`, `routing2.json` |
| Phase VIII Q, E-modular, MR | `python -m sci.p8_q q`; `python -m sci.p8_q emod`; `python -m sci.p8_extract`; `python -m sci.p8_mr_verify` | `runs/phase8/q.json`, `e_modular.json`, `extraction.json`, `mr.json` |
| figures and PDF | `python paper/make_figures.py && cd paper && latexmk -pdf main.tex` | `paper/main.pdf` |

## Tests

```
python -m pytest -q tests/test_concept.py tests/test_factorial.py tests/test_swap_recon.py
```

The tests cover:
- bitwise resume;
- the exact AdamW decomposition;
- the audit never touching the reserved test;
- the archive round-trip;
- held-out pair enumeration excluding test combinations;
- reconstruction resume on a shortened schedule.

## Re-run checks (Phase V)

- `sci.phase5_drivers d3` and `d5` reproduced `runs/phase5/tolerance.json` and `transplant.json` byte for byte.
- Every Phase IV intervention control is bitwise identical to the reconstruction at all 20 checks (doc 49 §49.4).
- Drivers D1/D2/D3 were run first as inline scripts. They were then recovered from the session log into
  `sci/phase5_drivers.py` with their logic unchanged.
