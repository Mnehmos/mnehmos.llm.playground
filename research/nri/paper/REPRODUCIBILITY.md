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
| 36 role-swap reconstructions (2,196 checkpoints, ≈ 2.8 GB) | `/home/user/swap_archives/*.tar.xz` | `runs/swap_recon/<run>/manifest.json` |
| 36 historical role-swap endpoints | `runs/zoo_swap/` (in Git, 11 MB) | commit `aaa67c6` |

**Owner-side verification of the delivered factorial archives is pending.** Checksums and manifests prove
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
