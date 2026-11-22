# 4. Experimental roadmap

Ordered by **decisiveness per unit cost**. Each stage has a pre-declared kill or pivot criterion,
so a negative result changes the plan instead of being explained away. Compute figures assume the
4-core CPU used here; "Colab" means a free T4 session.

## Scoring key

- **Value**: how much the result moves the central question.
- **Cost**: compute and API spend.
- **Difficulty**: engineering effort.
- **Decisive?**: can a single outcome falsify something?

| ID | Experiment (maps to brief) | Value | Cost | Difficulty | Decisive? | Status |
|---|---|---|---|---|---|---|
| **R0** | Prototype: typed priors vs causal localizability, editability and transfer on a toy task; 5 conditions incl. the **canonical** information control; 2 regimes (B0/C0/D0) | high | ~1 CPU-hour | low | **yes**, for the toy regime | **done** (doc 06) |
| **R1** | Blinded LLM interpreter, scored by NRI execution; **white-box vs black-box-only arms** (A, B) | very high | ~$20–100 API; CPU | medium | yes: if black-box ≈ white-box, "LLM reads weights" adds nothing here | harness done; n = 2 pilot (doc 06 §7) |
| **R2** | Move to a task where types *must* be contextual and nontrivial: unit-bearing arithmetic (NUMBER, UNIT, QUANTITY, DIMENSION, OPERATION), with a learned or noisy typer | high | CPU-hours | medium | partly | next |
| **R3** | Defect repair (C): planted defects (slice fine-tune, subspace rotation, head ablation); compare LLM-written programs, closed-form programs, fine-tuning and **random-direction controls**; collateral on held-out slices | high | CPU + API | medium | yes | design below |
| **R4** | Transfer under symmetry (D): permutation and rescaling suite (*Orbit Audits*, Git Re-Basin) + independent seeds; program rerun vs copied ΔW vs aligned ΔW | medium-high | CPU | low | yes | partly in R0 |
| **R5** | Optimizer geometry: AdamW vs **Muon** on the R0/R2 zoo → localizability and editability | medium | CPU | low | yes, for this question | cheap, novel |
| **R6** | Training for editability (E): `L = L_task + λ L_struct + μ L_edit` vs ERT-head vs weight-sparse baseline (Gao et al. 2025) | very high | Colab-hours | high | yes | after R2 |
| **R7** | ANLU curriculum (dictionary → textbook → unstructured) on a ~5–20M-param LM; interpretability measured with R1/R6 tools | high, but long | Colab days | high | rarely | last |

## R1: the LLM interpreter study (highest priority after R0)

- **Population:** the R0/R2 zoo (≥ 5 seeds × conditions), blinded with `pilot/make_blind_package.py`.
- **Arms:** (a) white-box (weights + activations + code), (b) **black-box** (input → output queries
  only), (c) white-box with the type channel's meaning revealed. Arm (b) is the control demanded by
  *Computation, not weight access* (2611.07847). Arm (c) answers your question of whether any
  advantage persists when labels are withheld.
- **Interpreters:** at least two LLM families. Cap tool calls and record token cost.
- **Outcomes** (all executable; doc 03 §3.5): variable recall, verified-claim rate, causal
  precision, edit success − collateral, and cost.
- **Kill criterion:** if typed-condition networks give no gain in variable recall or edit score
  over `none` *and* over `canonical` across ≥ 10 networks per arm, drop the claim that ANLU priors
  aid LLM interpretation and keep the NRI as the contribution.

## R3: repair (Experiment C)

| Defect | Planted by | Ground truth |
|---|---|---|
| D1 slice corruption | 200 steps of fine-tuning on mislabeled ADD with a = 3 | slice + the parameter diff |
| D2 subspace rotation | rotate the DAS op subspace by 60° | the rotation matrix |
| D3 head damage | zero one head's OV circuit | the head |

Compare four repair methods: (i) LLM-written NRI edit program, (ii) closed-form ROME-style edit at
the localized site, (iii) fine-tuning on the defect slice (matched examples), (iv) norm-matched
random-direction edit. Measure repair, collateral on 10 held-out slices, ΔW norm and rank, cost, and
whether the stated diagnosis passes an NRI `necessity` claim. Hase et al. predict that (ii)
can succeed even where localization is wrong. Test this directly by editing at the localized site
and at a non-localized site.

## R5: Muon (cheap, plausibly novel)

Muon orthogonalizes each update, which flattens update spectra. **Hypothesis (either direction is
interesting):** Muon-trained weights spread variables across more directions, making them less
low-rank and less editable; or they make class-mean subspaces better conditioned and more editable.
Swap the optimizer in `nri/train.py` and rerun R0's metrics. About 1 CPU-hour.

## R6: training for editability (Experiment E), a concrete first objective

- **`L_struct`** (localization): an IIA loss on a *fixed* random orthonormal subspace per variable,
  the training-time version of DAS (an interchange-intervention training objective, as in Geiger et
  al.'s IIT). It forces each variable into a known, disjoint subspace.
- **`L_edit`** (editability): sample an internal rank-one edit from the op_swap program family,
  apply it differentiably, and penalize (1 − success) plus collateral. This is ERT moved from the
  head to the **internal** site.
- **Baselines:** ERM, ERT-head (2611.26408), weight-sparse training (Gao et al. 2025), BIMT.
- **Primary outcome:** LLM edit score (R1 harness) and closed-form edit success on *held-out*
  edit types not used in `L_edit` (otherwise the result is circular).
- **Kill criterion:** if gains appear only on the trained edit family, `L_edit` buys a specific
  editability, not general programmability.

## Decision gates

1. **After R0** (now): if typed ≈ canonical on every causal metric, the "type *geometry*"
   claim is unsupported in this regime. Continue only with R2, where types carry information *not*
   recoverable from position. See doc 06 for the result.
2. **After R1:** if black-box ≈ white-box, frame all later work as "*computation-assisted*
   interpretation" and drop "native weight reading".
3. **After R2/R6:** publish whichever of {typed priors, IIT-style structure loss, internal ERT}
   gives the largest held-out LLM edit-score gain, with every arm reported.

## Reproducibility rules (already implemented)

- Fixed seeds; `torch.use_deterministic_algorithms(True)`; single-threaded training. A test checks
  bitwise-identical retraining.
- Checkpoints are never overwritten (`train()` skips existing runs unless `--force`).
- Every run directory holds `config.json` (including torch version and parameter count) and
  `log.jsonl`. Every NRI document records the weights' sha256.
- Results are regenerated only by scripts; tables are auto-generated, not hand-edited.
