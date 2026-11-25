# 41 — Protocol (committed before running): Phase II instruments and mechanisms, Phase III historical replay

Written and committed **before** any analysis below is run. Every analysis here is **retrospective and
exploratory**, except the replay qualification criteria (§41.6), which are pass/fail rules fixed now. Nothing here
changes the registered factorial (doc 37), its primary endpoint or its five contrasts. It does not re-use the
reserved test (doc 38). It does not modify any original trajectory or checkpoint measurement. The doc-15
registered results are also left unchanged.

Owner authorisation: Phases II and III were authorised with owner-side archive verification still pending (see
`runs/concept_factorial/PRESERVATION.md`, which stays "pending owner check"). No original archive is deleted or
modified.

## 41.1 Resource and safety check (before any reconstruction)

| item | value |
|---|---|
| free disk | 34 GB, against a 15 GB archive set that is not touched |
| reconstruction budget | ≤ 3 GB (36 × 61 checkpoints × ~1.1 MB uncompressed) |
| CPU | 4 cores, 15 GB RAM |
| original replay cost | 642 s per declared network, single thread (`config.json`) |
| 36-network reconstruction | ≈ 2.5 h wall-clock on 4 workers |

**Archive and weight protections:**
- **Archives:** `/home/user/nri_archives` is read-only (mode a-w). The session runs as root, so file modes are
  advisory only. The binding protection is procedural: nothing here writes to that path. In addition, the
  SHA-256 of all 96 archives is re-verified against `SHA256SUMS` **after** Phases II–III; any mismatch is
  reported and stops further work.
- **Extracted weights:** `/home/user/nri_weights` (step-0…6,000 model weights for all 96 runs, previously
  checked against the Git manifests) is also read-only. Analyses read only those copies.

**Where new outputs go:**
- reconstruction outputs: `/home/user/swap_recon` (outside Git);
- per-run compressed archives and SHA-256 manifests: the same as for the factorial;
- small JSON and Markdown outputs: `runs/swap_recon/` in Git.

## 41.2 Phase II.1 — validated held-out causal instruments

### Instrument

`sci/causal_instruments.py`. A generic pair-and-stratum engine: the doc-39 pair enumeration with the reserved
test excluded everywhere, and strata of base pool × canonical-implied combination. It runs against an adapter
exposing `states(batch)`, `patch(states_base, states_donor, condition)` and `readout(states, batch)`. All
rates are reported as **excess over the no-intervention baseline**, the correction adopted after the doc-39
control failure.

### Seven intervention conditions

All are applied to the source-number token unless stated.

| id | intervention | what it tests |
|---|---|---|
| I1 | full residual after block 1 (x1) | doc-39 instrument |
| I2 | number **and** unit tokens at x1 | whether unit information outside the number token adds anything |
| I3 | full residual after block 2 (x2) | whether the code persists downstream |
| I4 | **canonical subspace only** at x1: projection onto the span of canonical-class means, fitted on *trained* bases only (rank ≤ 12) | is the effect carried by the canonical-class structure? |
| I5 | **complement** of the I4 subspace at x1 | is the effect carried elsewhere? |
| I6 | **same-canonical donor** at x1: donor has the base's canonical quantity but a different number and unit | the canonical code predicts *unchanged*; the number code predicts a change |
| I7 | **random subspace** of the same rank as I4 at x1 (10 draws) | the dimensionality control for I4; this guards against the subspace-patching illusion (§41.3) |

### Controls

All must hold before any network result is interpreted:

- **(a) input-level swaps at x0.** Swapping the number token must follow the number prediction; swapping the
  number + unit tokens must follow the canonical prediction. This is required wherever the model answers the
  implied combination correctly.
- **(b) untrained networks (step 0).** |excess| ≤ 0.05 in every stratum, under every condition.
- **(c) no-intervention baseline.** Reported for every cell.

### Planted-mechanism validation

The instrument must recover each **known** mechanism, implemented as explicit adapters with no learning:

| planted | state at the number token | readout | required instrument result |
|---|---|---|---|
| P1 canonical-compositional | one-hot canonical q (+ 20 noise dims) | answer = q · s_t⁻¹ on every route | I1 canonical excess ≥ 0.9 in all strata; I6 unchanged ≥ 0.9 |
| P2 number + route lookup | one-hot v | answer table on (route, v) | I1 number excess ≥ 0.9; canonical excess ≤ 0.05 |
| P3 canonical code, **trained-only reader** | one-hot q | correct on trained (route, v′); a fixed wrong answer on held-out (route, v′) | canonical excess ≥ 0.9 when implied-trained; **≤ 0.10 when implied-held-out** (the discriminating case) |
| P4 rotated, distributed canonical code | random rotation of [one-hot q; one-hot u; noise] in 64 dims | as P1 | I4 canonical excess ≥ 0.9; I5 ≤ 0.10; I7 ≤ 0.10 |
| P5 null | arbitrary | ignores the number-token state | all excess ≤ 0.05 |

**Gate.** If any criterion fails, the instrument is fixed and re-validated before use, and the failure is
logged in doc 41R.

### Application

- **I1:** every 100 steps for all 96 networks, giving complete developmental trajectories across all 8
  conditions.
- **I2–I7:** every 500 steps for all 96 networks.
- **Pairs:** n ≤ 300 per stratum per network.
- **Reporting:** the held-out | held-out stratum (33 pairs) is reported descriptively only.

## 41.3 Literature on spurious alignments (to be cited, verified in Phase V)

Subspace activation patching can produce effects through dormant or orthogonal pathways rather than the
hypothesised variable (Makelov, Lange & Nanda 2023, *interpretability illusion*; reply by Wu et al. 2024).
Unconstrained or non-linear alignment maps can align almost any network with almost any algorithm (Sutter et al.
2025). Patching conclusions depend on the corruption method and metric (Zhang & Nanda 2023; Heimersheim & Nanda
2024). Causal abstraction and interchange interventions are framed by Geiger et al. (2021, 2023/2024).

**Design responses:**
- Patching uses real donor activations, not constructed ones, except in I4/I5.
- The I4 subspace is fitted on trained bases only.
- Random-subspace (I7) and same-canonical (I6) controls are included.
- Planted mechanisms are validated with known ground truth.
- The decisive stratum requires a **behavioural** consequence on never-trained (input, route) combinations.

## 41.4 Phase II.2 — why semantic supervision helps only with strong weight decay (H-A … H-E)

All measures are computed on the archived weights and audits. The seed is the unit of analysis. Comparisons:

- C1.0 vs C0.1 (the interaction's driver);
- C1.0 vs D1.0 (coherent vs permuted target);
- C1.0 vs B1.0 (aux vs none).

Tests are exact paired sign-flip, with Holm correction across the five primary contrasts (one per hypothesis),
α = 0.05. **Exploratory throughout.**

| | hypothesis | measure (number token, x1m) | prediction |
|---|---|---|---|
| H-A | **Compression:** strong decay plus a coherent target yields a lower-dimensional shared code | participation ratio of the 120 trained (route, value) class means at step 6,000 | C1.0 < C0.1 |
| H-B | **Route-specific components are pruned:** decay removes (route × value) structure beyond the canonical quantity | route-specificity index RSI = 1 − R²(class means ~ one-hot q) | C1.0 RSI < C0.1 RSI at 6,000; RSI falls over training in C1.0 |
| H-C | **Gradient cooperation:** the aux and task gradients at the number token cooperate more under strong decay | mean task–aux cosine at x1m over steps 0–2,000 (audits) | C1.0 > C0.1 |
| H-D | **The reader differs, not the code:** at matched decodability, C1.0's downstream reads the code on unseen combinations and C0.1's does not | I1 implied-held-out excess at the first audit where q kNN decodability ≥ 0.9 | C1.0 > C0.1 |
| H-E | **Norm / effective step:** strong decay shrinks norms, enabling late re-organisation | total parameter norm at 6,000; correlation of the norm-trajectory knee with the compositional onset | C1.0 norm < C0.1; knee precedes onset in ≥ 9/12 |

Contrasts:
- **Primary**, one per hypothesis: C1.0 − C0.1.
- **Secondary**, descriptive: C1.0 − D1.0 and C1.0 − B1.0.
- **Within C1.0 across seeds:** Spearman of each measure with development held-out accuracy (n = 12; reported
  with this caveat).

## 41.5 Phase II.3–II.5

- **Mixed-code composers (2002, 2007, 2008):** I1–I7 trajectories, with the timing of number-following vs
  canonical-following in held-out strata, compared with the 9 other C1.0 networks. Descriptive.
- **Collapse and recovery:** the 21 registered collapses (doc 38). For each, I1 implied-held-out excess and RSI
  are measured before, during and after. Hypothesis: collapses that recover restore the canonical reader before
  accuracy recovers. The test is per network (seed-clustered).
- **Dependency-aware statistics:** every longitudinal claim is made on per-network summaries (one number per
  network), with a seed bootstrap and exact sign-flip. Transitions are never pooled across networks as if
  independent. Where a trajectory-level model is needed, a per-seed random intercept and a block bootstrap over
  networks are used.

## 41.6 Phase III — replay qualification (pass/fail rules fixed now)

**Historical objects.** `runs/zoo_swap/<group>_s<seed>/{config.json, log.jsonl, model.pt}` for 36 networks
(commit `aaa67c6`, doc 15). These are final weights only. Code: `run_swap.py` at commit `aaa67c6`, exported
pristine to `/home/user/replay_src_aaa67c6`. The training-relevant files (`run_swap.py`, `run_dimensional.py`,
`nri/dimtask.py`, `nri/model.py`, `nri/train.py`) have no diffs between `aaa67c6` and HEAD. Recorded
environment: torch 2.15.0+cu132, the same as this container.

### Focus networks

All are `mass_only`: identical hyperparameters, different seed.

| role | seeds | historical OOD accuracy |
|---|---|---|
| focus | 307, 311 | 0.998 / 0.999 |
| comparators | 304, 310 | 0.263 / 0.272, despite ID 1.0 |
| intermediate | 305 | 0.711 |

### Qualification ladder

- **Q1 — plain replay.** Run the original `train_one`, unmodified, for 307, 311, 304 and 310.
  - **EXACT** if every `log.jsonl` line is byte-identical to the historical one **and** every tensor in the final
    state_dict is bitwise equal to `model.pt`.
  - **FUNCTIONAL** if not exact, but predictions are identical on the frozen ID and OOD evaluation sets, and
    acc_id / acc_ood agree to 4 decimals. Functional replay is labelled as such, and every later claim inherits
    the label.
  - Otherwise **FAIL**: the cause is diagnosed and reported, and no reconstruction proceeds.
- **Q2 — instrumented replay.** The reconstruction trainer adds checkpoint and audit hooks to an otherwise
  identical loop.
  - Checkpoints every 100 steps store model, AdamW state, OneCycle state, both data generators' states, and
    torch/python RNG.
  - Hooks may not consume any RNG used by training: evaluation uses its own generators, and the global RNG state
    is restored after each audit.
  - **Pass:** the instrumented run's final state_dict is bitwise equal to the plain replay's, for 307 and 311.
- **Q3 — resume.** Resuming from the step-3,000 checkpoint reproduces the final weights bitwise (307).
- **Q4 — reconstruction.** Begins only after Q1–Q3 pass (EXACT or FUNCTIONAL, stated). It covers all 36
  networks (3 groups × seeds 300–311), with complete checkpoints every 100 steps (61 per network, 2,196 total).
  - Each checkpoint is audited for:
    - ID and OOD accuracy, plus per-OOD-combination accuracy;
    - designed-block interchange accuracy;
    - M/A canonical decodability inside the block and outside it;
    - copy-into-block on OOD bases;
    - the route-specificity index;
    - gradient norms per parameter group;
    - the AdamW update decomposition.
  - Per-run compressed archives are written with SHA-256 manifests, and `tar --diff` is verified before any live
    copy is removed.
  - The endpoint of every reconstruction is compared with the historical `model.pt` (bitwise / functional, as
    above).
- **Q5 — cohort reviews.** Every 100 steps, a 36-network cohort review is generated from the audit data. Written
  scientific commentary is added at each phase transition (a change in ID or OOD rate, a declared-block
  formation event, a divergence between 307/311 and 304/310). Commentary is never written before reading the
  data (the doc-38 discipline).

## 41.7 Phase III.5 — why do some networks reach ≈ 1.0 OOD while comparable ones fail?

Retrospective only. The candidate explanations, each with a measurable signature in the reconstructed
trajectories:

- **G1 — timing:** OOD rises only if the declared block forms before the task is memorised.
  - **Measure:** the step at which designed-block IIA ≥ 0.9 relative to ID ≥ 0.95.
- **G2 — route-specific memorisation:** failing networks retain route × value structure for the OOD unit.
  - **Measure:** RSI at the encoder site, and the OOD-unit representation's alignment with the in-distribution
    units.
- **G3 — undeclared variable carriage:** generalisers compute A canonically too, outside the block.
  - **Measure:** A decodability and copy tests outside the block.
- **G4 — instability:** failing networks pass through loss spikes or episodes that destroy a forming shared code.
  - **Measure:** the episode counter (doc 37) on the 250-step logs and the 100-step audits.
- **G5 — initial conditions:** differences are present at step 0 (or by step 100–500).
  - **Measure:** early-checkpoint predictors (§ Phase IV).

**Generalization vs editability.** Across all 36 networks, OOD accuracy is related to the doc-15 edit
specificity and to copy-into-block, at the endpoint (Spearman, seed-clustered by group) and over training.

**What is not done here.** Any new training beyond exact reconstruction of these 36 histories (e.g. crossing
init seed × data seed to separate G5 from data order) is a **prospective** study. It is written as a separate
registration proposal (Phase IV) and requires owner approval before it runs.

---

# 41R — Results (appended after running)

Results are appended in stages. Sections marked *(final)* will not change. Sections marked *(interim)* will be
replaced when all runs complete.

## 41R.1 Instrument validation *(final)*

Output: `runs/instruments/planted_validation.json`, `runs/instruments/real_controls.json`.

### Planted mechanisms

- 3,074 cross pairs and 319 same-canonical pairs, with no reserved-test combination anywhere.
- **P1, P2, P4 and P5 passed** at the first run.
- **P3** (correct canonical code, trained-only reader) **failed** its pre-stated criterion on the *excess* metric:
  - held-out base | implied-trained excess was 0.88 against a criterion of 0.9;
  - cause: the unpatched wrong answer coincides with the canonical prediction in 12% of pairs, and the excess is
    bounded by 1 − baseline.

**Deviation 1.** A **switch rate** was added: among pairs whose unpatched prediction is not the target, the
fraction moved onto it. P3's criterion was restated on it. P3 then passed: 1.00 implied-trained, 0.00
implied-held-out. This change was made before any real-network result was read.

- The I6 control (same canonical donor) returned unchanged = 1.00 for P1 and P4. For P2, number-following was 1.00.
- **I4 is not pure.** For a number code (P2), the canonical-mean subspace transfers some number information
  (number excess ≈ 0.3), because canonical and number values correlate within the trained set. I4 is therefore
  interpreted only through its *canonical* excess, which is ≤ 0.017 for P2.

### Real-network controls

| control | networks | result |
|---|---|---|
| untrained (step 0), every condition and stratum | 4 networks | max \|excess\| = 0.020 (criterion ≤ 0.05): **pass** |
| x0 number-token swap | C wd 1.0 | number-switch 0.82–1.00 |
| x0 number + unit swap | C wd 1.0 | canonical-switch 0.52–0.92, bounded by model accuracy on the implied combinations |

## 41R.2 Probe validity for never-trained inputs *(final)*

| probe | fitted on | read on OOD inputs | known-positive case |
|---|---|---|---|
| ridge (linear) | ID inputs | 0.00 | `mass_only_s307`, OOD-source block interchange 0.994 |
| nearest centroid | ID inputs | 0.00 | same network |

Both probes read **0.00** on OOD inputs in the known-positive case.

**Deviation 2.** Probes are not used as evidence about never-trained inputs. They are kept in the audits as a
documented false-negative illusion. The validated measure is causal: block interchange with a never-trained
**source** (`iia_V_oodsrc`).

## 41R.3 Replay qualification *(final)*

Output: `runs/swap_recon/qualification.json`.

| step | test | result |
|---|---|---|
| Q1 | original code from aaa67c6, 304 / 307 / 310 / 311 | **EXACT**: every final tensor bitwise equal to the 2026-11-22 `model.pt`; all 24 log lines byte-identical |
| Q2 | instrumented trainer (checkpoint every 100 steps, update decomposition), 307 / 311 | **EXACT** |
| Q3 | resume from step 3,000 (307) | **EXACT**: final weights bitwise equal; the 12 post-resume log lines byte-identical |
| Q4 | reconstruction | every endpoint is compared bitwise with the historical `model.pt` (see 41R.6) |

## 41R.4 Mixed-code composers at 100-step resolution *(final for C wd 1.0)*

Measure: the validated I1 switch rate in the trained-base | implied-held-out stratum, every 100 steps.

- **Timing.** Canonical reading of never-trained implied combinations (switch ≥ 0.2 at two consecutive audits)
  begins **before** the registered compositional onset in 11/12 C wd 1.0 networks and at the **same** checkpoint in
  1 (seed 2002). It never begins after.
- **The three "mixed-code" networks** (2002, 2007, 2008) vs the other 9, steps 4,000–6,000:

  | | canonical-switch | number-switch | development held-out accuracy |
  |---|---|---|---|
  | mixed-code | 0.38 | 0.12 | 0.63 |
  | other 9 | 0.45 | 0.13 | 0.62 |

- **Conclusion.** The registered M1 "exceptions" came from the trained-context limitation of the registered
  measure, not from a different developmental order. This confirms doc 39 R4 at 5× finer resolution.
- **Note.** Seed 2002 composes well (development held-out 0.90) while reading the x1 code only moderately (0.39), and
  barely at x2 (I3 0.05). This points to a parallel pathway, consistent with R3's weak network-level correlation.

## 41R.5 Endpoint audit of the 36 historical networks *(final)*

Output: `runs/swap_recon/historical_endpoints_audit.json`.

Declared networks that fit the training data (ID = 1.0) split sharply on the validated causal measure:

| | seeds | OOD-source IIA | declared-variable OOD accuracy |
|---|---|---|---|
| generalisers | `mass_only` 307, 311 | 0.99 / 1.00 | 1.00 |
| failures | `mass_only` 304, 310 | 0.09 | 0.09 (chance 0.077) |

`acc_only` shows the same pattern (e.g. 308: 0.94; 310/311: 0.21/0.19).

## 41R.6 Why semantic supervision needs strong weight decay (H-A … H-E) *(final)*

Output: `runs/instruments/mechanism_sswd.json`, from `sci/mechanism_sswd.py`. The operationalisations were fixed
in the module docstring before running.

**Bug fix (not a change of definition).** A and B have no auxiliary head, so the H-C cosine is absent for them.
It is now read as optional. H-C concerns only C and D.

### Primary contrasts

C wd 1.0 vs C wd 0.1, 12 paired seeds, exact sign-flip, Holm over the five hypotheses.

| | measure (number token, after block 1) | C1.0 | C0.1 | D1.0 | B1.0 | d (as predicted) | 95% CI | p_Holm | seeds | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| H-A | participation ratio of the 120 trained class means | 3.70 | 4.64 | 4.43 | 5.77 | +0.94 | 0.73–1.16 | **0.0012** | 12/12 | supported |
| H-B | route-specificity index (1 − R² on canonical q) | **0.078** | 0.179 | 0.805 | 0.650 | +0.10 | 0.086–0.117 | **0.0012** | 12/12 | supported |
| H-C | task–aux gradient cosine, steps 0–2,000 | −0.014 | −0.006 | −0.011 | — | −0.008 | −0.021–0.004 | 0.89 | 5/12 | **not supported** (gradients orthogonal throughout, as in doc 38) |
| H-D | I1 canonical switch, implied-held-out, at decodability onset | 0.185 | 0.133 | 0.049 | 0.058 | +0.052 | −0.006–0.131 | 0.17 | 8/12 | **not supported at onset** (see trajectory) |
| H-E | total parameter norm at 6,000 | 19.2 | 66.0 | 21.0 | 18.3 | +46.8 | 46.3–47.2 | **0.0012** | 12/12 | supported but **not explanatory** |

- **H-E.** Strong decay shrinks the norm in every population (B1.0 18.3, A1.0 16.9), including those that do not
  compose. The norm is a consequence of the decay setting, not specific to the interaction.
- **The H-E "knee" is degenerate.** The maximum total norm is at step 0 in every network, so "knee before onset
  12/12" carries no information. It is reported as measured; the definition was not changed after seeing the data.

**Secondary contrasts** (descriptive):
- H-A and H-B separate C1.0 from D1.0 and from B1.0 (p ≤ 0.004).
- H-D separates C1.0 from D1.0 and B1.0 (+0.13, p = 0.001).

### Trajectories (means over 12 seeds)

| step | RSI C1.0 | RSI C0.1 | RSI D1.0 | RSI B1.0 | I1 switch C1.0 | I1 switch C0.1 |
|---|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0.00 | 0.00 |
| 1,000 | 0.17 | 0.27 | 0.78 | 0.73 | 0.14 | 0.10 |
| 2,000 | 0.11 | 0.24 | 0.78 | 0.71 | 0.22 | 0.11 |
| 4,000 | 0.09 | 0.20 | 0.80 | 0.64 | 0.41 | 0.12 |
| 6,000 | 0.08 | 0.18 | 0.81 | 0.65 | 0.47 | 0.14 |

### Account (retrospective, exploratory)

1. **The coherent semantic target builds the code.** In both C conditions, the canonical auxiliary target turns
   the number-token code from route-specific into canonical within about 1,000 steps (RSI 0.75 → 0.17–0.27). Without
   a coherent target (B, or the entropy-matched scrambled D), the code stays route-specific even under strong
   decay (0.65–0.81).
2. **Strong decay then removes the residual route-specific structure** (0.17 → 0.08; under weak decay it plateaus
   near 0.18–0.24). It also compresses the code (participation ratio 3.7 vs 4.6).
3. **The downstream reader becomes a reusable conversion only under strong decay.**
   - Its canonical reading of never-trained implied combinations grows steadily from 0.14 to 0.47 under strong
     decay, and stays flat at 0.10–0.14 under weak decay.
   - At the onset of decodability the two conditions barely differ (H-D). The difference develops over the
     following thousands of steps.
4. **Gradient cooperation is not the mechanism (H-C).** The task and auxiliary gradients stay orthogonal, so the
   auxiliary loss does not pull the task pathway. It shapes the representation, and decay selects the reader that
   reuses it.

**Consistency with the literature.** This fits the efficiency account of weight decay (Varma et al. 2023): a reader
that reuses one canonical code is cheaper in parameter norm than a route-by-route lookup, but only once a shared
code exists to be reused.

**Within-condition limits.**
- Across the 12 C1.0 seeds, neither RSI (ρ = −0.05) nor PR (ρ = −0.08) predicts development held-out accuracy.
- These measures separate *conditions*, not *seeds*. What distinguishes better from worse composers within C1.0
  remains open; the strongest within-condition correlate is the final parameter norm (ρ = −0.52, n = 12).
