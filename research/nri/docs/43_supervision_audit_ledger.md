# 43 — Supervision audit ledger (Birth of a Concept, docs 35–42)

Every methodological decision, deviation, defect and correction in the program, with:
- **who** made it: O = owner, A = supervising agent;
- **when**, relative to the data it concerns: **pre** = before the data were seen, **post** = after;
- its **effect** on the claims.

**Provenance.** The supervising agent is Claude Code running in a cloud container. No specific model version is
inferred from attribution metadata.

**Purpose.** To let a reader check that no registered hypothesis, threshold or endpoint was changed after its data
were seen, and that every post-hoc change is labelled.

| # | date (2026) | item | by | pre/post | effect on claims | where |
|---|---|---|---|---|---|---|
| 1 | Nov | Pilot scorecard defect: held-out patch used trained combinations → `novel_combo_patch` | A | post (pilot) | pilot only; fixed before registration | doc 36 |
| 2 | Nov | Pilot: patch assumed number semantics → `patch_semantics` (number vs canonical) | A | post (pilot) | pilot only | doc 36 |
| 3 | Nov | Pilot: A's practiced score measured on others' combinations → `own_practiced` | A | post (pilot) | pilot only | doc 36 |
| 4 | Nov | Checkpoint requirement: full state every 100 steps (weights, optimizer, RNG, update decomposition) | O | pre (factorial) | enabled all retrospective work | doc 37 |
| 5 | Nov | `aux.*` parameter-name collision corrupted resume → `auxhead.*` names + uniqueness assert; name-sorted digest | A | pre (factorial) | none (caught by resume test) | doc 37 |
| 6 | Nov | Registration of populations, endpoints, H1–H5, M1–M4, onset thresholds (commit 1b54b0b) | A, O-approved | pre | defines confirmatory scope | doc 37 |
| 7 | Nov | Wrong `.gitignore` path committed 68 checkpoint files (37829c5); untracked in fa8ec10 (still in history) | A | — | none on results; repository size | git |
| 8 | Nov | Cohort-600 review written before reading output; corrected; rule: never write commentary before reading the data | A | during | review text only | reviews/ |
| 9 | Nov | Truncated output hid seed 2012's onset; onsets now recomputed directly | A | during | corrected | doc 38 |
| 10 | Nov | Doc-38 number errors corrected: new-frame 0.44→0.37; C ≥ 0.5 on 15/18 (not 16); 11/257 audits (not 10) | A | post | corrected before PR review | doc 38 |
| 11 | Nov | Owner review: `patch_semantics` uses trained contexts only; M2 ignores clustering | O | post | causal claim was limited to trained contexts; M2's precision overstated | doc 39 |
| 12 | Nov | Plan for held-out causal test + clustered M2 committed before running (51c546e) | A | pre (doc 39) | — | doc 39 |
| 13 | Nov | Doc-39 step-0 control failed (0.13–0.18) → excess over the no-intervention baseline | A | post (control) | reported as a deviation; all doc-39 rates are excess | doc 39 |
| 14 | Nov | "Archive preservation is the highest priority; do not launch training" | O | — | training paused; 96 archives delivered; owner verification pending | PRESERVATION.md |
| 15 | Nov | Independent re-derivation of all registered results from archived weights (separate code) | A | post | no discrepancy | doc 40 |
| 16 | Nov 25 | Phases II–III authorised with owner-side verification still pending; originals untouched | O | — | scope of this work | doc 41 |
| 17 | Nov 25 | Protocol doc 41 (instruments, H-A–H-E, replay ladder) committed before running (0b7c6b1) | A | pre | — | doc 41 |
| 18 | Nov 25 | Planted P3 failed the excess criterion (0.88): excess is bounded by 1 − baseline → switch rate added; P3 criterion re-stated on the switch rate | A | post (validation) | validation-driven instrument change, before any real-network result was read | doc 41R |
| 19 | Nov 25 | Probes fitted on ID inputs read 0.00 on OOD for networks whose block causally carries OOD values (0.99) → probes flagged invalid for OOD; causal OOD-source IIA used instead | A | post (validation) | probes not used as evidence about OOD | doc 41R |
| 20 | Nov 25 | Replay ladder Q1–Q3: EXACT before any reconstruction was launched | A | pre (Q4) | reconstruction claims carry the EXACT label | qualification.json |
| 21 | Nov 25 | Prospective experiments (doc 42 §42.2) written as proposals only; none run | A | — | requires owner approval | doc 42 |

## Invariants checked

* No registered hypothesis, threshold, endpoint or analysis in doc 37 was changed after training began. The
  reserved test was evaluated once (doc 38). Doc 40 re-derived it from weights with no discrepancy.
* No seed was excluded in any analysis.
* Exploratory results stay labelled: docs 39 and 41R are retrospective and exploratory.
* No original trajectory, checkpoint or measurement was modified. Archives and extracted weights are read-only.
  Their SHA-256 values are re-verified at the end of Phases II–III (doc 41R).

## Additions (2026-11-25, second session block)

| # | item | by | pre/post | effect | where |
|---|---|---|---|---|---|
| 22 | Owner corrections to the generalisation standard: partial vs near-perfect (≥ 95% exhaustive + causal ≥ 90%); n = 12 cannot give a 0.80 lower bound | O | — | doc 42 rev. 1; 108-network design withdrawn; about 50 per certified condition | doc 42 |
| 23 | Critical-window replay registered before running (doc 45); bitwise qualification at steps 1,000 and 2,000 passed for all 5 | A | pre | 705 states at 10-step resolution | doc 45 |
| 24 | Mechanism battery: attention route and computation-first ordering contradicted as explanations | A | post | reported as contradicted candidates C4/C5 | doc 46 |
| 25 | Class-mean edit construction erased context (minimum-norm least squares) → in-span edit used; the original construction is documented | A | post (method) | edit ladder uses in-span maps | doc 46 §46.3, `sci/edit_ladder.py` |
| 26 | Subspace edits: full 13-point fit is uninformative (R² = 1 by construction) → held-out-class fit | A | post (method), before reading results | geometry-linearity measure | `sci/subspace_math.py` |
| 27 | P4 per-part edit: prediction stated in doc 46 §46.5 before the measurement | A | pre | supported for factorised networks; "reverse in failures" not supported | doc 46R.2 |
| 28 | H-C cosine absent for A/B: bug fix, not a definition change | A | post | — | doc 41R.6 |
| 29 | H-E knee degenerate (max norm at step 0): reported as is, not redefined | A | post | — | doc 41R.6 |
| 30 | Step-0 control fails in the n = 33 descriptive stratum (24/2,304 cells; \|excess\| 0.061) | A | post | reported; stratum not used for claims | doc 41R.7 |

## Scientific decision ledger: observed / inferred / demonstrated

| claim | status |
|---|---|
| Registered factorial effects H1–H5 | **demonstrated** (registered, reserved test, independently re-derived) |
| Historical networks replay bitwise (36/36; window; resume) | **demonstrated** (exact reproduction) |
| Canonical code converted along never-trained combinations (C1.0) | observed (validated instrument; retrospective) |
| Present / readable / controlling separation | observed |
| Factorisation kept vs lost during the escape separates generalisers from failures | observed (24 networks); **causal necessity during development demonstrated** by registered interventions in 307 and 311 (superseded row below) |
| Strong decay prunes route-specific structure; the reader becomes reusable | observed (condition level); in role-swap, P3 (decay → 1.0 from step 2,000) helped only marginally (304: 0.25) — the factorial mechanism does not transfer as a rescue |
| Per-part edit succeeds in factorised networks | observed (single-pass interventions on fixed networks) |
| Head composition is a second requirement (C2) | inferred from observed partial cases (300, 305) |
| Cross-network transplant of the canonical code | observed |

## Additions (Phase IV and Phase V)

| # | item | by | pre/post | effect | where |
|---|---|---|---|---|---|
| 31 | Phase IV registration (P1–P3, rescue, reverse, programming) committed before any run | A | pre | doc 48 | doc 48 |
| 32 | P3 planted-mechanism criterion failed on excess (bounded by 1 − baseline) → switch rate added | A | post (method) | deviation 1 of doc 48; both measures reported | doc 49 |
| 33 | Addendum A (earlier starts for 304/310; extra break variants) registered before its runs | A | pre | 6 runs added; prediction "earlier rescues more" supported for 310 | doc 48 addendum A |
| 34 | All 17 matched controls bitwise identical at 20/20 checks | A | — | interventions are the only difference from history | doc 49 §49.4 |
| 35 | Registered rescues R1–R5 failed → R6, R7, R7L, R7A added **after** registration | A | post | labelled post-registration; 302 repair (0.93) is a post-registration result | doc 49 §49.1 |
| 36 | `compiled_value_edit` failed (≤ 0.44); a mathematical impossibility for a one-step linear form was shown | A | post | kept as a negative result | doc 49 §49.6 |
| 37 | Phase V registration (D1–D7) committed before any run | A | pre | doc 51 | doc 51 (664a6c7) |
| 38 | D1 registered synthetic construction failed its fidelity check (trained cells 0.33–0.89 except 307) → exact-context variant used; logged **before** interpreting classes | A | post (method) | doc 52 deviation 1; both outputs kept | doc 52 §52.1 |
| 39 | D4 explicit programs (single-frequency, additive, pair, per-order) all failed full-domain validation | A | post | negative result; no compact program claimed | doc 52 §52.3 |
| 40 | D5 transplants all at chance; no qualifying repair | A | post | negative result | doc 52 §52.5 |
| 41 | D6 constructed decoder not built (its precondition, an identified program, failed) | A | — | recorded as open | doc 52 §52.7 |
| 42 | D2/D3/D5 run as inline drivers, later recovered into `sci/phase5_drivers.py`; D5 re-run byte-identical | A | post | reproducibility | doc 52 §52.8 |
| 43 | Archive manifests for reconstruction (36) and window (5) committed; delivered copies; owner check **pending** | A | — | not marked complete | `runs/swap_recon/archives/PRESERVATION.md` |

## Ledger after Phase V: observed / inferred / demonstrated

| claim | status |
|---|---|
| Rewarding the interaction during the window destroys generalisation (307, 311) | **demonstrated** (registered causal intervention, bitwise controls; one run per network) |
| After consolidation factorisation is not necessary (311 keeps 0.92 at factorisation 0.15) | **demonstrated** for 311; 307 not testable (the interaction did not form) |
| Preventing the interaction before lock-in partly rescues failures (0.52–0.61), later does not | **demonstrated** (registered); rescue criterion not met |
| Failures' decoders are intact for canonical inputs; representation missing for unseen pairings | observed (oracle diagnostic) |
| Training-free repair of 302 (0.19 → 0.93) | observed, post-registration method, no oracle |
| Token-level and output-level programs compile exactly, compose and invert | **demonstrated** (mathematical identities, verified over the full domain) |
| Execution of programs on unseen inputs is bounded by generalisation (ρ = 0.96) | observed (7 networks) |
| 307's decoder is algorithmic; 300's interpolates | observed (deviation-1 construction) |
| The composing decoder forms gradually, together with fitting (307) | observed |
| Generalisers multiply by adding per-token discrete-log phases at one frequency coprime to 12 | observed structure (R² 0.90/0.92); the algorithmic reading is **proposed**; no exact program |
| Decoder transplant or construction | not achieved |

## Additions (Phase VI)

| # | item | by | pre/post | effect | where |
|---|---|---|---|---|---|
| 44 | Phase VI registration (derivations, designs, Q-criteria, predictions) committed before any evaluation | A | pre | doc 53 | doc 53 (446bbeb) |
| 45 | Designs C and L passed every Q-criterion on first build; float32 activations bitwise symbolic | A | — | "exactly programmed" claimed on Q-exact, not accuracy | doc 54 §54.2 |
| 46 | `unit()` contained a dead destination computation; removed and re-run, identical results | A | post (code) | — | `sci/instructions.py` |
| 47 | Residual prediction partly unsupported: effect not via ReLU-boundary cells | A | post | reported as is | doc 54 §54.4 |
| 48 | Per-pair split of the pairwise residual family added after R2 (exploratory refinement) | A | post | labelled exploratory | `residual_pairs.json` |
| 49 | H2 "wrong-code" control invalid (column permutation is absorbed by the affine fit); replaced by random regrouping before interpreting H2 | A | post (method) | deviation 3 | doc 54 §54.7 |
| 50 | H1 and H2 fail the repair criterion; H1's random-encoder control beats every learned encoder | A | post | negative results kept; double dissociation only directional | doc 54 §54.7 |
| 51 | Compiler exhaustive cap lowered 3·10⁶ → 1.2·10⁶ prompts for runtime | A | post (method) | deviation 2; larger domains sampled and labelled | doc 54 §54.6 |

## Ledger after Phase VI

| claim | status |
|---|---|
| The task's computation can be compiled into the existing architecture without training, exactly | **demonstrated** (exhaustive, float32 bitwise-symbolic; real-arithmetic bound < 10⁻¹⁰⁰) |
| Internal operators (6), unit conversions (60), binding, decoding compile exactly, independently, reversibly, composably | **demonstrated** (exhaustive) |
| The compiler generalises across primes, operand counts, unit scales and expressions | **demonstrated** for the grid (exhaustive ≤ 1.2·10⁶ prompts, sampled beyond) |
| One ReLU layer cannot add phase vectors exactly; integer logs plus hats can | mathematical (D0, D1) |
| The learned decoders' residual is essential readout structure made of pairwise token interactions | observed (7 networks) |
| Generalisers' M code is factorised, 2-D and interchange-equivalent to the construction, but circular | observed |
| Constructed parts repair learned networks | **not supported** |
| Constructed networks are exact but fragile; learned ones robust | observed |

## Additions (Phase VI addendum)

| # | item | by | pre/post | effect | where |
|---|---|---|---|---|---|
| 52 | Addendum registered (doc 55) before measurement; Anthropic posts not readable (network policy), only owner's summary used | A | pre | citations flagged unverified | doc 55, CITATION_CHECKS |
| 53 | Validation: S-lens (V4) and averaged-lens sectors (V5) failed on constructed nets → not used for learned-net interpretation | A | pre-interpretation | measures reported but excluded | doc 56 §56.1 |
| 54 | V3 count criterion mis-specified (omitted competitor hats) | A | post | support criterion passes; count reported | doc 56 §56.1 |
| 55 | N6(a) failure traced post hoc to the direct residual path (margin metric hid it); extra measurement `direct_path.json` | A | post | reported as finding | doc 56 §56.5 |
| 56 | Predictions not supported: variance ordering not worst; failures also purify writes | A | post | reported | doc 56 §56.3, §56.6 |

## Additions (Phase VII)

| # | item | by | pre/post | effect | where |
|---|---|---|---|---|---|
| 57 | Phase VI dark-sector statements marked provisional before Phase VII | A | pre | doc 56 status note | doc 56 |
| 58 | Phase VII registration (doc 57) and addendum A (the exact H/K configuration) committed before runs | A | pre | — | doc 57 |
| 59 | Clarification before any training run: the constructed heads' full q/k/v rows are frozen (not only columns 0–63) | A | pre | documented in `sci/hybrid.py` | doc 58 |
| 60 | I2 failed V5′ narrowly (2.25 against ≤ 2) and I3 failed; per the rule, neither was applied to learned networks; the dark residual is unresolved | A | post | criteria not relaxed | doc 58 §58.2 |
| 61 | Background jobs were stopped at the 30-minute and 2-hour harness limits; incomplete runs were resumed with skip-completed (no completed run was rerun or replaced) | A | post (operational) | 48/48 hybrid runs and all robustness variants complete | doc 58 |
| 62 | Condition A first build had an undersized stage-2 ladder (an IndexError before any evaluation); fixed | A | pre-eval | A exact | `sci/hybrid.py` |
| 63 | Predictions not supported: D2 "partial" (it was complete); R1+R2+R3 robustness target; hybrid data-efficiency at 25% | A | post | reported | doc 58 |

## Ledger after Phase VII

| claim | status |
|---|---|
| Computational graphs compile exactly, with observable, interchangeable, programmable intermediates; cost linear in nodes | **demonstrated [E]** (4 named + 60 random expressions) |
| Fourier (periodic) construction is exact | **demonstrated** (real arithmetic; float32 argmax-exact [E]) |
| Interventional and integrated-gradient S-lenses are valid | **demonstrated** on constructed networks (V4′) |
| Exact per-prompt Jacobians predict ablations in learned networks; averaged lenses under-predict 4–8× | observed [T] |
| Constructed networks are fragile because of saturated sparse routing; balanced minimal gains help about 10×; geometry, margins and readout redundancy do not | observed [S] |
| Compiled and learned computation coexist with residual-subspace protection (full data) | **demonstrated [T]**, 3 seeds × 2 conditions, exact sub-circuit at every evaluation |
| Protection needs both write-blocking and output-level preservation when data are scarce | observed [T] (25% regime) |
| Hybrids learn the unknown operation from less data | **not supported** at 25% |

## Additions (Phase VIII)

| # | item | by | pre/post | effect | where |
|---|---|---|---|---|---|
| 64 | Phase VII effective trainable count was overstated (191,516 counted the 38,656 write-blocked entries); correct value 152,860 | A | post (audit) | corrected in doc 60 §60.1; doc 58 not altered | doc 60, `runs/phase8/accounting.json` |
| 65 | `spec_terms('lin')` bug (constant only for one force unit) found during MR verification; fixed and re-run before reporting | A | pre-report | compiled R4 exact | `sci/modular.py` |
| 66 | G2 select test with a 3-valued S operand failed; operands must be 13-valued (assert added); case changed to A; suite re-run | A | pre-report | 3/3 select cases exact | `sci/graph2.py` |
| 67 | E-modular coverage key collision (`M*100+A*10+fu`); fixed and re-run (sampler unaffected) | A | pre-report | coverage figures corrected | `sci/p8_q.py` |
| 68 | Not supported / not met: P matches K5 on OP₂; R2 robustness target (0.52 vs 0.95); MR registered extraction criterion (0.81 vs 0.99); 25% learning | A | post | reported as failures | doc 60 |
| 69 | MR trained-pairs-only extraction is post hoc; no automated-discovery claim | A | post | labelled post hoc | doc 60 §60.8 |
| 70 | Phase VII hybrid results reproduced bitwise (11/11 runs) | A | pre-analysis | audit passed | `runs/phase8/audit/comparison.json` |
| 71 | **Flag for owner review (Track B, CPU-01, outside this phase).** The historical role-swap zoo reproduces bitwise on this machine (mass_only s307 replay = historical `model.pt`). But rounding-level trainer changes (gradients equal to ~1e-7 at the same weights) altered per-seed outcomes: mass_only s307/s311 OOD 0.998/0.999 → 0.14–0.79 across fast-path variants; the neither arm was stable. The s307/s311 outcomes studied as a "critical window" (doc 46 and following) are therefore reproducible but not robust to numerics; seed-specific mechanistic claims should be read as describing rare trajectories, and arm-level claims should rest on population statistics. No historical document altered. | A | post | owner approved committing the report (25 Nov 2026); manuscript distinguishes the reproducibility tiers | doc 61, `trackb/` |
| 72 | Track C report (owner-supplied, external, package not verified) independently reproduced Phase VI/VII constructions at `f5ab660` (9,126/9,126; R1 gain 0.02371374) and reports a new negative result: three binding edits fail on top of the rebalanced R1 | external | post | cited as corroboration, unverified | doc 60 §60.7 |

| 73 | E structured-sampler cell key collides across operations (`op*10000` vs `M*1000`, M ≤ 12): 1,014 cells merged into 897; runs reported as executed, coverage measured post hoc | A | post | structured coverage 0.98–0.99 rather than 1.00 | doc 60 §60.6, `sci/p8_ecov.py` |
| 74 | E group stopped by the 2-hour background limit at 40/48 runs; resumed from checkpoints, no completed run re-run | A | post (operational) | 48/48 complete | doc 60 |
| 75 | E prediction "25% failure = canonical-coverage failure" not supported for the transformers (coverage ≥ 0.96 at 25%; training accuracy 1.00 with familiar 0.44–0.75 = memorisation); supported for E-modular. Read-restriction interpretation is post hoc | A | post | reported | doc 60 §60.6 |

| 76 | Phase VIII approved as complete by the owner (25 Nov 2026); PR #36 kept unmerged | owner | — | — | — |
| 77 | Independent-track updates recorded as reported, **unverified**: Track C decoder continuation (ternary and factored-binary decoders at 100% under 1% additive / 5% multiplicative noise, 20 seeds; factored decoder 8,125 parameters versus 33,920); Track A MM-02AC (≈ 4.39% nominal savings, slower than GEMM); longitudinal 5M-parameter reuse experiment (primitive preserved bitwise; transfer on familiar but not new states) | external | pre | manuscript robustness conclusions unchanged pending §62.V | doc 62 §62.0 |
| 78 | Phase IX registration (doc 62) drafted and committed before any Phase IX run; awaiting owner approval; Track C package not yet available to this session | A | pre | — | doc 62 |

## Ledger after Phase VIII

| claim | status |
|---|---|
| Phase VII hybrid results | **reproduced bitwise**; trainable count corrected to 152,860 |
| Hybrid advantage over parameter- and compute-matched baselines | observed [T]: 3/3 vs 1/3 competent; ~3–15× fewer training FLOPs to competence |
| The advantage comes from the canonical semantic variables (S3), not the compiled arithmetic | observed [T] (S ablation, 3 seeds) |
| Compiled correctness guaranteed for all trainable parameter values | **demonstrated [E, M]** with output ownership; fails without it (0.07) |
| Structural protection learns the new operation as well as K5 | **not supported** (1/3 vs 3/3 competent) |
| Retention of all earlier computations across stages | **demonstrated [E]** (bitwise) |
| Fast, compositional learning of combinations of compiled operators | observed [T] (registers visible) |
| Generalising an unknown (non-compositional) relationship | **not supported** (memorisation) |
| Extraction under the registered criterion | **not met**; post-hoc success labelled |
| Robust distributed routing at 1% multiplicative noise | **not met** (0.52) |
| Learning a related operation from 25% data | observed [T] with a protected register and read-restricted module (≤ 25%); protected transformers need ≥ 55% (K5s) or never qualify (P, B) |
| CSE / register allocation / shared operators / select compile exactly | **demonstrated [E]** (63 trees × 4 variants; loop n ≤ 6) |
