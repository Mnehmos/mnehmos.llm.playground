# 50 — Handoff: Birth of a Concept, state at 2026-11-25 06:10 UTC

**Purpose.** This is a self-contained snapshot for whoever continues this work: what is established, what is only
observed, what is running, and what is decided or pending.

**Where things are.**
- Branch: `claude/awesome-archimedes-9foebw` (draft **PR #33**), stacked on PR #32. Neither is merged.
- Independent study: **PR #34** (ChatGPT Work, `codex/generalization-seed-cross`). Read-only for us; its crossing
  results are not yet posted.
- **Owner-side verification of the 96 factorial archives is still pending.** The originals are intact and re-verified
  96/96 OK at the end of Phases II–III (doc 41R.8). The container restarted at about 05:22 UTC; re-run
  `verify_archives.py` before relying on them.

**Labels.**
- **[R]** registered or confirmatory;
- **[X]** exact (bitwise or mathematical identity);
- **[O]** observed: retrospective, exploratory measurement;
- **[C]** causal intervention in training, exploratory, registered in doc 48;
- **[I]** inferred.

---

## 1. The questions and short answers

| question | answer so far |
|---|---|
| What makes small networks learn reusable computation instead of a lookup? | Two different routes were found. **Factorial:** a coherent semantic target plus strong weight decay [R]. **Role-swap:** a declared representation that stays *factorised* (value part + unit part), plus a head that *composes* the parts [O, C]. |
| Why do identically configured networks split into perfect and failed generalisers? | During a few hundred steps, as training accuracy escapes a shared plateau, failures build a value×unit *interaction* (a lookup over trained pairs) into the declared block. Generalisers keep it factorised [O]. Rewarding the interaction during training breaks a generaliser [C]. Preventing it early partly rescues a failure; preventing it later does not [C]. |
| Can a memorising network be repaired without retraining? | **Yes for one of three:** 302, 0.19 → 0.93 unseen, familiar 1.00, no oracle. **Partly** for 310 (0.54). **No** for 304 (0.30). Removing the interaction, the obvious surgery, makes things worse [O]. |
| Can networks be programmed directly? | **Yes, exactly, for token-level edits.** Compiled into embedding rows, reversible bit for bit, composable [X]. Execution on never-trained inputs is perfect in generalisers and degrades with generalisation: ρ = 0.96 across 7 networks [O]. Mass-only value edits cannot be compiled into one linear change, and approximations failed [X, O]. |

---

## 2. Established results

### 2.1 Registered factorial (96 networks; docs 37, 38, 40) [R]

- Canonical-quantity supervision promotes compositional generalisation **only with strong weight decay**: reserved
  test 0.655 vs 0.101 for the scrambled target.
  - H1 +0.55; interaction H2 +0.48; all five contrasts Holm p ≤ 0.0015.
  - Independently re-derived from archived weights with no discrepancy.

### 2.2 Exact replay [X]

- All 36 historical role-swap networks replay bit for bit: 2,196 checkpoints, archived and verified.
- The instrumented runs, the resume from step 3,000, and the 10-step window replay of steps 600–2,000 (5 networks,
  705 states) are all bitwise.
- Every training-intervention control below is bitwise identical to the reconstruction, so the interventions are the
  only difference.

### 2.3 Exact semantic programming (doc 49 §49.6) [X]

- A token-level edit ("read token t as t′") compiled as embedding-row replacement is an identity of functions.
- It applies to:
  - written values, which act on both quantities because the value tokens are shared;
  - mass, acceleration and output units, which have role-specific tokens.
- Sequences compose exactly, and permutations invert bit for bit.
- **Impossibility:** no token-level program can repair a failure while preserving meaning, because the required unit
  scales (12; and 5, 10, 9 for acceleration) do not exist.

---

## 3. Observed and causal findings (exploratory)

### 3.1 Factorial mechanism (docs 41R, 46R) [O]

- **Validated instruments.** Seven causal intervention conditions were validated on five planted mechanisms.
  - Probes fitted on familiar inputs can read 0.00 on unseen inputs that the network demonstrably processes.
- **Weight decay mechanism.**
  - The semantic target builds a canonical code under either decay.
  - Strong decay prunes route-specific structure (0.18 → 0.08) and makes the reader reusable (0.14 → 0.47 on
    never-trained combinations).
  - Gradient cooperation was ruled out.
- **Present, readable and controlling dissociate.**
  - In the composing condition, the code is present and readable at step 1,000 (0.87) but controls only 0.14.
  - About 4 of 12 dimensions do the causal work.
  - Linear edits generalise across classes only for multiplicative edits (invert 0.78 vs add-one 0.06).
  - Codes transfer across networks by rotation (0.41 vs 0.47 within a network).

### 3.2 Role-swap: what separates generalisers from failures (docs 45, 46) [O]

- **Factorisation.** Declared-block factorisation R²(value + unit) ≥ 0.98 is **necessary** for near-perfect
  generalisation in 24/24 declared networks, but **not sufficient**: 300 and 305 are partial.
- **Developmental pattern.**
  - Failures lose factorisation during their escape (304: 0.98 → 0.71; 310: 1.00 → 0.68) and freeze.
  - Generalisers dip and recover.
- **No predictor before the window:** none of the measures at steps 0–1,000 tested.
- **Not the cause:**
  - attention routes, which differ between failures;
  - which loss falls first.

### 3.3 Three distinct defects (doc 49 §49.1, §49.3) [O]

| group | representation | decoder (head MLP) |
|---|---|---|
| generalisers 307, 311 | factorised, correct for unseen pairings | composes: correct even for unseen cells far from any trained twin |
| partial 300, 305 | factorised, correct for unseen pairings (error ≤ 0.26) | **interpolates**: correct only for unseen cells near a trained twin |
| failures 302, 304, 310 | **canonical code built by an interaction; missing for unseen pairings** | intact for canonical inputs: oracle placement gives 0.71–0.93 |

### 3.4 Rescue and break without training (doc 49 §49.1, §49.5) [O]

- **Removing the interaction makes failures worse:** familiar accuracy falls to 0.33–0.40. Their decoder depends on
  the interaction.
- **Class completion rescues 302 at the registered criterion.** It uses training labels, the bijectivity of unit
  rescaling, and the network's own geometry (0.93 / 1.00). This method was added after the registered methods failed.
- **Injecting interaction into generalisers breaks them dose-dependently.** Placed on unit-2 cells it hurts unseen
  accuracy about 2× more than familiar (0.45 vs 0.85).

### 3.5 Training-time interventions (doc 48 E4; `/home/user/dev_intervene`) [C]

| network | resume | intervention | unseen at end (control) | familiar | reading |
|---|---|---|---|---|---|
| **307** | 850 | **reward interaction** (λ = −1) | **0.35 (0.99)** | 1.00 | **generalisation destroyed, training accuracy kept**: meets the registered disruption criterion; factorisation 0.03 |
| 304 | **600** (before its dip) | penalise interaction, 500 steps | **0.52 (0.10)** | 1.00 | partial prevention |
| 304 | 700, 850, 1,000, 1,500, 2,000, 4,000 | penalise | 0.10 (0.10) | 1.00 | ineffective once the dip has begun, even with factorisation restored to 0.99 |
| 310 | **1,350** (before its dip) | penalise, 500 steps | **0.55 (0.10)** | 1.00 | partial prevention |
| 310 | 1,450 | penalise, 500 steps | 0.22 (0.10) | 1.00 | mostly ineffective |
| 310 | 2,000 | penalise | 0.34 (0.10) | 1.00 | small |
| 302 | 2,000 | weak penalty (λ = 0.1) | **0.52 (0.19)** | 1.00 | prevents post-fit erosion |
| 304 | 2,000 | weight decay → 1.0 | 0.25 (0.10) | 1.00 | small; the code becomes *less* factorised (the factorial's other route) |

**[I] Reading:**
- **Factorisation is causally *necessary*** for generalisation in 307: breaking it during the window destroys
  generalisation while training accuracy is kept.
- **It is not sufficient:**
  - preventing the interaction from the start of the window yields only partial generalisation (0.52–0.55);
  - restoring it after lock-in yields none.
- **Path dependence:** the timing matters. Each failure has a point, at or before its dip onset, after which
  prevention no longer helps.
- **The missing ingredient is the composing decoder.** How the head comes to compose rather than interpolate is the
  main open question.

### 3.6 Population [O]

- Programmability on never-trained renderings correlates with generalisation at ρ = 0.96 (7 fitted mass-declared
  networks).
- Generalisation correlates *negatively* with editability under class-mean edits (ρ = −0.55, 24 networks). A per-part
  edit reverses this: 307 is edited at 1.00 on seen and unseen inputs.

---

## 4. Contradictions and open questions

1. **C1 (interaction lock-in as *the* cause) is half right.** It is necessary (P2a) but not sufficient (P1). A second
   mechanism, the decoder's composition, is required, and nothing measured predicts it yet.
2. **304's unit-2 rotation** is consistent on trained values but extrapolates to the wrong classes (geometric
   completion 0/4). Unexplained.
3. **Two generalising routes:**
   - factorial: a canonical code with a reusable reader, under strong decay;
   - role-swap: factorised parts with a composing head.

   Which one a network takes depends on supervision and regularisation; the conditions are not mapped.
4. **Representations and decoders are network-specific in the role-swap task.** Affine transplants fail even on
   familiar inputs, yet in the factorial a rotation transfers a low-dimensional code.
5. **Value edits for one variable** (value tokens shared by mass and acceleration) are not compilable into a single
   linear change. A permanent version of the successful per-part activation edit does not exist yet.

---

## 5. Running at handoff time

- `python -m sci.dev_intervene /home/user/dev_intervene --workers 4`: the registered E4 set, 30/43 done.
  Remaining include:
  - **P2a on 311**, the replication of the break;
  - **P2b** (interchange weight ×3) and **P2c** (break after consolidation, step 2,000);
  - **F1** (freeze attention-2 q/k) and **G1** (gradient projection).
- `... --only ...addendum A`: earlier start points, 2/6 done (304 from 600 is done; 310 from 1,200 and 1,300 are
  pending).
- **Both resume safely:** each run writes `result.json` only when complete, and the driver skips completed runs.
  Re-launch with the same command after a restart.

**To finish:**
1. Collect the final table: `python` over `/home/user/dev_intervene/*/result.json`.
2. Replace doc 49 §49.4 (marked interim).
3. Check P2a on 311 against the disruption criterion (unseen ≤ 0.5, familiar ≥ 0.95).

## 6. Pending decisions for the owner

1. Whether to approve a registered study of the **decoder-composition** question: why a head composes or
   interpolates. This is now the central open mechanism.
2. **Reliability design** (doc 42 rev. 1): about 50 networks per certified condition. Still deferred, as instructed.
3. **Registry proposals** (doc 47; to be updated with Phase IV). Proposed new records:
   - exact token-level programming [X];
   - single-network repair without retraining (302) [O];
   - causal necessity of factorisation (307 break) [C];
   - non-sufficiency (304, 310) [C].
4. Whether to deliver the reconstruction and window archives (2.8 GB + 0.55 GB) to durable storage, as with the
   factorial archives.

## 7. Remaining deliverables

| deliverable | state |
|---|---|
| causal intervention results | single-pass done (doc 49); training interventions 30+2 of 49 runs done |
| representation–decoder compatibility matrix | done (all off-diagonal fail) |
| critical-window intervention atlas | partial: §3.5 above; head-level atlas done (`atlas_heads.json`) |
| permanent semantic editing | done for token-level edits (exact); negative for single-variable value edits |
| compositional editing benchmark | done (40 programs × 6 networks; population ρ = 0.96) |
| updated preprint | **not yet updated with Phase IV**: still reflects docs 41–47 |
| updated registry proposals | **not yet updated with Phase IV** |
| archive re-verification after the container restart | **to do** |

## 8. File map

| what | where |
|---|---|
| registrations | docs 37, 39, 41, 45, 48 (+ addendum A) |
| results | docs 38, 39R, 40, 41R, 46/46R, 49 |
| standards, ledger, literature, registry | docs 42 (rev. 1), 43, 44, 47 |
| this handoff | doc 50 |
| Phase IV code | `sci/rescue.py`, `sci/compile_edit.py`, `sci/dev_intervene.py`, `sci/decoder_trace.py`, `sci/atlas_heads.py`, `sci/part_edit.py`, `sci/factorization.py` |
| Phase IV data | `runs/phase4/*.json`; training interventions at `/home/user/dev_intervene/` (outside Git; to be summarised into `runs/phase4/`) |
| reconstructions | `/home/user/swap_recon/zoo` (live), `/home/user/swap_archives` (verified), manifests `runs/swap_recon/` |
| window replay | `/home/user/swap_window`, archives `/home/user/swap_window_archives`, manifests `runs/swap_recon/window/` |
| factorial weights / archives | `/home/user/nri_weights` (read-only copies), `/home/user/nri_archives` (originals) |
| preprint | `paper/main.tex`, `paper/main.pdf`, `paper/REPRODUCIBILITY.md` |
