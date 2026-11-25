# 38 — Results: Birth of a Concept, 4 × 2 factorial (96 networks)

Registration: doc 37 (commit 1b54b0b, before training). Pilot: doc 36 (preserved). Code: `sci/concept.py`,
`sci/concept_audit.py`, `sci/factorial.py`, `sci/cohort.py`, `sci/episodes.py`, `sci/factorial_analysis.py`.
Per-cohort supervisor reviews: `runs/concept_factorial/reviews/cohort_00000.md` … `cohort_06000.md` (61 reviews);
cohort tables `runs/concept_factorial/cohorts/`; per-run manifests, audit summaries and interval logs
`runs/concept_factorial/<run>/`; reserved-test results `final_test.json`; registered analysis `analysis.json`.
Task: modular unit conversion (mod 13, fictional units — not real-world arithmetic).

## 38.1 Answers to the five questions

1. **Does canonical supervision causally promote compositional generalisation? Yes, at weight decay 1.0.** C beats its
   entropy-matched control D by **+0.55** on the reserved test (95% CI 0.45-0.66, Holm p = 0.0012); 12/12 C networks
   composed during training, 0/12 D networks did. Supervising the *semantic structure* of the quantity (shared across
   units), not merely adding an auxiliary loss of the same size, is what helps.
2. **Does weight decay interact with semantic supervision? Yes, strongly.** Interaction **+0.48** (0.37-0.59, p_Holm
   0.0012). C at weight decay 0.1 has the representation (canonical quantity decodable in 11/12 at the end) but 0/12
   compose.
3. **Why does a decodable representation precede compositional use?** Decodability arrived at steps 300-1,200,
   causal use of the code as a *quantity* at 1,200-3,500, composition at 2,000-5,000. The task and auxiliary gradients
   at the code's site stayed orthogonal throughout (cosine −0.02…+0.04), so the task loss never "pulled" toward the
   representation; what distinguished the composing condition is weight decay, and the registered mechanistic test
   M3 shows weight decay changes *how downstream layers read* the code (canonical − number reading +0.39 higher at
   wd 1.0, p = 0.0002) — not whether the code exists.
4. **What accompanies the transition in weight space and gradient dynamics?** At wd 1.0 every network entered a
   decay/Adam equilibrium (weight norm flat ≈ 17-21 from ~3,500; per-100-step adaptive update almost exactly
   opposite to the decay update in the readout, cosine ≈ −0.74 in calm blocks) and **instability episodes** became
   frequent in every population (120-155 episode blocks per population; at wd 0.1: 7 blocks in total). No single
   weight-space precursor of compositional onset was identified (exploratory).
5. **Collapses and recoveries.** 21 collapses (held-out drop ≥ 0.15 after onset) in composers; **19/21 fell in
   instability episodes**; 18/21 recovered (median 2 cohorts, range 1-8). During collapse the readout equilibrium
   loosens (cosine −0.87 → −0.50) and the downstream canonical reading drops (−0.12); in about half the collapses the
   representation itself is also hit (decodability drop ≥ 0.10 in 10/21). The mechanism that *triggers* the episodes
   is identified only descriptively (thin margins at the equilibrium norm at wd 1.0; near-zero gradients at wd 0.1).

> **Update (doc 39, retrospective).** An owner review found that the registered utilisation measure
> (`patch_semantics`) draws base and donor only from trained combinations, so the causal account in §38.1 (Q3) was
> supported only in trained contexts, and that M2's binomial test ignored clustering by seed. Doc 39 adds held-out
> intervention evidence (the canonical code is converted correctly along never-trained (input, route) combinations in
> C wd 1.0: excess 0.47 vs ≤ 0.14 in other conditions; precedes composition in 11/12 networks) and a seed-clustered M2
> (mean 0.70, CI 0.60-0.80, p = 0.003). Primary results unchanged.

## 38.2 Primary endpoint (registered; reserved test, run once; 72 items per network; best constant 0.167)

| | A narrow | B diverse | C canonical aux | D scrambled aux |
|---|---|---|---|---|
| wd 0.1 | 0.061 | 0.098 | 0.134 | 0.058 |
| wd 1.0 | 0.065 | **0.307** | **0.655** | 0.101 |

(new-frame test: wd 1.0 C 0.310, B 0.218, D 0.079, A 0.046; wd 0.1 ≤ 0.125.)

| ID | Contrast (paired by seed, n = 12) | Mean | 95% CI | p (exact sign-flip) | p (Holm) |
|---|---|---|---|---|---|
| H1 | C − D @ wd 1.0 | +0.554 | 0.450-0.659 | 0.00024 | **0.0012** |
| H2 | (C − D)@1.0 − (C − D)@0.1 | +0.478 | 0.372-0.587 | 0.00024 | **0.0012** |
| H3 | C@1.0 − C@0.1 | +0.521 | 0.431-0.613 | 0.00024 | **0.0012** |
| H4 | C − B @ wd 1.0 | +0.348 | 0.199-0.479 | 0.0015 | **0.0015** |
| H5 | B − A @ wd 1.0 (two-sided) | +0.242 | 0.176-0.332 | 0.00049 | **0.0012** |

Secondary (new frame): C − D @ wd 1.0 +0.23 (0.13-0.33, p = 0.002); interaction +0.15 (0.02-0.29, p = 0.033).

**Unexpected: B at wd 1.0 partially generalises on the reserved test (0.31) although its development held-out
accuracy never exceeded ~0.08.** The gain is concentrated in a few test combinations (e.g. mass 1→2 value 8: B
0.83, D 0.33; acceleration 0→2 value 9: B 0.58, D 0.42) — the test combinations are heterogeneous in difficulty and
some are partly solvable without canonical supervision. The development-based onset definitions therefore missed
B's partial composition; H5 (diverse > narrow at wd 1.0) is a real effect but its size is specific to these
combinations. C is ≥ 0.50 on 15 of the 18 test combinations.

## 38.3 Registered mechanistic predictions

| | Prediction | Result |
|---|---|---|
| M1 | causal utilisation precedes compositional onset | 9 precede / 3 follow among 12 networks with both; sign test p = 0.073 — **direction supported, not significant**; not falsified |
| M2 | ≥ 70% sign agreement of held-out and x1 canonical-semantics changes after onset | 88/127 = **69.3%** (vs 50%: p ≈ 8 × 10⁻⁶) — far above chance, narrowly below the predicted 70% |
| M3 | x1 canonical − number larger at wd 1.0 than 0.1 in C | **+0.39** (0.27-0.52), p = 0.0002 — confirmed |
| M4 | C wd 0.1 decodable (≥ 9/12) but not compositional (≥ 9/12) | 11/12 decodable, 12/12 not compositional — confirmed |

## 38.4 Transition timing (C, wd 1.0; registered onset definitions)

| seed | decodable | utilised | compositional | lead (util → comp) |
|---|---|---|---|---|
| 2004 | 300 | 1,500 | 2,000 | +500 |
| 2009 | 400 | 2,000 | 2,200 | +200 |
| 2002 | 400 | 2,800 | 2,500 | −300 |
| 2007 | 300 | 2,800 | 2,500 | −300 |
| 2001 | 400 | 1,600 | 2,900 | +1,300 |
| 2008 | 1,200 | 3,500 | 2,900 | −600 |
| 2012 | 300 | 1,200 | 3,000 | +1,800 |
| 2003 | 300 | 1,800 | 3,100 | +1,300 |
| 2006 | 400 | 1,500 | 3,300 | +1,800 |
| 2011 | 500 | 3,100 | 3,600 | +500 |
| 2010 | 300 | 1,400 | 4,100 | +2,700 |
| 2005 | 300 | 2,600 | 5,000 | +2,400 |

(all onsets from `analysis.json`, registered definitions.) Three "mixed-code"
composers (2002, 2007, 2008) composed while their x1 code was read about equally as raw number and canonical
quantity, and reached the canonical criterion only afterwards. **Composition does not require the canonical reading
at x1**, though in 9/12 networks it came first. Mixed-code composers spent more audits below criterion after onset
(22/104) than the others (11/257) but transferred *better* to the unseen frame at step 6,000 on development items
(mean 0.37 vs 0.21 for the four most canonical composers 2001/2004/2010/2012) — the opposite of my in-run
prediction (n = 3 vs 4; descriptive only). *Correction:* my chat summary during the analysis quoted 0.44 for the
mixed group, which used seed 2005 instead of 2008; 0.37 is the value for the group defined here.

## 38.5 Exploratory findings (declared in doc 37 §37.9; not confirmatory)

* **Episodes.** Episode block = post-fit block with batch loss > 0.2 and ≥ 3 steps above 3× median gradient. wd 1.0:
  every network, 120-155 blocks per population; wd 0.1: 5 networks, 7 blocks. Two regimes: wd 1.0 episodes escalate
  over ~25-30 steps at the decay/Adam equilibrium; wd 0.1 episodes are explosive (gradient up to 73 within 2 steps)
  and every one (4/4 examined) followed a block with median gradient < 0.02.
* **Readout equilibrium.** Mean W_in adaptive/decay cosine: episode blocks −0.54, calm blocks −0.74.
* **Robustness of the canonical code.** Median auxiliary-loss rise in episode blocks: C ×1.29, D ×1.89.
* **Own-trained drop at the audit in episode blocks after 2,500:** A 44%, B 26%, C 19%, D 18% (my prediction "C
  lowest" — falsified; C ≈ D).
* **Collapse → recovery "ratchet".** Post-recovery held-out above the pre-collapse level in 10/18 recoveries
  (prediction ≥ 2/3 — not supported).
* **Episodes as the mechanism of the weight-decay effect — not supported at population level.** In C at wd 1.0 the
  canonical − number reading changed by +0.004 across episode blocks vs +0.013 across calm blocks (paired per
  network −0.008, p = 0.81); held-out accuracy in episode blocks 0.37 vs calm 0.33. Several striking single cases
  (seeds 2007, 2010, 2004 in-run, and the two C wd 0.1 networks whose only utilisation onsets followed their only
  episodes) suggested the mechanism; the population test does not. Recorded as a lesson in anecdote selection.
* **Slow drift at low decay.** C wd 0.1 canonical − number rose from −0.09 (step 1,000) to +0.08 (6,000) as its weight
  norm fell from ~102 to ~66; 2/12 reached the utilisation criterion late (5,300, 5,600), none composed.

## 38.6 Instrument problems, corrections and failed predictions (all logged in the reviews)

* Canonical-grouped circle score has a structural null of ≈ 0.45 at initialisation (shuffled baseline 0.26): class
  means over multiplicatively related values carry discrete-log structure even from random codes (cohort 0).
* The registered relative gradient-spike rule fires on every converged network; a run-length classifier and the
  episode counter were built during the run and tested on planted logs.
* A review paragraph (cohort 600) was written before the instrument output was read and was wrong; corrected the
  same cohort and the rule "anomaly text only after the measurement is on screen" adopted.
* A truncated screen output hid a registered onset (seed 2012 at 1,200) for one cohort; onsets were thereafter
  recomputed directly every cohort.
* `patch_semantics` "number" vs "canonical" measures which variable the code carries, not lookup vs rule (cohort 2,200).
* Subspace mean-ablation of class-mean spans could not discriminate pathways (found before registration; kept
  exploratory).
* In-run predictions: 30+ recorded; confirmed include the 5/12-by-3,000 composer count, seed-level recoveries,
  the small-gradient precursor at wd 0.1, the C/D auxiliary-robustness asymmetry; falsified include the early
  episode-frequency predictions (both directions at different times), "readout-first always", "canonical composers
  more robust/better transfer", the task-aux cosine turning positive before composition, the ratchet, and several
  seed-level timing predictions.

## 38.7 Limitations

One task family; one small architecture (129,280 parameters) and optimiser; the registered endpoint is a single
snapshot at step 6,000 in a regime with frequent collapses (seed 2012 was mid-recovery at 0.35 on development);
the 18 reserved test combinations are heterogeneous in difficulty; M1 is directionally but not significantly
supported; precursor analyses are exploratory and must be tested on new seeds. Checkpoint archives
(`/home/user/nri_archives/*.tar.xz`, ~15.6 GB, per-file SHA-256 manifests in Git, each verified byte-for-byte with
`tar --diff` before the live copy was removed) exist only in this session's container and must be copied to
durable storage to be preserved.

## 38.8 Reproduce

See doc 37 §37.11; analysis: `python -m sci.factorial_analysis`; cohort tables: `python -m sci.cohort <step>`;
episode counts: `python -m sci.episodes 6000`. Restore a run: `tar -xJf <run>.tar.xz` and check against
`runs/concept_factorial/<run>/manifest.json`.
