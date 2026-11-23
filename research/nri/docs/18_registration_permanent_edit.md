# 18. Registration: can the unit-aware edit be installed once, and can it be installed within the original network?

*This document, together with `run_perm.py` and `train_perm.py` in the same commit, is the
registration. It was committed **before** any editor was fitted or scored on the new networks.
Training of the new networks had started with the procedure frozen in doc 14; training uses no
editor or evaluation code. Earlier registrations and results (docs 11–17) are preserved
unchanged.*

## 18.1 Questions

Doc 17 found that a **stored table**, where the network's own activation selects a unit-specific
adjustment on each question, edits the declared variable almost as well as copying a real
pattern. A **single linear weight change** gained only about a tenth as much. Two questions
follow:

- **Q1. Can the ability be installed once?** Write the table into the network as a **small added
  layer** that reads only the activation. There is no per-question information from outside and
  no separate lookup step.
- **Q2. Can it be installed within the original network?** Change **only the network's existing
  weights**, with no new units.

The doc-17 result that a single *linear* weight edit fell short does not settle Q2. A better
weight edit may close the gap.

## 18.2 What is new compared with docs 16–17

1. **New networks.** 24 networks, `mass_only` and `acc_only` × fresh seeds **400–411**, trained
   with the doc-14 procedure (warm-up schedule, λ = 3, 6,000 steps), in `runs/zoo_perm/`. They
   were never used for development. No restarts: every network counts, converged or not.
2. **Reserved questions.** The task has exactly 7,350 distinct in-distribution questions. Before
   anything was fitted, a fixed random 30% (**2,205 questions**, seed 13131) was **reserved for
   evaluation**.
   - The location search, every editor fit, the distillation and all tuning draw only from the
     other 70%, using rejection sampling from the task distribution.
   - Each reserved question is scored **once**. Of the 2,205, **1,011** are eligible: their
     unchanged, mass-edited and acceleration-edited answers are pairwise distinct.
   - The networks were trained on the full task, as before. "Reserved" means reserved **from the
     editors**.
3. **Location.** The DAS subspace (k = 16) is searched per network with the doc-15 algorithm, on
   non-reserved questions.

Unchanged from docs 14–17:
- the scorer (`run_swap.score`: intended / wrong-variable / unchanged / other);
- the edit (inversion mod 13 of the declared variable);
- the route (unit × position) and the table construction (`run_editors.Tables`).

## 18.3 Editors

All editors are fitted on the same non-reserved fitting draw.

| editor | installed how | needs at run time |
|---|---|---|
| current (doc 15) | pooled affine map in the subspace, folded into `W_in` | nothing |
| linear (doc 17) | unit-aware affine map from the full activation, folded into `W_in` | nothing |
| pooled table / unit-aware table (doc 17) | stored table; nearest centroid in the subspace selects the entry | a lookup on every question, chosen from the network's own activation |
| **added layer** (Q1) | a softmax key-value layer with one unit per (value, route) cluster, about 70–78 units. Keys are the cluster centroids, values are the stored adjustments, and the output is added to the activation. Written once. | nothing beyond the layer itself |
| **existing weights only** (Q2) | the existing post-site weights (`W_in`, `b_in`, `W_out`, `b_out`, `W_U`), trained by KL distillation to reproduce the **unit-aware table**'s outputs. No new units. | nothing |
| per question (reference) | table entry chosen from the question's known value and unit | outside information |
| copy (benchmark) | a real source pattern | a real example per question |

**Fairness.**
- The added layer and the existing-weights edit learn only from the table. The table is built
  from the fitting examples' value and unit labels.
- Neither is ever shown the edited task's answers.
- Neither sees a reserved question.

**Locked by tuning on the OLD networks** (the 24 declared networks of docs 14–15; non-reserved
questions only). The criterion was **agreement with the hard table's predictions** on a separate
non-reserved check draw. It never used reserved questions or intended-answer rates.
Data: `results/perm/dev.json`.

| setting | grid | agreement with the table | locked |
|---|---|---|---|
| added layer sharpness β | 1, 3, 10, 30, 100 | 0.81, 0.89, 0.94, 0.97, **0.99** | **β = 100** |
| distillation (steps, lr) | (1000, 1e-3), (1000, 3e-3), (3000, 1e-3), (3000, 3e-3) | 0.929, 0.936, **0.954**, 0.944 | **3,000 steps, lr 1e-3**, batch 512, Adam |

## 18.4 Hypotheses and tests

**Population:** the declared variable (mass in `mass_only`, acceleration in `acc_only`), **all 12
networks each**.

**Primary family** (6 tests, Holm, α = 0.05, two-sided exact sign-flip over 12 networks, 95%
bootstrap CIs). Each Δ is the intended-answer rate on the 1,011 eligible reserved questions:

| test | Δ | prediction |
|---|---|---|
| R (mass, acceleration) | unit-aware table − pooled table | > 0: doc 17 replicates on new networks and reserved questions |
| Q1 (mass, acceleration) | added layer − linear weight edit | > 0 |
| Q2 (mass, acceleration) | existing-weights edit − linear weight edit | > 0 |

**Wrong-variable Δ for each pair.** A gain is *specific* only if wrong-variable changes do not
rise significantly (p < 0.05 with mean > 0).

**"Installs the table's ability"** (pre-specified secondary, per population): the installed edit
"matches the table" if mean(installed − table) ≥ −0.05 **and** the 95% CI lower bound is ≥ −0.10.

**Also reported:**
- all four outcome proportions for every editor;
- **preservation**: on reserved questions whose intended edited answer *equals* the original
  answer, the share whose answer stays unchanged;
- agreement with the table's answers;
- the relative size of the existing-weights change;
- per-network results, and converged vs struggling networks separately.

## 18.5 Decision rules, fixed in advance

| outcome | conclusion |
|---|---|
| R holds | The doc-17 unit-aware advantage replicates on new networks and unseen questions. |
| Q1 holds **and** the layer matches the table | **The ability can be installed once**, as a small added layer, with no outside information per question. |
| Q2 holds **and** the existing-weights edit matches the table | **It can be installed within the original network**: the doc-17 shortfall was the *linear* edit, not weight edits as such. |
| Q2 holds but does not match the table | Existing weights can carry part of it. Report the gap; the added layer is then the practical route. |
| Q2 fails | Within this network, existing post-site weights cannot carry the unit-aware edit by this method. |
| Any gain that is non-specific | Report it as such; it is not evidence of variable-specific editing. |

## 18.6 Procedure and compute

- **Procedure:** `python train_perm.py` (training, about 2 CPU-hours) → `python run_perm.py analyse` →
  `python run_perm.py report` → docs/19.
- **Analysis cost:** about 1 CPU-minute per network.
- **Out of scope:** the paraphrase experiment remains deferred.
