# 35 — Birth of a Concept, Stage 1: report-level re-analysis of the 6,000-step conversion trainee

Source: PR #27 (doc 34) and the owner-supplied archive `NRI_Report_Data_and_Code.zip` (report, 61-row
`trajectory.csv`, cross-route patch diagnostics at steps 900/1100/2400/6000, development data, implementation
patch). The checkpoint tensors (weights, optimizer, traces) were **not** available, so this stage re-reads recorded
measurements only; no new causal test was run. Exploratory; one seed (482111).

## 35.1 What the trainee was actually tested on

* Training covered mass conversion route 1→0 first (later reverse 0→1 and source 2→0 were added). Acceleration
  conversion was never trained in any direction.
* On route 1→0 the tuple hash leaves 10 values for training, value 4 for development and values 0 and 1 sealed.
  "Practiced" = 10 items; "new_values" = **one** item (value 4), which stayed wrong (answer 6, gold 7) at every
  checkpoint.
* The broader held-out score (34/384, all hits on mass, 0/139 on acceleration) is **below** a best constant answer
  (96/245 on mass alone). No checkpoint shows compositional generalisation; there was no "birth" to time.

## 35.2 The swap trajectory is quantised and noisy

`swap_value` = all ordered pairs of the 10 practiced items on one route (10 × 9 = 90) with the layer-1 residual at
the source-number token copied from donor to base. The 90 trials share 10 donor codes, so one value's code becoming
(un)readable moves the score by ~0.1-0.2. Large changes (|Δ| ≥ 0.15 in one block):

| Block | Δ swap | Curriculum changed at the start of the block? |
|---|---|---|
| 400→500 | −0.16 | yes |
| 900→1000 | −0.20 | yes |
| 1900→2000 | −0.22 | no |
| 3100→3200 | +0.15 | no |
| 3400→3500 | −0.16 | no |
| 4100→4200 | +0.20 | no |
| 5500→5600 | −0.21 | no |

5 of 7 large moves happened with **no** curriculum change. With ~10 effective units the metric cannot time a
transition; part of the trajectory may be checkpoint-to-checkpoint drift of individual value codes.

## 35.3 Where the number is causally available

Cross-route patches (270 discriminating pairs, base route kept, donor number patched):

| Step / site | answer follows donor number under base route | answer unchanged | donor number + donor route |
|---|---|---|---|
| 900, x1 | 137 | 41 | 36 |
| 1100, x1 | 91 | 23 | 54 |
| 6000, x1 | 105 | 32 | 56 |
| 2400, x2 | 2 | 258 | 1 |
| 6000, x2 | 1 | 266 | 2 |

Unit-token-only patches never changed an answer at either site. Reading: after layer 1 the number token carries a
code that layer 2 can partly re-use under another route (34-51% of pairs) — the beginning of a route-independent
number representation — but by layer 2 the number is no longer at that token (it has been consumed/moved), and the
unit has already been read into other positions by layer 1. The cross-route transfer fell from 137 to 91 between
steps 900 and 1100, while practiced accuracy stayed 10/10.

## 35.4 Consequences for the Birth-of-a-Concept design

1. **Purpose-built split.** Withhold (value, route) combinations so that every value and every route is trained
   elsewhere, with dozens of held-out items per seed; never a single item.
2. **High-resolution causal panel.** Hundreds of distinct base/donor pairs over all trained routes, including
   held-out bases, scored separately for intended, unchanged and wrong-variable outcomes; patch at x1 (where the
   number is available), with x0/x2 as controls.
3. **An explicit mechanistic signature of the concept.** For conversion by a fixed scale in Z/13, the reusable rule
   is "multiply the number by a route constant". In discrete-log coordinates that is a *rotation*, and earlier work
   (doc 29) found discrete-log circles in this student family. A concept is therefore detectable directly: value
   codes shared across routes that lie on a discrete-log circle, with each route acting as a consistent rotation.
   Lookup predicts route-specific, unstructured codes.
4. **Learning regime must allow generalisation.** Learning a modular operation from a partial table is the
   setting where delayed generalisation ("grokking") is reported to depend on the training fraction, weight decay
   and training length. Neither PR #27 nor doc 31 trained in a regime known to permit it. The pilot must find a
   regime in which at least one population generalises, or the four-population comparison is uninformative.
5. **Seeds.** Every trajectory claim needs replication across seeds; one seed's swap curve is a hypothesis
   generator only.
