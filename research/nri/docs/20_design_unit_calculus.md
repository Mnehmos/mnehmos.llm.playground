# 20. Design: mapping how labels, values and the calculator correspond ("unit calculus")

*Design document, NOT a registration. The registration will be a separate, timestamped
document committed before any network for this study is trained. This document includes an
exploratory feasibility probe on existing networks (§20.2), which motivates the design.*

## 20.1 Goal

**Can we determine, from a trained network, how the relationships between labels and values
correspond to calculations?**

In the dimensional task, a question such as "3 g × 5 cm/s² → N" contains values (3, 5) and labels
(g, cm/s², N). Each label stands for a calculation applied to its value: "g" means *multiply by
the gram-to-base scale*. The task then combines the converted quantities with a **calculator**:
force = mass × acceleration, expressed in the output unit.

The study tracks **every quantity in that computation**, the "units" of the calculation:

| level | quantities |
|---|---|
| inputs | raw values (m, a); labels (mass unit, acceleration unit, output unit, operator) |
| label meaning | each label's scale, as a log-scale λ, and its direction: multiply, or divide for "per" units |
| converted quantities | canonical mass M, canonical acceleration A, output factor S |
| **calculator** | the combining step (product M·A, quotient F/M), its intermediate and the answer |

It then asks whether the network's internal structure reveals:
1. **where** each quantity is formed;
2. **what operation** forms it (multiply, divide, add);
3. **what each label means**: its scale, recovered *without* being told the scale table.

All three are checked causally, by predicting and then performing interventions.

## 20.2 Feasibility probe (exploratory, existing networks)

**Setup.** `feasibility_unit_calculus.py`, run on the 12 undeclared (`neither`) networks from
docs 14–15. Linear probes and class-mean geometry were measured at every role-aligned token
position and layer. Output: `results/unitcalc_feasibility/summary.md`.

**1. The calculator is visible, and it multiplies by adding logs.** Arrange the 12 nonzero
values mod 13 by their discrete logarithm (base 2), so that multiplying two values adds their
logs. Then fit the average activation pattern of each value class with *multiplicative
characters* (circles in log space) or *additive characters* (circles in ordinary value order).

| quantity, site | multiplicative characters (best 2 frequencies) | additive characters (any frequency) |
|---|---|---|
| product M·A, final position before the MLP | **0.94** (f = 2 and 5) | ≤ 0.21 (flat, ≈ uniform) |
| answer F, after the MLP | **0.92** (f = 2 and 5) | ≤ 0.19 (flat) |
| unembedding rows | 0.76 (f = 2 and 5) | ≤ 0.19 (flat) |

The network represents the product as a **rotation in discrete-log space**. That is the
multiplicative analogue of the Fourier circuits found in networks that learn modular addition.

**2. Labels are not converted first.**
- Unit labels are linearly readable at every position after the first attention layer
  (0.88–1.00).
- Each raw value is perfectly readable at its own position (1.00).
- Canonical mass is readable only at 0.43–0.74.
- The raw (value, label) pair explains **2–6× more** activation variance than canonical mass at
  every site tested.

So the network does not appear to rewrite "3 g" as a canonical mass and then multiply. More
likely, each label contributes **its own rotation** (its log-scale) directly to the calculator.

If so, a label's meaning should be readable as a phase. That is the label↔calculation
correspondence this study sets out to test.

**Caveats.**
- Linear probes, 12 networks, one task with only 3 units per dimension.
- The additive spectrum over nonzero classes is not orthogonal.
- The frequency-6 (sign) term is fitted with a degenerate basis.
- These numbers motivate the design. They are not results.

## 20.3 Task: more units, more kinds of label relationship

The current task has 3 units per dimension, too few to identify a label's meaning from its
effect. The new task family, "unit calculus v2", keeps arithmetic mod 13. The changes below are
proposed for registration.

| element | proposal | why |
|---|---|---|
| **multiplicative labels** | 6 mass, 6 acceleration and 6 output units. Their scales are chosen to cover the multiplicative group, including **label pairs with equal ratios** | identifiability; tests whether "same ratio ⇒ same internal change" |
| **inverse labels** ("per" units) | some labels divide rather than multiply (log-scale −λ) | does the network represent the *direction* of the relationship? |
| **two calculator modes** | an operator token selects product (F = M·A) or quotient (A = F / M) | is the calculator one mechanism with a sign switch (add vs subtract logs) or two separate ones? |
| **additive labels** (phase B) | an offset dimension, like temperature, where the label adds a constant (T = t + c_u) | a *different* relationship type. It should appear in additive, not multiplicative, geometry |
| **held-out structure** | some (label, value) combinations and some **label pairs never seen together** in training | can the recovered label meanings predict behaviour on unseen combinations? (composition) |

**Phases.**
- **Phase A:** multiplicative and inverse labels, plus both calculator modes.
- **Phase B:** adds the additive-label dimension, run only if phase A works.

## 20.4 Networks (arms)

| arm | training | role |
|---|---|---|
| **natural** (primary) | ordinary training, nothing declared | how the network organises labels and calculation on its own |
| **declared** (positive control) | interchange training declares M and A at known blocks (doc 14 recipe) | ground truth for *where*; checks the instruments |
| **meaningless labels** (negative control) | same task, but labels carry **no** scale; every unit means ×1 | label "meanings" recovered here must be ≈ uninformative |
| **untrained** | random initialisation | floor for every measurement |

**Size:** 12 seeds per arm, the trained network being the unit of evidence. Architecture as in
docs 14–19 (2 attention layers + MLP, d = 64), with the vocabulary extended for the new labels.

## 20.5 What we measure: a map of every quantity, plus causal checks

**Sites:** every token position × layer (embedding, after attention 1, after attention 2, MLP
hidden units, after the MLP), plus attention-head routing (which head carries each label to
which value).

| measurement | question | method |
|---|---|---|
| **M1 readability** | where can each quantity be read? | linear probes on held-out data, against a control-task baseline |
| **M2 causal location** | where is each quantity *used*? | interchange interventions and DAS per site: sufficiency (swap in → output follows) and necessity (ablate → output breaks) |
| **M3 calculator geometry** | what operation forms each quantity? | character spectrum: multiplicative (log-space circles) vs additive (value-order circles). Product and quotient predicted to share the same frequencies, with opposite rotation for the divisor |
| **M4 label meaning recovery** (core) | what does each label do? | estimate each label's contribution as a **rotation angle** in the calculator's character plane, *without* the scale table. Compare with the true log-scales. Scored by circular error, the share recovered exactly, and the **equal-ratio test** (pairs with equal true ratio must give equal recovered rotation) |
| **M5 algebra** | do recovered meanings compose? | inverse labels give opposite rotations; label combinations add; the quotient mode subtracts. Predict behaviour on **held-out label combinations** from the recovered meanings alone |
| **M6 causal validation by editing** | does the recovered map work as a control knob? | (a) **relabel**: rotating a label's contribution by Δλ should change answers as if the label's scale changed by e^Δλ; (b) **redefine a unit** permanently: fold that rotation into the weights and score with the frozen scorer (intended / wrong / unchanged / other), with copying a real pattern as the ceiling |
| **M7 interpreter** (optional) | can an LLM recover the same map? | blinded white-box sessions under the docs/blind protocol: recover the label→scale table and the calculator rule from the network alone, scored against ground truth |

M4 is the direct answer to the goal. If a label's internal effect is a rotation equal to its
log-scale, then "the relationship between a label and a value" corresponds exactly to "multiply
by the scale". We can then read the unit table back out of the network.

## 20.6 Hypotheses to register (draft)

| | hypothesis | prediction | control |
|---|---|---|---|
| H1 | the calculator uses discrete-log geometry | product and answer ≥ 0.8 multiplicative-character concentration at their best site; additive ≈ flat | untrained ≈ flat for both |
| H2 | labels act as rotations equal to their log-scales | recovered λ matches the true log-scale for ≥ 80% of labels (circular error < half a step); equal-ratio pairs give equal rotations | meaningless-label arm: recovered rotations ≈ 0 |
| H3 | label meanings compose | accuracy on held-out label combinations is predicted by the recovered rotations (rank correlation across networks) | — |
| H4 | relationship *direction* and *mode* are represented | inverse labels rotate oppositely; the quotient mode reverses the divisor's rotation | — |
| H5 | the map is causal | relabel interventions and unit-redefinition edits succeed at ≥ the copy-benchmark-relative threshold fixed at registration | declared arm as the positive control |
| H6 (phase B) | relationship type determines geometry | offset labels appear in additive, not multiplicative, characters | — |

**Statistics:** Holm across H1–H5. Exact sign-flip tests over seeds. All scheduled seeds count,
with no restarts. Pilot seeds are used only to fix budgets.

## 20.7 Protocol lessons carried forward (docs 14–19, the blind pilot)

- **Register before training.** Lock settings on pilot networks before analysing the main
  networks.
- **Reserve questions from every fitted component.** Score each question once (doc 17 §17.8).
- **Report every outcome category,** not just success rates.
- **Report every network,** including ones that struggled.
- **Separate tiers:** per-question lookups, stored tables, and edits written into the weights
  are different capabilities.
- **Independence for M7:** fresh isolated sessions, no repository or memory, new networks (blind
  pilot lessons).

## 20.8 Compute

- **Training:** about 8 CPU-minutes per network → 48 networks (4 arms × 12) ≈ 6.5 CPU-hours.
  Phase B adds about a third.
- **Analysis:** minutes per network.

## 20.9 Decisions for the owner before registration

1. **Scope:** phase A only, or A + B (additive labels) in one registration?
2. **Interpreter (M7):** include the blinded LLM recovery now, or run it after the mechanistic
   map is established?
3. **Calculator modes:** product + quotient (proposed), or also a sum mode, e.g. total mass of
   two masses?
4. **Edit threshold for H5:** absolute (e.g. ≥ 0.8 intended) or relative to copying (e.g.
   ≥ 90% of the copy benchmark)?
