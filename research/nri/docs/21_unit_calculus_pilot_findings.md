# 21. Unit-calculus pilot findings (exploratory; not a registration)

*Everything here comes from **pilot seeds only** (900–904, 950–951). Most of the mechanistic
results come from **one or two networks**. These are development findings that shape a future
registration. They are not confirmatory results. No main-study network was trained.*

**Data:**
- `results/unitcalc/learnability_sweep.json`: every pilot training curve;
- `results/unitcalc/dev/`: development analyses and the scripts that produced them;
- `runs/zoo_unitcalc_pilot/`: the pilot checkpoints.

**Code:** `nri/unitcalc.py` (task), `run_unitcalc.py` (training), `analyse_unitcalc.py`
(analysis draft).

## 21.1 Plain-English summary

1. **The full design from doc 20 did not learn.** That design had 6 labels per dimension, four
   calculator modes, scale, "per" and offset labels.
   - Multiply and divide stayed at 23–61% accuracy even after 30,000–56,000 steps.
   - Wider networks, strong weight decay and multiply-only training did not fix it.
   - With **4 labels per dimension** (the owner's pre-agreed fallback), the task does learn: two
     networks reached 95% and 100%. But it learns unevenly: a third reached only 67% in the same
     budget. Learning arrives suddenly ("grokking") and at different times in different networks.
2. **The kind of calculation shows up in the geometry of the answer.**
   - Multiply, divide and mass + mass produce answers on **discrete-log circles** (0.94–0.99).
     There, multiplying adds angles.
   - Temperature addition produces answers on **value-order circles** (1.00).
   - **Where** it happens differs too. The offset sum is already formed **before** the MLP, by the
     attention layers (0.94 after layer 1). The scaled calculations appear only **after** the MLP.
3. **In the trained network, no converted quantity (value × scale) exists as a separate,
   swappable variable at any bottleneck.** That held in every mode tested: multiply, mass + mass,
   temperature addition.
   - The label stays separately swappable through the first layer in multiply mode (0.94), and
     partly in temperature addition (0.63).
   - The raw value is already fused with other information by then.
   - At the calculator's input, a 16-dimension swap transplants a **whole answer** 96% of the
     time, but never a single factor (≤ 0.20).
   - The network does not convert units first and then calculate. Nor does it keep value and label
     cleanly separate until the calculator. It **fuses them early** into a code from which the MLP
     reads the answer.
4. **The planned positive control failed.** Networks trained to place converted mass in a
   designated block stayed at chance on multiplication. The interchange loss never came down.

## 21.2 What changed from the design (doc 20), and why

| design element | what happened | consequence |
|---|---|---|
| 6 labels per dimension | not learnable in budget (§21.3) | 4 labels per dimension (owner's pre-agreed fallback) |
| core measurement: read each label's meaning as a rotation | the rotation readout fails before the calculator: even a **known** value change is recovered at only about half its size (value circles are only 23–60% clean). After the MLP, it merely restates the output | the owner chose a **causal comparison of two algorithms** instead (convert-first vs route) |
| declared arm as a positive control | the interchange loss did not train (§21.3) | no working positive control yet for "convert-first" |
| a single 16-dimension subspace at one position | information bypasses a single position: the second attention layer re-reads every position | the causal test must use **true bottlenecks**: the whole layer-1 stream (8 × 64 numbers) or the MLP input |

## 21.3 Learnability (natural networks unless stated; final logged step)

| label set | setting | seed | step | multiply | divide | mass + mass | temperature | held-out label pairs |
|---|---|---|---|---|---|---|---|---|
| 6 labels | d = 64, 30k steps | 950 | 30,000 | 0.31 | 0.23 | 0.17 | **1.00** (sudden at ~19k) | 0.46 |
| 6 labels | d = 128, weight decay 0.1 | 950 | 15,000 | 0.24 | 0.15 | 0.10 | 0.10 | 0.12 |
| 6 labels | d = 64, weight decay 0.1, lr 3e-3 | 950 | 15,000 | 0.30 | 0.22 | 0.14 | 0.08 | 0.16 |
| 6 labels | multiply only, 15k / 40k steps | 950 | 15,000 / 40,000 | 0.38 / 0.36 | — | — | — | — |
| 6 labels | 80k steps (stopped early) | 950 | 43,000 | 0.35 | 0.30 | 0.19 | 0.08 | 0.19 |
| 6 labels | 80k steps (stopped early) | 951 | 56,500 | 0.58 | 0.54 | 0.38 | 0.09 | 0.28 |
| **4 labels** | 30k steps | 950 | 30,000 | 0.88 | 0.92 | 0.99 | 1.00 | 0.47 |
| **4 labels** | 30k steps | **951** | 30,000 | **1.00** | **1.00** | **1.00** | **1.00** | 0.65 |
| 4 labels | 30k steps (locking pilot) | 900 | 30,000 | 0.71 | 0.68 | 0.27 | 1.00 | 0.64 |
| 4 labels | 30k steps (locking pilot, stopped) | 901 | 15,000 | 0.35 | 0.30 | 0.25 | 1.00 | 0.47 |
| 4 labels, **declared** | 30k steps (stopped) | 900–902 | 17,500–25,000 | 0.22–0.23 | 0.14–0.17 | 0.10–0.43 | 1.00 | 0.37–0.40 |

**Notes on this table.**
- Two other 6-label runs are missing from the table. A wider model without weight decay (d =
  128) stalled at 0.25 on multiplication, at 5,500 steps when stopped. A constant learning rate
  with weight decay 1.0 stayed at 0.22–0.24 through 24,500 steps. Both were stopped and their
  directories deleted, so neither appears in `learnability_sweep.json`. Their final readings
  were recorded during the session.
- **Composition is partial.** Held-out label pairs reach 0.47–0.65 overall even in fully trained
  networks. In seed 951: multiply 0.93, temperature 0.91, divide 0.53, mass + mass only 0.18.
- **Temperature addition always learns.** Mass + mass is the hardest mode to generalize.

## 21.4 Calculator geometry by calculation type (seed 951, fully trained)

Share of class-mean variance in the best two frequencies. Multiplicative characters are
discrete-log circles over the nonzero classes. Additive characters are value-order circles. Rows
are the **answer**, at the query position.

| mode | after layer 1 | MLP input | after MLP |
|---|---|---|---|
| multiply | mult 0.44 / add 0.34 | mult 0.57 / add 0.37 | **mult 0.99** / add 0.36 |
| divide | 0.43 / 0.34 | 0.53 / 0.37 | **0.99** / 0.36 |
| mass + mass | 0.48 / 0.42 | 0.50 / 0.45 | **0.94** / 0.42 |
| temperature + temperature | 0.39 / **0.91** | 0.41 / **0.98** | 0.37 / **1.00** |

- **Scaled calculations** (including mass + mass, whose final output-unit conversion is a
  multiplication) end on discrete-log circles, formed by the MLP.
- **The offset sum** ends on value-order circles, formed already by attention.

This reproduces the doc 20 feasibility probe and extends it to new calculation types. But it is
**one network**.

## 21.5 Where labels meet values: causal checks (seed 951, operand 1)

Interchange intervention accuracy on reserved, eligible pairs. A learned 16-dimension subspace is
swapped from a donor question. For each operand, the answer should then follow one of three
counterfactuals:
- **converted:** the donor's converted quantity;
- **value only:** the donor's raw value with the base label;
- **label only:** the donor's label with the base value.

| site (bottleneck?) | multiply: converted / value / label | mass + mass | temperature + temperature |
|---|---|---|---|
| embeddings at the value token (route by construction: control) | 0.24 / **0.99** / 0.06 | 0.15 / **1.00** / 0.06 | 0.17 / **0.97** / 0.08 |
| whole layer-1 stream, 512 numbers (**bottleneck**) | 0.11 / 0.13 / **0.94** | 0.07 / 0.08 / 0.14 | 0.15 / 0.21 / **0.63** |
| MLP input (**bottleneck**) | 0.10 / 0.11 / 0.19 | 0.07 / 0.07 / 0.07 | 0.07 / 0.07 / 0.18 |

**Instrument checks:**
- At the MLP input, the same procedure transplants the donor's **whole answer** 0.96 of the time
  with 16 dimensions and 0.98 with 32. A random 16-dimension subspace achieves 0.05. So the
  optimizer works.
- With 64 dimensions (a full swap), "converted" reaches only 0.14, the rate at which it coincides
  with the donor's answer.
- Raising the subspace from 16 to 32 or 48 dimensions changed nothing at the sites tested.

**Reading.**
- After the first layer, the label can still be swapped on its own in multiply mode, and partly in
  temperature addition. The raw value cannot, and neither can the converted quantity.
- At the MLP input, only the whole answer is swappable.
- The network seems to **fuse value information early and keep the label separate a little
  longer**. It never forms the converted quantity as a separable variable.
- Mass + mass, where the sum needs conversion first, shows **no** separable variable even after
  layer 1. The doc 20 "convert-first in sums" prediction found no support in this network.

## 21.6 Edits (seed 951): giving a label a new meaning with one embedding row

| edit | intended answer |
|---|---|
| mass label → a scale no mass label has | 0.69 (copying an existing label's row: 1.00) |
| acceleration label → a scale no acceleration label has | 0.85 (copy: 0.99) |
| temperature label → an offset no temperature label has | **0.06** (copy: 0.93) |

- A single 64-number change (an embedding row) can teach a scale label a partly new meaning.
- It cannot do the same for a temperature label. That label appears in both operand slots and the
  output slot, with opposite signs.

## 21.7 Limits

- **Sample size:** one or two networks for every mechanistic statement; no statistics.
- **Training:** grokking-style variance means the same budget yields networks at very different
  stages.
- **Missing control:** without the declared positive control, a "convert-first" result could not
  be distinguished from a failure of the instrument. The embedding-layer route control and the
  whole-answer transplant check do work.

## 21.8 Options for a confirmatory study (owner to choose)

| option | what | cost |
|---|---|---|
| A. two-mode study | multiply + temperature addition: "scale label + multiplicative calculator" vs "offset label + additive calculator". Train fast, test the geometry and bottleneck findings at n = 12 | lowest; most likely to learn reliably |
| B. four modes, longer training | re-lock a budget (e.g. 60k) on pilot seeds; natural + meaningless arms; bottleneck test for 3 modes; resume-after-restart | about 12–15 hours |
| C. fix the positive control first | find interchange-training settings under which converted mass can be declared in this task (e.g. a longer warm-up, a lower λ, multiply-only pretraining), so that "convert-first" is detectable when present | a short pilot, then A or B |
