# 52 — Results: Phase V, the birth of compositional computation (the decoder)

**Registration.** Doc 51 (commit 664a6c7), made before any experiment ran. Everything here is **exploratory**:
single-pass measurements and interventions on fixed networks, with no training. Data: `runs/phase5/*.json`,
`runs/phase5/decoder_birth.json`. Code: `sci/decoder.py`, `sci/algorithm.py`, `sci/operator_edit.py`;
`/tmp`-free versions of the D1–D3 scripts are listed in §52.8.

## Plain-language summary

**What distinguishes a network that has learned the algorithm.** In the clearest case, seed 307, the computation is
**phase addition**: multiplication done by adding positions on circles.
- Each written token pushes the decoder's hidden units by an amount that depends only on that token.
- That amount follows a circle: the token's discrete logarithm times a single frequency (1 in 307, 5 in 311). Both
  frequencies are coprime to 12, so every value has its own position.
- The decoder turns the *sum* of these positions into the answer.

Because each token contributes independently, a never-trained combination is just a new sum of familiar
contributions, and the network computes it correctly.

In the networks that memorise, the decoder's input is instead explained by the already-multiplied quantity. That
quantity can only be looked up for trained combinations. One failure (302) uses frequency 2, which folds 12 values
onto 6.

**What we could not do.**
- Write the complete computation as a short exact program. The additive part explains 83–90% of the decoder's input,
  and the rest, which the readout depends on, resisted every simple description.
- Transplant a decoder between networks. Every variant failed at chance, because the networks share no coordinate
  system.
- Show by run-time perturbation that factorisation becomes dispensable after the critical window. 307 stays equally
  fragile throughout.

**What we could do.** Program the decoder's output permanently and exactly: "add one to the answer", "invert the
answer", and so on. These compose with the token-level programs and invert bit for bit.

## 52.1 Decoder isolation (D1)

**Deviation 1** (logged before interpreting). The registered synthetic-input construction (D1) uses parts fitted on
familiar data plus an estimated context term. Applied to the networks' own *trained* cells, it reproduces their
behaviour only in 307 (1.00); the others score 0.33–0.89. The construction therefore fails its validity check for
those networks.

The analysis uses the **exact-context variant**:
- each example keeps its own deviation from its cell mean;
- only the cell mean is replaced by the familiar-fitted additive prediction.

For unseen cells this uses the encoder's own activations, which is permitted for characterising a decoder and never
for a qualifying repair.

| network | class | synthetic: trained / near / far / all unseen | real unseen | reading |
|---|---|---|---|---|
| **307** | **algorithmic** | 1.00 / 1.00 / 0.99 / 0.99 | 0.99 | factorised encoder + composing decoder |
| 300 | **interpolative** | 1.00 / 0.97 / 0.29 / 0.46 | 0.56 | the decoder fails far from trained twins |
| 311 | mixed | 0.90 / 0.50 / 0.11 / 0.40 | **1.00** | the composition is shared with the encoder (its code is frequency 5, see §52.3) |
| 305 | mixed | 0.87 / 0.70 / 0.79 / 0.76 | 0.48 | the decoder is better than its own encoder's representation of unseen inputs |
| 302 / 304 / 310 | canonical-reading | 0.32–0.41 additive-only; **1.00** with their interaction restored / 0.09–0.31 | 0.09–0.19 | the decoder requires the interaction (canonical) code |

## 52.2 Birth of the decoder (D2)

Source: `runs/phase5/decoder_birth.json`, 20-step resolution over steps 600–2,000 plus selected 100-step states.

**307:**
- Decoder composition (synthetic unseen accuracy) rises **together with** training accuracy and real unseen accuracy:

  | step | 900 | 1,000 | 1,100 | 1,200 | 1,400 | 2,000 | 6,000 |
  |---|---|---|---|---|---|---|---|
  | synthetic unseen | 0.26 | 0.24 | 0.57 | 0.89 | 0.94 | 0.98 | 1.00 |
  | synthetic trained (same steps) | 0.47 | 0.66 | 0.82 | 0.96 | 1.00 | 1.00 | 1.00 |
- There is no phase in which the decoder fits trained cells without composing.
- **Gradual by the registered criterion:** no rise of ≥ 0.5 within 50 steps. The largest 100-step increases are
  +0.33 and +0.32.

**304 and 310.** Composition stays at 0.08 from their escape onward.

**311.** The decoder alone never exceeds 0.39, while the network is at 0.94–1.00.

**305.** About 0.73 from step 1,400 on, while its real unseen accuracy erodes. Its decoder outperforms its encoder.

**Not measured yet:** the other D2 properties (invariance, operand reusability, resistance) over development.

## 52.3 Reverse-engineering the learned algorithm (D4)

Domain: all 9,126 written prompts.

### Decoder input (hidden pre-activations, 256 units)

| network | additive in written-token phases | additive in canonical phases | dominant frequency (units) |
|---|---|---|---|
| **307** | **0.895** | 0.37 | k = 1 (256/256) |
| **311** | **0.916** | 0.18 | k = 5 (256/256) |
| 300 | 0.76 | 0.13 | k = 1 (248) |
| 305 | 0.67 | 0.22 | k = 1 (254) |
| 304 | 0.46 | 0.33 | k = 1 (255) |
| 310 | 0.47 | **0.66** | k = 5 (256) |
| 302 | 0.34 | **0.67** | **k = 2** (199) |

"Written-token phases" means Fourier features of the five written phases: mass value, mass-unit scale, acceleration
value, acceleration-unit scale, and output factor. Additivity in written-token phases equals token-additivity, since
the one-hot token model gives the same R². So **each token contributes a point on a circle indexed by its discrete
logarithm**.

The hidden activations, after the ReLU, are explained by the separate operands (0.27–0.61) and hardly by the product
phase (≤ 0.05). The logits are fully explained by the product phase (0.98), as in any correct network.

### Interpretation [I]

The generalisers compute F = M·A·S by **phase addition**:
- additive per-token circle features at one frequency coprime to 12;
- ReLU units;
- a readout that is a function of the summed phase.

This is the arithmetic counterpart of the "clock"-type algorithms reported for modular addition. Here it acts on
discrete logarithms, so it implements multiplication. Mass value and mass unit enter as *separate* additive phases.
That is the mechanistic meaning of factorisation, and it is why unseen unit–value pairings work.

### Explicit program: negative results

Programs fitted on **familiar prompts only** and validated over the full domain against the network's own readout:

| program | 307 agreement with network | 307 unseen accuracy | 311 agreement |
|---|---|---|---|
| single frequency per unit (written phases) | 0.19 | 0.18 | 0.17 |
| full additive token parts (all harmonics) | 0.51 | 0.58 | 0.46 |
| + acceleration or mass pair cells | 0.49 | 0.20–0.56 | ≤ 0.89 (familiar), ≤ 0.40 (unseen) |
| additive, fitted per word order | 0.51 | 0.57 | 0.52 |
| network | — | 0.998 | — |

**What the failures show:**
- The additive part predicts the decoder input for never-trained mass renderings well in the generalisers (R² 0.91
  in 307 vs 0.19 in 304, fitted without them). Composition-relevant structure is additive.
- But the remaining 10–17% of pre-activation variance carries information that the ReLU readout needs. It is not a
  single-quantity interaction and not an order effect.
- **The learned algorithm is "approximately additive phase encoding", but no compact closed-form program reproducing
  the network exactly was found.** The program's errors are spread evenly over familiar and unseen prompts, so the
  gap concerns readout precision, not composition.

## 52.4 Developmental scaffolding (D3)

**Run-time tolerance.** In-subspace interaction injected at κ = 0.5; unseen-mass accuracy, injected / clean:

| step | 1,200 | 1,400 | 2,000 | 3,000 | 6,000 |
|---|---|---|---|---|---|
| 307 | 0.46 / 0.92 | 0.47 / 0.98 | 0.41 / 0.98 | 0.43 / 0.98 | 0.44 / 1.00 |
| 311 | 0.48 / 0.55 | 0.77 / 0.95 | 0.75 / 0.94 | 0.83 / 1.00 | 0.84 / 1.00 |

Controls at matched magnitude are less damaging: the out-of-subspace injection gives 307 0.78–0.86 and 311 0.89–0.96.
Isotropic noise gives 0.63–0.69.

**Reading.**
- 307's run-time fragility does **not** change after consolidation.
- 311 becomes somewhat more tolerant over time.
- Combined with Phase IV, "factorisation is temporary scaffolding" holds **developmentally**: training pressure
  toward interaction cannot destroy generalisation after consolidation. In 307 the interaction never forms; in 311
  factorisation is destroyed but generalisation is kept.
- It does **not** hold as run-time robustness of the trained network: inserting an interaction still breaks 307.
- **Interpretation [I]:** after consolidation, the decoder's training dynamics protect the composing solution; the
  finished decoder does not become insensitive to its input format.

## 52.5 Decoder transplantation (D5)

| variant | result |
|---|---|
| direct head copy (307/311 → 300/305/302/304/310) | familiar 0.08–0.18, unseen 0.04–0.12 (chance) |
| W_in only | same |
| W_out + W_U only | same |
| affine-aligned full residual / block only (Phase IV §49.2) | familiar 0.24–0.43, unseen ≤ 0.52 |

**No qualifying repair.**

There is also a **theoretical limit.** On familiar inputs, an algorithmic decoder and an interpolating one give the
same answers. A repair fitted only on familiar data therefore cannot select composition over interpolation. Such a
repair needs a structural prior, for example forcing pre-activations to be additive in token phases. Imposing exactly
that prior on 300's decoder (the additive program, §52.3) reduced its unseen accuracy (0.17), because the readout
needs the non-additive residual.

## 52.6 Programming the decoder: exact output operators (D7)

The output layer's rows are permuted so the network outputs g(F) for its computed F. This edits the decoder's own
parameters, with no token remapping and no hooks.

| network | single operators (F+1, 2F, F⁻¹, −F), familiar / unseen | 30 random compositions with token programs, familiar / unseen | inverse restores bitwise |
|---|---|---|---|
| 307 | 1.00 / 0.998 | 1.00 / 0.999 | yes |
| 311 | 1.00 / 0.999 | 1.00 / 0.999 | yes |
| 300 | 1.00 / 0.434 (= its baseline) | 0.94 / 0.66 | yes |
| 304 | 1.00 / 0.269 (= its baseline) | 0.91 / 0.58 | yes |

These operators are exact, reversible and composable. They neither add nor remove competence.

**Distinction (doc 51).** They act *after* the core computation:
- token programs substitute inputs;
- output operators relabel outputs;
- **neither modifies the algorithm itself.**

An exact edit of the core operator (e.g. F = M²·A·S) was not achieved:
- token maps cannot square the unit scales, since 5² = 12 is not a unit scale;
- the additive phase code is not exactly separable (§52.3), so the per-token phase contributions cannot be
  rewritten in closed form.

## 52.7 Constructed decoder (D6)

**Not built.** D6 required an identified explicit program, and §52.3 found none that reproduces the network. A
hand-constructed head implementing the *task* algorithm (decode token phases, add, read out) remains feasible in
principle, but would be a construction from the task's mathematics, not from the learned algorithm. It is deferred and
recorded as open.

## 52.8 Reproducibility notes

- **Scripts.** D1 exact-context, D2 birth, D3 tolerance and D5 transplant were run as small driver scripts on top of
  `sci/decoder.py` and `sci/rescue.py`. The D2/D3 drivers are preserved as `sci/phase5_drivers.py`. Every reported
  number is in `runs/phase5/*.json` and `runs/phase5/decoder_birth.json`.
- **Data.** All inputs are archived checkpoints, with manifests in `runs/swap_recon/`.

## 52.9 Answer to the central question (provisional)

**What mathematical structure distinguishes a network that has learned the algorithm?** In these networks, it is the
*separability of operand contributions at the decoder's input*, in a periodic code at a frequency coprime to the
group order. Each token's contribution is independent of the others, so the decoder's nonlinearity can combine any
combination of familiar contributions.

Memorising networks deliver instead a pre-combined, canonical code. Their decoders are correct on it, but their
encoders can produce it only for trained combinations.

**What we could do with that structure:**
- **Identify it:** yes.
- **Destroy it causally during development:** yes (Phase IV).
- **Repair it without training:** for one network (302).
- **Program around it exactly at its inputs and outputs:** yes.
- **Transplant or construct it:** not yet.
