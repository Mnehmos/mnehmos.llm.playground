# 49 — Results: Phase IV, causal engineering of generalisable computation

**Registration.** Doc 48 (commit 7eceebb), made before any experiment ran.
**Status.** Exploratory, except the pass/fail criteria fixed in doc 48.
**Data.** `runs/phase4/*.json`, `/home/user/dev_intervene/*/result.json`.
**Scope.** Seven role-swap networks (307, 311 generalise; 300, 305 partial; 302, 304, 310 fail), so seeds are not
selected for success.

## Plain-language summary

1. **The obvious surgery fails, and teaches why.**
   - Deleting the value×unit interaction from a failing network's declared representation makes things worse:
     familiar answers fall from 100% to 33–40%, and unseen answers stay at chance.
   - The failing networks do not compute with composable parts at all. Their downstream layers read a finished
     canonical code, and their defect is that the code is *missing* for unseen pairings.
   - Supplying the missing code, using the true answer as a diagnostic oracle, fixes them (unseen accuracy 0.71–0.93).
2. **A failing network was repaired without retraining and without the answer** (302: unseen accuracy 0.19 → 0.93,
   familiar unchanged at 1.00).
   - The repair reads which canonical classes are missing from the network's own trained structure.
   - It places the unseen pairings using the network's own geometry.
   - The same procedure only partly repairs 310 (0.54) and fails on 304 (0.30), where the geometry misleads it.
3. **The reverse works too.**
   - Injecting value×unit interactions into the generalisers' representation breaks unseen accuracy in a smooth
     dose–response.
   - Placed on one unit, it breaks generalisation about twice as fast as familiar accuracy.
4. **Three distinct defects** were identified by tracing the computation:
   - failures have a defective *representation*;
   - partial generalisers have a correct representation but a *decoder that interpolates rather than composes*;
   - generalisers have neither defect.
5. **Training-time intervention (final, §49.4).**
   - Rewarding the interaction during the critical window broke **both** generalisers (unseen 0.99/1.00 → 0.35/0.33,
     familiar 1.00): factorisation is causally necessary during development.
   - After consolidation it no longer is: 311 kept 0.92 with its factorisation destroyed.
   - Preventing the interaction before each failure's dip gave only partial generalisation (0.52–0.61); later
     prevention gave none. Factorisation is necessary but not sufficient, and the timing matters.
6. **Permanent programming works exactly where the mathematics allows it.**
   - Edits that act on tokens: units relabelled, written values mapped, output unit changed. They compile *exactly*
     into embedding rows.
   - They are reversible bit for bit, and any sequence composes exactly.
   - In the generalisers every such program is executed correctly, including on never-trained inputs: 40 random
     2–3-step programs, 1.00 familiar / 0.999–1.00 unseen.
   - Edits that must act on one variable's value but not another's (mass-only increment) cannot be compiled into a
     single linear change. Least-squares approximations failed (≤ 0.44).
   - **The extent to which a network can be programmed is bounded by the extent to which it generalises.**

## 49.1 Rescue without training (E1)

Unseen-mass accuracy / familiar accuracy / discriminating unseen-source interchange. 8,192-item panels; the swap
uses 2,738 eligible pairs.

| intervention | 304 | 310 | 302 | 300 (partial) | 305 (partial) | 307 (gen.) | 311 (gen.) |
|---|---|---|---|---|---|---|---|
| R0 none | 0.08 / 1.00 / 0.00 | 0.09 / 1.00 / 0.00 | 0.19 / 1.00 / 0.10 | 0.56 / 1.00 / 0.49 | 0.46 / 1.00 / 0.30 | 1.00 / 1.00 / 0.99 | 1.00 / 1.00 / 0.99 |
| R1 remove interaction | 0.08 / **0.40** / 0.00 | 0.08 / **0.33** / 0.00 | 0.24 / 0.34 / 0.16 | 0.49 / 0.76 / 0.40 | 0.27 / 0.79 / 0.06 | 1.00 / 1.00 / 0.99 | 0.35 / 0.88 / 0.30 |
| R2 keep 75% | 0.08 / 0.97 | 0.15 / 0.93 | 0.09 / 0.98 | 0.59 / 1.00 | 0.41 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 |
| R3 remove only inside causal subspace | 0.08 / 0.40 | 0.08 / 0.33 | 0.24 / 0.34 | 0.49 / 0.76 | 0.27 / 0.79 | 1.00 / 1.00 | 0.35 / 0.88 |
| R3 remove only outside it | 0.08 / 1.00 | 0.09 / 1.00 | 0.19 / 1.00 | 0.57 / 1.00 | 0.46 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| R4 additive reconstruction | = R1 | = R1 | = R1 | = R1 | = R1 | = R1 | = R1 |
| R6 geometric completion (best k) | 0.52 / 1.00 | 0.54 / 1.00 | 0.46 / 1.00 | 0.59 / 1.00 | 0.63 / 1.00 | 1.00 / 1.00 | 0.98 / 1.00 |
| **R7L class completion** | 0.32 / 1.00 (0/4 assigned) | 0.32 / 1.00 (1/4) | **0.93 / 1.00 / 0.93 (4/4)** | 0.16 / 1.00 | 0.10 / 1.00 | 0.08 / 1.00 | 0.08 / 1.00 |
| R7A class completion by angle | 0.30 (0/4) | 0.54 (2/4) | **0.93 (4/4)** | 0.25 | 0.26 | 0.08 | 0.08 |
| D1 oracle canonical placement | **0.71** / 0.86 | **0.85** / 0.94 | **0.93** / 0.90 | 0.56 / 0.61 | 0.60 / 0.67 | 0.54 / 0.75 | 0.54 / 0.66 |
| control: random rank-matched removal | 0.10 / 0.65 | 0.11 / 0.50 | 0.13 / 0.67 | 0.24 / 0.35 | 0.28 / 0.62 | 0.08 / 0.18 | 0.40 / 0.58 |
| control: magnitude-matched noise | 0.11 / 0.83 | 0.18 / 0.77 | 0.19 / 0.72 | 0.54 / 0.97 | 0.47 / 0.97 | 1.00 / 1.00 | 0.91 / 0.99 |

**Registered rescue criterion** (unseen ≥ 0.90, familiar ≥ 0.95, both controls ≤ R0 + 0.05):
- **Met by 302 under R7L/R7A.** This is the first repair of a memorising network without retraining and without
  oracle information.
- Not met by 304 or 310.
- R6, R7 and R7A were **added after the registered R1–R5 failed**, so they are post-registration analyses. The
  choice of k in R6 was not fixed in advance; all k are reported, and R7L's result is the same for every k.

**Diagnosis (doc 48 rule):**
- **Failures:** the representation is defective, and the decoder is intact for canonical inputs.
  - D1 rescues to 0.71–0.93, while R1/R4 (making the block additive) destroy familiar accuracy. The failure's
    decoder reads a canonical, interaction-built code.
  - Its defect is that the code for unseen pairings is absent. The block for those cells does not match either the
    additive or the canonical model (block error 0.26–1.41; `decoder_trace.json`).
- **Partial generalisers:** the representation is correct, and the decoder interpolates rather than composes
  (§49.3).
- **Generalisers:** neither is defective. D1 *hurts* them (0.54): their decoder composes parts and does not expect
  a canonical code.

**Why R7 works on 302 and not on 304:**
- The repair relies on three things:
  - the network's own trained classes (training labels);
  - bijectivity of unit rescaling, which identifies the 4 classes unit 2 never uses;
  - the network's own geometry, to assign the 4 unseen pairings to those classes.
- In 302 the code is close to purely canonical (R²_canonical 0.92) and the unit acts as a clean rotation, so the
  assignment is right 4/4.
- In 304 (R²_canonical 0.63) the rotation is consistent across trained values (spread 0.04), yet it predicts the
  wrong classes for the unseen values (0/4). Its geometry does not extrapolate.
- **Not explained:** why 304's rotation is consistent on trained values but wrong on unseen ones.

## 49.2 Representation × decoder compatibility (E2)

Affine maps fitted on 16,384 familiar inputs from network R's last-token residual into network D's:
- **The diagonal is 1.00 on familiar inputs**, a sanity check.
- **Every off-diagonal pair fails, even on familiar inputs** (0.24–0.43). Unseen accuracy is at chance (0.08–0.36).
- **Block-only transplants also fail** (≤ 0.52).

**Conclusion.** Within this task, representations and decoders are *network-specific*. No affine map aligns one
network's full representation well enough to drive another network's decoder. This contrasts with the factorial,
where a rotation transferred a low-dimensional canonical code at one site (0.41 vs 0.47; doc 46R.1). Moving a code
between networks requires more than linear alignment of the whole residual.

## 49.3 Where correct information stops being used (decoder trace, E5 §5)

Per unseen cell (mass unit 2, value 9–12):

| network | cells correct | far cells (block distance to trained twin > 2): accuracy | near cells (< 0.3): accuracy | block error vs additive prediction |
|---|---|---|---|---|
| 307 | 4/4 | 0.98–1.00 | 1.00 | 0.02–0.03 |
| 311 | 4/4 | 1.00 | 1.00 | 0.09–0.21 |
| 300 | 1/4 fully | **0.17, 0.33, 0.77** | 1.00 | 0.07–0.19 |
| 305 | partial | 0.22, 0.27, 0.74 | 0.66 | 0.08–0.26 |
| 304 / 310 | 0/4 | 0.08–0.09 | 0.11 | **0.26–1.41** |

- **Partial networks (300, 305).** The block for every unseen cell is where the factorised code predicts it (error
  ≤ 0.26).
  - The head answers correctly only when that block lies near a trained rendering of the same mass.
  - When the block is far, the head's active hidden units overlap the trained twin's by only 0.08–0.14, and it
    answers wrongly.
  - The head **interpolates by similarity; it does not compose.**
- **Generalisers (307).** The head answers correctly even for far cells with an equally small overlap (0.09–0.10):
  it computes the answer by a different internal route.
- **Information stops being used correctly inside the head MLP.** There is no other computation after the block.
  The attention layers precede the block, so they do not cause the decoder failure.

## 49.4 Developmental interventions (E4; training continuations) — final

All 46 registered runs completed, none failed: 40 from doc 48 and 6 from addendum A. Each run is 2,000 steps from a
bitwise-exact checkpoint. All **17 matched no-intervention controls are bitwise identical to the reconstruction at
every 100-step check (20/20 each)**, so each intervention is the only difference from history. Data:
`runs/phase4/dev_intervene/*.json`.

Columns:
- **unseen:** unseen-mass accuracy at the end, with the matched control in parentheses;
- **familiar:** familiar accuracy at the end;
- **fact.:** declared-block factorisation R²(value + unit), start → end;
- **swap:** discriminating unseen-source interchange at the end.

### Breaking generalisers (P2: reward the interaction, λ = −1; or interchange weight × 3)

| network | resume | intervention | unseen (control) | familiar | fact. | swap | registered disruption (unseen ≤ 0.5, familiar ≥ 0.95) |
|---|---|---|---|---|---|---|---|
| 307 | 850 (before dip) | reward interaction | **0.35 (0.99)** | 1.00 | 0.96 → 0.03 | 0.35 | **met** |
| 311 | 850 (before dip) | reward interaction | **0.33 (1.00)** | 1.00 | 0.93 → 0.04 | 0.21 | **met** |
| 307 | 850 | interchange ×3 | 0.96 (0.99) | 1.00 | 0.96 → 1.00 | 0.95 | not met |
| 311 | 850 | interchange ×3 | 0.32 (1.00) | 1.00 | 0.93 → 0.40 | 0.24 | met |
| 307 | 2,000 (consolidated) | reward interaction | 1.00 (1.00) | 1.00 | 1.00 → 1.00 | 0.99 | not met; the interaction did not form |
| 311 | 2,000 (consolidated) | reward interaction | **0.92** (1.00) | 1.00 | 0.98 → **0.15** | 0.91 | not met, although factorisation was destroyed |

### Rescuing failures (P1: penalise the interaction; P3: strong decay; F1 freeze; G1 projection)

| network | resume | intervention | unseen (control 0.10 unless noted) | familiar | fact. |
|---|---|---|---|---|---|
| 304 | **600** | λ = 1 for 500 steps | **0.52** | 1.00 | 0.98 → 0.99 |
| 304 | 700 / 850 | λ = 1 for 500 steps | 0.10 / 0.10 | 1.00 | → 0.85 / 0.88 |
| 304 | 700 / 850 / 1,000 / 1,500 / 4,000 | λ = 1 throughout | 0.11 / 0.10 / 0.10 / 0.10 / 0.10 | 0.97–1.00 | → 0.97–0.99 |
| 304 | 2,000 | λ = 1 / 0.1 / wd 1.0 | 0.10 / 0.10 / 0.25 | 1.00 | → 0.99 / 0.88 / 0.41 |
| 304 | 700 | freeze attention-2 q/k for 500 steps (F1) | 0.19 | 1.00 | 0.94 → 0.44 |
| 304 | 700 | gradient projection on attention-2 q/k (G1) | 0.10 | 1.00 | 0.94 → 0.74 |
| 310 | **1,200** | λ = 1 for 500 steps | **0.61** | 1.00 | 1.00 → 0.97 |
| 310 | 1,300 / 1,350 | λ = 1 for 500 steps | 0.55 / 0.55 | 1.00 | → 0.96 / 0.97 |
| 310 | 1,450 | λ = 1 for 500 steps | 0.22 | 1.00 | 0.88 → 0.88 |
| 310 | 2,000 | λ = 1 / 0.1 / wd 1.0 | 0.34 / 0.17 / 0.10 | 1.00 | → 0.98 / 0.84 / 0.48 |
| 302 | 2,000 | λ = 1 / 0.1 / wd 1.0 | 0.39 / **0.52** / 0.51 (control **0.19**) | 0.63 / 1.00 / 0.58 | → 0.99 / 0.91 / 0.11 |

No rescue run met the registered criterion (unseen ≥ 0.90 with familiar ≥ 0.95).

### Reading [C: causal interventions, exploratory]

1. **Factorisation is causally necessary *during development*.**
   - Rewarding the interaction from step 850 destroyed generalisation in **both** generalisers (0.99/1.00 → 0.35/0.33).
   - Familiar accuracy stayed at 1.00.
   - Both meet the registered disruption criterion.
2. **After consolidation it is no longer necessary.**
   - In 311, the same intervention from step 2,000 destroyed factorisation (0.98 → 0.15) but generalisation survived
     (0.92). The head, once composing, keeps composing.
   - 307 resisted entirely.
   - Necessity holds in the developmental window, not after it: **developmental path dependence in the breaking
     direction.**
3. **Factorisation is not sufficient, and prevention works only before the dip.**
   - Penalising the interaction before each failure's dip onset gave partial generalisation: 304 0.52; 310 0.55–0.61.
   - Starting once the dip had begun gave 0.10–0.22.
   - Restoring factorisation after lock-in gave 0.10–0.34, even at factorisation 0.98–0.99.
   - Addendum A's prediction (an earlier start rescues more) is **supported**: 310 shows 0.61 → 0.55 → 0.55 → 0.22
     as the start moves later.
   - The rescue criterion was **not met**.
4. **The missing ingredient is the composing decoder.**
   - Every partial rescue keeps the block factorised (≥ 0.96), yet unseen accuracy stalls at 0.5–0.6.
   - This matches the partial generalisers 300 and 305, whose decoders interpolate (§49.3).
5. **Other levers.**
   - Strong weight decay helps only marginally (304: 0.25), and it makes the code less factorised: the factorial's
     other route.
   - Freezing (F1) and gradient projection (G1) on attention-2 query/key did not help.
   - A weak penalty prevented 302's post-fit erosion (0.52 vs 0.19).

## 49.5 Reverse intervention without training (E3)

The interaction is synthetic, double-centred and random, scaled by κ × the mean part norm. Means over ranks 1–8.

| | κ = 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| 307, all cells, inside causal subspace: unseen / familiar | 0.88 / 0.95 | 0.65 / 0.77 | 0.45 / 0.51 | 0.26 / 0.32 |
| 307, all cells, outside causal subspace | 0.95 / 0.97 | 0.83 / 0.88 | 0.67 / 0.72 | 0.48 / 0.53 |
| 307, unit-2 cells, inside causal subspace | 0.88 / 0.99 | 0.65 / 0.93 | **0.45 / 0.85** | 0.26 / 0.79 |
| 311, unit-2 cells, inside causal subspace | 0.87 / 0.96 | 0.64 / 0.90 | **0.43 / 0.83** | 0.28 / 0.78 |

**Registered predictions:**
- **Monotone in κ:** confirmed.
- **Inside the causal subspace hurts more than outside:** confirmed (0.45 vs 0.67 at κ = 1).
- **Unit-specific interactions hurt unseen accuracy more than familiar:** confirmed, by about 2× at κ = 1.

The "unseen cells only" variant leaves familiar accuracy at 1.00 by construction, so it is not counted as evidence.
Rank had no monotone effect.

Together with §49.1, the same representation can be **repaired** (302, by completing its canonical code) and
**broken** (307/311, by injecting interaction) by single-pass interventions. The repair and the break act on the two
different coding schemes that the generalising and failing networks use.

## 49.6 Permanent weight-level editing and semantic programming (E5, E6)

### Mathematical conditions for exact compilation (no hooks, no fitting, no training)

1. **Token-level edits compile exactly.** "Read token t as token t′" for any map of token identities compiles as
   replacing embedding row t by the original row t′.
   - The edited network computes *exactly* what the original computes on the re-written input. This is an identity
     of functions, not an approximation.
   - It covers: written-value maps, which apply to both quantities because value tokens are shared; mass-unit,
     acceleration-unit and output-unit relabelling, which are role-specific tokens.
   - **Composition is exact:** a sequence of row replacements is a single row replacement whose map is the
     composition of the maps.
   - **Reversibility:** permutation programs are exactly invertible, verified bit for bit.
2. **Linear maps on the last-token residual compile exactly into the head.** For T invertible:
   - W_in ← W_in T;
   - W_U ← W_U T;
   - W_out ← T⁻¹ W_out.

   Affine translations do not compile exactly, because W_U has no bias.
3. **Edits that must distinguish variables sharing a token cannot be a single linear change at any one site.**
   - Example: mass-only increment, since value tokens are shared by mass and acceleration.
   - The change needed depends on the token's *role*, which is available only as a nonlinear function of the
     context.
   - Least-squares compilation into the attention-2 value path failed: intended 0.06–0.44, with an ill-conditioned
     parameter displacement of 5–75× the network norm.
   - Reported as a negative result.
4. **No token-level program can repair a failure while preserving meaning.**
   - A semantics-preserving recoding of "unit 2, value v" into a trained rendering would need a mass unit with scale
     5/8 ≡ 12 (absent).
   - It would also need acceleration scales multiplied by 5 ({5, 10, 9}, absent).

### Instruction set results

Compiled into weights, no hooks. Correct = matches the mathematically composed target.

| network | single instructions, familiar | single instructions, unseen masses | 40 random 2–3-step programs, familiar / unseen | inverse restores bitwise |
|---|---|---|---|---|
| **307** | 1.00 (all 10) | 0.996–1.00 | **1.00 / 0.999** | yes |
| **311** | 1.00 | 0.997–1.00 | **1.00 / 1.00** | yes |
| 300 | 0.89–1.00 | 0.53–1.00 | 0.95 / 0.80 | yes |
| 305 | 0.93–1.00 | 0.48–1.00 | 0.97 / 0.79 | yes |
| 304 | 0.84–1.00 | 0.09–1.00 | 0.92 / 0.64 | yes |
| 310 | 0.83–1.00 | 0.10–1.00 | 0.92 / 0.65 | yes |

The instructions are increment, decrement, doubling, inversion and constant (written values), mass-unit relabelling,
mass-unit rotation, acceleration-unit rotation, output-unit swap, and identity.

**Programmability is bounded by generalisation.** The compiled programs are exactly correct as programs. What fails
in 304/310 is the network's computation on the renderings the program routes inputs into. When a program maps
familiar inputs onto never-trained renderings, the failures give wrong answers (e.g. 304 "read unit 0 as unit 2":
0.66). The generalisers execute every program correctly.

## 49.7 Concept Formation Atlas (heads; `atlas_heads.json`)

**Layer-2 head share of the declared block's variance:**
- 304: one head writes 0.95 (it reads the mass-unit token, which layer 1 had already bound to its value with
  attention 0.46);
- 310: one head writes 0.92 (it reads the mass-value token, with pair-dependent attention);
- 300: one head writes 0.90;
- 307 and 311: the block is split across two heads (0.49/0.49; 0.58/0.36).

**Zero-ablating any single head** in any network costs 8–75 points of familiar accuracy, so no head is redundant.

**No single head-level signature separates the groups.** The two failures build their interaction by different
circuits: a layer-1 unit-token binding in 304, and pair-dependent layer-2 attention in 310.

## 49.8 Levels of a concept (doc 48 §9)

| property | 307/311 (generalise) | 300/305 (partial) | 302/304/310 (fail) |
|---|---|---|---|
| present (decodable on familiar inputs) | yes | yes | yes |
| accessible (designed-block interchange) | yes (1.00) | yes (1.00) | yes (1.00) |
| causally used on **unseen** inputs (discriminating swap) | yes (0.99) | partly (0.30–0.49) | no (0.00–0.10) |
| independently editable (per-part activation edit, doc 46R.2) | 307 yes (1.00); 311 partly | partly | no |
| compositionally reusable (unseen accuracy) | yes | partly | no |
| permanently programmable (token-level programs, unseen) | yes (≥ 0.996) | partly (0.48–1.00) | only where programs avoid unseen renderings |

These six properties dissociate in these networks; none implies the next.

## 49.9 Contradictions and open questions

- **C1 is refuted as sufficient.** It was the doc-46 primary candidate: enforcing or restoring factorisation did not
  produce generalisation (304).
  - The decisive missing ingredient is a *composing decoder*: how the head comes to compose rather than
    interpolate.
  - Nothing measured so far predicts which head will form.
- **Unexplained:** why 304's geometry extrapolates wrongly even though its unit-2 rotation is consistent across
  trained values.
- **Factorial vs role-swap:**
  - The factorial's generalising route is a *canonical* code read by a reusable reader, under strong decay.
  - The role-swap's generalising route is a *factorised* code composed by the head.
  - The two studies therefore show different solutions to compositional generalisation. Which one a network finds
    depends on supervision and regularisation.
- **PR #34's crossing (pending) will show which random stream carries the outcome.** Its results can be scored for
  factorisation and for the decoder-interpolation signature without retraining.
