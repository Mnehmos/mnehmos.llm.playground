# Cohort review — step 300 (96/96)

**What changed.** Every population now fits its own trained combinations (0.99-1.00; prediction (i) from step 100
**confirmed**). Held-out accuracy is still ≤ 0.03 and held-out task loss keeps climbing (B 21.5 at wd 0.1 / 16.9 at
wd 1.0; C 16.4 / 13.3): confident wrong answers on unseen combinations. Weight decay 1.0 keeps held-out loss lower
in every population (≈ 20-25% lower) — the first behavioural-adjacent weight-decay effect.

**Representation.** C's canonical quantity is now decodable on held-out combinations at 0.52 (wd 0.1) and 0.46
(wd 1.0) (kNN; onset criterion 0.5) — the "representation" transition is happening now, slightly earlier at
wd 0.1; D carries a weaker version (0.32-0.36: its scrambled-per-unit labels still require reading value and
unit, which makes q partly recoverable); A/B ≈ 0.2. C's raw-grouped circle 0.46-0.47 at both weight decays.

**Causal reading of x1.** C's number-token code is still read as the raw number (canonical − number −0.18/−0.19),
as predicted at step 200. B/D −0.30…−0.32: strongly number-read.

**Gradients.** Task-aux gradient cosine at the x1m source token ≈ 0 in C and D (−0.01…0.01): the auxiliary and
task losses push x1m in *unrelated* directions — they neither reinforce nor conflict on average. This is a
candidate explanation for Q3 (why representation precedes use): the task loss, while trained combinations can be
fitted by lookup, exerts no pressure toward reading x1m as a quantity, so the representation the auxiliary
builds is orthogonal to what the task gradient currently needs.

**Update dynamics.** W_in adaptive/decay cosine at wd 1.0 reached −0.37…−0.44 (wd 0.1: −0.18…−0.36): the readout
keeps growing against a decay that is now ~10× stronger at wd 1.0; weight norm 88 vs 109.

**Prediction (testable at 400-1,000).** The task-aux cosine at x1m in C stays within ±0.05 until the task loss on
trained items is near zero, and becomes **positive** (> 0.05) before compositional onset at wd 1.0 — i.e. when the
task gradient starts to *use* the canonical code it will align with the auxiliary gradient. Competing: the cosine
stays ≈ 0 throughout and composition is driven downstream of x1m (then M3 should also fail).
