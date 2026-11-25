# Cohort review — step 2,500 (96/96)

**COLLAPSE OF THE FIRST COMPOSER, captured at 100-step resolution (C wd 1.0 seed 2004).** Measured:

| | 2,300 | 2,400 | 2,500 |
|---|---|---|---|
| own-trained | 1.00 | 1.00 | **0.53** |
| held-out (dev) | 0.57 | 0.61 | **0.26** |
| held-out, new frame | 0.22 | 0.22 | 0.33 |
| canonical decodable (x1m, kNN) | 1.00 | 1.00 | 0.89 |
| x1 patch: canonical / number | 0.46 / 0.15 | 0.46 / 0.16 | 0.35 / 0.12 |
| dlog circle raw / canonical (x1m) | 0.58 / 0.57 | 0.58 / 0.57 | 0.58 / 0.59 |
| auxiliary loss (panel) | 0.013 | 0.012 | 0.108 |
| W_in: cos(adaptive, decay) per block | −0.84 | −0.86 | **−0.07** |
| W_in: adaptive path / decay path | 1.03 / 0.75 | 1.01 / 0.76 | **4.69** / 0.76 |
| W_out: cos(adaptive, decay) | −0.80 | −0.82 | **−0.10** |

Interval log: small bursts begin at step 2,471 (gradient 2-4 vs block median 0.65, batch loss ≈ 0.01), escalate, and
the final steps 2,496-2,500 are a full breakdown (gradient 5.6-12.3, batch task loss 1.4-1.6, auxiliary loss 0.19-0.41).

**Reading (Q5 mechanism candidate).** In the stable composing regime the readout is in a **weight-decay/Adam
equilibrium**: each block, the adaptive update almost exactly cancels the decay (cosine ≈ −0.85, adaptive path ≈
decay path). The collapse is the **breakdown of that equilibrium** — the adaptive update stops opposing the decay
and becomes 4-5× longer, in a direction nearly orthogonal to it. The breakdown is readout-first (own-trained and
held-out fall sharply) while the representation is largely spared (geometry unchanged, decodability 1.00 → 0.89,
canonical reading 0.46 → 0.35). The small-burst escalation over the preceding ~25-30 steps is the second recorded
instance of the precursor window (first: C 2002 at 1,165-1,195).

**Competing explanations.** (a) Equilibrium breakdown is a *consequence* of item-level failures (a few items cross
their decision boundary, gradients spike, Adam's normalised step on low-second-moment parameters is large), not a
cause; (b) a slow drift of the adaptive/decay balance precedes the episode and could serve as a precursor. Under (b)
the per-block cosine should creep toward zero in blocks *before* the episode; here it was −0.84 → −0.86 (no creep)
→ favours (a) at block resolution. A finer instrument (per-step cosine in the readout) can be computed from the saved
checkpoints + interval batches on copies — scheduled for the analysis.

**Prediction.** Seed 2004 recovers within 300 steps: own-trained ≥ 0.99, held-out ≥ 0.50, readout cosine back
≤ −0.7 by step 2,800, with canonical decodability ≥ 0.95. If held-out does not recover, the pilot's "collapse and
recovery" generalisation fails for this seed.

**Episode count through 2,500** (counter): wd 1.0 — every B, C and D network has had ≥ 1 episode block (30-32 blocks
each population), A 10/12 (15 blocks); wd 0.1 — none.
