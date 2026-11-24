# Cohort review — step 1,300 (96/96)

**Previous prediction — confirmed.** C wd 1.0 seed 2002 recovered: own-trained 0.925 → 1.00, decodability 0.88 →
0.89, raw circle 0.59 → 0.63 (all three registered thresholds met).

**Second mid-episode capture, same signature.** C wd 1.0 seed 2007: own-trained 1.00 → 0.94 at this audit while
canonical decodability stays 0.90 — again a readout-first failure with the canonical code intact. Two of two
captured episodes now fit the "readout-first" reading.

**What changed.** Episodes continue to spread at wd 1.0 only: 12 sustained flags this cohort (B 2003/2011/2012,
C 2002/2007/2008/2009, D 2001/2002/2003/2010/2011; gradients up to 52). Zero at wd 0.1 since step 600. C wd 1.0:
canonical − number +0.13 (per seed 0.00…+0.27), held-out 0.06, decodability 0.89 (seed 2008's decodable onset
finally at 1,200 — the latest in C wd 1.0). Weight norm at wd 1.0 ≈ 37-39 and still falling slowly; at wd 0.1 ≈ 99-100.

**Emerging picture (exploratory).** At wd 1.0 the weight norm keeps shrinking toward an equilibrium and
instability episodes become common in B, C and D (not A); at wd 0.1 neither happens. In C wd 1.0 the canonical code
survives every episode observed; what breaks is the lookup of trained answers. Hypothesis H-episode: episodes are
the mechanism by which weight decay destabilises lookup solutions — each episode forces a re-fit, and a re-fit in
a network that can read x1 as a quantity has a cheaper canonical route available. Prediction: in C wd 1.0, held-out
accuracy and x1 canonical − number increase more across episode blocks than across calm blocks (paired within
network, to be tested at the end); in B and D, episodes re-fit the lookup without held-out gain.
