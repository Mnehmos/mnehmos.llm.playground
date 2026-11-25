# Cohort review — step 2,000 (96/96)

**Predictions resolved.** Step 900 (≥ 4/12 B wd 1.0 vs ≤ 1/12 B wd 0.1 with an episode by 2,000): **confirmed**
(B wd 1.0 ≥ 6/12 since 1,100; B wd 0.1 0/12 through 2,000, episode counter). Step 1,800/1,900 (seed 2004
compositional onset at 1,900-2,000): **not yet** — 0.36 at 1,900, **0.38 at 2,000**; the registered two-audit rule
can place the onset at 2,000 only if step 2,100 is also ≥ 0.37.

**What changed.** C wd 1.0 held-out mean 0.15 (from 0.11), new-frame 0.10; per seed 0.00-0.38. Episodes are now
large and frequent at wd 1.0: own-trained at the audit fell to 0.29 (A 2012), 0.67 (B 2003), 0.79 (C 2001), 0.91
(D 2007), 0.94 (B 2001). Condition means of own-trained at wd 1.0 dipped (A 0.94, B 0.97, C 0.98): the cost of
weight decay 1.0 on the *trained* task is now visible in the means.

**Notable: held-out rose during an episode.** C wd 1.0 seed 2001: own-trained 1.00 → 0.79 while held-out
0.01 → 0.15 (canonical − number 0.37 → 0.21). A breaking lookup coinciding with *more* generalisation is what
H-episode predicts (episodes dislodge the lookup; the canonical route covers more combinations); seed 2012 at
1,700 showed the opposite (held-out fell). Both directions occur; the population-level test decides.

**Prediction for 2,100.** Seed 2004 stays ≥ 0.37 (onset recorded at 2,000); seed 2001 recovers own-trained ≥ 0.98
with held-out ≥ 0.10 retained.
