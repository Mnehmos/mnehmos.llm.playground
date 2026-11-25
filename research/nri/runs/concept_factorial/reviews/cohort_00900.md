# Cohort review — step 900 (96/96)

**What changed.** C wd 1.0: x1 canonical − number +0.05 (from +0.02), held-out 0.04, decodability 0.82; C wd 0.1
flat (−0.10, 0.02, 0.68). Weight norm at wd 1.0 now ≈ 51-52 (half of initial), at wd 0.1 ≈ 103. D decodable
onsets at 800: seed 2010 (wd 0.1), seed 2001 (wd 1.0).

**Anomalies investigated (measured).** B wd 1.0 seed 2002, steps 838-854+: second large instability episode
(batch task loss up to 0.50, gradient up to 13.9 vs median 0.83, trained items to loss 6.9-7.7). B wd 1.0 seed
2001: small tail of its step-760 episode (gradients 0.46-0.91, batch loss ≤ 0.045). Both large episodes so far are
in **B at wd 1.0**, consistent with explanation (b) from step 800 (decay drifts a fitted lookup until items cross a
decision boundary), not with (a) alone. Still anecdotal; whole-run count pending.

**Prediction.** By step 2,000 at least 4 of 12 B wd 1.0 networks will have had an episode with batch task loss
> 0.2, versus at most 1 of 12 B wd 0.1.
