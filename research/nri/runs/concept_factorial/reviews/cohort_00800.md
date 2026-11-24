# Cohort review — step 800 (96/96)

**What changed.** C at wd 1.0: mean x1 canonical − number crossed zero (+0.02), held-out accuracy 0.04 (first
cohort mean above the 0.02-0.03 floor). C wd 0.1 unchanged (−0.11, held-out 0.02). D decodable onset at 700 for
seed 2004 (wd 1.0). B at wd 1.0 remains strongly number-read (−0.25…−0.55 across seeds).

**Anomaly investigated — largest episode so far.** B wd 1.0 seed 2001, steps 759-780 (measured from interval log):
batch task loss jumps from ≈ 0.03 to 0.17-0.38 for ~18 steps, gradient norm up to 12.5 (median 0.31), single trained
items reaching loss 5-8.7, alternating with near-baseline steps; fully re-fitted by the audit (own-trained 1.00,
held-out 0.00). An instability episode of the item-flip type, but an order of magnitude larger and at **wd 1.0** —
so the step-500 prediction ("sustained episodes more frequent at wd 0.1") is in doubt; to be settled by the whole-run
episode count, not by anecdotes.

**Competing explanations for episodes.** (a) Adam instability when the second-moment estimate has decayed on a
near-zero-loss lookup (a single rare item's gradient then produces a large step); (b) weight-decay-driven drift of a
fitted lookup until an item crosses its decision boundary, then a corrective burst. (a) predicts episodes
independent of weight decay and correlated with how small the median gradient has become; (b) predicts episode
rate rising with weight decay. Both predict that episodes in C coincide with transient losses of held-out accuracy
once C composes.

**Prediction.** Over the whole run, episode rate (sustained flags per 100 steps) will be higher at wd 1.0 than wd
0.1 in B (favouring b) — replacing my step-500 prediction, which I mark as likely wrong but not yet tested.
