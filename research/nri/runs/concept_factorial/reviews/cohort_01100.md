# Cohort review — step 1,100 (96/96)

**What changed.** C wd 1.0 continues to consolidate: canonical − number +0.10, held-out 0.05, decodability 0.86.
C wd 0.1 flat (−0.09 / 0.02 / 0.71). D decodable onset at 1,000 for seed 2007 (wd 0.1).

**New instrument — `sci/episodes.py`** (built this cohort, tested on planted logs: a calm log yields no episode,
a planted 6-step burst is found in exactly its block). Episode block = post-fit 100-step block with batch task loss
> 0.2 and ≥ 3 steps above 3× the block-median gradient. Retrospective count, steps 300-1,100 (all networks):

| | A | B | C | D |
|---|---|---|---|---|
| wd 0.1 — networks with ≥ 1 episode | 0/12 | 0/12 | 0/12 | 0/12 |
| wd 1.0 — networks with ≥ 1 episode | 0/12 | **6/12** | 2/12 | 2/12 |

(The step-500 B wd 0.1 bursts stayed below batch loss 0.2 and are not counted.) Step-900 prediction (≥ 4/12 B
wd 1.0 vs ≤ 1/12 B wd 0.1 by step 2,000) is met at 1,100 for the B wd 1.0 side; the wd 0.1 side stays open until
2,000. Explanation (b) — weight decay drives fitted lookups to their decision boundaries — is favoured over pure
Adam small-gradient instability (a), which would not single out wd 1.0. A (42 combinations, one frame) shows no
episodes at either weight decay: a small lookup sits further from its boundaries.

**This cohort's sustained flags** (B wd 1.0 seeds 2006, 2009, 2010; C wd 1.0 seed 2001; D wd 1.0 seed 2006; gradients
up to 23.0) are all at wd 1.0 and will be included in the next count.

**Prediction (Q5 link).** When C wd 1.0 networks compose, their held-out collapses will coincide with episode
blocks (≥ 70% of held-out drops ≥ 0.10 fall in or right after an episode block). Falsifier: collapses mostly in
calm blocks.
