# Cohort review — step 5,400 (96/96)

**First utilisation onset at weight decay 0.1 — triggered by an episode (C wd 0.1 seed 2006, measured).**
Block 5,101-5,200 median gradient 0.002 (no hot steps); block 5,201-5,300: explosive episode at 5,229-5,240
(gradient 21 → **58**, batch task loss 1.4-8.4, items to loss 64, auxiliary loss 0.001 → 0.29); x1 canonical −
number 0.09 (5,200) → **0.20** (5,300) → 0.20 (5,400) = registered utilisation onset at 5,300. Held-out unchanged
(0.00-0.06), decodability 0.71 → 0.64. Small-gradient precursor: **4/4** wd 0.1 episodes.

**Synthesis — a candidate mechanism for the weight-decay interaction (Q2/Q4, exploratory).** Across the run, a
step change toward canonical use (and, in composers, toward composition) has repeatedly followed an instability
episode: C wd 1.0 seeds 2007 (1,300-1,400), 2010 (3,700-3,800, then composition at 4,100), 2004 (ratchet after each
collapse), 2001 (held-out rising through episodes), and now C wd 0.1 seed 2006. Episodes are frequent at wd 1.0
(decay/Adam equilibrium with thin margins) and rare at wd 0.1 (only after the gradient decays to ~0.001-0.005). If
episodes are the events that dislodge route-specific lookup and let the re-fit settle on the canonical code, then
weight decay promotes composition largely *by producing episodes*. Falsifiable predictions for the final analysis:
(1) in C networks, the change in x1 canonical − number across episode blocks exceeds that across calm blocks (paired
within network); (2) in C wd 0.1, the only networks with any utilisation onset are those that had an episode.

**Other.** Episodes back at wd 1.0 (16 flags) — the two quiet cohorts at 5,200-5,300 were not a trend (step-5,300
revision withdrawn). C wd 1.0 held-out 0.59; seed 2005's canonical − number fell to 0.03 during its episode while
held-out stayed 0.46.
