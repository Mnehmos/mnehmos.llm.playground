# Cohort review — step 1,200 (96/96)

**What changed.** C wd 1.0: canonical − number +0.12, held-out 0.06, decodability 0.89. C wd 0.1 unchanged. Sustained
flags in 10 networks, **all at wd 1.0** (B 2003/2005/2006/2008/2010, C 2002, D 2001/2006/2009/2010; gradients up to
32-47).

**Anomaly investigated individually — first episode caught in progress at an audit (C wd 1.0 seed 2002).**
Measured: small bursts start at step 1,165 (gradient 1.1-2.1, batch loss ≤ 0.03), escalate after step 1,181 and
peak at 1,195-1,200 (gradient 6.7-14.9 in the last six steps, 46.8 at the block maximum; batch task loss 0.38-0.70;
worst items 3.7-11.1). The auxiliary loss rises 0.019 → 0.074 during the escalation. Audit at 1,200 (mid-episode):
own-trained 1.00 → 0.925; held-out 0.00 (unchanged — this seed had not composed); canonical decodability 0.89 → 0.88
(unchanged); raw-value circle at x1m 0.66 → 0.59; x1 canonical − number 0.04 → 0.08.

**Reading.** An episode is a *readout/lookup* failure first: trained answers break, while the canonical code at x1m
stays decodable and is still read as a quantity; its geometry is perturbed (circle −0.07) and the auxiliary loss
quadruples, so the disturbance does reach x1m. The ~30-step escalation from small bursts to a large episode is a
candidate **early-warning signature** (precursor window).

**Prediction.** Seed 2002 recovers own-trained accuracy to ≥ 0.99 by step 1,300 with decodability unchanged, and its
raw circle returns to ≥ 0.63. Generally: in episode blocks, the drop in own-trained accuracy at the audit is larger
than any change in canonical decodability (readout-first failure).
