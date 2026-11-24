# Cohort review — step 600 (96/96)

**What changed.** C canonical − number at x1: wd 1.0 −0.07 → **−0.04**, wd 0.1 −0.14 → −0.12: the weight-decay gap
in C's causal reading of x1 persists (0.08) and both drift toward "quantity". C held-out decodability 0.74 (wd 1.0)
vs 0.66 (wd 0.1) — now also slightly higher at wd 1.0. C's x2 canonical semantics creeping up at wd 1.0 (0.10).
Held-out accuracy unchanged (≤ 0.03); held-out loss flat (C 12.2 / 19.4).
Decodable onsets recorded at 500: C wd 0.1 three more seeds (2003, 2004, 2011), C wd 1.0 one more (2011).

**Update dynamics.** At wd 1.0 the W_in adaptive/decay cosine keeps falling (A −0.72, B/C/D −0.54…−0.56): as
the decay-limited weights approach equilibrium, the adaptive update is increasingly the direct *opposite* of the
decay (Adam restoring what decay removes) — a sign of an equilibrium-norm regime. Weight norms: wd 1.0 66-67, wd 0.1
105-107.

**Anomaly investigated.** C wd 0.1 seed 2001 (sustained flag): a single burst at steps 515-518 (gradient 1.5 → 5.9
with the *auxiliary* loss jumping 0.00 → 0.18 and the worst item's task loss to 2.8), then quiet. A C-specific
event involving the auxiliary head — the first case where the canonical code itself is perturbed; logged as a
candidate precursor to collapse.

**Prediction (step 1,000).** C wd 1.0 reaches canonical − number ≥ 0 (x1 read as quantity at least as often as
raw number) by step 1,000 in ≥ 6/12 seeds; C wd 0.1 in ≤ 3/12.
