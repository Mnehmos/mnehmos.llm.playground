# 10. R1: white-box vs black-box LLM interpreters

*Packages: `pilot/make_blind_package.py --mode white|black`. Scoring: `pilot/score_pilot.py`.
Cross-evaluation: `pilot/cross_eval.py`. Artifacts (each interpreter's NRI document, edit program,
report, scratch scripts and scores): `results/r1/`. Cross-evaluation data:
`results/r1/cross_eval.json`.*

## 10.1 Question

Can an LLM learn something from a network's weights and internals that it cannot learn by querying
the network? *Computation, not weight access* (2611.07847) found black-box audits matching
white-box ones on clinical networks. Here the test uses executable, refutable NRI claims, plus edit
programs scored against a hidden goal.

## 10.2 Design (pilot: 2 networks × 2 access modes, n = 1 per cell)

| | **baseline network** (`none_s1`, doc 08) | **role-supervised network** (`aux_only_s1`, doc 08) |
|---|---|---|
| **white-box**: weights, architecture code, activations, Python | `net_P4` | `net_W8` |
| **black-box**: `predict()` queries only; told the architecture shape and the edit API | `net_B3` | `net_J6` |

**Common to all four:**

- No task description, no type semantics, no training code. The auxiliary head was stripped from
  the shipped weights.
- The behavioral edit goal: "wherever token 11 occurs, output what you would if it were 12".
- The corrected instructions, which state that `linear_rep` is tested by interchange intervention.

**Black-box arm:**

- The network sits behind a subprocess oracle in a hidden directory.
- Its edit program is never run by the agent. The evaluator runs it once against the real network,
  where the program may read activations at run time. So the question is whether the *agent* needs
  weight access to write a working weight program.

**Blinding audit:** a transcript scan of all four runs (25–34 tool calls each) found no access to
the hidden directory, weights, answer keys or any path outside the package.

## 10.3 Results on the network each interpreter studied

| | P4 (white, baseline) | W8 (white, role-sup.) | B3 (black, baseline) | J6 (black, role-sup.) |
|---|---|---|---|---|
| exact rule recovered (20,000/20,000) | ✓ | ✓ | ✓ | ✓ |
| variable recall op / a / b | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |
| claims verified | **7 / 7** | **9 / 10** | 4 / 7 | 5 / 9 |
| internal claims (`linear_rep`, `necessity`) verified | **5 / 5** | **6 / 7** | 2 / 4 | 2 / 6 |
| edit vs true goal: success / preserve | 0.11 / 0.99 | 0.78 / 0.99 | 0.31 / 0.82 | 0.76 / 0.98 |
| edit rank, ‖ΔW‖ / ‖W_in‖ | 58, 1.2 | 50, 4.1 | 64, 5.8 | 64, 3.1 |

**What the white-box interpreters found that black-box ones could not.** Both white-box agents
discovered the actual representational fact at the site, that `a`, `b` and `op` are jointly
entangled. Each made a *joint* subspace claim (rank 18–20), which verified at 1.00 by interchange.
They declined to claim per-variable subspaces, and set their edit thresholds to match what they
measured (P4 claimed edit success ≥ 0.08 and got 0.11). That is calibrated, falsifiable
self-knowledge about another network. The black-box agents guessed per-variable `linear_rep` claims
that mostly failed.

## 10.4 Cross-evaluation: every program on networks it never saw

Success / preservation on the true goal. Hosts are other seeds of the baseline and role-supervised
networks, plus **IIT networks from doc 09**.

| program ↓ / host → | none s0 | none s2 | aux s0 | aux s2 | **iit s0** | **iit s2** |
|---|---|---|---|---|---|---|
| P4 (white, written on baseline) | 0.43 / 1.00 | 0.31 / 1.00 | 0.95 / 1.00 | 0.82 / 1.00 | 0.88 / 1.00 | **1.00 / 1.00** |
| B3 (black, written on baseline) | 0.50 / 0.99 | 0.52 / 0.97 | 0.98 / 1.00 | 0.87 / 1.00 | 0.92 / 1.00 | **1.00 / 1.00** |
| W8 (white, written on role-sup.) | 0.49 / 0.98 | 0.40 / 0.97 | 0.98 / 1.00 | 0.83 / 1.00 | 0.89 / 1.00 | **1.00 / 1.00** |
| J6 (black, written on role-sup.) | 0.61 / 0.89 | 0.66 / 0.82 | 0.99 / 1.00 | 0.89 / 0.99 | 0.91 / 1.00 | **1.00 / 1.00** |

Means by access mode and host type:

| | baseline hosts | role-supervised hosts | IIT hosts |
|---|---|---|---|
| white-box programs | 0.41 (preserve 0.99) | 0.90 (1.00) | 0.94 (1.00) |
| black-box programs | 0.57 (preserve 0.92) | 0.93 (1.00) | 0.96 (1.00) |

## 10.5 Findings

1. **Behavioral understanding needs no weight access.** All four interpreters recovered the exact
   rule and every ground-truth variable, black-box included. This replicates the *Computation,
   not weight access* result in a new setting.
2. **Mechanistic understanding does need internal access, and the NRI measures it.** White-box
   interpreters made 11 of 12 internal claims that survived interchange and ablation tests. That
   includes a correct discovery of the network's joint (entangled) code, which no query can
   reveal. Black-box interpreters' internal claims were guesses: 4 of 10 survived. Note that the
   white-box advantage here comes from **experiments on activations**, not from reading raw weights
   alone; the agents used both (§10.6).
3. **Editability is a property of the network, not of the interpreter.**
   - The same four LLM-written programs succeed 0.41–0.57 of the time on baseline networks,
     0.90–0.93 on role-supervised ones, and **0.94–0.96 on IIT networks**. Every program reaches
     1.00 on one IIT seed.
   - Host type explains far more variance than access mode or author.
   - Black-box programs were *as good as* white-box programs at editing. An agent that never saw a
     weight can write a working weight program, because the program reads the network at run
     time, as WeightCode's do. Their preservation on baseline networks is lower (0.82–0.89), so they
     are less careful when the network is hard.
4. **All agent-written edits are high-rank regression fits** (rank 50–64, 1–6× the weight norm),
   not targeted rank-one programs. On IIT networks they still work. Training for structure makes
   even crude editing methods reliable.

## 10.6 Limitations, and the test that remains for "reading raw weights"

- n = 1 per cell, one LLM family, honor-system blinding (audited).
- The **white-box arm gives activations and code execution**, so it does not isolate *raw-weight
  reading*. The belief this experiment was meant to address (that an LLM understands something from
  raw weights that cannot be learned by querying) needs a third arm: **weights only, no forward
  passes, no code execution on the network**, as in *Sparks* experiments 1, 2 and 4. The
  packaging code supports it with one change: ship the weights as text and omit `arch.py`'s
  forward pass. That is the next R1 step.
- Black-box agents knew the architecture shape and the edit API. A strict black-box arm would hide
  those too.
