# 15. Role-swap results: declaring a variable makes *that* variable easier to edit

*Registration: doc 14, commit `ac005c8`, made before the main run. Analysis: the frozen
`run_swap.py analyse` and `report` modes. Full auto-generated report: `results/swap/summary.md`.
Data: `results/swap/metrics.json`. 36 networks (3 groups × 12 fresh seeds, 300–311). No
deviations from the registration.*

## 15.1 Verdict in plain English

**Yes, with an important limit.**

When we gave **mass** its own place in the network, the mass edit became more accurate than the
acceleration edit. When we gave **acceleration** its own place instead, the advantage **moved to
acceleration**. That happened in all 12 matched pairs of networks.

Declaring a variable helped that variable's edit and did not measurably hurt the other one. So
the credit belongs to *declaration*, not to mass being easier to edit.

The limit: even for the declared variable, the rewritten network gave the intended answer only
18% (mass) to 31% (acceleration) of the time on eligible examples, and about half the time it
gave some other answer.

The diagnostic points at the cause. **Copying a real internal pattern into the block works
81–91% of the time.** The fitted edit in the same block works only 19–32%. The network carries
the variable in a usable place; our *method for constructing the replacement pattern* is what
falls short.

## 15.2 What happened when each variable was declared (inversion edit, all 12 seeds)

| | mass edit specificity | acceleration edit specificity | mass edit: intended / wrong / unchanged / other | accel. edit: intended / wrong / unchanged / other |
|---|---|---|---|---|
| **mass declared** | **+0.124** | +0.012 | **0.18** / 0.06 / 0.27 / 0.49 | 0.05 / 0.04 / 0.43 / 0.48 |
| **acceleration declared** | +0.005 | **+0.268** | 0.05 / 0.05 / 0.40 / 0.51 | **0.31** / 0.04 / 0.21 / 0.45 |
| **neither declared** | −0.001 | +0.002 | 0.01 / 0.01 / 0.97 / 0.02 | 0.01 / 0.01 / 0.93 / 0.06 |

7,385 of 16,384 evaluation examples are eligible: the same set for both variables. Specificity =
P(intended) − P(wrong-variable).

## 15.3 Registered tests

| test | mean | 95% CI (bootstrap over seeds) | p (exact sign-flip) | seeds > 0 | verdict |
|---|---|---|---|---|---|
| **PRIMARY:** gap(mass declared) − gap(accel. declared), gap = spec(M) − spec(A) | **+0.374** | [+0.214, +0.579] | **0.0005** | **12/12** | **supported** |
| direction M: mass edit, mass declared − accel. declared | +0.118 | [+0.029, +0.216] | 0.047 | 9/12 | supported (Holm) |
| direction A: accel. edit, accel. declared − mass declared | +0.256 | [+0.102, +0.438] | 0.001 | 11/12 | supported (Holm) |
| help: mass edit, mass declared − neither | +0.125 | [+0.042, +0.216] | 0.012 | 10/12 | declaration helps its own variable |
| side effect: accel. edit, mass declared − neither | +0.011 | [−0.010, +0.042] | 0.67 | 5/12 | no measurable damage |
| help: accel. edit, accel. declared − neither | +0.267 | [+0.099, +0.462] | 0.005 | 9/12 | declaration helps its own variable |
| side effect: mass edit, accel. declared − neither | +0.006 | [−0.016, +0.032] | 0.65 | 8/12 | no measurable damage |

**Pre-specified secondary: add-one edit.** It shows the same pattern, much weaker:
- primary gap +0.071, CI [+0.026, +0.122], p = 0.012, 9/12 seeds;
- declared-variable specificity +0.04 / +0.05, against about 0 for `neither`.

So add-one did not fail outright this time, but it is far weaker than inversion.

## 15.4 Diagnostic: where does editing fail? (inversion, intended-answer rate)

| group | edited variable | **copy real pattern into block** | fitted edit, known block | fitted edit, searched (DAS) location | copy into DAS location | designed-block interchange |
|---|---|---|---|---|---|---|
| mass declared | M | **0.807** | 0.189 | 0.184 | 0.804 | 0.842 |
| mass declared | A | 0.034 | 0.033 | 0.052 | 0.368 | 0.081 |
| accel. declared | M | 0.014 | 0.013 | 0.050 | 0.490 | 0.080 |
| accel. declared | A | **0.912** | 0.321 | 0.305 | 0.911 | 0.920 |
| neither | M | 0.023 | 0.005 | 0.005 | 0.090 | 0.156 |
| neither | A | 0.026 | 0.007 | 0.010 | 0.092 | 0.166 |

Applying the registered rule:

1. **Copy works; the fitted edit fails, so the edit's construction is the problem.** In the
   declared block, a real pattern carries the edited value 81–91% of the time, but the fitted
   affine edit reaches only 19–32%.
2. **The location search is not the problem.** The DAS-located edit performs like the
   known-block edit (0.184 vs 0.189; 0.305 vs 0.321), and copying into the DAS location works as
   well as copying into the block (0.80 / 0.91).
3. **The block carries only the declared variable.** Copying the *undeclared* variable into the
   block does nothing (0.01–0.03). Copying into a DAS subspace searched for the undeclared
   variable partly works (0.37–0.49): that variable is carried, just not in the block.

This matches PR #4's account. A declared block is causally sufficient for its variable, but a
class-mean affine map cannot reproduce the real patterns, which differ by the value–unit route
that produced them. The next improvement is an edit that constructs *realistic* patterns, for
example by mapping each route-specific cluster to its target. More localization is not needed.

## 15.5 Per-seed results (inversion specificity, searched location)

| seed | mass declared: M | mass declared: A | accel. declared: M | accel. declared: A | neither: M | neither: A | primary Δ | converged (mass/accel./neither) |
|---|---|---|---|---|---|---|---|---|
| 300 | +0.030 | +0.001 | +0.092 | +0.731 | +0.000 | +0.000 | +0.668 | y/y/y |
| 301 | +0.013 | +0.019 | +0.010 | +0.161 | +0.000 | +0.012 | +0.144 | n/y/y |
| 302 | +0.314 | −0.012 | +0.012 | −0.015 | +0.000 | +0.001 | +0.299 | y/n/y |
| 303 | +0.366 | +0.011 | −0.020 | +0.026 | +0.000 | −0.002 | +0.401 | n/n/y |
| 304 | +0.028 | +0.001 | −0.012 | +0.147 | −0.017 | +0.009 | +0.186 | y/n/n |
| 305 | +0.006 | −0.008 | +0.001 | +0.046 | +0.000 | −0.000 | +0.059 | y/y/y |
| 306 | +0.369 | +0.041 | +0.002 | +0.962 | +0.000 | −0.000 | +1.287 | n/y/y |
| 307 | +0.000 | −0.035 | +0.000 | −0.003 | +0.000 | +0.000 | +0.031 | y/y/y |
| 308 | +0.296 | −0.015 | −0.029 | −0.012 | +0.000 | +0.000 | +0.328 | n/y/y |
| 309 | +0.007 | −0.004 | −0.071 | +0.101 | +0.000 | −0.002 | +0.183 | n/y/y |
| 310 | +0.095 | +0.160 | +0.083 | +0.694 | +0.000 | +0.000 | +0.547 | y/y/y |
| 311 | −0.043 | −0.009 | −0.009 | +0.383 | +0.000 | +0.000 | +0.358 | y/y/y |

The primary Δ is positive in every seed, but its size varies a lot. In several seeds the declared
variable's edit is still near zero (for example mass declared in seeds 300, 305, 307 and 311). The
effect is reliable in direction, not in magnitude.

## 15.6 Costs and caveats

- **Convergence** (ID accuracy ≥ 0.95, all scheduled seeds, no restarts): mass declared **7/12**,
  acceleration declared **9/12**, neither **11/12**.
  - In the pilot, declaring acceleration was the harder case. In the main run the order reversed,
    so neither variable is reliably harder.
  - On the 5 seeds where all three groups converged, the primary Δ is +0.333, positive in 5/5
    (p = 0.0625 with only 5 seeds). The direction holds; power is lower.
- **Generalization to held-out unit–value combinations is worse with declaration:** OOD accuracy
  0.53 (mass) and 0.51 (acceleration) vs 0.85 (neither). This repeats doc 12. Declaring a variable
  where it is *read* does not help the network *compute* it for unseen combinations, and here
  it hurt.
- **Toy scale:** 69.8K-parameter networks, one task, one edit construction. The 12 seeds per group
  are the unit of evidence. The thousands of test questions per network are not independent
  evidence.

## 15.7 Against the pre-registered decision rules (doc 14 §14.7)

- **The advantage switches sides with declaration.** The primary test is significant and both
  directions are positive and significant after Holm correction. **This supports "declaration
  itself helps" for the inversion edit.**
- The add-one rule ("inversion replicates and add-one fails again") does not apply as written:
  add-one shows a small effect in the same direction. The honest summary is that **declaration
  helps both tested transformations, strongly for inversion and weakly for add-one.** In absolute
  terms both edits remain far from reliable.
- The hypothesis that different units leave different internal patterns remains a candidate
  explanation for the gap between copying (≈ 0.8–0.9) and fitted editing (≈ 0.2–0.3). It has not
  yet predicted a result on fresh networks.

## 15.8 Next step

The bottleneck is now the **edit construction**, not localization or declaration. The next
registered test should compare the frozen affine class-mean edit against an edit that writes
*realistic* patterns, such as per-route cluster mapping. It should use the same networks, the same
eligible examples and the same scorer, with copy-into-block (≈ 0.8–0.9) as the ceiling. The
paraphrase experiment stays deferred until an edit reliably follows the declared variable.
