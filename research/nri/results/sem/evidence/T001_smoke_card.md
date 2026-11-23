# Evidence card — T001_edit_decomposer v2 (NON-STUDY smoke, seed 999)

Status: exploratory instrument demonstration; not part of the registered comparison.

**Why built.** Smoke ckpts showed within-pretrain-family block IIA = 1.0 while the pooled affine
inversion edit of the same block succeeded ~2%. The prebuilt editor cannot say *why*. Competing
hypotheses: H1 per-route (unit-family) rotation; H2 non-linear but class-separable codes; H3 block not
sufficient; H4 class regions extended so the class mean is off-manifold.

**Validation (sandboxed, `python -m toolsmith.runner validate`).** Five synthetic gold cases each recover
their planted mechanism: additive→editable, route_rotated→H1, scrambled→H2, ring (class-dependent
radius, readout valid on-ring only)→H4, random→H3. v1→v2 history: the first ring design (class-
independent radius) wrongly came out "editable" and was redesigned — logged in the manifest changelog.

**Replay (copies of ckpts, nonmutation hash-checked: OK).**

| ckpt | unedited | pooled | per-route full | per-route rank-2 | copy mean | copy exemplar | verdict |
|---|---|---|---|---|---|---|---|
| inv s2400 | .28 | .08 | .12 | .11 | .19 | .17 | H3 |
| inv s2600 | .92 | .02 | .15 | .06 | .44 | .49 | undetermined |
| inv s2800 | 1.0 | .02 | .11 | .03 | .48 | .54 | undetermined |
| inv s3100 | 1.0 | .02 | .11 | .03 | .47 | .54 | undetermined |
| add1 s2800 | 1.0 | .10 | .12 | .10 | .43 | .61 | H4 |
| add1 s3100 | 1.0 | .09 | .11 | .09 | .43 | .60 | undetermined |

**Reading.** H1 (per-route rotation) is falsified: route-specific linear maps barely beat pooled. H3
falsified after s2600 (copying real block contents moves the answer ~50–60%). H4 is directionally
supported (exemplar > mean at every learned ckpt) but sits at the 0.6 threshold, so ~40% of block
copies still fail: the block carries something beyond the mass class (likely unit/route identity that
must match the base) — a new falsifiable prediction for a follow-up instrument: *copying an exemplar
matched on mass-unit should exceed 0.9*.

## Update — T002 falsified the "route identity" follow-up; T001 v3 (canonical-style pool)

The follow-up prediction above ("route-matched exemplar > 0.9") was already implied by v2, whose exemplars
*were* route-matched — so it was effectively falsified at 0.54–0.60. A new instrument, T002_swap_reconciler,
stratified swap success by source attributes and found the real cause: sources with **word numbers or
abbreviated units** (never shown in pretraining) swap at 0.14, canonical-style sources at **1.00**; source
pool, family and unit are irrelevant. With `canonical_style_only` (v3):

| ckpt | pooled | per-route full | per-route rank-2 | copy mean | copy exemplar | verdict |
|---|---|---|---|---|---|---|
| inv s2800 | .32 | .72 | .54 | 1.0 | 1.0 | H2 |
| inv s3100 | .34 | .75 | .55 | 1.0 | 1.0 | H2 |
| add1 s2800 | .11 | .28 | .10 | 1.0 | 1.0 | H2 |
| add1 s3100 | .10 | .31 | .11 | 1.0 | 1.0 | H2 |

Corrected reading: H4 rejected (class means are on-manifold). **H2**: the mass code is class-separable but
curved within each route — linear (affine) edits fail, lookup edits succeed; inversion is "more linear" than
add-one, consistent with doc 29's finding that editability follows the representation's algebra.
**Instrument caveat for the study:** the registered card metric `observe.pooled_edit` fits on the styled fit
pool, so for a student that has not yet learned the surface forms it is contaminated; it is kept unchanged
(registered) and this caveat is reported in doc 31.
