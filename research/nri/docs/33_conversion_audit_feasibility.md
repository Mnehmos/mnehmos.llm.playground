# Conversion feasibility with four scientific checkpoints

Seed 482111. Fresh semantic student, 300 updates. The supervised exploratory curriculum kept the mass conversion direction at klot to brel throughout, with 20% force replay. Start with digit and full-unit text in C01. After review at step 100, add word and abbreviation variation. After the step-200 review, add C02. The step-300 review closes the run. Only the training split was sampled. No persistent weight intervention ran.

| Step | Original form | Word + abbreviation | C02 frame | Withheld value | Reverse direction | New source unit | Layer-one value swap | Same-unit swap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1/10 | 1/10 | 1/10 | 0/1 | 1/9 | 0/8 | 9/90 | 9/90 |
| 100 | 10/10 | 1/10 | 5/10 | 0/1 | 0/9 | 1/8 | 90/90 | 0/90 |
| 200 | 10/10 | 10/10 | 6/10 | 0/1 | 1/9 | 1/8 | 87/90 | 0/90 |
| 300 | 10/10 | 10/10 | 10/10 | 0/1 | 1/9 | 1/8 | 89/90 | 0/90 |

The word/abbreviation and C02 evaluations become familiar-form measures after those lesson changes. Their improvement shows acquisition under direct practice. The reverse and new-source-unit panels use directions omitted from conversion training. The withheld number is a single eligible repository development tuple. The 90 ordered pairs reuse ten examples, so they provide intervention coverage on one route. The same-unit intervention can still move a contextual activation. It is a comparison site and does not test a change to the input unit's meaning.

The value patch uses the residual after attention and MLP in layer one. It copies the source-number activation between different values while preserving the rest of the base forward pass. The donor's correct answer is valid here because the source and destination units match. This demonstrates causal control through the source-number activation on a practiced route. A memorized mapping remains consistent with these results.

The controlled trace records the exact same conversion in three renderings. On the word form, the source-occurrence gradient norm dropped from 1.52 at step 100 to .00106 at step 200 after style practice. During that interval the answer MLP's W_out had the largest mean actual AdamW update norm, .0538. Those are separate measurements. The local trace does not attribute the historical update uniquely to that token. The first three predictions in the checkpoint ledger were confirmed. The step-300 prediction is prospective and pending an extension.

Reproduce from research/nri with `python -m sci.loop init RUN --seed 482111 --budget 300 --arm toolsmith --conversion`. Supply a scientific review for each completed checkpoint before invoking the next advance. The archive has all four full snapshots, evidence, reviews and 300 per-step records. The evidence supports a supervised narrow pilot. Qualification of the broader student and a matched comparative registration remain pending.
