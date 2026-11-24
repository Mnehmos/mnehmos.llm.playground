# Paper Three · Evidence audit and Experiment 30 qualification

## Experiment 29 · Fresh reconstruction

I recovered the original code, frozen protocol, telemetry, reviews and all 84 model/optimizer archives. The ZIP and source digests, checkpoint hashes, matching batch streams, and training exclusion masks passed. A saved 500–600 interval reproduced exactly. An independent calculation of the modular conversion oracle plus fresh checkpoint scoring reproduced the following in the three unprotected runs:

| Measurement | Step 0 | Step 100 | Step 600 |
| --- | ---: | ---: | ---: |
| Practiced conversion | 100% | 99.65% | 100% |
| Full-model conversion swap | 100% | 10.31% | 8.77% |
| Same trained converter evaluated alone | 100% | 100% | 100% |

Restoring the original maps at step 600 yielded 4.94% swap success. Zeroing the conversion rows of the output matrix restored exact conversion and, on all available force rows, left force logits bit-for-bit unchanged. The restored influence was evaluated as an endpoint repair, while isolated accuracy and swap success were retrospectively recomputed at every checkpoint. The original fixed comparison combines freezing with either disabling or bounding the residual. It provides no freeze-only control. Held force pairs stayed weak despite a known-good converter, and fitness of this learner on modular multiplication remained an open question.

## Gradient-source replay · Exploratory

From the same original model, optimizer and data streams at steps 0 and 100, I ran the next 100 updates with conversion and force losses independently enabled or disabled. The both-loss setting exactly reproduced the archived model weights for all six replay intervals. Starting from zero, force alone left swap success at 100% through step 100. Conversion alone reduced it to 82.0% at step 20 and 10.2% at step 50. Joint training produced 94.1% at step 20, 10.4% at 50 and 10.3% at 100. The sampled early trajectory bounds onset, rather than identifying an exact update.

The architecture gives force loss no direct gradient into the conversion readout rows. Those rows were initialized at zero. With only force learning, the trunk can change but cannot activate an initially zero conversion readout. At step 100, with prior AdamW moments already present, turning off both fresh loss signals still moved those rows by a mean norm of 0.0062 on the very next update. With no fresh conversion loss, force-only replay remained at 9.2% authority after another 100 updates, versus 12.8% for zero new gradients. This separates a sufficient trigger (conversion loss from the original initialization) from contributors after displacement (existing optimizer state and changes to the shared trunk). The replay does not separately identify momentum, adaptive scaling, and decoupled weight decay.

## Experiment 30 · Non-study qualification

Two fresh matched seeds (830401 and 830409), six arms, 600 AdamW updates, and 84 reviewed full checkpoints. A–E test the five proposed policies. F is D's matched extra ordinary-supervision control. All models use the Experiment 29 token trunk with a third readout on a small independently learnable routing task as a plasticity check. Learning that task shows capacity to acquire a new behavior under a protection policy. It does not establish retained ability on hard tasks.

Development results at step 600, averaged across two seeds:

| Arm | Practiced conversion | Held conversion authority | Isolated authority | Held routing accuracy | Held canonical force pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| A · Trainable program, unrestricted | 100.0% | 8.4% | 100% | 100.0% | 14.8% |
| B · Frozen program, unrestricted | 97.9% | 3.5% | 100% | 98.8% | 14.6% |
| C · Frozen and bounded | 100.0% | 100% | 100% | 100.0% | 13.5% |
| D · Causal supervision | 99.5% | 7.1% | 100% | 100.0% | 14.2% |
| E · Architectural isolation | 100.0% | 100% | 100% | 100.0% | 14.2% |
| F · Extra ordinary supervision | 100.0% | 3.5% | 100% | 99.5% | 15.1% |

B was intact through step 10, at 79.1–81.5% authority at step 20, and at 9.1–9.4% at step 50. Freezing alone failed even on practiced conversion swaps (0.2% mean at step 600). C has a mathematical minimum margin of 0.2 under valid ordinary and one-hot counterfactual inputs. Its perfect performance is an instrument check and a guaranteed engineering reference. At the late checkpoint D's average of each seed's maximum residual spread was 4.95 versus 70.61 for F, but both spreads exceeded the installed unit margin on at least some inputs. D did not preserve control. The two-seed outcome supports retaining the current training recipe as a fixed test of weak protection, rather than treating it as a qualified practical replacement. A different objective or coefficient would be a separately declared treatment.

Retrospective endpoint repair zeroed only the conversion readout rows. Both B and both D clones recovered 100%. Moving unrelated routing-readout weights by the same tensor norm left conversion authority unchanged. Attenuating conversion weights to one eighth retained perfect control in D but gave 14.6% and 26.5% in B. D therefore reduced the amount of suppression needed at these endpoints. These perturbations are local in architectural scope and may still be large in block-relative norm. Repair durability was not tested.

## Feasibility and limits before lock

The recorded training-plus-measurement cost was approximately 14.5 CPU seconds per run on this executor. At 72 runs the same configuration projects to about 17.4 CPU-minutes and 466 MB of uncompressed checkpoint payload, excluding review time, archival overhead and follow-up tests. Measure throughput again if the final holdout or instruments expand. The qualification exercised the matched run, audit gate, hash linking, restart, balanced traces and restoration. One final D train interval reproduced bit-for-bit, including the saved optimizer, RNG and traces.

The pilot measured stepwise raw gradients and effective parameter changes, signed checkpoints on 14 controlled prompt instances per model, per-tensor alignment on selected panels, and separate path interventions. It did not demonstrate full functional lineage of all hidden features, independently blinded evaluation, or absence of training-related plasticity costs. The final conversion universe is small (117 written prompts). An example split leaves only two crossed bases in the final set. The endpoint should be stated against that specified finite task, while architecture or task-family validation is needed for broad claims. The force reuse study with all four input conditions and joint operand swaps, durable repairs, pretrained-host installation, and the second task family are separate prospective stages.
