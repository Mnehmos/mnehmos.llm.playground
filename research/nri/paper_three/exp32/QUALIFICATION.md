# Experiment 32 qualification · Two matched seeds

200 updates on the public pilot split. These measurements do not enter confirmatory estimates.

| Treatment | Authority | Ordinary | Flagged extra task |
| --- | ---: | ---: | ---: |
| CE | 7.3% | 30.4% | 100% |
| Satisfied natural margin | 87.0% | 88.0% | 100% |
| Scale 5 | 99.7% | 100% | 100% |
| Previous IIT loss | 39.1% | 75.0% | 100% |
| Soft counterfactual margin | 99.7% | 100% | 100% |
| Routed natural conversion | 12.1% | 9.2% | 99.5% |
| Matched ordinary margin | 11.0% | 64.7% | 97.3% |

The routed objective sets natural conversion gradients to exactly zero. Its competing path still receives flagged-task gradients and may extend that behavior to the ordinary view. Thus a natural confidence gradient cannot be the unique possible trigger. The satisfied objective allows reactivation when the ordinary margin falls below 0.5 after other learning. The stronger initial scale is a confidence/margin comparison and does not claim intrinsically different causal organization. At pilot initialization the sampled raw conversion readout norm was .904 under natural CE, .083 under scale, and zero under satisfied and routed. Fixed signs and dimensions of the automaton transitions passed exhaustive oracle checks on 20,480 start-state/word combinations.

The pilot supports retaining the six mechanistic/soft comparisons, with the earlier IIT retained as a pilot and Experiment 30 reference. It did not establish meaningful routing transfer because the held source had a single binary routing label. The flagged extra task is an intentionally simple but shared-output test. Confirmation is projected at 8–12 CPU-minutes and 250 MB uncompressed before packaging.
