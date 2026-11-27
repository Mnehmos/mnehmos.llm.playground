# H01: inspect a 196-parameter hybrid

This directory is a portable, narrow inspection demo of the actual frozen H01 weights. It is not the full historical runtime and not a newly trained model.

- 13 learned phase coordinates: predict addition modulo 13.
- 183 learned method-selection weights: 14 input features, 8 tanh hidden units, 7 output scores. The features are supplied normalized numerical differences.
- Conventional exact family solvers: fit the first 12 observations, check four additional observations, and evaluate the fitted program using Python rational arithmetic. These algorithms are supplied, not learned.
- The neural adder and family-selection path are separate tests in this demo. The full H01 archive connects fitted programs to the preserved arithmetic substrate.

## Run

From the repository root:

```sh
python -m pip install numpy torch
python research/hybrid_math/h01_smoke.py
```

Expected output from the locally tested adaptation:

```text
Frozen 13-parameter digit adder: 169/169
Frozen 183-parameter router: reciprocal; 1 solver attempt(s)
Conventional exact evaluation at n=31: 27/13
PASS: narrow inspection demo, not full H01 qualification.
```

The adapter was also evaluated on the archived H01 observation panels: 700/700 IID, 700/700 OOD supported candidates accepted, 0/300 controls accepted, 1,427 supported attempts. These are existing exposed panels, not fresh qualification. The archived data is not bundled here. The full archived runtime executed 5,600 further supported predictions with the 13-parameter arithmetic core; this narrow demo does not claim to reproduce that full execution engine.

## Provenance

Weights are stored as hex of the original little-endian FP32 payload in [weights.json](weights.json), with SHA-256 verified at load. They were exported from the archived source checkpoints:

| Component | Archived checkpoint | Source checkpoint SHA-256 |
| --- | --- | --- |
| Addition | historical/source/runs/38001_structured/inferred/s0800.pt | `e57c66aac84f9398ea11461546d6edd2d0fe85b35da94b4fcb83a5ed7e8595b9` |
| Router | router_checkpoints/8/77762/0600.pt | `adf0076f59af46028b91357911ee917d10752463d8eff3998944612c81175ff5` |

Full archive identities: [artifact registry](../ARTIFACTS.md). Results and caveats: [evidence ledger](../RESULTS.md).

The 196 count is 784 raw neural-weight bytes, not the total code, data, storage or computation. The selected router ranks seven known families; exact fitting, validation and execution are conventional. The zero-neural baseline solves the supplied arithmetic panel, and a hand-written structural heuristic is competitive with the router. No unrestricted theorem proving, autonomous family discovery, or minimum-size theorem is claimed.
