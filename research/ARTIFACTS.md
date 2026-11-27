# Artifact registry

Inventory snapshot: 27 November 2026. This is a partial migration inventory, not a complete release manifest. An archive hash identifies a particular file; it does not independently authenticate every scientific claim inside it.

## Git revisions

| Source | Immutable reference |
| --- | --- |
| Original main lineage | [af17da2](https://github.com/Mnehmos/mnehmos.llm.playground/tree/af17da246aa47e81570ccfc74ff5bd2b75b8a090) |
| Conversion / early concept studies | [484a30a](https://github.com/Mnehmos/mnehmos.llm.playground/tree/484a30a02a3000c029c4c58a9241708f7afbcdec) |
| Phases IV–VIII / CPU-01 / Phase IX registration | [5958e2d](https://github.com/Mnehmos/mnehmos.llm.playground/tree/5958e2d9e22b234f94353e386bf1c4d0a734efc6) ([draft PR #36](https://github.com/Mnehmos/mnehmos.llm.playground/pull/36)) |
| Generalization seed-cross | [0f13b59](https://github.com/Mnehmos/mnehmos.llm.playground/tree/0f13b596c61842f19abbf5d4fd67fc2bcfad64b3) |
| Causal authority manuscript branch | [d34aa61](https://github.com/Mnehmos/mnehmos.llm.playground/tree/d34aa61cb5826b5287380ad96f33af96ad219bf9) |
| Matrix research | [48f75c1](https://github.com/Mnehmos/mnehmos.llm.playground/tree/48f75c17c2521d96548e8155bc4e46bf967f9a5b) |

## Separate retained archives

The following files were located in the research workspace and hashed during the documentation audit. They are **not published download links or committed archives**. A maintainer can match a recovered copy against these hashes. A public artifact release with source, provenance, licensing and review remains pending.

| Archive | SHA-256 |
| --- | --- |
| nri_G04_mathematics_runtime.zip | `0486a039987d041a9f8b443b9e47fb5783e64210323478db0b8f782ba30cae6d` |
| nri_G04_session06_neural_planning.zip | `bec4ccfe70930a78c112dad979ef7558ffefef550f04659d79700b8f09345a9f` |
| nri_G05_session07_residual_discovery.zip | `56bd86ebd2bb447258e4474e40f80a300035cf47900bfafb046356e005002e41` |
| nri_G05_session08_adaptive_allocation.zip | `c5d9d82fbab1025930802224560b9e529241656e5f2bcd0e3f9ab1e3102f9915` |
| Study05_research_package.zip | `96afe1338216405eff8b1a0d3159538b93e085766474a47dc2209e2bc734e367` |
| Track_L_Session01.zip | `e7d0f0f415019af2c266bfb4bbc2bb7794f7dc268bf30c1494ec934f1b466620` |
| Gap_Closure_Study01_Research.zip | `8bc7ed269a8cfc756a4aaa3c93dc8ff721827b26dd2d8fad774014622bb370d7` |
| Gap_Closure_Study02.zip | `480c45b2ca8c467a6eb817bc96270da65cd448e66c16a7beb8151133ffdd5aa0` |
| Hybrid_Math_196_Runtime.zip | `1ca0357af4f7a4364bf78fa66f7fdc9af0d34dc6e49aa50ec2a22dcd1c15c60e` |
| Hybrid_Math_H01_Research.zip | `c4c3ddadca1d004a2d5b0dd6256d928aa6593df53c3e9127aeb3f81dbee2fbac` |

G01–G03, G04 intermediate sessions, G05 S01–S06, earlier parallel studies, Track C, and large checkpoint packs require a fuller migration inventory. Do not infer completeness from this table.

## Small checked-in artifact

[H01 inspection demo](hybrid_math/README.md): the actual two frozen FP32 weight vectors are encoded as hexadecimal with individual SHA-256 receipts. The portable inspection code is an adaptation of the H01 arithmetic and router source, not a replacement for the full archived runtime.

## Release policy (proposed)

Preserve all original source archives and checkpoints without rewriting their hashes. Release small code, protocols, summaries and manifests in Git. Use separately versioned artifact storage or release assets for large weights/checkpoints, with explicit hashes and source revisions. Do not expose reserved evaluation sets before decisions are locked. Review rights, dependencies, provenance and public release scope before distribution.
