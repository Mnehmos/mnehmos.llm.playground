# Experimental idea atlas — hypothesis backlog, NOT observed discoveries

**Ideas:** 120, covering 12 tracks. **Stage:** brainstorm only. A catchy title is not a novelty claim. Every item requires the scientific controls and prior-art check in [README](README.md) before evaluation. Any numerical success criterion must be frozen **before** confirmatory runs, never chosen from the outcome.

**Budget tiers:** A = analyze existing checkpoints / tiny counterfactuals (often seconds–minutes, not benchmarked); B = short CPU pilot or small checkpoint run; C = independently seeded CPU study / possibly Colab; D = larger training requiring GPU. These are relative planning tiers, not verified runtime estimates.

**Paper tags:** P01 causal unit semantics; P02 training genesis and recurrent computation; P03 neural weight programming; P04 ANLU, algebraic transfer and optimization; P05 NRI benchmark; P06 auditable research automation; P07 exploratory wildcards. The tags indicate **potential homes**, not seven guaranteed publishable papers.

**Quick wins:** screen all tier-A proposals on immutable existing checkpoints first, except experiments requiring new task generators. Use exploratory labels and provenance. **Do not** combine the best 20 exploratory p-values into one claimed discovery.

**Triage score to assign before running:** information value (0–3) + falsifiability (0–3) + causal precision (0–3) + reproducibility (0–3) − cost (0–3) − prior-art overlap (0–3). Pilot ranking is for scheduling only; a confirmatory claim needs fresh evidence.


## U. Unit meaning & semantic equivalence

| ID | Experimental question / intervention | Quantified outcome | Tier |
|---|---|---|---|
| **U01** Unit-scale blind decoding | recover hidden log-scale without scale table | exact label-scale accuracy | A |
| **U02** Equal-ratio fingerprint | compare activation deltas for equal scale ratios | circular delta error | A |
| **U03** Canonical vs raw-route rivalry | fit class-mean representations with matched capacity | held-out partial R² difference | A |
| **U04** Unit relabel causal swap | change label only, not raw value | intended/wrong-answer fractions | A |
| **U05** Change-of-unit triangle | compose A→B→C vs A→C swaps | composition residual | A |
| **U06** Invisible unit collision | same canonical quantity via two unseen routes | interchange preservation accuracy | B |
| **U07** Hidden unit aliases | two unrelated tokens encode identical scale | alias identity score vs shuffled control | B |
| **U08** New unit interpolation | introduce one scale absent from train | novel unit transfer accuracy | C |
| **U09** Noncommutative unit path | compare conversion order under multiple unit rules | order-specific error | B |
| **U10** Numeric magnitude stress | shift values by orders of magnitude in real-number arm | relative error vs digit length | C |

## T. Learning-time concept genesis

| ID | Experimental question / intervention | Quantified outcome | Tier |
|---|---|---|---|
| **T01** Pre-generalization meaning | test if causal label feature appears before OOD accuracy | checkpoint lead/lag steps | B |
| **T02** Concept birth certificate | earliest checkpoint meeting preregistered behavior+causality | emergence step distribution | B |
| **T03** Gradient omen | predict future causal circuit from early update directions | future-site prediction AUC | B |
| **T04** Feature migration | track aligned subspace across layers during training | subspace drift and causal score | B |
| **T05** Circuit death | identify once-used features discarded after convergence | necessity change across time | B |
| **T06** Semantic compression | measure representation intrinsic dimension during acquisition | dimension vs OOD accuracy | B |
| **T07** Learning as phase locking | measure cross-neuron spectral synchrony before generalization | synchrony lead time | B |
| **T08** One example flips mechanism | remove one training example family, compare trajectories | emergence-time effect | C |
| **T09** Curriculum reversal | train conversions→arithmetic and arithmetic→conversions | sample efficiency and editability | C |
| **T10** Early-vs-late checkpoint edit | apply identical semantic edit across learning stages | success per relative edit norm | B |

## L. Looped and recurrent computation

| ID | Experimental question / intervention | Quantified outcome | Tier |
|---|---|---|---|
| **L01** Compute-matched looping | recurrent 4 unique vs unrolled 8 vs shallow | held-out accuracy per FLOP | C |
| **L02** Loop-count dose response | execute same learned block 1–7 passes | accuracy/overthinking curve | B |
| **L03** Pass-wise semantic itinerary | track unit/value/calculator probes per iteration | causal emergence pass | B |
| **L04** Recurrent attractor test | look for hidden-state convergence or cycles | spectral radius and state distance | A |
| **L05** Pass-3 bypass | skip only second or third recurrence | accuracy delta and intended change | B |
| **L06** Cross-pass graft | insert loop-2 state from matched counterfactual | intervention IIA / collateral | B |
| **L07** Loop schedule randomization | train fixed vs randomized recurrence lengths | OOD loop-count stability | C |
| **L08** Frozen first pass | freeze early pass mapping by structural gate | held-out accuracy at equal compute | C |
| **L09** Recurrent error correction | inject a unit corruption at pass k | recovery probability by remaining passes | B |
| **L10** Recurrent saturation | map when additional loops cease useful change | accuracy per added pass / entropy | A |

## E. Weight editing as programming

| ID | Experimental question / intervention | Quantified outcome | Tier |
|---|---|---|---|
| **E01** Semantic one-row rewrite | redefine one unit via embedding-row edit | intended/preserve and norm | B |
| **E02** Zero-shot calculator patch | install product→quotient edit without task retraining | specificity and rank | C |
| **E03** Edit budget ladder | constrain ΔW to rank 1/2/4/8 | Pareto success vs norm | B |
| **E04** Two edits in either order | mass edit followed by acceleration edit and reverse | commutator error | B |
| **E05** Edit interference threshold | increase overlap between declared subspaces | collateral vs overlap | C |
| **E06** Weight firewall | train protected subspace that should resist unrelated edits | preservation after adversarial edit | C |
| **E07** Semantic checksum | predict target shift from ΔW analytically before applying | prediction calibration | B |
| **E08** Semantic undo | subtract stored delta after edit and test recovery | restoration accuracy | A |
| **E09** Edit transfer across seeds | fit edit program on one model, run on another | program vs copied-delta gap | B |
| **E10** Edit error-correcting code | encode one concept redundantly then damage one site | robustness vs editability tradeoff | C |

## N. ANLU and contextual meaning

| ID | Experimental question / intervention | Quantified outcome | Tier |
|---|---|---|---|
| **N01** Polysemy force test | same surface word means physics quantity or compel | unseen-context sense accuracy | B |
| **N02** Type-dropout curve | remove type hints at controlled training fractions | untyped inference performance | B |
| **N03** Aux-type placement sweep | supervise type at embed vs attention vs MLP | editability by site | C |
| **N04** Type-alias cross-domain | swap physical unit label for role-equivalent new symbol | zero-shot transfer accuracy | C |
| **N05** Syntactic shortcut audit | hold out entire context templates from train | OOD minus IID gap | B |
| **N06** Ambiguous label under adversarial context | one token carries two physical dimensions | contextual dimension error | B |
| **N07** Context-to-type head | predict types from raw tokens without teacher at inference | type and task accuracy | B |
| **N08** Spurious type injection | train random fake labels as placebo | gain vs shuffled prior | C |
| **N09** Typed vs canonical information control | same information supplied in different embeddings | causal editability difference | C |
| **N10** Type-free compression | remove type embedding after aux-trained model convergence | accuracy with zero external type | B |

## S. Symmetries, invariance and algebra

| ID | Experimental question / intervention | Quantified outcome | Tier |
|---|---|---|---|
| **S01** Gauge-respecting unit encoding | add explicit dimensional equivariance | unit-rescale equivariance error | C |
| **S02** Gauge-choice stability | rotate representational reference frame | inferred-scale invariance | A |
| **S03** Group-size sweep | change modulus 7/11/13/17 | recovery rate vs group order | C |
| **S04** Prime vs composite arithmetic | compare Z_p and non-field cases | OOD formula performance | C |
| **S05** Zero as algebraic singularity | treat zero input independently of log-space rules | zero-specific failure rate | A |
| **S06** Inverse identity ablation | remove x×x⁻¹ examples from train | zero-shot identity accuracy | B |
| **S07** Dimensional nonsense rejection | train valid dimensional equations, test impossible ones | false acceptance rate | B |
| **S08** Distributive law generalization | compose plus and multiply without triple patterns in train | identity accuracy | C |
| **S09** Unit-conversion cycle consistency | compare output after A→B→C→A | cycle drift / KL | B |
| **S10** Affine units versus multiplicative units | add temperature-like offsets with distinct encoding | linear vs rotation spectrum separation | C |

## R. Fault injection, repairs and invariants

| ID | Experimental question / intervention | Quantified outcome | Tier |
|---|---|---|---|
| **R01** Break just one scale | corrupt mapping for a single unit token | detection/repair precision | B |
| **R02** Damaged attention head triage | ablate head, compare NRI diagnoses | causal localization F1 | A |
| **R03** Wrong-unit implant | swap unit embedding without touching values | edit specificity | A |
| **R04** Rotated latent coordinate attack | apply function-preserving hidden permutation | invariance of interpreter output | A |
| **R05** Circuit transplant | move a learned conversion subspace across models | transfer benefit vs random transplant | C |
| **R06** Detect silent corruption | perturb weights without changing IID accuracy | OOD/counterfactual sensitivity | A |
| **R07** Repair with calibration budget 1/5/20 | vary examples supplied to editor | success vs examples | B |
| **R08** Self-healing after damage | continue training after precise lesion | steps to restore accuracy | B |
| **R09** Minimal cut set | find smallest set of heads/sites whose ablation breaks one concept | cut size and specificity | B |
| **R10** Robustness to quantization | compare FP32/FP16/int8-like weight rounding | causal map stability | B |

## O. Optimizer & resource efficiency

| ID | Experimental question / intervention | Quantified outcome | Tier |
|---|---|---|---|
| **O01** Muon vs AdamW editability | match training compute and compare optimizer geometry | edit-specificity gap | C |
| **O02** Warm-start from rule table | initialize unit scales from symbolic metadata | samples to accuracy | B |
| **O03** One-shot curriculum lesson | single new relation taught, no full retrain | data efficiency | B |
| **O04** Gradient accumulation effects | vary batch/accumulation keeping tokens fixed | causal structure variance | C |
| **O05** Weight decay as curriculum | sweep regularization, track feature emergence | time-to-generalization | C |
| **O06** Low-precision tiny training | compare bfloat16 versus float32 where supported | convergence per joule | C |
| **O07** Freeze embeddings mid-course | lock token semantics at selected checkpoint | OOD gain or loss | B |
| **O08** Two-phase learning schedule | convert then compose vs joint learning | compute to same OOD accuracy | C |
| **O09** Rare unit oversampling | sample unusual labels nonuniformly | held-out minority-unit accuracy | B |
| **O10** Data-efficient teacher/student | distill explicit rule network into free network | causal-map transfer vs cost | C |

## I. Interpreters and executable explanations

| ID | Experimental question / intervention | Quantified outcome | Tier |
|---|---|---|---|
| **I01** Probe versus causal explanation | compare linear-probe top site with intervention top site | localization rank agreement | A |
| **I02** Blind LLM unit table recovery | give weights without labels table | accuracy of executable claims | C |
| **I03** Black-box versus white-box parity | same task, bounded calls for both interpreters | verified claims per dollar | C |
| **I04** Wrong explanation trap | construct two networks with same behavior, different mechanisms | interpreter mechanism discrimination | C |
| **I05** NRI proof compression | minimize typed causal claims needed to predict behavior | claims per covered behavior | B |
| **I06** Counterexample generator | search input that refutes candidate NRI mechanism | counterexamples per compute | B |
| **I07** Interpreter self-consistency | same weights, independent fresh interpreter runs | verified-claim agreement | C |
| **I08** Predict an unseen ablation | freeze explanation before performing test | causal prediction Brier score | B |
| **I09** Editor source code portability | reuse edit program on symmetry-transformed model | success retention ratio | B |
| **I10** Circuit translation to executable math | fit minimal symbolic operation to discovered activations | held-out mechanism agreement | B |

## F. Research factory and metascience

| ID | Experimental question / intervention | Quantified outcome | Tier |
|---|---|---|---|
| **F01** Blind future outcome forecast | researcher predicts effect sign before new seed | forecast calibration | A |
| **F02** How often probes lie | compare exploratory probe significance with causal held-out result | false discovery rate | B |
| **F03** Null result publication audit | track failed ideas as proportion of all registered ideas | reporting completeness | A |
| **F04** Compute-to-discovery curve | record compute for every hypothesis screened | verified effects per CPU-hour | A |
| **F05** Adversarial reviewer generator | generate falsification tests before first training | caught flaw rate | B |
| **F06** Prior-art tripwire | track overlap between proposed claims and references | novelty conflicts resolved | A |
| **F07** Tuner contamination test | compare frozen vs hindsight parameter selection | held-out metric inflation | B |
| **F08** Paper reproducibility one-command | fresh environment replays all paper numbers | checksum/metric concordance | A |
| **F09** Researcher disagreement | independent interpretation of same opaque run | claim convergence after checks | B |
| **F10** Meta-experiment: random ideas | compare curated vs randomized experiment ranking | yield per budget | B |

## X. High-risk wildcards, all quantifiable

| ID | Experimental question / intervention | Quantified outcome | Tier |
|---|---|---|---|
| **X01** Neural semantic periodic table | cluster operations by reusable causal transformation signatures | cluster stability and transfer | C |
| **X02** Algorithm vaccination | teach inverse task before corruption, then lesion | damage tolerance improvement | C |
| **X03** Artificial concept birth | insert new semantic operation with no gradient training | OOD intended-answer accuracy | C |
| **X04** Latent compiler | compile symbolic unit algebra into low-rank weight edits | equivalence test pass rate | C |
| **X05** Concept fusion | make one shared subspace jointly encode mass and acceleration | independent edit selectivity | C |
| **X06** Neural counterfactual time machine | transplant earlier checkpoint subspace into later model | regressed behavior fraction | B |
| **X07** Knowledge without names | randomize every symbol, retain only algebraic relationships | scale-recovery transfer | B |
| **X08** Causal map compression limit | find minimum dimension preserving 95% intervention performance | rank-versus-fidelity frontier | B |
| **X09** Executable concept checksum across seeds | hash canonicalized causal signatures, not raw weights | false-match rate | C |
| **X10** Semantics from unlabeled gradients | recover which unit changed using only ΔW time series | label identification AUC | C |

## Q. Surprising cheap diagnostics

| ID | Experimental question / intervention | Quantified outcome | Tier |
|---|---|---|---|
| **Q01** Logit-lens false positive | compare layerwise early answers with causal answer formation | early-decode vs patch agreement | A |
| **Q02** Activation nuisance subtraction | remove raw token identity and re-probe canonical quantity | delta chance-adjusted accuracy | A |
| **Q03** Outlier neuron monopoly | ablate top-1/top-5 high-activation neurons | causal effect vs random set | A |
| **Q04** Permutation stress | permute hidden neurons with matching inverse matrices | identical logits and map invariance | A |
| **Q05** One-feature semantic knockoff | fit synthetic feature matching probe score but causally inert | edit success difference | A |
| **Q06** Head-to-role traffic map | patch head outputs by operand/token role | route-specific IIA | A |
| **Q07** Swap label twice | perform A→B→A intervention composition | restoration rate | A |
| **Q08** Answer-preserving adversarial paraphrase | change context without changing math result | internal drift and accuracy | B |
| **Q09** One-seed robustness alarm | estimate how often exploratory best-site flips across seeds | selection instability | A |
| **Q10** Spectral coordinate mismatch | compare additive power before/after valid basis orthogonalization | false spectral preference rate | A |

## Stop / publication rules

1. **A screening result is not a confirmatory result.** Confirm only on fresh seeds or frozen held-out rows that were not used to design the method. Existing docs 14–20 results remain the foundation, not a new sample.
2. **Prevent fake independence.** Network seed is the primary unit of replication; thousands of prompts from a single weight file are not thousands of independent networks.
3. **Keep alternatives explicit.** For every promising idea: raw lookup vs reusable mechanism, probe-vs-causal discrepancy, distribution shift, random intervention, parameter/FLOP imbalance, gauge/coordinate invariance, and negative controls.
4. **Report all attempted ideas.** Registry should retain abandoned, null and failed pilot hypotheses with reasons; no publication-count KPI.
5. **Paper boundaries:** promote a group of experiments only when it supports a distinct question and independent evidence, not because a report generated a polished PDF.

A machine-readable copy lives at [ideas.json](ideas.json); use IDs as stable references in commits, issue titles, paths, and experiment manifests.
