# 44 — Literature and novelty review (Phase V, in progress)

Citations are checked against primary sources; the log is `paper/CITATION_CHECKS.md`.

## 44.1 What is known

- **Seed-dependent out-of-distribution generalization.** Models with near-identical in-distribution accuracy can
  differ enormously out of distribution.
  - 100 BERT fine-tunes, differing only in classifier initialisation and data order: MNLI dev 83.6–84.8%, but HANS
    subcategories from 0% to 66% (McCoy, Min & Linzen 2020).
  - The same pattern holds for hierarchical generalisation in seq2seq networks (McCoy, Frank & Linzen 2020).
  - These studies tie initialisation and data order to the same seed, so they cannot say which matters. A vision
    same-different study with fixed weights attributes the outcome to batch order alone (search result; to be
    verified before citing).
- **Weight decay and generalising circuits.** Grokking: generalisation emerges long after memorisation, and weight
  decay drives the switch (Power et al. 2022; Nanda et al. 2023; Liu et al. 2023). Varma et al. (2023) explain this
  by efficiency: weight decay favours the generalising circuit, which is cheaper in parameter norm than a memorising
  one.
- **Causal abstraction and interchange interventions.**
  - The framework: Geiger et al. 2021, 2025 (JMLR).
  - Training a declared variable into a network (IIT): Geiger et al. 2022.
  - Limits:
    - subspace patching can work through dormant pathways (Makelov et al. 2024; reply by Wu et al. 2024);
    - unrestricted non-linear alignment maps can align any network with any algorithm (Sutter et al. 2025);
    - patching results depend on the corruption method and metric (Zhang & Nanda 2024; Heimersheim & Nanda 2024).

- **Complexity control decides reasoning vs memorisation.**
  - Zhang et al. (arXiv 2501.08537): initialisation scale and weight decay determine whether transformers learn
    reasoning-based (compositional) or memorising solutions.
  - A May 2026 preprint (arXiv 2605.04396): the decision is made within a sharp, observable *window* of training,
    and weight decay matters only within it; reportedly the effect does not appear for grokking on modular
    arithmetic.
  - **These are the closest prior work to both our results.** The factorial's weight-decay requirement fits their
    complexity-control account. The role-swap finding that the outcome is decided at the escape from a shared plateau
    and never revised is an independent observation of a "decision window", in a setting with a declared causal
    variable.
  - Our novelty claims must be limited accordingly (§44.2). Author lists and figures are to be verified from the
    PDFs before citing; summaries seen so far are partly third-party.

## 44.2 What appears new here (subject to a fuller search before any submission)

1. **The semantic-supervision × weight-decay interaction for compositional generalisation**, established under a
   registered design (96 networks, paired seeds, entropy-matched control target).
   - The grokking literature links weight decay to generalisation, but not to *what is supervised*. Here,
     supervising the canonical quantity has no compositional effect without strong decay, and strong decay
     produces no composition without a coherent semantic target.
2. **Held-out causal instruments validated on planted mechanisms**, including a planted *trained-only reader*: a
   correct canonical code read by a lookup. Patching studies usually measure in trained contexts, where these
   alternatives are indistinguishable.
3. **Exact (bitwise) replay of historical networks and full 100-step developmental records**, which turn
   seed-dependent generalisation from an endpoint statistic into an observable developmental event.
   - Prior seed-variance work reports endpoints.
   - Here the split is decided within a few hundred steps, at the escape from a shared plateau, and is never
     revised afterwards (doc 41R, provisional).
   - Decision windows have been reported before (arXiv 2605.04396). Our contribution is the causal
     characterisation of what is decided: whether the declared representation of never-trained inputs is canonical.
4. **A false-negative illusion for probes on never-trained inputs.** Probes fitted on trained inputs read 0.00 on OOD
   inputs that the network demonstrably processes correctly through the probed block. This complements the
   false-positive illusions of subspace patching.

## 44.3 To do

- Search specifically for:
  - auxiliary-target (multi-task) supervision combined with weight decay;
  - "representation vs use" dissociations (decodable but unused codes);
  - developmental studies of OOD generalisation with complete checkpoints.
- Verify the vision batch-order study before citing it.
