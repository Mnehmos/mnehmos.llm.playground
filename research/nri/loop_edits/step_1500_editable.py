"""Step 1,500 (after the collapse at 1,200 and full recovery: accuracy 0.999, block IIA 0.94; the
network re-formed the control's organization -- interaction 0.63, unit 1 split (CKA 0.36), pooled
editability 0.06; organization is frozen from here, learning rate decaying).

Post-formation, shrinking the interaction costs held-out accuracy (OBS pre-checks), so I target
editability directly. Bespoke method (loop_edits/common.rewire_edit), 400 Adam steps on FIT over all
attention + MLP weights:
  * translation term: shifting the block by class-mean differences mu[pi(M)] - mu[M] must produce
    the answer for pi(M), for RANDOM class permutations pi (never inversion or add-one: inversion is
    the gate target, add-one is the held-out final test);
  * KL x30 to the original predictions (behaviour) and block-interchange CE (causal sufficiency);
  * noise-KL x10: the original readout's outputs on noise-perturbed block activations are kept
    (function preserved OFF the data manifold, where held-out unit-value combinations land; without
    it OOD_DEV fell 0.13);
  * route_keep x5: a frozen probe of route (unit x position), fitted on the original block, must still
    read the edited block (without it route information fell 0.04-0.06, violating S5);
  * L2 leash; the learned delta is then scaled by 0.4 (editability gain is superlinear in scale;
    0.4 is the largest scale whose OBS pre-check kept OOD_DEV within tolerance with margin).
OBS pre-check: editability +0.070, keep 1.000, acc 0.000, IIA -0.004, OOD_DEV 0.000, wrong +0.014,
route info +0.19, unit info +0.02. Target: edit_inv.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import rewire_edit  # noqa: E402

TARGET = "edit_inv"
RATIONALE = __doc__
UP = ["attn1.qkv.weight", "attn1.qkv.bias", "attn1.o.weight", "attn1.o.bias", "attn2.qkv.weight",
      "attn2.qkv.bias", "attn2.o.weight", "attn2.o.bias"]
POST = ["W_in.weight", "W_in.bias", "W_out.weight", "W_out.bias"]
SCALE = 0.4


def propose(model, ctx):
    P = dict(model.named_parameters())
    up = {k: torch.zeros_like(P[k]) for k in UP}
    d = rewire_edit(model, ctx, up, post=POST, also_upstream=True, kl=30, iit=1, translate=1.0, l2=3,
                    steps=400, lr=1e-3, noise_kl=10, route_keep=5)
    return {k: v * SCALE for k, v in d.items() if v.abs().sum() > 0}
