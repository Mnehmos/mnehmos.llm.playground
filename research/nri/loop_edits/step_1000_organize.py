"""Step 1,000 (mass just became causal: block IIA 0.65; unit 1 has split off: unit 0-1 CKA 0.52,
interaction 0.45 and rising; 84% of the interaction is written by layer-2 attention heads 0/2).

Method (bespoke, loop_edits/common.organize_edit): optimize the attention and MLP weights for 300
Adam steps on FIT to minimise a DIFFERENTIABLE mass x route interaction share of the block (two-way
cell decomposition, residual-maker on the cell design), while keeping the original predictions
(KL x30), the block's causal sufficiency (block-interchange CE) and a relative L2 leash.
Route (unit x position) MAIN effects are not penalized, so unit information stays in the block.

Why now: OBS pre-checks on the control's future showed the same method at 1,200-1,500 removes more
interaction but costs 0.04-0.12 held-out-combination accuracy; at 1,000-1,100 it is nearly free.
OBS pre-check here: interaction -0.14, keep 0.993, acc +0.02, IIA +0.08, OOD_DEV -0.007, edit +0.012.
Target: interaction (decrease >= 0.02 with editability not falling).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import organize_edit  # noqa: E402

TARGET = "interaction"
RATIONALE = __doc__
PARAMS = ["attn1.qkv.weight", "attn1.qkv.bias", "attn1.o.weight", "attn1.o.bias", "attn2.qkv.weight",
          "attn2.qkv.bias", "attn2.o.weight", "attn2.o.bias", "W_in.weight", "W_in.bias", "W_out.weight", "W_out.bias"]


def propose(model, ctx):
    return organize_edit(model, ctx, PARAMS, w_inter=2, kl=30, iit=1, l2=3)
