"""Step 1,100 (accuracy 0.95, block IIA 0.83, interaction 0.50, unit 0-1 CKA 0.55; 89% of the
interaction written by attn2).

Same bespoke method as at 1,000 (organize_edit: differentiable interaction penalty, KL to original
predictions, block interchange, L2 leash) with a stronger KL (x60) because at 1,000 the VAL keep-rate
was the binding constraint. OBS pre-check: interaction -0.10, unit-transfer ratio +0.039, keep 0.995,
accuracy +0.002, IIA +0.018, OOD_DEV -0.003, editability -0.004.
Target: unit_transfer (the mass code learned on two units reads correctly on the third), the
organizational property I am trying to create; editability is NOT claimed here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import organize_edit  # noqa: E402

TARGET = "unit_transfer"
RATIONALE = __doc__
PARAMS = ["attn1.qkv.weight", "attn1.qkv.bias", "attn1.o.weight", "attn1.o.bias", "attn2.qkv.weight",
          "attn2.qkv.bias", "attn2.o.weight", "attn2.o.bias", "W_in.weight", "W_in.bias", "W_out.weight", "W_out.bias"]


def propose(model, ctx):
    return organize_edit(model, ctx, PARAMS, w_inter=2, kl=60, iit=1, l2=3)
