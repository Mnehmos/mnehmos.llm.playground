"""Step 700. Observation: mass is decodable at the value token (kappa 0.6) and the three units share
one block geometry (per-unit CKA 0.99), but nothing is causal yet (block IIA 0.16) and accuracy is at
chance-level for the task (0.25). From the pilot we know that at steps 900-1,000 the code becomes
causal and unit 1's route splits off (interaction 0.05-0.2 -> 0.45 -> 0.66), after which no small
edit can undo it.

Hypothesis: the split arises because the readout learns to use unit-specific patterns. If, while the
shared geometry still exists, we make the network USE the block additively (translating the block by
class-mean differences changes the answer accordingly), the shared code may become the causal one.

Method: constrained optimization (loop_edits/common.mean_translation_edit) over the attention and MLP
weights, 200 Adam steps on FIT, with KL and task terms to preserve behaviour; scaled to <= 8% of the
model norm. Target: block IIA(M) (the representation becomes causal in the block).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import mean_translation_edit  # noqa: E402

TARGET = "iia"
RATIONALE = __doc__
PARAMS = ["attn1.qkv.weight", "attn1.qkv.bias", "attn1.o.weight", "attn1.o.bias", "attn2.qkv.weight",
          "attn2.qkv.bias", "attn2.o.weight", "attn2.o.bias", "W_in.weight", "W_in.bias", "W_out.weight", "W_out.bias"]


def propose(model, ctx):
    return mean_translation_edit(model, ctx, PARAMS, steps=200, lr=1e-3)
