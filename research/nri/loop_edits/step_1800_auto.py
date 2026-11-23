"""Step 1800: auto-cycle submission (journal: codified procedure from step 1,700).
Packet: acc OBS                              1.000   ctrl   1.000  Δ +0.000; acc OOD_DEV                          0.738   ctrl   0.731  Δ +0.007; block IIA(M)                         0.940   ctrl   0.939  Δ +0.001; pooled edit inv (OBS)                0.109   ctrl   0.102  Δ +0.007; interaction share (block)            0.640   ctrl   0.637  Δ +0.003
Method: the step-1,500 editability edit (rewire_edit: random-permutation class-mean translation,
KL x30, block interchange, noise-KL x10, frozen route probe x5, L2 leash), scaled 0.4.
OBS pre-check: {'gain': 0.059, 'keep': 1.0, 'd_acc': 0.0, 'd_iia': -0.001, 'd_ood_dev': -0.002, 'd_edit_inv': 0.059, 'd_interaction': -0.015, 'd_wrong': 0.007, 'd_route_probe': 0.072, 'd_unit_probe': 0.016}
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import rewire_edit  # noqa: E402

TARGET = "edit_inv"
RATIONALE = __doc__
UP = ['attn1.qkv.weight', 'attn1.qkv.bias', 'attn1.o.weight', 'attn1.o.bias', 'attn2.qkv.weight', 'attn2.qkv.bias', 'attn2.o.weight', 'attn2.o.bias']
POST = ['W_in.weight', 'W_in.bias', 'W_out.weight', 'W_out.bias']


def propose(model, ctx):
    P = dict(model.named_parameters())
    up = {k: torch.zeros_like(P[k]) for k in UP}
    d = rewire_edit(model, ctx, up, post=POST, also_upstream=True, **{'kl': 30, 'iit': 1, 'translate': 1.0, 'l2': 3, 'steps': 400, 'lr': 0.001, 'noise_kl': 10, 'route_keep': 5})
    return {k: v * 0.4 for k, v in d.items() if v.abs().sum() > 0}
