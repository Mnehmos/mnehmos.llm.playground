"""A tiny, LayerNorm-free transformer chosen for clean linear algebra.

    x0 = E_c[tok] + E_t[type] + E_p[pos]
    x1 = x0 + Attn1(x0)
    x2 = x1 + Attn2(x1)             <- routing (gather operands + op at "=")
    h  = x2[:, -1]                  <- site "resid_pre_mlp"  (d_model)
    z  = W_in h + b_in              <- site "mlp_pre"        (d_mlp)
    r  = h + W_out relu(z) + b_out  <- computation of OP(a, b)
    logits = W_U r

No LayerNorm, so an edit to W_in is an exact affine change of z given h.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tasks


class Attn(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, return_pattern=False):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)  # bidirectional
        pat = att.softmax(-1)
        out = (pat @ v).transpose(1, 2).reshape(B, L, d)
        out = self.o(out)
        return (out, pat) if return_pattern else out


class TinyTyped(nn.Module):
    def __init__(self, condition: str, d: int = 64, n_heads: int = 4, d_mlp: int = 256,
                 type_dims: int = 16, n_types: int | None = None, aux_classes: int = 0):
        super().__init__()
        self.condition, self.d, self.d_mlp = condition, d, d_mlp
        self.E_c = nn.Embedding(tasks.VOCAB, d)
        self.E_t = nn.Embedding(n_types or tasks.N_TYPES, d)
        # optional auxiliary role-prediction head on the layer-1 residual (training only;
        # never used at inference). Present in every condition of an experiment that uses it,
        # so parameter counts stay matched.
        self.aux = nn.Linear(d, aux_classes) if aux_classes else None
        self.E_p = nn.Embedding(tasks.SEQ_LEN, d)
        self.attn1 = Attn(d, n_heads)
        self.attn2 = Attn(d, n_heads)
        self.W_in = nn.Linear(d, d_mlp)
        self.W_out = nn.Linear(d_mlp, d)
        self.W_U = nn.Linear(d, tasks.P, bias=False)
        # factored geometric prior: content and type live in disjoint coordinate blocks
        mc = torch.ones(d)
        mt = torch.ones(d)
        if condition == "typed_fact":
            mc[d - type_dims:] = 0
            mt[: d - type_dims] = 0
        self.register_buffer("mask_c", mc)
        self.register_buffer("mask_t", mt)

    # ---- pieces exposed for inspection / intervention ----
    def embed(self, tokens, types):
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        return self.E_c(tokens) * self.mask_c + self.E_t(types) * self.mask_t + self.E_p(pos)

    def resid_layer1(self, tokens, types):
        x = self.embed(tokens, types)
        return x + self.attn1(x)

    def resid_pre_mlp(self, tokens, types):
        x = self.embed(tokens, types)
        x = x + self.attn1(x)
        x = x + self.attn2(x)
        return x[:, -1]

    def head_from_resid(self, h, W_in=None, b_in=None):
        W_in = self.W_in.weight if W_in is None else W_in
        b_in = self.W_in.bias if b_in is None else b_in
        z = h @ W_in.T + b_in
        r = h + self.W_out(F.relu(z))
        return self.W_U(r)

    def forward(self, tokens, types):
        return self.head_from_resid(self.resid_pre_mlp(tokens, types))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
