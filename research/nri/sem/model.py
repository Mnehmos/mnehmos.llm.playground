"""Language-capable student for Issue #25 (word-level; see docs/30 for size and compute).

    x0 = E[tok] + P[pos]                         (left-padded; the last position is the <q> marker)
    x1 = x0 + Attn1(LN(x0));  x1 = x1 + MLP1(LN(x1))
    x2 = x1 + Attn2(LN(x1));  x2 = x2 + MLP2(LN(x2))
    x3 = x2 + Attn3(LN(x2))
    h  = x3[:, -1]                               site 'h' (no normalisation: folds into W_in are exact)
    r  = h + W_out relu(W_in h + b_in) + b_out   readout MLP
    logits = W_U r                               17 answers: 0..12, yes, no, phys, other
Declared mass block: h[0:16] (interchange supervision on force examples, as in docs 14-29).
Training-only auxiliary head: contextual semantic role of every token, read from x1 (ANLU 'aux_only',
doc 08); no types are ever given as input.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sem import data as D
from sem import lang as L

BLOCK = (0, 16)


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, keymask, ret=False):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att.masked_fill(~keymask[:, None, None, :], float("-inf"))
        pat = att.softmax(-1)
        pre = (pat @ v).transpose(1, 2).reshape(B, T, d)
        out = self.o(pre)
        return (out, pat, pre) if ret else out


class MLP(nn.Module):
    def __init__(self, d, hdim):
        super().__init__()
        self.fc = nn.Linear(d, hdim)
        self.proj = nn.Linear(hdim, d)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class SemStudent(nn.Module):
    def __init__(self, vocab_size, d=64, heads=4, mlp=128, d_read=256, n_out=len(L.ANSWERS), n_roles=len(L.ROLES),
                 seq_len=D.MAX_LEN):
        super().__init__()
        self.d = d
        self.E = nn.Embedding(vocab_size, d)
        self.P = nn.Embedding(seq_len, d)
        self.ln1a, self.ln1m, self.ln2a, self.ln2m, self.ln3a = (nn.LayerNorm(d) for _ in range(5))
        self.attn1, self.attn2, self.attn3 = Attn(d, heads), Attn(d, heads), Attn(d, heads)
        self.mlp1, self.mlp2 = MLP(d, mlp), MLP(d, mlp)
        self.W_in = nn.Linear(d, d_read)
        self.W_out = nn.Linear(d_read, d)
        self.W_U = nn.Linear(d, n_out, bias=False)
        self.aux = nn.Linear(d, n_roles)

    def trunk(self, tok, ret=False):
        keymask = tok != D.PAD
        pos = torch.arange(tok.shape[1], device=tok.device)
        x0 = self.E(tok) + self.P(pos)
        a1 = self.attn1(self.ln1a(x0), keymask, ret=ret)
        x1 = x0 + (a1[0] if ret else a1)
        x1m = x1 + self.mlp1(self.ln1m(x1))
        a2 = self.attn2(self.ln2a(x1m), keymask, ret=ret)
        x2 = x1m + (a2[0] if ret else a2)
        x2m = x2 + self.mlp2(self.ln2m(x2))
        a3 = self.attn3(self.ln3a(x2m), keymask, ret=ret)
        x3 = x2m + (a3[0] if ret else a3)
        h = x3[:, -1]
        if not ret:
            return h, x1m
        return dict(x0=x0, x1=x1m, x2=x2m, x3=x3, h=h, pat1=a1[1], pat2=a2[1], pat3=a3[1], pre3=a3[2][:, -1])

    def head(self, h, W_in=None, b_in=None):
        W_in = self.W_in.weight if W_in is None else W_in
        b_in = self.W_in.bias if b_in is None else b_in
        r = h + self.W_out(F.relu(h @ W_in.T + b_in))
        return self.W_U(r)

    def forward(self, tok):
        h, _ = self.trunk(tok)
        return self.head(h)

    def n_params(self, include_aux=True):
        return sum(p.numel() for n, p in self.named_parameters() if include_aux or not n.startswith("aux"))
