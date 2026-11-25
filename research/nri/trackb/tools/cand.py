"""Candidate encoders for run_swap training (same parameters, same math; NOT historical semantics).
core: ref (model's own Attn code path) | manual (exp/max/sum softmax) | sdpa (F.scaled_dot_product_attention)
last: layer 2 computes only the last query row (keys/values for all rows).
batch: base and source encoded in one call, then split.
"""
import math
import torch
import torch.nn.functional as F


def _attn(a, x, core, last, split=False):
    B, L, d = x.shape
    if last and split:          # q only for the last row; k, v for all rows (weight row-slices of the same qkv)
        W, b = a.qkv.weight, a.qkv.bias
        q = (x[:, -1:] @ W[:d].T + b[:d]).view(B, 1, a.h, a.dh)
        k, v = (x @ W[d:].T + b[d:]).view(B, L, 2, a.h, a.dh).unbind(2)
    else:
        q, k, v = a.qkv(x).view(B, L, 3, a.h, a.dh).unbind(2)
    if last and not split:
        q = q[:, -1:]
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if core == "sdpa":
        out = F.scaled_dot_product_attention(q, k, v)
    else:
        s = (q @ k.transpose(-1, -2)) / math.sqrt(a.dh)
        if core == "ref":
            p = s.softmax(-1)
        else:
            e = (s - s.amax(-1, keepdim=True)).exp()
            p = e / e.sum(-1, keepdim=True)
        out = p @ v
    Lq = q.shape[2]
    return a.o(out.transpose(1, 2).reshape(B, Lq, d))


def encode(m, tok, core="ref", last=False, split=False):
    zero = torch.zeros_like(tok)
    x0 = m.embed(tok, zero)
    x1 = x0 + _attn(m.attn1, x0, core, False)
    if last:
        return x1, (x1[:, -1:] + _attn(m.attn2, x1, core, True, split))[:, 0]
    return x1, (x1 + _attn(m.attn2, x1, core, False))[:, -1]


def pair(m, D, btok, stok, variant):
    """variant string e.g. 'baseline', 'last', 'sdpa', 'manual+last', 'sdpa+last+batch'."""
    if variant == "baseline":
        return D.encode(m, btok)[1], (D.encode(m, stok)[1] if stok is not None else None)
    parts = set(variant.split("+"))
    core = "sdpa" if "sdpa" in parts else ("manual" if "manual" in parts else "ref")
    last = "last" in parts or "split" in parts
    sp = "split" in parts
    if "batch" in parts and stok is not None:
        h = encode(m, torch.cat((btok, stok)), core, last, sp)[1]
        return h[: len(btok)], h[len(btok):]
    return encode(m, btok, core, last, sp)[1], (encode(m, stok, core, last, sp)[1] if stok is not None else None)
