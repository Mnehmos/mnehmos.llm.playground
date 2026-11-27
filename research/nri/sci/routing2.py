"""Doc 59 R2: constructed distributed routing (no optimisation).

Base: Design C (MAS) embedded in a d=128, 8-head TinyTyped (dims 0-63 identical; heads 0-3 = Design C heads).
  R2a spread : each head's score is distributed over all 16 head dims (rotary pair replicated 7x at gain/7, gate row 2x)
  R2b dup    : every routing head duplicated into head h+4; both OVs scaled by 1/2 (consensus = sum)
  R2c mingain: balanced q/k + minimal argmax-exact gain (as Phase VII R1) applied on top
  R2d alt    : positional-only binding (no content-gated selection) + unit conversion by exact AND-units in an MLP stage
               (relu(v_i + u_j - 1) = [v=i and unit=j]; one 39-unit table per role, i.e. the per-role conversion table,
               not a table over input combinations), then the Design-C decoder ladder (StagedNet, 2 blocks)
Controls separating architecture from rescaling: R1 on the unmodified Design C (Phase VII), R2a/R2b at original gains.

    python -m sci.routing2      # -> runs/phase8/routing2.json
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri.model import TinyTyped  # noqa: E402
from sci import construct as C  # noqa: E402
from sci import graph_compiler as G  # noqa: E402
from sci import robust as RB  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase8"


def embed128(spread=False, dup=False):
    src, _ = C.build_C()
    m = TinyTyped("mass_only", d=128, n_heads=8, d_mlp=256, **C.MODEL_KW)
    d, d0 = 128, 64
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.E_c.weight[:, :d0] = src.E_c.weight; m.E_p.weight[:, :d0] = src.E_p.weight
        for L, sa in (("attn1", src.attn1), ("attn2", src.attn2)):
            a = getattr(m, L)
            W, b = a.qkv.weight, a.qkv.bias
            for part in range(3):
                for h in range(4):
                    rs, rb = part * 64 + 16 * h, part * d + 16 * h
                    W[rb:rb + 16, :d0] = sa.qkv.weight[rs:rs + 16]; b[rb:rb + 16] = sa.qkv.bias[rs:rs + 16]
            a.o.weight[:d0, :64] = sa.o.weight; a.o.bias[:d0] = sa.o.bias
            if spread:
                for h in range(4):
                    q0, k0 = 16 * h, d + 16 * h
                    Wq, Wk, bq, bk = W[q0:q0 + 16].clone(), W[k0:k0 + 16].clone(), b[q0:q0 + 16].clone(), b[k0:k0 + 16].clone()
                    used = [r for r in range(16) if Wq[r].abs().sum() + bq[r].abs() > 0]
                    if not used:
                        continue
                    W[q0:q0 + 16] = 0; W[k0:k0 + 16] = 0; b[q0:q0 + 16] = 0; b[k0:k0 + 16] = 0
                    reps = 16 // len(used)
                    for i, r in enumerate(used):
                        for c in range(reps):
                            row = i * reps + c
                            W[q0 + row] = Wq[r] / reps; b[q0 + row] = bq[r] / reps
                            W[k0 + row] = Wk[r]; b[k0 + row] = bk[r]
            if dup:
                for h in range(4):
                    for part in range(3):
                        W[part * d + 16 * (h + 4):part * d + 16 * (h + 4) + 16] = W[part * d + 16 * h:part * d + 16 * h + 16]
                        b[part * d + 16 * (h + 4):part * d + 16 * (h + 4) + 16] = b[part * d + 16 * h:part * d + 16 * h + 16]
                    a.o.weight[:, 64 + 16 * h:64 + 16 * h + 16] = a.o.weight[:, 16 * h:16 * h + 16] / 2
                    a.o.weight[:, 16 * h:16 * h + 16] /= 2
        m.W_in.weight[:, :d0] = src.W_in.weight; m.W_in.bias[:] = src.W_in.bias
        m.W_out.weight[:d0] = src.W_out.weight; m.W_out.bias[:d0] = src.W_out.bias
        m.W_U.weight[:, :d0] = src.W_U.weight
    return m


def build_alt():
    """R2d: positional binding only (copy head; no content-gated correction heads) + AND-unit conversion in an MLP stage."""
    src, _ = C.build_C()
    D = C.DIM
    m = copy.deepcopy(src)
    with torch.no_grad():
        # remove layer-1 correction heads (1-3) and layer-2 scale-7 head (2): binding/gather only
        for h in (1, 2, 3):
            m.attn1.qkv.weight[[16 * h + i for i in range(16)] + [64 + 16 * h + i for i in range(16)] + [128 + 16 * h + i for i in range(16)]] = 0
            m.attn1.qkv.bias[[16 * h + i for i in range(16)] + [64 + 16 * h + i for i in range(16)]] = 0
            m.attn1.o.weight[:, 16 * h:16 * h + 16] = 0
        m.attn2.qkv.weight[[32 + i for i in range(16)] + [96 + i for i in range(16)] + [160 + i for i in range(16)]] = 0
        m.attn2.qkv.bias[[32 + i for i in range(16)] + [96 + i for i in range(16)]] = 0
        m.attn2.o.weight[:, 32:48] = 0
        # layer-2 gathers also copy the unit flags (M-gather: MU flags; A-gather: AU flags) into new dims 51..56
        # M-gather head 0 value rows 0..12 = Q, 13..15 = FU (full) -> put MU flags through the A-gather head's spare rows
        W, O = m.attn2.qkv.weight, m.attn2.o.weight
        # head 3 attends the mass-unit position and copies the MU flags -> dims 51..53; AU flags via head 1 rows 13..15
        b = m.attn2.qkv.bias
        b[48] = C.G2
        for d_ in D["MU"]:
            W[64 + 48, d_] = 1
        for j in range(3):
            W[128 + 48 + j, D["MU"][j]] = 1; O[51 + j, 48 + j] = 1
        for j in range(3):
            W[128 + 16 + 13 + j, D["AU"][j]] = 1; O[54 + j, 16 + 13 + j] = 1
    # StagedNet: block 1 = AND-unit conversion of the raw gathered values into canonical one-hots (new regions),
    # block 2 = Design-C MAS ladder reading the canonical regions
    Dd = 64 + 2 * P
    Mc, Ac = list(range(64, 64 + P)), list(range(64 + P, 64 + 2 * P))
    net = G.StagedNet(m, Dd, [2 * 3 * P, 34])
    with torch.no_grad():
        for p in net.blocks.parameters():
            p.zero_()
        net.W_U.weight.zero_()
        b1 = net.blocks[0]
        u = 0
        for raw, flags, scales, dest in ((D["Q"], [51, 52, 53], T.MASS_S, Mc), (D["A"], [54, 55, 56], T.ACC_S, Ac)):
            for j, s in enumerate(scales):
                for v in range(P):
                    b1["W_in"].weight[u, raw[v]] = 1; b1["W_in"].weight[u, flags[j]] = 1; b1["W_in"].bias[u] = -1
                    b1["W_out"].weight[dest[v * s % P], u] = 1
                    u += 1
        op = C.OPS["MAS"]
        tmp = TinyTyped("mass_only", d=Dd, n_heads=1, d_mlp=34, n_types=1, vocab=T.VOCAB, seq_len=8, n_out=P)
        C.compile_mlp(tmp, op, dict(M=Mc, A=Ac, S=D["FU"]), D["OUT"])
        net.blocks[1]["W_in"].weight.copy_(tmp.W_in.weight); net.blocks[1]["W_in"].bias.copy_(tmp.W_in.bias)
        net.blocks[1]["W_out"].weight.copy_(tmp.W_out.weight)
        net.W_U.weight.copy_(tmp.W_U.weight)
    return net


class _Adapt(nn.Module):
    """lets StagedNet be called as m(tokens, types) by the robustness tools."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, tokens, types=None):
        return self.net(tokens)


def exact_check(m):
    L = C.domain()
    b = T.build(L)
    with torch.no_grad():
        lg = m(b.tokens, torch.zeros_like(b.tokens))
    oh = torch.nn.functional.one_hot(b.y, P).float()
    return dict(acc=float((lg.argmax(-1) == b.y).float().mean()), logits_exactly_onehot=float((lg == oh).all(1).float().mean()))


def main():
    torch.set_num_threads(1)
    res = {}
    variants = {"R2_base128": embed128(), "R2a_spread": embed128(spread=True), "R2b_dup": embed128(dup=True),
                "R2ab": embed128(spread=True, dup=True)}
    for k in ("R2a_spread", "R2b_dup", "R2ab"):
        bal = RB.rebalance(variants[k])
        gam = RB.minimal_gain(bal)
        variants[k + "_mingain"] = RB.scale_gain(bal, gam)
        res[k + "_gamma"] = gam
    alt = _Adapt(build_alt())
    variants["R2d_alt"] = alt
    for k, m in variants.items():
        ex = exact_check(m)
        r = dict(kind="constructed", exact=ex, dense_params=sum(p.numel() for p in m.parameters()),
                 nonzero_params=int(sum((p != 0).sum() for p in m.parameters())))
        if ex["acc"] == 1.0:
            r.update(a=RB.curve(m, "a"), b=RB.curve(m, "b"))
            if k != "R2d_alt":
                r["c"] = RB.curve(m, "c")
        res[k] = r
        print(k, {a: b for a, b in r.items() if a not in ("a", "b", "c")}, "| a", {s: round(v, 2) for s, v in r.get("a", {}).items()},
              "| b", {s: round(v, 2) for s, v in r.get("b", {}).items()}, flush=True)
        (OUT / "routing2.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
