"""Doc 59 Q / E / MR: ModularNet — protected compiled stage + read-only semantic interface + gated learned modules.

Encoder: Design C copied into a d=80, 5-head TinyTyped; head 4 (layer 2) gathers the operation one-hot from position 5
(op tokens carry a one-hot in dims 64-70 and an is-op flag in dim 71; the gathered op one-hot lands in dims 72-78).
Compiled stage (frozen, exact): OUT1 = M*A*S, OUTadd = M+A, registers X = M*A, Y = M+A (+ recompiled nodes later).
Learned modules: one small MLP per new operation, reading ONLY listed read-only dims (no write path into the residual at
all); logits = sum_op g_op(x) * logits_op(x) with g the exact op one-hot (output ownership).

Ops (token): OP1 23 (M*A*S, compiled) | OPadd 25 (M+A, compiled) | OP2 26 (M*A+S) | OP3 27 ((M+A)*S) |
             OP4 28 (independently generated rule, see p8_rules; NOT imported here except by the trainer's label oracle)
             | OP5 29 (OP4*M) | OPR 30 (random table; negative control for extraction)

    python -m sci.modular q | e | mr
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri.model import TinyTyped  # noqa: E402
from sci import construct as C  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase8"
OPS = ["OP1", "OPadd", "OP2", "OP3", "OP4", "OP5", "OPR"]
OPTOK = {"OP1": 23, "OPadd": 25, "OP2": 26, "OP3": 27, "OP4": 28, "OP5": 29, "OPR": 30}
VOCAB = 31
OPE, ISOP, OPG = list(range(64, 71)), 71, list(range(72, 79))
LG = lambda x: C.DLOG[x] if x else 0


# ------------------------------------------------------------------ encoder (d=80, 5 heads)
def encoder():
    src, _ = C.build_C()
    m = TinyTyped("mass_only", d=80, n_heads=5, d_mlp=1, n_types=1, vocab=VOCAB, seq_len=8, n_out=P)
    d = 80
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.E_c.weight[:25, :64] = src.E_c.weight
        for i, op in enumerate(OPS):
            m.E_c.weight[OPTOK[op], OPE[i]] = 1
            m.E_c.weight[OPTOK[op], ISOP] = 1
        m.E_p.weight[:, :64] = src.E_p.weight
        for L, sa in (("attn1", src.attn1), ("attn2", src.attn2)):
            a = getattr(m, L)
            for part in range(3):
                for h in range(4):
                    rs, rb = part * 64 + 16 * h, part * d + 16 * h
                    a.qkv.weight[rb:rb + 16, :64] = sa.qkv.weight[rs:rs + 16]
                    a.qkv.bias[rb:rb + 16] = sa.qkv.bias[rs:rs + 16]
            a.o.weight[:64, :64] = sa.o.weight
            a.o.bias[:64] = sa.o.bias
        a = m.attn2                                   # head 4: gather op one-hot
        a.qkv.bias[64] = C.G2
        a.qkv.weight[d + 64, ISOP] = 1
        for i in range(len(OPS)):
            a.qkv.weight[2 * d + 64 + i, OPE[i]] = 1
            a.o.weight[OPG[i], 64 + i] = 1
    return m


# ------------------------------------------------------------------ compiled stage
def add_ladder(Win, bi, Wo, u0, terms, g, gate, dest):
    vs = {0}
    for _, tab, _ in terms:
        vs = {a + b for a in vs for b in tab}
    tmin, tmax = min(vs), max(vs)
    B = float(tmax - tmin + 10)
    js = list(range(tmin - 1, tmax + 2))
    for u, j in enumerate(js):
        for dims, tab, zero in terms:
            for d_, v in zip(dims, tab):
                Win[u0 + u, d_] += v
            if gate:
                for d_, z in zip(dims, zero):
                    if z:
                        Win[u0 + u, d_] -= B
        bi[u0 + u] = -j
    for k in range(tmin, tmax + 1):
        c = dest[g(k)]
        Wo[c, u0 + k - tmin] += 1; Wo[c, u0 + k - tmin + 1] -= 2; Wo[c, u0 + k - tmin + 2] += 1
    n = len(js)
    if gate:
        for uu, off in ((u0 + n, 0.0), (u0 + n + 1, -1.0)):
            for dims, tab, zero in terms:
                for d_, z in zip(dims, zero):
                    if z:
                        Win[uu, d_] += 1
            bi[uu] = off
        Wo[dest[0], u0 + n] += 1; Wo[dest[0], u0 + n + 1] -= 1
        n += 2
    return n


def ladder_units(terms, gate):
    vs = {0}
    for _, tab, _ in terms:
        vs = {a + b for a in vs for b in tab}
    return (max(vs) - min(vs) + 3) + (2 if gate else 0)


M_, A_, S_ = (C.DIM["Q"], list(range(P))), (C.DIM["A"], list(range(P))), (C.DIM["FU"], C.SINV)


def spec_terms(kind, coef=None):
    """compiled node statistics. kind in mas, add, mul(M,A), lin(a,b,c,d) [mod 13 on integer values], dlog(a,b,c)."""
    z = lambda vals: [v == 0 for v in vals]
    if kind == "mas":
        return [(M_[0], [LG(v) for v in M_[1]], z(M_[1])), (A_[0], [LG(v) for v in A_[1]], z(A_[1])),
                (S_[0], [LG(v) for v in S_[1]], z(S_[1]))], (lambda t: pow(2, t % 12, P)), True
    if kind == "mul":
        return [(M_[0], [LG(v) for v in M_[1]], z(M_[1])), (A_[0], [LG(v) for v in A_[1]], z(A_[1]))], (lambda t: pow(2, t % 12, P)), True
    if kind == "add":
        return [(M_[0], list(M_[1]), z(M_[1])), (A_[0], list(A_[1]), z(A_[1]))], (lambda t: t % P), False
    if kind == "lin":
        a, b, c, d = coef
        return [(M_[0], [a * v for v in M_[1]], z(M_[1])), (A_[0], [b * v for v in A_[1]], z(A_[1])),
                (S_[0], [c * v + (d if i == 0 else 0) for i, v in enumerate(S_[1])], z(S_[1]))], (lambda t: t % P), False
    raise ValueError(kind)


class ModularNet(nn.Module):
    def __init__(self, nodes):
        """nodes: list of (name, kind, coef) compiled into the stage; each gets a 13-dim region."""
        super().__init__()
        self.enc = encoder()
        self.regions, nxt = {}, 80
        for name, _, _ in nodes:
            self.regions[name] = list(range(nxt, nxt + P)); nxt += P
        self.D = nxt
        units = []
        built = []
        for name, kind, coef in nodes:
            terms, g, gate = spec_terms(kind, coef)
            units.append(ladder_units(terms, gate)); built.append((name, terms, g, gate))
        self.W_in = nn.Linear(self.D, sum(units))
        self.W_out = nn.Linear(sum(units), self.D)
        with torch.no_grad():
            for p in (self.W_in.weight, self.W_in.bias, self.W_out.weight, self.W_out.bias):
                p.zero_()
            u0 = 0
            for name, terms, g, gate in built:
                u0 += add_ladder(self.W_in.weight, self.W_in.bias, self.W_out.weight, u0, terms, g, gate, self.regions[name])
        for p in self.parameters():
            p.requires_grad_(False)
        self.modules_ = nn.ModuleDict()
        self.reads = {}
        self.owner = {}                                  # op -> ("compiled", region) or ("module", name)

    @torch.no_grad()
    def features(self, tokens, batch=4096):
        out = []
        for i in range(0, len(tokens), batch):
            t = tokens[i:i + batch]
            h = self.enc.resid_pre_mlp(t, torch.zeros_like(t))
            h = torch.cat([h, torch.zeros(len(h), self.D - 80)], 1)
            out.append(h + self.W_out(torch.relu(self.W_in(h))))
        return torch.cat(out)

    def logits(self, h):
        lg = torch.zeros(len(h), P)
        for i, op in enumerate(OPS):
            g = h[:, OPG[i]:OPG[i] + 1]
            own = self.owner.get(op)
            if own is None:
                continue
            if own[0] == "compiled":
                lg = lg + g * h[:, self.regions[own[1]]]
            else:
                lg = lg + g * self.modules_[own[1]](h[:, self.reads[own[1]]])
        return lg


def read_dims(net, regs=()):
    return C.DIM["Q"] + C.DIM["A"] + C.DIM["FU"] + [d for r in regs for d in net.regions[r]]


# ------------------------------------------------------------------ data
def domain_ops(ops):
    L = C.domain()
    n = len(L["m"])
    Lk = {k: v.repeat(len(ops)) for k, v in L.items()}
    Lk["op"] = torch.arange(len(ops)).repeat_interleave(n)
    b = T.build({k: v for k, v in Lk.items() if k != "op"})
    tok = b.tokens.clone()
    tok[:, 5] = torch.tensor([OPTOK[ops[i]] for i in Lk["op"]])
    M, A, S = C.canon_scales(Lk, dict(MU=T.MASS_S, AU=T.ACC_S))
    return Lk, tok, M, A, S


def heldout_pairs(seed=59, frac=0.2):
    g = torch.Generator().manual_seed(seed)
    pairs = torch.randperm(P * P, generator=g)[: int(P * P * frac)]
    hm = torch.zeros(P * P, dtype=torch.bool); hm[pairs] = True
    return hm


# ------------------------------------------------------------------ training a module (features precomputed: frozen trunk)
def train_module(feats, targets, read, train_idx, eval_sets, seed, steps=4000, hidden=64, eval_every=200):
    torch.manual_seed(seed)
    mod = nn.Sequential(nn.Linear(len(read), hidden), nn.ReLU(), nn.Linear(hidden, P))
    opt = torch.optim.AdamW(mod.parameters(), lr=2e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(1000 + seed)
    X = feats[:, read]
    hist = []
    t0 = time.time()
    for step in range(steps + 1):
        if step % eval_every == 0:
            with torch.no_grad():
                pr = mod(X).argmax(-1)
                e = {k: float((pr[idx] == targets[idx]).float().mean()) for k, idx in eval_sets.items()}
                e["train"] = float((pr[train_idx] == targets[train_idx]).float().mean())
                e["step"] = step
                hist.append(e)
        if step == steps:
            break
        b = train_idx[torch.randint(0, len(train_idx), (256,), generator=g)]
        loss = F.cross_entropy(mod(X[b]), targets[b])
        opt.zero_grad(); loss.backward(); opt.step()
    n_par = sum(p.numel() for p in mod.parameters())
    return mod, dict(history=hist, final=hist[-1], params=n_par, wall=time.time() - t0,
                     module_flops_per_step=3 * 256 * 2 * (len(read) * hidden + hidden * P))
