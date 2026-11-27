"""Doc 53: mathematically CONSTRUCTED networks for F = M*A*S mod 13 (no data, no training).

Weights are written from the task definition only (P, unit scales, token layout) into the ordinary TinyTyped
architecture (d=64, 4 heads x 16, MLP 256). The forward pass is the model's own; no hooks.

Design C (canonical): one-hot codes; per-unit permutations P_s applied by attention heads (unit conversion), so the
  declared regions at "=" hold one-hot(M), one-hot(A), one-hot(force unit) -- identical for equivalent renderings.
  The MLP computes any g(phi_M(M) + phi_A(A) + phi_S(S)) exactly with ReLU "hats" (doc 53 D1).
Design L (log-lift): scalar unwrapped discrete logs; supports only the monomial family M^a A^b S^c.

    from sci.construct import build_C, build_L, OPS, qualify
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri import dimtask as T  # noqa: E402
from nri.model import TinyTyped  # noqa: E402

P = T.P
DLOG = {}
_x = 1
for _i in range(P - 1):
    DLOG[_x] = _i
    _x = _x * 2 % P
SINV = [T.INV[s] for s in T.FORCE_S]          # canonical S per force unit
MODEL_KW = dict(n_types=1, aux_classes=T.N_ROLES, vocab=T.VOCAB, seq_len=T.SEQ_LEN, n_out=T.P)


def perm(s):
    """P_s one-hot(v) = one-hot(v*s mod P)."""
    M = torch.zeros(P, P)
    for v in range(P):
        M[v * s % P, v] = 1
    return M


# ------------------------------------------------------------------ operation specs (the "instruction set")
@dataclass
class Op:
    """F = g(t) with t = phiM[M] + phiA[A] + phiS[fu]; gated variables force output `zero_out` when they are 0."""
    name: str
    phiM: list
    phiA: list
    phiS: list
    g: Callable[[int], int]
    gate: tuple = ()
    zero_out: int = 0
    target: Callable = None        # reference semantics on canonical (M, A, S) in Python arithmetic


def _lg(x):
    return DLOG[x] if x else 0


def monomial(a, b, c=1, name=None):
    return Op(name or f"M^{a}A^{b}S^{c}", [a * _lg(x) for x in range(P)], [b * _lg(x) for x in range(P)],
              [c * DLOG[s] for s in SINV], lambda t: pow(2, t % (P - 1), P), gate=("M", "A"),
              target=lambda M, A, S: 0 if (M == 0 or A == 0) else pow(M, a, P) * pow(A, b, P) * pow(S, c, P) % P)


OPS = {
    "MAS": monomial(1, 1, 1, "M*A*S"),
    "M2AS": monomial(2, 1, 1, "M^2*A*S"),
    "MA2S": monomial(1, 2, 1, "M*A^2*S"),
    "M+A": Op("M+A", list(range(P)), list(range(P)), [0, 0, 0], lambda t: t % P,
              target=lambda M, A, S: (M + A) % P),
    "M-A": Op("M-A", list(range(P)), [-x for x in range(P)], [0, 0, 0], lambda t: t % P,
              target=lambda M, A, S: (M - A) % P),
    # M/A with the explicit convention x/0 := 0 (and 0/A = 0)
    "M/A": Op("M/A", [_lg(x) for x in range(P)], [-_lg(x) for x in range(P)], [0, 0, 0],
              lambda t: pow(2, t % (P - 1), P), gate=("M", "A"),
              target=lambda M, A, S: 0 if (M == 0 or A == 0) else M * pow(A, -1, P) % P),
}


# ------------------------------------------------------------------ Design C layout (residual dims)
DIM = dict(Q=list(range(0, 13)), A=list(range(13, 26)), FU=[26, 27, 28], MU=[29, 30, 31], AU=[32, 33, 34],
           NULL=35, POS=[36, 37], OUT=list(range(38, 51)))
G1, H1, G2 = 4000.0, 12000.0, 2000.0          # layer-1 rotary gain, null gate, layer-2 key gain (doc 53 D2)
TH = 2 * math.pi / T.SEQ_LEN


def _blank(cond="mass_only"):
    m = TinyTyped(cond, **MODEL_KW)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    return m


def _pos_embed(m, dims):
    for p in range(T.SEQ_LEN):
        m.E_p.weight.data[p, dims[0]] = math.cos(TH * p)
        m.E_p.weight.data[p, dims[1]] = math.sin(TH * p)


def _rotary_query(Wq, row0, dims, gain):
    """q . k = gain * cos(theta (j - (p - 1))): attend to the previous position."""
    c, s = math.cos(TH), math.sin(TH)
    Wq[row0, dims[0]], Wq[row0, dims[1]] = gain * c, gain * s
    Wq[row0 + 1, dims[0]], Wq[row0 + 1, dims[1]] = -gain * s, gain * c


def _qkv(attn):
    W, b = attn.qkv.weight.data, attn.qkv.bias.data      # rows: q 0..63, k 64..127, v 128..191; head h: +16h
    return W, b, (lambda h: 16 * h), (lambda h: 64 + 16 * h), (lambda h: 128 + 16 * h)


def compile_mlp(m, op: Op, in_dims, out_dims, B=None):
    """D1: exact readout of g(t) with hats; zero gate. in_dims: dict M,A (13 dims each), S (3 dims)."""
    vals = [a + b + c for a in op.phiM for b in op.phiA for c in op.phiS]
    tmin, tmax = min(vals), max(vals)
    B = B or float(tmax - tmin + 10)
    Win, bin_, Wout = m.W_in.weight.data, m.W_in.bias.data, m.W_out.weight.data
    Win.zero_(); bin_.zero_(); Wout.zero_(); m.W_out.bias.data.zero_()
    js = list(range(tmin - 1, tmax + 2))
    n_units = len(js) + (2 if op.gate else 0)
    assert n_units <= m.d_mlp, f"needs {n_units} hidden units > {m.d_mlp}"
    for u, j in enumerate(js):
        for i in range(P):
            Win[u, in_dims["M"][i]] = op.phiM[i]
            Win[u, in_dims["A"][i]] = op.phiA[i]
        for f in range(3):
            Win[u, in_dims["S"][f]] = op.phiS[f]
        if "M" in op.gate:
            Win[u, in_dims["M"][0]] -= B
        if "A" in op.gate:
            Win[u, in_dims["A"][0]] -= B
        bin_[u] = -j
    unit = {j: u for u, j in enumerate(js)}
    for k in range(tmin, tmax + 1):
        c = out_dims[op.g(k)]
        Wout[c, unit[k - 1]] += 1
        Wout[c, unit[k]] -= 2
        Wout[c, unit[k + 1]] += 1
    if op.gate:
        z1, z2 = len(js), len(js) + 1
        for u, off in ((z1, 0.0), (z2, -1.0)):
            if "M" in op.gate:
                Win[u, in_dims["M"][0]] = 1
            if "A" in op.gate:
                Win[u, in_dims["A"][0]] = 1
            bin_[u] = off
        Wout[out_dims[op.zero_out], z1] += 1
        Wout[out_dims[op.zero_out], z2] -= 1
    m.W_U.weight.data.zero_()
    for c in range(P):
        m.W_U.weight.data[c, out_dims[c]] = 1
    return dict(t_range=(tmin, tmax), hidden_units=n_units)


def build_C(op: Op = OPS["MAS"], scales=None):
    """Design C. `scales` = dict(MU=(...), AU=(...)) to compile other unit conversions (default: task's)."""
    scales = scales or dict(MU=T.MASS_S, AU=T.ACC_S)
    m = _blank()
    D = DIM
    E = m.E_c.weight.data
    for v in range(P):
        E[v, D["Q"][v]] = 1
    for i in range(3):
        E[T.MU[i], D["MU"][i]] = 1
        E[T.AU[i], D["AU"][i]] = 1
        E[T.FU[i], D["FU"][i]] = 1
    E[T.TIMES, D["NULL"]] = 1
    _pos_embed(m, D["POS"])
    # ---- layer 1: copy head (0) + correction heads (1..3) for the non-identity scales done in layer 1
    W, b, qr, kr, vr = _qkv(m.attn1)
    O = m.attn1.o.weight.data
    corr1 = [(D["MU"][1], scales["MU"][1]), (D["MU"][2], scales["MU"][2]), (D["AU"][1], scales["AU"][1])]
    for h in range(4):
        _rotary_query(W, qr(h), D["POS"], G1)
        W[kr(h), D["POS"][0]], W[kr(h) + 1, D["POS"][1]] = 1, 1
        for i in range(P):
            W[vr(h) + i, D["Q"][i]] = 1
        if h == 0:
            for i in range(P):
                O[D["Q"][i], 16 * h + i] = 1
        else:
            flag, s = corr1[h - 1]
            W[qr(h) + 2, flag] = -H1
            b[qr(h) + 2] = H1
            W[kr(h) + 2, D["NULL"]] = 1
            C = perm(s) - torch.eye(P)
            for i in range(P):
                for j in range(P):
                    O[D["Q"][i], 16 * h + j] = C[i, j]
    # ---- layer 2: M-gather (0), A-gather (1), scale correction for acceleration unit 2 (2); head 3 unused
    W, b, qr, kr, vr = _qkv(m.attn2)
    O = m.attn2.o.weight.data
    b[qr(0)] = G2
    for d_ in D["MU"] + D["FU"]:
        W[kr(0), d_] = 1
    for i in range(P):
        W[vr(0) + i, D["Q"][i]] = 1
        O[D["Q"][i], 0 + i] = 2
    for f in range(3):
        W[vr(0) + 13 + f, D["FU"][f]] = 1
        O[D["FU"][f], 13 + f] = 2
    b[qr(1)] = G2
    for d_ in D["AU"]:
        W[kr(1), d_] = 1
    for i in range(P):
        W[vr(1) + i, D["Q"][i]] = 1
        O[D["A"][i], 16 + i] = 1
    b[qr(2)], b[qr(2) + 1] = G2, G2 / 2
    W[kr(2), D["AU"][2]] = 1
    W[kr(2) + 1, D["NULL"]] = 1
    C = perm(scales["AU"][2]) - torch.eye(P)
    for i in range(P):
        W[vr(2) + i, D["Q"][i]] = 1
        for j in range(P):
            O[D["A"][i], 32 + j] = C[i, j]
    # mass scale 1 is the identity in the task; if a compiled edit makes it non-identity it is not supported here
    assert scales["MU"][0] == 1 and scales["AU"][0] == 1, "unit 0 must keep scale 1 in this layout"
    info = compile_mlp(m, op, dict(M=D["Q"], A=D["A"], S=D["FU"]), D["OUT"])
    return m, info


# ------------------------------------------------------------------ Design L (log-lift)
DL = dict(LV=0, ZV=1, LMU=2, LAU=3, LS=4, SM=(5, 6), SA=(7, 8), ISMU=9, ISAU=10, NULL=11, POS=(12, 13),
          HM=14, HZM=15, HA=16, HZA=17, HS=18, OUT=list(range(19, 32)))


def build_L(op: Op = OPS["MAS"]):
    """Design L: embeddings carry discrete logs; binding heads copy the value's log into the unit's slot;
    uniform layer-2 attention sums slots (x8) at '='; MLP reads the unwrapped sum. Monomials only."""
    m = _blank()
    D = DL
    E = m.E_c.weight.data
    for v in range(P):
        E[v, D["LV"]] = _lg(v)
        E[v, D["ZV"]] = 1.0 if v == 0 else 0.0
    for i in range(3):
        E[T.MU[i], D["LMU"]] = DLOG[T.MASS_S[i]]
        E[T.MU[i], D["ISMU"]] = 1
        E[T.AU[i], D["LAU"]] = DLOG[T.ACC_S[i]]
        E[T.AU[i], D["ISAU"]] = 1
        E[T.FU[i], D["LS"]] = DLOG[SINV[i]]
    E[T.TIMES, D["NULL"]] = 1
    _pos_embed(m, D["POS"])
    W, b, qr, kr, vr = _qkv(m.attn1)
    O = m.attn1.o.weight.data
    for h, (flag, slot) in enumerate(((D["ISMU"], D["SM"]), (D["ISAU"], D["SA"]))):
        _rotary_query(W, qr(h), D["POS"], G1)
        W[kr(h), D["POS"][0]], W[kr(h) + 1, D["POS"][1]] = 1, 1
        W[qr(h) + 2, flag] = -H1
        b[qr(h) + 2] = H1
        W[kr(h) + 2, D["NULL"]] = 1
        W[vr(h), D["LV"]], W[vr(h) + 1, D["ZV"]] = 1, 1
        O[slot[0], 16 * h], O[slot[1], 16 * h + 1] = 1, 1
    W, b, qr, kr, vr = _qkv(m.attn2)               # head 0, uniform (zero scores): exact 1/8
    O = m.attn2.o.weight.data
    srcs = [((D["SM"][0], D["LMU"]), D["HM"]), ((D["SM"][1],), D["HZM"]), ((D["SA"][0], D["LAU"]), D["HA"]),
            ((D["SA"][1],), D["HZA"]), ((D["LS"],), D["HS"])]
    for r, (ins, out) in enumerate(srcs):
        for i in ins:
            W[vr(0) + r, i] = 1
        O[out, r] = 8
    # MLP on the scalar lifts: phi coefficients a, b, c recovered from the monomial op (phi = coef * dlog)
    a, bb = op.phiM[2], op.phiA[2]          # dlog(2) = 1, so phi[2] is the exponent
    c = op.phiS[1] // DLOG[SINV[1]]
    assert op.phiM == [a * _lg(x) for x in range(P)] and op.phiA == [bb * _lg(x) for x in range(P)] \
        and op.phiS == [c * DLOG[s] for s in SINV], "Design L supports only monomials (doc 53 §53.2)"
    vals = [a * x + bb * y + c * z for x in range(23) for y in range(23) for z in range(12)]
    tmin, tmax = min(vals), max(vals)
    Bg = float(tmax - tmin + 10)
    Win, bin_, Wout = m.W_in.weight.data, m.W_in.bias.data, m.W_out.weight.data
    js = list(range(tmin - 1, tmax + 2))
    assert len(js) + 2 <= m.d_mlp
    for u, j in enumerate(js):
        Win[u, D["HM"]], Win[u, D["HA"]], Win[u, D["HS"]] = a, bb, c
        Win[u, D["HZM"]], Win[u, D["HZA"]] = -Bg, -Bg
        bin_[u] = -j
    unit = {j: u for u, j in enumerate(js)}
    for k in range(tmin, tmax + 1):
        o = D["OUT"][op.g(k)]
        Wout[o, unit[k - 1]] += 1
        Wout[o, unit[k]] -= 2
        Wout[o, unit[k + 1]] += 1
    z1, z2 = len(js), len(js) + 1
    for u, off in ((z1, 0.0), (z2, -1.0)):
        Win[u, D["HZM"]], Win[u, D["HZA"]] = 1, 1
        bin_[u] = off
    Wout[D["OUT"][op.zero_out], z1] += 1
    Wout[D["OUT"][op.zero_out], z2] -= 1
    for cl in range(P):
        m.W_U.weight.data[cl, D["OUT"][cl]] = 1
    return m, dict(t_range=(tmin, tmax), hidden_units=len(js) + 2)


# ------------------------------------------------------------------ evaluation over the complete domain
def domain():
    import itertools
    rows = list(itertools.product(range(P), range(3), range(P), range(3), range(3), range(2)))
    t = torch.tensor(rows)
    return dict(m=t[:, 0], mu=t[:, 1], a=t[:, 2], au=t[:, 3], fu=t[:, 4], order=t[:, 5])


def canon_scales(L, scales):
    ms = torch.tensor(scales["MU"])[L["mu"]]
    as_ = torch.tensor(scales["AU"])[L["au"]]
    S = torch.tensor(SINV)[L["fu"]]
    return L["m"] * ms % P, L["a"] * as_ % P, S


def unseen_mask(L):
    return ((L["mu"] == T.OOD_UNIT) & (L["m"] >= 9)) | ((L["au"] == T.OOD_UNIT) & (L["a"] >= 9))


def target(op, M, A, S):
    return torch.tensor([op.target(int(x), int(y), int(z)) for x, y, z in zip(M, A, S)])


@torch.no_grad()
def run(m, L, dtype=torch.float32):
    m = m.to(dtype)
    b = T.build(L)
    types = torch.zeros_like(b.tokens)
    h = m.resid_pre_mlp(b.tokens, types)
    z = h @ m.W_in.weight.T + m.W_in.bias
    r = h + m.W_out(torch.relu(z))
    logits = m.W_U(r)
    return dict(h=h, pre=z, logits=logits, pred=logits.argmax(-1))
