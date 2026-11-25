"""Doc 53 §53.8: a neural compiler. Specification -> TinyTyped parameters (Design C, generalised); no data, no training.

Spec: prime p; k roles, each with unit scales (unit 0 has scale 1); output-unit scales; an expression given as a ridge
statistic t = sum_r phi_r(X_r) + phi_S(S) and an output map g(t), with optional zero gates.
Task family (generalising nri.dimtask): prompt = x_1 u_1 x x_2 u_2 x ... x_k u_k -> f =, every role order allowed;
a role is identified only by its unit token. X_r = x_r * scale(u_r) mod p, S = scale(f)^-1.

Layout: layer 1 = copy head + one correction head per non-identity unit scale (P_s - I); layer 2 = one gather head
per role (role 0 also gathers the output unit). Width, heads and MLP size are computed from the spec.

    python -m sci.compiler     # -> runs/phase6/compiler.json
"""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri.model import TinyTyped  # noqa: E402

OUT = ROOT / "runs/phase6"


def primitive_root(p):
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in {q for q in range(2, p) if (p - 1) % q == 0 and all(q % r for r in range(2, q))}):
            return g
    return 1


@dataclass
class Spec:
    p: int
    unit_scales: list          # per role: list of scales (scales[r][0] == 1)
    out_scales: list           # output-unit scales
    expr: str                  # 'mono:e1,e2,...[;S]' | 'sum' | 'diff'

    @property
    def k(self):
        return len(self.unit_scales)


class Task:
    def __init__(self, s: Spec):
        self.s = s
        p, k = s.p, s.k
        self.g = primitive_root(p)
        self.dlog = {pow(self.g, i, p): i for i in range(p - 1)}
        self.VAL = list(range(p))
        nxt = p
        self.UNIT = []
        for r in range(k):
            self.UNIT.append(list(range(nxt, nxt + len(s.unit_scales[r]))))
            nxt += len(s.unit_scales[r])
        self.FU = list(range(nxt, nxt + len(s.out_scales))); nxt += len(s.out_scales)
        self.TIMES, self.ARROW, self.EQ = nxt, nxt + 1, nxt + 2
        self.vocab = nxt + 3
        self.seq = 3 * k + 2
        self.Sval = [pow(x, -1, p) for x in s.out_scales]

    def canon(self, x, u, f):
        return [x[r] * self.s.unit_scales[r][u[r]] % self.s.p for r in range(self.s.k)], self.Sval[f]

    def target(self, X, S):
        p, e = self.s.p, self.s.expr
        if e == "sum":
            return sum(X) % p
        if e == "diff":
            return (X[0] - X[1]) % p
        exps, useS = parse_mono(e)
        if any(x == 0 for x, a in zip(X, exps) if a != 0):
            return 0
        v = 1
        for x, a in zip(X, exps):
            v = v * pow(x, a, p) % p if a >= 0 else v * pow(pow(x, -1, p), -a, p) % p
        return v * (S if useS else 1) % p

    def tokens(self, x, u, f, order):
        """order: permutation of roles giving the written sequence."""
        t = []
        for i, r in enumerate(order):
            if i:
                t.append(self.TIMES)
            t += [self.VAL[x[r]], self.UNIT[r][u[r]]]
        return t + [self.ARROW, self.FU[f], self.EQ]


def parse_mono(e):
    body = e.split(":")[1]
    useS = body.endswith(";S")
    exps = [int(a) for a in body.replace(";S", "").split(",")]
    return exps, useS


def build(s: Spec):
    T_ = Task(s)
    p, k = s.p, s.k
    nf = len(s.out_scales)
    corr = [(r, i, sc) for r in range(k) for i, sc in enumerate(s.unit_scales[r]) if i > 0 and sc != 1]
    H = max(1 + len(corr), k, 1)
    # residual layout
    dims, nxt = {}, 0
    def alloc(name, n):
        nonlocal nxt
        dims[name] = list(range(nxt, nxt + n)); nxt += n
    alloc("Q", p)
    for r in range(1, k):
        alloc(f"R{r}", p)
    alloc("FU", nf)
    for r in range(k):
        alloc(f"U{r}", len(s.unit_scales[r]))
    alloc("NULL", 1); alloc("POS", 2); alloc("OUT", p)
    need = nxt
    dh = max(p + nf, 3)
    d = H * dh
    while d < need:
        dh += 1; d = H * dh
    # MLP statistic
    phi = []
    lg = lambda x: T_.dlog[x] if x else 0
    gate = []
    if s.expr == "sum":
        phi = [[x for x in range(p)] for _ in range(k)]; phiS = [0] * nf; gfun = lambda t: t % p
    elif s.expr == "diff":
        phi = [list(range(p)), [-x for x in range(p)]] + [[0] * p for _ in range(k - 2)]; phiS = [0] * nf; gfun = lambda t: t % p
    else:
        exps, useS = parse_mono(s.expr)
        phi = [[a * lg(x) for x in range(p)] for a in exps]
        phiS = [T_.dlog[v] if useS else 0 for v in T_.Sval]
        gate = [r for r, a in enumerate(exps) if a != 0]
        gfun = lambda t: pow(T_.g, t % (p - 1), p)
    vals = [sum(c) for c in itertools.product(*[sorted(set(f)) for f in phi], sorted(set(phiS)))]
    tmin, tmax = min(vals), max(vals)
    n_units = (tmax - tmin + 3) + (2 if gate else 0)
    m = TinyTyped("mass_only", d=d, n_heads=H, d_mlp=n_units, n_types=1, vocab=T_.vocab, seq_len=T_.seq, n_out=p)
    with torch.no_grad():
        for prm in m.parameters():
            prm.zero_()
    E = m.E_c.weight.data
    for v in range(p):
        E[v, dims["Q"][v]] = 1
    for r in range(k):
        for i, tok in enumerate(T_.UNIT[r]):
            E[tok, dims[f"U{r}"][i]] = 1
    for f, tok in enumerate(T_.FU):
        E[tok, dims["FU"][f]] = 1
    E[T_.TIMES, dims["NULL"][0]] = 1
    th = 2 * math.pi / T_.seq
    for q in range(T_.seq):
        m.E_p.weight.data[q, dims["POS"][0]] = math.cos(th * q)
        m.E_p.weight.data[q, dims["POS"][1]] = math.sin(th * q)
    scale = math.sqrt(dh)
    G1 = 1000 * scale / (1 - math.cos(th))          # >= 1000 / sqrt(dh) score gap after scaling -> exact float32
    H1 = 3 * G1
    G2 = 1000 * scale
    perm = lambda sc: torch.tensor([[1.0 if a == (b * sc) % p else 0.0 for b in range(p)] for a in range(p)])
    W, b = m.attn1.qkv.weight.data, m.attn1.qkv.bias.data
    O = m.attn1.o.weight.data
    q0, k0, v0 = 0, d, 2 * d
    c, sn = math.cos(th), math.sin(th)
    for h in range(1 + len(corr)):
        qr, kr, vr = q0 + dh * h, k0 + dh * h, v0 + dh * h
        W[qr, dims["POS"][0]], W[qr, dims["POS"][1]] = G1 * c, G1 * sn
        W[qr + 1, dims["POS"][0]], W[qr + 1, dims["POS"][1]] = -G1 * sn, G1 * c
        W[kr, dims["POS"][0]], W[kr + 1, dims["POS"][1]] = 1, 1
        for i in range(p):
            W[vr + i, dims["Q"][i]] = 1
        if h == 0:
            for i in range(p):
                O[dims["Q"][i], dh * h + i] = 1
        else:
            r, iu, sc = corr[h - 1]
            W[qr + 2, dims[f"U{r}"][iu]] = -H1
            b[qr + 2] = H1
            W[kr + 2, dims["NULL"][0]] = 1
            Cm = perm(sc) - torch.eye(p)
            for a in range(p):
                for bb in range(p):
                    O[dims["Q"][a], dh * h + bb] = Cm[a, bb]
    W, b = m.attn2.qkv.weight.data, m.attn2.qkv.bias.data
    O = m.attn2.o.weight.data
    for r in range(k):
        qr, kr, vr = q0 + dh * r, k0 + dh * r, v0 + dh * r
        b[qr] = G2
        for dd in dims[f"U{r}"]:
            W[kr, dd] = 1
        dest = dims["Q"] if r == 0 else dims[f"R{r}"]
        mult = 2 if r == 0 else 1
        if r == 0:
            for dd in dims["FU"]:
                W[kr, dd] = 1
        for i in range(p):
            W[vr + i, dims["Q"][i]] = 1
            O[dest[i], dh * r + i] = mult
        if r == 0:
            for f in range(nf):
                W[vr + p + f, dims["FU"][f]] = 1
                O[dims["FU"][f], dh * r + p + f] = 2
    # MLP
    regions = [dims["Q"]] + [dims[f"R{r}"] for r in range(1, k)]
    Win, bin_, Wout = m.W_in.weight.data, m.W_in.bias.data, m.W_out.weight.data
    B = float(tmax - tmin + 10)
    js = list(range(tmin - 1, tmax + 2))
    for u, j in enumerate(js):
        for r in range(k):
            for x in range(p):
                Win[u, regions[r][x]] = phi[r][x]
            if r in gate:
                Win[u, regions[r][0]] -= B
        for f in range(nf):
            Win[u, dims["FU"][f]] = phiS[f]
        bin_[u] = -j
    unit = {j: u for u, j in enumerate(js)}
    for t in range(tmin, tmax + 1):
        o = dims["OUT"][gfun(t)]
        Wout[o, unit[t - 1]] += 1; Wout[o, unit[t]] -= 2; Wout[o, unit[t + 1]] += 1
    if gate:
        z1, z2 = len(js), len(js) + 1
        for u, off in ((z1, 0.0), (z2, -1.0)):
            for r in gate:
                Win[u, regions[r][0]] = 1
            bin_[u] = off
        Wout[dims["OUT"][0], z1] += 1; Wout[dims["OUT"][0], z2] -= 1
    for cl in range(p):
        m.W_U.weight.data[cl, dims["OUT"][cl]] = 1
    info = dict(d=d, heads=H, d_head=dh, d_mlp=n_units, residual_dims_used=need, t_range=(tmin, tmax),
                n_params=m.n_params(), nonzero_params=int(sum((q != 0).sum() for q in m.parameters())))
    return m, T_, dims, regions, info


def domain_iter(T_):
    s = T_.s
    p, k = s.p, s.k
    for x in itertools.product(range(p), repeat=k):
        for u in itertools.product(*[range(len(sc)) for sc in s.unit_scales]):
            for f in range(len(s.out_scales)):
                for order in itertools.permutations(range(k)):
                    yield x, u, f, order


@torch.no_grad()
def verify(s: Spec, max_exhaustive=1_200_000, n_sample=200_000, seed=0, batch=65536):
    m, T_, dims, regions, info = build(s)
    p, k = s.p, s.k
    size = p ** k * math.prod(len(sc) for sc in s.unit_scales) * len(s.out_scales) * math.factorial(k)
    if size <= max_exhaustive:
        items = domain_iter(T_); mode = "exhaustive"
    else:
        rng = random.Random(seed)
        def gen():
            for _ in range(n_sample):
                x = tuple(rng.randrange(p) for _ in range(k))
                u = tuple(rng.randrange(len(sc)) for sc in s.unit_scales)
                yield x, u, rng.randrange(len(s.out_scales)), tuple(rng.sample(range(k), k))
        items = gen(); mode = f"sampled {n_sample}"
    n = ok = exact = 0
    iit_ok = iit_n = 0
    inv_ok = True
    buf = []
    def flush(buf):
        nonlocal n, ok, exact, iit_ok, iit_n, inv_ok
        tok = torch.tensor([T_.tokens(*it) for it in buf])
        canon = [T_.canon(it[0], it[1], it[2]) for it in buf]
        y = torch.tensor([T_.target(X, S) for X, S in canon])
        hh = m.resid_pre_mlp(tok, torch.zeros_like(tok))
        lg = m.head_from_resid(hh)
        oh = torch.nn.functional.one_hot(y, p).float()
        n += len(y); ok += int((lg.argmax(-1) == y).sum()); exact += int((lg == oh).all(1).sum())
        # invariance: each role region equals one-hot of its canonical value
        for r in range(k):
            Xr = torch.tensor([X[r] for X, _ in canon])
            inv_ok &= bool((hh[:, regions[r]] == torch.nn.functional.one_hot(Xr, p).float()).all())
        # interchange: every role region x every source value (on the first 4096 of each batch)
        sub = slice(0, min(4096, len(y)))
        for r in range(k):
            for v in range(p):
                h2 = hh[sub].clone(); h2[:, regions[r]] = 0; h2[:, regions[r][v]] = 1
                want = torch.tensor([T_.target([v if q == r else X[q] for q in range(k)], S) for X, S in canon[sub]])
                iit_ok += int((m.head_from_resid(h2).argmax(-1) == want).sum()); iit_n += len(want)
    for it in items:
        buf.append(it)
        if len(buf) == batch:
            flush(buf); buf = []
    if buf:
        flush(buf)
    return dict(spec=dict(p=s.p, unit_scales=s.unit_scales, out_scales=s.out_scales, expr=s.expr), mode=mode, domain_size=size,
                n=n, acc=ok / n, logits_exact_onehot_frac=exact / n, role_regions_equal_canonical_onehot=inv_ok,
                iit_acc=iit_ok / iit_n, iit_n=iit_n, **info)


def grid():
    rng = random.Random(53)
    specs = []
    for p in (5, 7, 11, 13, 17, 19, 23):
        for k in (2, 3):
            us = [[1] + rng.sample(range(2, p), 2) for _ in range(k)]
            os_ = [1] + rng.sample(range(2, p), 2)
            exprs = ["mono:" + ",".join(["1"] * k) + ";S", "mono:" + ",".join(str(rng.randint(1, 3)) for _ in range(k)) + ";S",
                     "mono:1,-1" + (",1" if k == 3 else ""), "sum"]
            if k == 2:
                exprs.append("diff")
            for e in exprs:
                specs.append(Spec(p, us, os_, e))
    # the original task reproduced by the compiler
    specs.insert(0, Spec(13, [[1, 5, 8], [1, 2, 7]], [1, 6, 11], "mono:1,1;S"))
    return specs


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for s in grid():
        r = verify(s)
        res.append(r)
        print(f"p={s.p:2d} k={s.k} {s.expr:14s} {r['mode']:16s} n={r['n']:8d} acc={r['acc']:.4f} exact={r['logits_exact_onehot_frac']:.4f} "
              f"inv={r['role_regions_equal_canonical_onehot']} iit={r['iit_acc']:.4f} d={r['d']} H={r['heads']} mlp={r['d_mlp']} nz={r['nonzero_params']}", flush=True)
        (OUT / "compiler.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
