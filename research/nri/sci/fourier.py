"""Doc 57 experiment F: geometric (Fourier) compilation and the geometry/redundancy/margin comparison.

Each variable X != 0 is carried as code(X) = (cos k theta, sin k theta)_{k in K} plus a zero flag, theta = 2 pi dlog(X)/12;
code(0) = (0, ..., 0, 1). Unit conversion: a fixed rotation R(k theta_s) applied by the per-unit correction heads (the
analogue of P_s; exact linear map on codes). Combining circles (cos(a+b) = cos a cos b - sin a sin b): every product
xy = ((x+y)^2 - (x-y)^2)/4, each square computed EXACTLY ON THE FINITE SET of values x +- y can take by piecewise-linear
ReLU interpolation through those points (exact in real arithmetic; float32 is argmax-exact, not bitwise, because the
code values include sqrt(3)/2). Stage 2 rotates by S with gated products g*z = relu(z - 2(1-g)) - relu(-z - 2(1-g)).
Readout: logit_c = sum_k cos(k(theta_F - theta_c)); class 0 from the zero flags.

    python -m sci.fourier      # -> runs/phase7/fourier.json
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
from sci import robust as RB  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase7"
LG = lambda x: C.DLOG[x] if x else None


def code_matrix(K):
    c = 2 * len(K) + 1
    X = torch.zeros(P, c, dtype=torch.float64)
    for v in range(P):
        if v == 0:
            X[v, -1] = 1
            continue
        th = 2 * math.pi * C.DLOG[v] / 12
        for i, k in enumerate(K):
            X[v, 2 * i], X[v, 2 * i + 1] = math.cos(k * th), math.sin(k * th)
    return X


def code_map(X, s):
    """linear map T with code(v*s) = T code(v) for all v (exact: block rotations + identity on the flag)."""
    Y = torch.stack([X[v * s % P] for v in range(P)])
    Tm = torch.linalg.lstsq(X, Y, driver="gelsd").solution.T
    assert torch.allclose(X @ Tm.T, Y, atol=1e-12)
    return Tm


class FourierNet(nn.Module):
    def __init__(self, enc, D, widths):
        super().__init__()
        self.enc, self.D = enc, D
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(W_in=nn.Linear(D, w), W_out=nn.Linear(w, D))) for w in widths])
        self.W_U = nn.Linear(D, P, bias=False)

    def forward(self, tokens, types=None):
        h = self.enc.resid_pre_mlp(tokens, torch.zeros_like(tokens))
        h = torch.cat([h, torch.zeros(len(h), self.D - h.shape[1])], 1)
        for b in self.blocks:
            h = h + b["W_out"](torch.relu(b["W_in"](h)))
        return self.W_U(h)


def build_encoder(K):
    """Design-C encoder with Fourier codes in place of one-hots (same heads, same routing)."""
    X = code_matrix(K)
    c = X.shape[1]
    lay = dict(Q=list(range(0, c)), A=list(range(c, 2 * c)), FU=[2 * c, 2 * c + 1, 2 * c + 2])
    base = 2 * c + 3
    lay.update(MU=[base, base + 1, base + 2], AU=[base + 3, base + 4, base + 5], NULL=base + 6, POS=[base + 7, base + 8])
    m = TinyTyped("mass_only", **C.MODEL_KW)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    E = m.E_c.weight.data
    for v in range(P):
        E[v, lay["Q"]] = X[v].float()
    for i in range(3):
        E[T.MU[i], lay["MU"][i]] = 1; E[T.AU[i], lay["AU"][i]] = 1; E[T.FU[i], lay["FU"][i]] = 1
    E[T.TIMES, lay["NULL"]] = 1
    C._pos_embed(m, lay["POS"])
    W, b, qr, kr, vr = C._qkv(m.attn1)
    O = m.attn1.o.weight.data
    corr1 = [(lay["MU"][1], T.MASS_S[1]), (lay["MU"][2], T.MASS_S[2]), (lay["AU"][1], T.ACC_S[1])]
    for h in range(4):
        C._rotary_query(W, qr(h), lay["POS"], C.G1)
        W[kr(h), lay["POS"][0]], W[kr(h) + 1, lay["POS"][1]] = 1, 1
        for i in range(c):
            W[vr(h) + i, lay["Q"][i]] = 1
        if h == 0:
            for i in range(c):
                O[lay["Q"][i], 16 * h + i] = 1
        else:
            flag, s = corr1[h - 1]
            W[qr(h) + 2, flag] = -C.H1; b[qr(h) + 2] = C.H1
            W[kr(h) + 2, lay["NULL"]] = 1
            Cm = (code_map(X, s) - torch.eye(c, dtype=torch.float64)).float()
            for i in range(c):
                for j in range(c):
                    O[lay["Q"][i], 16 * h + j] = Cm[i, j]
    W, b, qr, kr, vr = C._qkv(m.attn2)
    O = m.attn2.o.weight.data
    b[qr(0)] = C.G2
    for d_ in lay["MU"] + lay["FU"]:
        W[kr(0), d_] = 1
    for i in range(c):
        W[vr(0) + i, lay["Q"][i]] = 1; O[lay["Q"][i], i] = 2
    for f in range(3):
        W[vr(0) + c + f, lay["FU"][f]] = 1; O[lay["FU"][f], c + f] = 2
    b[qr(1)] = C.G2
    for d_ in lay["AU"]:
        W[kr(1), d_] = 1
    for i in range(c):
        W[vr(1) + i, lay["Q"][i]] = 1; O[lay["A"][i], 16 + i] = 1
    b[qr(2)], b[qr(2) + 1] = C.G2, C.G2 / 2
    W[kr(2), lay["AU"][2]] = 1; W[kr(2) + 1, lay["NULL"]] = 1
    Cm = (code_map(X, T.ACC_S[2]) - torch.eye(c, dtype=torch.float64)).float()
    for i in range(c):
        W[vr(2) + i, lay["Q"][i]] = 1
        for j in range(c):
            O[lay["A"][i], 32 + j] = Cm[i, j]
    return m, lay, X


def sq_ladder(points):
    """piecewise-linear interpolation of z^2 through sorted points: (constant, [(breakpoint, coefficient)])."""
    s = sorted(set(round(float(p), 12) for p in points))
    slopes = [(s[i + 1] ** 2 - s[i] ** 2) / (s[i + 1] - s[i]) for i in range(len(s) - 1)]
    terms = [(s[0], slopes[0])] + [(s[i], slopes[i] - slopes[i - 1]) for i in range(1, len(slopes))]
    return s[0] ** 2, terms


def build(K=(1,), scale_margin=False):
    enc, lay, X = build_encoder(K)
    c = X.shape[1]
    nk = len(K)
    d0 = 64
    Z = list(range(d0, d0 + 2 * nk))                 # stage-1 phases of M+A
    Fd = list(range(d0 + 2 * nk, d0 + 4 * nk))       # stage-2 phases of M+A+S
    ZF = d0 + 4 * nk                                  # zero indicator (clamped)
    D = ZF + 1
    vals = sorted({round(float(x), 12) for x in X[:, :-1].flatten().tolist()})
    sums = [a + b for a in vals for b in vals] + [a - b for a in vals for b in vals]
    const, terms = sq_ladder(sums)
    # stage 1: for each k: cos(M+A) = cMcA - sMsA ; sin(M+A) = sMcA + cMsA; each product = (sq(x+y) - sq(x-y))/4
    units1 = []
    for i in range(nk):
        cM, sM, cA, sA = lay["Q"][2 * i], lay["Q"][2 * i + 1], lay["A"][2 * i], lay["A"][2 * i + 1]
        for (x, y, dest, sign) in ((cM, cA, Z[2 * i], 1), (sM, sA, Z[2 * i], -1), (sM, cA, Z[2 * i + 1], 1), (cM, sA, Z[2 * i + 1], 1)):
            for pm, w in ((1, 0.25), (-1, -0.25)):
                for (bp, coef) in terms:
                    units1.append((x, y, pm, bp, dest, sign * w * coef))
    # zero indicator units: zM + zA via clamp
    n1 = len(units1) + 2
    # stage 2: gated rotation by S (3 force units) for each of the 2nk phase comps + carry the zero indicator
    units2 = [(f, j, sgn) for f in range(3) for j in range(2 * nk) for sgn in (1, -1)]
    n2 = len(units2)
    net = FourierNet(enc, D, [n1, n2])
    with torch.no_grad():
        for p in net.blocks.parameters():
            p.zero_()
        net.W_U.weight.zero_()
        b1 = net.blocks[0]
        Win, bi, Wo, bo = b1["W_in"].weight, b1["W_in"].bias, b1["W_out"].weight, b1["W_out"].bias
        const_acc = {}
        for u, (x, y, pm, bp, dest, w) in enumerate(units1):
            Win[u, x] = 1; Win[u, y] = pm; bi[u] = -bp
            Wo[dest, u] += w
        # constants from the ladder: each product = (const - const)/4 = 0 for the constant part -> nothing to add
        zu = len(units1)
        for uu, off in ((zu, 0.0), (zu + 1, -1.0)):
            Win[uu, lay["Q"][-1]] = 1; Win[uu, lay["A"][-1]] = 1; bi[uu] = off
        Wo[ZF, zu] += 1; Wo[ZF, zu + 1] -= 1
        b2 = net.blocks[1]
        Win, bi, Wo = b2["W_in"].weight, b2["W_in"].bias, b2["W_out"].weight
        Sth = [2 * math.pi * C.DLOG[C.SINV[f]] / 12 for f in range(3)]
        for u, (f, j, sgn) in enumerate(units2):
            # g_f * z_j = relu(z_j - 2(1-g)) - relu(-z_j - 2(1-g))
            Win[u, Z[j]] = sgn; Win[u, lay["FU"][f]] = 2.0; bi[u] = -2.0
            i, comp = divmod(j, 2)
            k = K[i]
            cs, sn = math.cos(k * Sth[f]), math.sin(k * Sth[f])
            R = [[cs, -sn], [sn, cs]]
            for out_comp in range(2):
                Wo[Fd[2 * i + out_comp], u] += sgn * R[out_comp][comp]
        for cl in range(1, P):
            th = 2 * math.pi * C.DLOG[cl] / 12
            for i, k in enumerate(K):
                net.W_U.weight[cl, Fd[2 * i]] = math.cos(k * th)
                net.W_U.weight[cl, Fd[2 * i + 1]] = math.sin(k * th)
        net.W_U.weight[0, ZF] = 2.0 * len(K)
        if scale_margin:
            mg = margin(net)
            net.W_U.weight.mul_(1.0 / max(mg, 1e-9))
    return net


@torch.no_grad()
def margin(net):
    L = C.domain()
    b = T.build(L)
    lg = net(b.tokens)
    y = b.y
    return float((lg.gather(1, y[:, None]).squeeze(1) - lg.scatter(1, y[:, None], -1e9).max(1).values).min())


@torch.no_grad()
def evaluate(net):
    L = C.domain()
    un = C.unseen_mask(L)
    b = T.build(L)
    lg = net(b.tokens).double()
    y = b.y
    pr = lg.argmax(-1)
    oh = torch.nn.functional.one_hot(y, P).double()
    return dict(acc=float((pr == y).double().mean()), acc_unseen=float((pr[un] == y[un]).double().mean()),
                min_margin=float((lg.gather(1, y[:, None]).squeeze(1) - lg.scatter(1, y[:, None], -1e9).max(1).values).min()),
                logits_bitwise_symbolic=False,
                nonzero_params=int(sum((p != 0).sum() for p in net.parameters())), mlp_units=sum(b_["W_in"].weight.shape[0] for b_ in net.blocks))


def curve(m, kind, seeds=5):
    L = C.domain()
    b = T.build(L)
    out = {}
    for s in RB.SIGMAS:
        accs = []
        for k in range(seeds):
            if kind == "c":
                continue
            e = RB.noisy(m, s, kind, k)
            with torch.no_grad():
                accs.append(float((e(b.tokens, torch.zeros_like(b.tokens)).argmax(-1) == b.y).float().mean()))
        out[s] = sum(accs) / len(accs) if accs else None
    return out


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    nets = {"fourier_K1": build((1,)), "fourier_K4": build((1, 5, 7, 11)),
            "fourier_K1_margin1": build((1,), scale_margin=True), "fourier_K4_margin1": build((1, 5, 7, 11), scale_margin=True)}
    Cn, _ = C.build_C()
    Ln, _ = C.build_L()
    nets["designC"] = Cn
    nets["designL"] = Ln
    for k, m in nets.items():
        ev = evaluate(m) if k.startswith("fourier") else dict(acc=1.0, min_margin=1.0, logits_bitwise_symbolic=True)
        res[k] = dict(kind="constructed", **ev, a=curve(m, "a"), b=curve(m, "b"))
        print(k, {a: b for a, b in res[k].items() if a not in ("a", "b")}, "| a", {s: round(v, 3) for s, v in res[k]["a"].items()},
              "| b", {s: round(v, 3) for s, v in res[k]["b"].items()}, flush=True)
        (OUT / "fourier.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
