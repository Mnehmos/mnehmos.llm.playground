"""Doc 57 experiment G: compositional neural compiler for computational graphs.

Architecture (smallest necessary alternative): the Design-C encoder (TinyTyped attention, exact canonical one-hot codes
of M, A and the force unit at '='), its 64-d residual zero-padded to D, followed by residual MLP blocks
  h <- h + W_out ReLU(W_in h + b),   logits = W_U h.
Each graph node computes g(sum_i phi_i(child_i)) exactly with a ReLU hat ladder (doc 53 D1) and WRITES THE ONE-HOT CODE
OF ITS VALUE into its own residual region, so later nodes read any function of it linearly. Nodes of equal depth share
a block. Statistics are sums of per-child terms: no lookup over input combinations.

    python -m sci.graph_compiler     # -> runs/phase7/graph_compiler.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import construct as C  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase7"
LG = lambda x: C.DLOG[x] if x else 0


# ------------------------------------------------------------------ expression trees
class Node:
    _id = 0

    def __init__(self, op, *kids, fused_unary=None):
        self.op, self.kids = op, list(kids)
        Node._id += 1
        self.id = Node._id

    def __repr__(self):
        if self.op in ("M", "A", "S"):
            return self.op
        if self.op == "sq":
            return f"({self.kids[0]})^2"
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[self.op]
        return f"({self.kids[0]} {sym} {self.kids[1]})"


def leaf(v):
    return Node(v)


def evaluate(n, env):
    if n.op in ("M", "A", "S"):
        return env[n.op]
    v = [evaluate(k, env) for k in n.kids]
    if n.op == "add":
        return (v[0] + v[1]) % P
    if n.op == "sub":
        return (v[0] - v[1]) % P
    if n.op == "mul":
        return v[0] * v[1] % P
    if n.op == "div":
        return 0 if (v[0] == 0 or v[1] == 0) else v[0] * pow(v[1], -1, P) % P
    if n.op == "sq":
        return v[0] * v[0] % P
    raise ValueError(n.op)


def internal(n):
    return [] if n.op in ("M", "A", "S") else [x for k in n.kids for x in internal(k)] + [n]


def depth(n):
    return 0 if n.op in ("M", "A", "S") else 1 + max(depth(k) for k in n.kids)


# ------------------------------------------------------------------ network
class StagedNet(nn.Module):
    def __init__(self, enc, D, widths):
        super().__init__()
        self.enc, self.D = enc, D
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(W_in=nn.Linear(D, w), W_out=nn.Linear(w, D))) for w in widths])
        self.W_U = nn.Linear(D, P, bias=False)

    def residual(self, tokens):
        h = self.enc.resid_pre_mlp(tokens, torch.zeros_like(tokens))
        h = torch.cat([h, torch.zeros(len(h), self.D - h.shape[1])], 1)
        states = [h]
        for blk in self.blocks:
            h = h + blk["W_out"](torch.relu(blk["W_in"](h)))
            states.append(h)
        return h, states

    def forward(self, tokens):
        return self.W_U(self.residual(tokens)[0])


def phi_for(op, which):
    """per-child statistic tables (value -> integer) and output map g, gate flags."""
    if op == "add":
        return [x for x in range(P)], (lambda t: t % P), False
    if op == "sub":
        return ([x for x in range(P)] if which == 0 else [-x for x in range(P)]), (lambda t: t % P), False
    if op == "mul":
        return [LG(x) for x in range(P)], (lambda t: pow(2, t % 12, P)), True
    if op == "div":
        return ([LG(x) for x in range(P)] if which == 0 else [-LG(x) for x in range(P)]), (lambda t: pow(2, t % 12, P)), True
    if op == "sq":
        return [x * x % P for x in range(P)], (lambda t: t % P), False
    raise ValueError(op)


def compile_graph(root, fuse_unary=False):
    """Returns (net, regions{node.id: dims}, info). fuse_unary folds 'sq' children into the parent's phi (one stage less)."""
    enc, _ = C.build_C()
    nodes = internal(root)
    if fuse_unary:
        nodes = [n for n in nodes if not (n.op == "sq" and n is not root)]
    # regions: leaves come from the encoder
    D0 = 64
    regions = {}
    nxt = D0
    for n in nodes:
        if n is root:
            continue
        regions[n.id] = list(range(nxt, nxt + P)); nxt += P
    out_dims = list(range(nxt, nxt + P)); nxt += P
    regions[root.id] = out_dims
    D = nxt
    leaf_region = {"M": (C.DIM["Q"], list(range(P))), "A": (C.DIM["A"], list(range(P))), "S": (C.DIM["FU"], C.SINV)}

    def source(k):
        """(dims, value-per-dim, extra phi transform) for a child, following fused unary nodes."""
        if fuse_unary and k.op == "sq" and k is not root:
            dims, vals, f = source(k.kids[0])
            return dims, vals, (lambda x, f=f: f(x) * f(x) % P)
        if k.op in ("M", "A", "S"):
            d, vals = leaf_region[k.op]
            return d, vals, (lambda x: x)
        return regions[k.id], list(range(P)), (lambda x: x)

    def ndepth(n):
        if n.op in ("M", "A", "S"):
            return 0
        if fuse_unary and n.op == "sq" and n is not root:
            return ndepth(n.kids[0])
        return 1 + max(ndepth(k) for k in n.kids)

    by_depth = {}
    for n in nodes:
        by_depth.setdefault(ndepth(n), []).append(n)
    L_ = max(by_depth)
    # per-node ladder specs
    specs = {}
    widths = [0] * L_
    for n in nodes:
        terms = []
        gate = False
        for w, k in enumerate(n.kids):
            phi, g, gt = phi_for(n.op, w)
            dims, vals, f = source(k)
            terms.append((dims, [phi[f(v)] for v in vals], [f(v) == 0 for v in vals]))
            gate = gate or gt
        vals_ = {0}
        for _, tab, _ in terms:
            vals_ = {a + b for a in vals_ for b in tab}
        tmin, tmax = min(vals_), max(vals_)
        units = (tmax - tmin + 3) + (2 if gate else 0)
        specs[n.id] = (n, terms, g, gate, tmin, tmax, units)
        widths[ndepth(n) - 1] += units
    net = StagedNet(enc, D, widths)
    with torch.no_grad():
        for p in net.blocks.parameters():
            p.zero_()
        net.W_U.weight.zero_()
        offs = [0] * L_
        for n in nodes:
            _, terms, g, gate, tmin, tmax, units = specs[n.id]
            blk = net.blocks[ndepth(n) - 1]
            Win, bin_, Wout = blk["W_in"].weight, blk["W_in"].bias, blk["W_out"].weight
            u0 = offs[ndepth(n) - 1]
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
                bin_[u0 + u] = -j
            dest = regions[n.id]
            for k in range(tmin, tmax + 1):
                c = dest[g(k)]
                Wout[c, u0 + (k - 1 - (tmin - 1))] += 1
                Wout[c, u0 + (k - (tmin - 1))] -= 2
                Wout[c, u0 + (k + 1 - (tmin - 1))] += 1
            if gate:
                z1, z2 = u0 + len(js), u0 + len(js) + 1
                for uu, off in ((z1, 0.0), (z2, -1.0)):
                    for dims, tab, zero in terms:
                        for d_, z in zip(dims, zero):
                            if z:
                                Win[uu, d_] += 1
                    bin_[uu] = off
                Wout[dest[0], z1] += 1
                Wout[dest[0], z2] -= 1
            offs[ndepth(n) - 1] += units
        for c in range(P):
            net.W_U.weight[c, out_dims[c]] = 1
    info = dict(expr=repr(root), D=D, blocks=L_, mlp_units=sum(widths), widths=widths, n_nodes=len(nodes),
                n_params=sum(p.numel() for p in net.parameters()), nonzero_params=int(sum((p != 0).sum() for p in net.parameters())),
                nonzero_params_decoder=int(sum((p != 0).sum() for p in list(net.blocks.parameters()) + [net.W_U.weight])))
    return net, regions, nodes, info


# ------------------------------------------------------------------ verification
@torch.no_grad()
def verify(root, fuse_unary=False):
    net, regions, nodes, info = compile_graph(root, fuse_unary)
    L = C.domain()
    un = C.unseen_mask(L)
    b = T.build(L)
    M, A, S = C.canon_scales(L, dict(MU=T.MASS_S, AU=T.ACC_S))
    envs = [dict(M=int(x), A=int(y), S=int(z)) for x, y, z in zip(M, A, S)]
    want = torch.tensor([evaluate(root, e) for e in envs])
    h, states = net.residual(b.tokens)
    logits = net.W_U(h)
    oh = torch.nn.functional.one_hot(want, P).float()
    res = dict(**info, acc=float((logits.argmax(-1) == want).float().mean()),
               acc_unseen=float((logits.argmax(-1)[un] == want[un]).float().mean()),
               logits_exact_onehot=bool((logits == oh).all()))
    # per-intermediate-node observability, causal access, programmability
    per = {}
    for n in nodes:
        if n is root:
            continue
        val = torch.tensor([evaluate(n, e) for e in envs])
        obs = bool((h[:, regions[n.id]] == torch.nn.functional.one_hot(val, P).float()).all())
        # causal access: set the node's region (in the state after its block) to every value, continue the forward
        dn = None
        for i, st in enumerate(states[1:]):
            if bool((st[:, regions[n.id]] != 0).any()):
                dn = i; break
        ok = tot = 0
        sub = torch.arange(0, len(envs), 7)
        for v in range(P):
            hh = states[dn + 1][sub].clone()
            hh[:, regions[n.id]] = 0; hh[:, regions[n.id][v]] = 1
            for blk in net.blocks[dn + 1:]:
                hh = hh + blk["W_out"](torch.relu(blk["W_in"](hh)))
            pred = net.W_U(hh).argmax(-1)
            w = torch.tensor([eval_with(root, envs[i], n, v) for i in sub.tolist()])
            ok += int((pred == w).sum()); tot += len(w)
        per[repr(n)] = dict(observable_bitwise=obs, interchange_acc=ok / tot)
    res["intermediates"] = per
    return res, net, regions, nodes


def eval_with(root, env, target, value):
    """evaluate root with node `target` forced to `value`."""
    def ev(n):
        if n is target:
            return value
        if n.op in ("M", "A", "S"):
            return env[n.op]
        v = [ev(k) for k in n.kids]
        return _apply(n.op, v)
    return ev(root)


def _apply(op, v):
    if op == "add": return (v[0] + v[1]) % P
    if op == "sub": return (v[0] - v[1]) % P
    if op == "mul": return v[0] * v[1] % P
    if op == "div": return 0 if (v[0] == 0 or v[1] == 0) else v[0] * pow(v[1], -1, P) % P
    if op == "sq": return v[0] * v[0] % P


def programmability(root):
    """swap the operation of one internal (non-root) node by recompiling; only that node's semantics may change."""
    nodes = [n for n in internal(root) if n is not root and n.op in ("add", "sub", "mul", "div")]
    if not nodes:
        return None
    n = nodes[0]
    old = n.op
    n.op = {"add": "mul", "mul": "add", "sub": "div", "div": "sub"}[old]
    r, *_ = verify(root)
    n.op = old
    return dict(node=repr(n), swapped_to={"add": "mul", "mul": "add", "sub": "div", "div": "sub"}[old], acc=r["acc"], exact=r["logits_exact_onehot"])


def random_tree(d, rng):
    ops2 = ["add", "sub", "mul", "div"]
    if d == 0:
        return leaf(rng.choice(["M", "A", "S"]))
    if rng.random() < 0.2:
        return Node("sq", random_tree(d - 1, rng))
    deep = random_tree(d - 1, rng)
    other = random_tree(rng.randrange(0, d), rng)
    kids = [deep, other] if rng.random() < 0.5 else [other, deep]
    return Node(rng.choice(ops2), *kids)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    named = {
        "M*A+S": Node("add", Node("mul", leaf("M"), leaf("A")), leaf("S")),
        "(M+A)*S": Node("mul", Node("add", leaf("M"), leaf("A")), leaf("S")),
        "M^2+A^2": Node("add", Node("sq", leaf("M")), Node("sq", leaf("A"))),
        "(M-A)/(M+A)": Node("div", Node("sub", leaf("M"), leaf("A")), Node("add", leaf("M"), leaf("A"))),
    }
    res = {"named": {}, "random": []}
    for k, root in named.items():
        r, *_ = verify(root)
        rf, *_ = verify(root, fuse_unary=True)
        res["named"][k] = dict(unfused=r, fused=dict(blocks=rf["blocks"], mlp_units=rf["mlp_units"], acc=rf["acc"], exact=rf["logits_exact_onehot"],
                                                      nonzero_params_decoder=rf["nonzero_params_decoder"]),
                               program_swap=programmability(root))
        print(k, {a: b for a, b in r.items() if a != "intermediates"}, "| inter", r["intermediates"], "| fused", res["named"][k]["fused"], "| swap", res["named"][k]["program_swap"], flush=True)
    rng = random.Random(57)
    for d in range(1, 7):
        for _ in range(10):
            root = random_tree(d, rng)
            r, *_ = verify(root)
            inter = r.pop("intermediates")
            r.update(depth=depth(root), all_intermediates_observable=all(v["observable_bitwise"] for v in inter.values()),
                     min_intermediate_interchange=min([v["interchange_acc"] for v in inter.values()], default=1.0))
            res["random"].append(r)
            print(d, r["expr"][:70], "acc", r["acc"], "exact", r["logits_exact_onehot"], "obs", r["all_intermediates_observable"],
                  "iit", round(r["min_intermediate_interchange"], 3), "blocks", r["blocks"], "units", r["mlp_units"], "D", r["D"], "nz_dec", r["nonzero_params_decoder"], flush=True)
            (OUT / "graph_compiler.json").write_text(json.dumps(res, indent=1))
    # scaling fit: decoder nonzero params and units vs n_nodes
    X = torch.tensor([[r["n_nodes"], 1.0] for r in res["random"]])
    for key in ("mlp_units", "nonzero_params_decoder", "D"):
        Y = torch.tensor([float(r[key]) for r in res["random"]])
        coef = torch.linalg.lstsq(X, Y[:, None]).solution.flatten()
        pred = X @ coef
        res[f"fit_{key}_vs_nodes"] = dict(slope=float(coef[0]), intercept=float(coef[1]), r2=float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()))
    print({k: v for k, v in res.items() if k.startswith("fit_")})
    (OUT / "graph_compiler.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
