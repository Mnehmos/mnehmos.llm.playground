"""Doc 59 G2: reusable compiled operators — CSE, lifetime analysis + register allocation, weight-tied function blocks,
conditional execution. Built on the Phase VII graph compiler (Design-C encoder + residual MLP blocks, one-hot registers).

  CSE        : hash-consing of structurally identical subtrees (a DAG; each distinct node computed once)
  REGALLOC   : a register is cleared after its last read by 13 units in the last-reader block (ReLU(x_d) -> -x_d, exact
               for one-hot codes; readers in that block still see the value, since all read the pre-block residual);
               a node computed in block b may reuse any register cleared in a block <= b-1 (greedy linear scan)
  SELECT     : out_i = ReLU(X_i + c - 1) + ReLU(Y_i - c) for one-hot X, Y and c in {0,1} (exact)
  LOOP       : one weight-tied block  ARG <- ARG - ARG + onehot(f(ARG))  applied n times (unique parameters of one block)

    python -m sci.graph2      # -> runs/phase8/graph2.json
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
from sci import graph_compiler as G  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase8"
LEAF = {"M": (C.DIM["Q"], list(range(P))), "A": (C.DIM["A"], list(range(P))), "S": (C.DIM["FU"], C.SINV)}


class N:
    def __init__(self, op, *kids):
        self.op, self.kids = op, list(kids)

    def __repr__(self):
        if self.op in LEAF:
            return self.op
        if self.op == "sq":
            return f"({self.kids[0]})^2"
        if self.op == "select":
            return f"(if {self.kids[0]}==0 then {self.kids[1]} else {self.kids[2]})"
        return f"({self.kids[0]} {dict(add='+', sub='-', mul='*', div='/')[self.op]} {self.kids[1]})"


def ev(n, env, memo=None):
    if n.op in LEAF:
        return env[n.op]
    v = [ev(k, env) for k in n.kids]
    if n.op == "select":
        return v[1] if v[0] == 0 else v[2]
    return G._apply(n.op, v)


def from_tree(t):
    """convert a graph_compiler.Node tree to N."""
    if t.op in LEAF:
        return N(t.op)
    return N(t.op, *[from_tree(k) for k in t.kids])


def dag(root, cse=True):
    """returns (unique internal nodes in topological order, root, key->node) ; leaves are shared."""
    table, order = {}, []

    def visit(n):
        if n.op in LEAF:
            return n.op
        ks = tuple(visit(k) for k in n.kids)
        key = (n.op, ks) if cse else (n.op, ks, id(n))
        if key not in table:
            node = N(n.op, *[table[k] if not isinstance(k, str) else N(k) for k in ks])
            node.key = key
            table[key] = node
            order.append(node)
        return key
    rk = visit(root)
    return order, table[rk] if not isinstance(rk, str) else N(rk), table


def compile_dag(root, cse=True, regalloc=True):
    order, rnode, table = dag(root, cse)
    enc, _ = C.build_C()
    # depth / block of each node (leaves: 0)
    depth = {}
    for n in order:
        depth[id(n)] = 1 + max((depth.get(id(k), 0) for k in n.kids if k.op not in LEAF), default=0)
    L_ = max(depth.values())
    readers = {id(n): [] for n in order}
    for n in order:
        for k in n.kids:
            if k.op not in LEAF:
                readers[id(k)].append(depth[id(n)])
    last_use = {i: (max(r) if r else None) for i, r in readers.items()}
    # register allocation (root writes OUT region)
    D0 = 64
    regs, free = {}, []          # free: list of (available_from_block, base)
    nxt = D0
    clears = {}                  # block -> list of base dims to clear
    for n in sorted(order, key=lambda x: depth[id(x)]):
        if n is rnode:
            continue
        b = depth[id(n)]
        base = None
        if regalloc:
            for i, (avail, bs) in enumerate(free):
                if avail <= b - 1:
                    base = bs; free.pop(i); break
        if base is None:
            base = nxt; nxt += P
        regs[id(n)] = list(range(base, base + P))
        lu = last_use[id(n)]
        if regalloc and lu is not None and n is not rnode:
            clears.setdefault(lu, []).append(base)
            free.append((lu, base))
    out_dims = list(range(nxt, nxt + P)); nxt += P
    regs[id(rnode)] = out_dims
    D = nxt

    def src(k):
        if k.op in LEAF:
            return LEAF[k.op]
        return regs[id(k)], list(range(P))

    # unit counts per block
    specs, widths = {}, [0] * L_
    for n in order:
        b = depth[id(n)]
        if n.op == "select":
            units = 2 * P
            specs[id(n)] = ("select", units)
        else:
            terms, gate = [], False
            for w, k in enumerate(n.kids):
                phi, g, gt = G.phi_for(n.op, w)
                dims, vals = src(k)
                terms.append((dims, [phi[v] for v in vals], [v == 0 for v in vals]))
                gate = gate or gt
            vs = {0}
            for _, tab, _ in terms:
                vs = {a + c for a in vs for c in tab}
            tmin, tmax = min(vs), max(vs)
            units = (tmax - tmin + 3) + (2 if gate else 0)
            specs[id(n)] = ("ladder", units, terms, g, gate, tmin, tmax)
        widths[b - 1] += units
    for b, bases in clears.items():
        widths[b - 1] += P * len(bases)
    net = G.StagedNet(enc, D, widths)
    with torch.no_grad():
        for p in net.blocks.parameters():
            p.zero_()
        net.W_U.weight.zero_()
        offs = [0] * L_
        for n in order:
            b = depth[id(n)]
            blk = net.blocks[b - 1]
            Win, bi, Wo = blk["W_in"].weight, blk["W_in"].bias, blk["W_out"].weight
            u0 = offs[b - 1]
            dest = regs[id(n)]
            sp = specs[id(n)]
            if sp[0] == "select":
                c_dim = src(n.kids[0])[0][0]            # condition: child is zero  <=> its region's value-0 dim
                X, Y = src(n.kids[1])[0], src(n.kids[2])[0]
                for i in range(P):
                    Win[u0 + i, X[i]] = 1; Win[u0 + i, c_dim] += 1; bi[u0 + i] = -1; Wo[dest[i], u0 + i] += 1
                    Win[u0 + P + i, Y[i]] = 1; Win[u0 + P + i, c_dim] -= 1; Wo[dest[i], u0 + P + i] += 1
            else:
                _, units, terms, g, gate, tmin, tmax = sp
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
                    Wo[c, u0 + (k - tmin)] += 1; Wo[c, u0 + (k - tmin) + 1] -= 2; Wo[c, u0 + (k - tmin) + 2] += 1
                if gate:
                    z1 = u0 + len(js)
                    for uu, off in ((z1, 0.0), (z1 + 1, -1.0)):
                        for dims, tab, zero in terms:
                            for d_, z in zip(dims, zero):
                                if z:
                                    Win[uu, d_] += 1
                        bi[uu] = off
                    Wo[dest[0], z1] += 1; Wo[dest[0], z1 + 1] -= 1
            offs[b - 1] += sp[1]
        for b, bases in clears.items():
            blk = net.blocks[b - 1]
            for base in bases:
                for i in range(P):
                    u = offs[b - 1]
                    blk["W_in"].weight[u, base + i] = 1
                    blk["W_out"].weight[base + i, u] = -1
                    offs[b - 1] += 1
        for c in range(P):
            net.W_U.weight[c, out_dims[c]] = 1
    return net, regs, order, rnode, dict(D=D, blocks=L_, widths=widths, n_nodes=len(order), clears=sum(len(v) for v in clears.values()))


def cost(net, enc_d=64):
    dense = sum(p.numel() for p in net.parameters())
    nz = int(sum((p != 0).sum() for p in net.parameters()))
    D = net.D
    flops = sum(2 * D * b["W_in"].weight.shape[0] * 2 for b in net.blocks) + 2 * D * P
    return dict(dense_params=dense, nonzero_params=nz, decoder_flops_per_prompt=flops)


@torch.no_grad()
def verify(root, cse=True, regalloc=True):
    net, regs, order, rnode, info = compile_dag(root, cse, regalloc)
    L = C.domain()
    b = T.build(L)
    M, A, S = C.canon_scales(L, dict(MU=T.MASS_S, AU=T.ACC_S))
    envs = [dict(M=int(x), A=int(y), S=int(z)) for x, y, z in zip(M, A, S)]
    want = torch.tensor([ev(root, e) for e in envs])
    h, states = net.residual(b.tokens)
    lg = net.W_U(h)
    exact = bool((lg == torch.nn.functional.one_hot(want, P).float()).all())
    acc = float((lg.argmax(-1) == want).float().mean())
    # observability: each intermediate is one-hot-correct in the state right after its block (live interval)
    depth = {}
    for n in order:
        depth[id(n)] = 1 + max((depth.get(id(k), 0) for k in n.kids if k.op not in LEAF), default=0)
    obs = True
    for n in order:
        if n is rnode:
            continue
        val = torch.tensor([ev(n, e) for e in envs])
        st = states[depth[id(n)]]
        obs &= bool((st[:, regs[id(n)]] == torch.nn.functional.one_hot(val, P).float()).all())
    return dict(expr=repr(root)[:120], acc=acc, exact=exact, intermediates_observable_in_live_interval=obs, **info, **cost(net))


# ------------------------------------------------------------------ weight-tied loop (function call)
class LoopNet(nn.Module):
    def __init__(self, enc, D, n_iter, block, move_in, W_U):
        super().__init__()
        self.enc, self.D, self.n = enc, D, n_iter
        self.move_in, self.block, self.W_U = move_in, block, W_U

    def forward(self, tokens):
        h = self.enc.resid_pre_mlp(tokens, torch.zeros_like(tokens))
        h = torch.cat([h, torch.zeros(len(h), self.D - h.shape[1])], 1)
        h = h + self.move_in["W_out"](torch.relu(self.move_in["W_in"](h)))
        for _ in range(self.n):                                       # ONE set of weights, applied n times
            h = h + self.block["W_out"](torch.relu(self.block["W_in"](h)))
        return self.W_U(h)


def build_loop(n_iter):
    """ARG register <- M (move), then ARG <- (ARG)^2 n times with one tied block; output ARG -> M^(2^n)."""
    enc, _ = C.build_C()
    D = 64 + P
    ARG = list(range(64, 64 + P))
    mi = nn.ModuleDict(dict(W_in=nn.Linear(D, P), W_out=nn.Linear(P, D)))
    phi = [x * x % P for x in range(P)]                               # t = phi(ARG) in 0..12, g = identity
    tmin, tmax = 0, P - 1
    js = list(range(tmin - 1, tmax + 2))
    width = len(js) + P                                               # ladder + clear units
    blk = nn.ModuleDict(dict(W_in=nn.Linear(D, width), W_out=nn.Linear(width, D)))
    W_U = nn.Linear(D, P, bias=False)
    with torch.no_grad():
        for p in list(mi.parameters()) + list(blk.parameters()) + list(W_U.parameters()):
            p.zero_()
        for i in range(P):
            mi["W_in"].weight[i, C.DIM["Q"][i]] = 1; mi["W_out"].weight[ARG[i], i] = 1
        for u, j in enumerate(js):
            for i in range(P):
                blk["W_in"].weight[u, ARG[i]] = phi[i]
            blk["W_in"].bias[u] = -j
        for k in range(tmin, tmax + 1):
            c = ARG[k % P]
            blk["W_out"].weight[c, k - tmin] += 1; blk["W_out"].weight[c, k - tmin + 1] -= 2; blk["W_out"].weight[c, k - tmin + 2] += 1
        for i in range(P):                                            # clear the old value
            u = len(js) + i
            blk["W_in"].weight[u, ARG[i]] = 1; blk["W_out"].weight[ARG[i], u] = -1
        for c in range(P):
            W_U.weight[c, ARG[c]] = 1
    return LoopNet(enc, D, n_iter, blk, mi, W_U)


@torch.no_grad()
def verify_loop():
    L = C.domain()
    b = T.build(L)
    M, _, _ = C.canon_scales(L, dict(MU=T.MASS_S, AU=T.ACC_S))
    out = {}
    for n in range(1, 7):
        net = build_loop(n)
        lg = net(b.tokens)
        want = torch.tensor([pow(int(x), 2 ** n, P) for x in M])
        unique = sum(p.numel() for p in list(net.block.parameters()) + list(net.move_in.parameters()) + list(net.W_U.parameters()))
        unrolled = unique + (n - 1) * sum(p.numel() for p in net.block.parameters())
        out[n] = dict(exact=bool((lg == torch.nn.functional.one_hot(want, P).float()).all()),
                      unique_decoder_params=unique, unrolled_equivalent_params=unrolled)
    return out


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"cse_regalloc": [], "select": [], "loop": None}
    # same 60 random trees as doc 58, plus constructed shared-subexpression cases
    rng = random.Random(57)
    trees = []
    for d in range(1, 7):
        for _ in range(10):
            trees.append(from_tree(G.random_tree(d, rng)))
    MA = lambda: N("mul", N("M"), N("A"))
    trees += [N("add", MA(), N("mul", MA(), N("S"))), N("sub", N("sq", MA()), N("add", MA(), N("M"))),
              N("div", N("add", MA(), N("S")), N("add", MA(), N("S")))]
    for t in trees:
        rows = {}
        for cse in (False, True):
            for ra in (False, True):
                rows[f"cse={cse},regalloc={ra}"] = verify(t, cse, ra)
        res["cse_regalloc"].append(rows)
        r0, r1 = rows["cse=False,regalloc=False"], rows["cse=True,regalloc=True"]
        print(r0["expr"][:60], "| exact", all(v["exact"] for v in rows.values()), "obs", all(v["intermediates_observable_in_live_interval"] for v in rows.values()),
              "| nodes", r0["n_nodes"], "->", r1["n_nodes"], "| D", r0["D"], "->", r1["D"], "| nz", r0["nonzero_params"], "->", r1["nonzero_params"],
              "| flops", r0["decoder_flops_per_prompt"], "->", r1["decoder_flops_per_prompt"], flush=True)
    sel = [N("select", N("M"), N("A"), N("add", N("M"), N("A"))),
           N("select", N("A"), N("mul", N("M"), N("S")), N("div", N("M"), N("A"))),
           N("add", N("select", N("sub", N("M"), N("A")), N("S"), N("mul", N("M"), N("A"))), N("M"))]
    for t in sel:
        r = verify(t, True, True)
        res["select"].append(r)
        print("select", r, flush=True)
    res["loop"] = verify_loop()
    print("loop", res["loop"], flush=True)
    (OUT / "graph2.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
