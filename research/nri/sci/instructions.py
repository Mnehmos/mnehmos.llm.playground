"""Doc 53 §53.4: a neural instruction system on the constructed Design-C network.

Each instruction rewrites ONE parameter block of the network in place (no hooks); semantics are checked exhaustively
over all 9,126 prompts against Python modular arithmetic.

  component          instruction                       parameters rewritten
  operation          op(name)                          W_in, b_in, W_out (ridge statistic + readout table)
  unit conversion    unit(role, i, s)                  OV block of that unit's correction head
  operand binding    swap_roles()                      layer-2 gather keys (+ the scale-7 head's destination)
  output decoding    out(g)                            W_U rows (permutation)
  operand identity   token programs (Phase IV)         embedding rows

    python -m sci.instructions    # -> runs/phase6/instructions.json
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import compile_edit as CE  # noqa: E402
from sci import construct as C  # noqa: E402

T, P, D = C.T, C.P, C.DIM
OUT = ROOT / "runs/phase6"
OUTOPS = {"id": lambda f: f, "F+1": lambda f: (f + 1) % P, "2F": lambda f: 2 * f % P,
          "F^-1": lambda f: pow(f, -1, P) if f else 0, "-F": lambda f: (-f) % P}
# unit -> (attention layer, head) holding its correction (Design C layout; MU0 uses the spare layer-2 head 3)
UNIT_HEAD = {("MU", 1): (1, 1), ("MU", 2): (1, 2), ("AU", 1): (1, 3), ("AU", 2): (2, 2), ("MU", 0): (2, 3)}


# ------------------------------------------------------------------ semantics (Python arithmetic)
class Spec:
    def __init__(self):
        self.op = "MAS"
        self.scales = dict(MU=list(T.MASS_S), AU=list(T.ACC_S))
        self.swap = False
        self.out = lambda f: f
        self.prog = CE.Program()

    def target(self, L):
        val, mu, au, fu = self.prog.val, self.prog.mu, self.prog.au, self.prog.fu
        m = torch.tensor([val[int(x)] for x in L["m"]])
        a = torch.tensor([val[int(x)] for x in L["a"]])
        mu_ = torch.tensor([mu[int(x)] for x in L["mu"]])
        au_ = torch.tensor([au[int(x)] for x in L["au"]])
        fu_ = torch.tensor([fu[int(x)] for x in L["fu"]])
        M = m * torch.tensor(self.scales["MU"])[mu_] % P
        A = a * torch.tensor(self.scales["AU"])[au_] % P
        S = torch.tensor(C.SINV)[fu_]
        if self.swap:
            M, A = A, M
        y = C.target(C.OPS[self.op], M, A, S)
        return torch.tensor([self.out(int(f)) for f in y])


# ------------------------------------------------------------------ instructions (weight rewrites)
def op(m, name):
    e = copy.deepcopy(m)
    C.compile_mlp(e, C.OPS[name], dict(M=D["Q"], A=D["A"], S=D["FU"]), D["OUT"])
    return e


def unit(m, role, i, s):
    """Reassign the scale of unit (role, i) to s by rewriting its correction head's OV block."""
    e = copy.deepcopy(m)
    layer, h = UNIT_HEAD[(role, i)]
    Cm = C.perm(s) - torch.eye(P)
    if layer == 1:
        O = e.attn1.o.weight.data
        O[:, 16 * h:16 * h + 13] = 0
        for a in range(P):
            for b in range(P):
                O[D["Q"][a], 16 * h + b] = Cm[a, b]
        return e
    O = e.attn2.o.weight.data
    W, bq = e.attn2.qkv.weight.data, e.attn2.qkv.bias.data
    # destination: the region this role is currently gathered into (follows swap_roles)
    dest = "Q" if (role == "MU") != _is_swapped(e) else "A"
    O[:, 16 * h:16 * h + 13] = 0
    for a in range(P):
        for b in range(P):
            O[D[dest][a], 16 * h + b] = Cm[a, b]
    if h == 3:   # spare head: build its routing once (query, keys, values)
        flag = D["MU"][i] if role == "MU" else D["AU"][i]
        bq[48], bq[49] = C.G2, C.G2 / 2
        W[64 + 48, :] = 0
        W[64 + 48, flag] = 1
        W[64 + 49, D["NULL"]] = 1
        for a in range(P):
            W[128 + 48 + a, D["Q"][a]] = 1
    return e


def _is_swapped(m):
    """True if the mass-unit keys feed the A region (roles swapped)."""
    W = m.attn2.qkv.weight.data
    return bool(W[64 + 16, D["MU"][0]] == 1)


def swap_roles(m):
    """Operand binding: gather mass into the A region and acceleration into the M region (keys swapped), and move the
    layer-2 correction heads' destinations with their roles."""
    e = copy.deepcopy(m)
    W, O = e.attn2.qkv.weight.data, e.attn2.o.weight.data
    swapped = _is_swapped(e)
    mass_head, acc_head = (1, 0) if swapped else (0, 1)
    # head 0 always also gathers the force unit; move only the unit keys
    for d_ in D["MU"]:
        W[64 + 16 * mass_head, d_], W[64 + 16 * acc_head, d_] = 0, 1
    for d_ in D["AU"]:
        W[64 + 16 * acc_head, d_], W[64 + 16 * mass_head, d_] = 0, 1
    for h in (2, 3):
        blk = O[:, 16 * h:16 * h + 13].clone()
        O[D["Q"], 16 * h:16 * h + 13], O[D["A"], 16 * h:16 * h + 13] = blk[D["A"]], blk[D["Q"]]
    return e


def out(m, g):
    e = copy.deepcopy(m)
    W = m.W_U.weight.data
    for f in range(P):
        e.W_U.weight.data[g(f)] = W[f]
    return e


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def check(m, spec, L=None):
    L = L or C.domain()
    un = C.unseen_mask(L)
    b = T.build(L)
    logits = m(b.tokens, torch.zeros_like(b.tokens))
    pred = logits.argmax(-1)
    y = spec.target(L)
    oh = torch.nn.functional.one_hot(y, P).to(logits.dtype)
    return dict(acc=float((pred == y).float().mean()), acc_familiar=float((pred[~un] == y[~un]).float().mean()),
                acc_unseen=float((pred[un] == y[un]).float().mean()), logits_exact_onehot=bool((logits == oh).all()))


def changed(m0, m1):
    return {k: int((a != b).sum()) for (k, a), (_, b) in zip(m0.state_dict().items(), m1.state_dict().items())
            if not torch.equal(a, b)}


def same(m0, m1):
    return all(torch.equal(a, b) for a, b in zip(m0.state_dict().values(), m1.state_dict().values()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = C.domain()
    base, _ = C.build_C(C.OPS["MAS"])
    res = {"single": {}, "independence": {}, "composites": [], "reversibility": {}}
    # --- operations
    for name in C.OPS:
        e = op(base, name)
        s = Spec(); s.op = name
        res["single"][f"op:{name}"] = dict(**check(e, s), changed=changed(base, e))
        res["reversibility"][f"op:{name}"] = same(base, op(e, "MAS"))
    # --- unit conversions: every unit with a correction head x every scale
    ucount = uok = 0
    for (role, i) in UNIT_HEAD:
        for s_new in range(1, P):
            e = unit(base, role, i, s_new)
            s = Spec(); s.scales[role][i] = s_new
            r = check(e, s)
            ucount += 1; uok += int(r["acc"] == 1.0 and r["logits_exact_onehot"])
            if s_new in (2, 12):
                res["single"][f"unit:{role}{i}->{s_new}"] = dict(**r, changed=changed(base, e))
        orig = (T.MASS_S if role == "MU" else T.ACC_S)[i]
        res["reversibility"][f"unit:{role}{i}"] = same(unit(unit(base, role, i, 7), role, i, orig), unit(base, role, i, orig))
    res["unit_edits_exact"] = dict(n=ucount, exact=uok)
    # --- binding swap (use a non-symmetric operation so the swap is observable)
    for name in ("M2AS", "M-A", "M/A"):
        e = swap_roles(op(base, name))
        s = Spec(); s.op = name; s.swap = True
        res["single"][f"swap:{name}"] = dict(**check(e, s), changed=changed(op(base, name), e))
    res["reversibility"]["swap"] = same(base, swap_roles(swap_roles(base)))
    # --- output decoding
    for k, g in OUTOPS.items():
        e = out(base, g)
        s = Spec(); s.out = g
        res["single"][f"out:{k}"] = dict(**check(e, s), changed=changed(base, e))
        if k != "id":
            ginv = {g(f): f for f in range(P)}
            res["reversibility"][f"out:{k}"] = same(base, out(e, lambda f: ginv[f]))
    # --- independence: each instruction's changed tensors are disjoint from the others'
    blocks = {k: set(v["changed"]) for k, v in res["single"].items()}
    fam = {}
    for k, v in blocks.items():
        fam.setdefault(k.split(":")[0], set()).update(v)
    res["independence"]["blocks_by_component"] = {k: sorted(v) for k, v in fam.items()}
    # --- random composites (all five components, plus Phase IV token programs)
    rng = random.Random(0)
    tok_keys = [k for k in CE.INSTR if k != "identity"]
    for _ in range(50):
        s = Spec(); e = base
        s.op = rng.choice(list(C.OPS)); e = op(e, s.op)
        for (role, i) in rng.sample(list(UNIT_HEAD), rng.choice((0, 1, 2))):
            sn = rng.randrange(1, P); s.scales[role][i] = sn; e = unit(e, role, i, sn)
        if rng.random() < 0.5:
            s.swap = True; e = swap_roles(e)
        ok = rng.choice(list(OUTOPS)); s.out = OUTOPS[ok]; e = out(e, s.out)
        seq = [rng.choice(tok_keys) for _ in range(rng.choice((0, 1, 2)))]
        for k in seq:
            e = CE.INSTR[k].compile(e); s.prog = s.prog.then(CE.INSTR[k])
        r = check(e, s, L)
        res["composites"].append(dict(op=s.op, scales=copy.deepcopy(s.scales), swap=s.swap, out=ok, tokens=seq, **r))
    comp = res["composites"]
    res["composite_summary"] = dict(n=len(comp), all_exact=all(c["acc"] == 1.0 and c["logits_exact_onehot"] for c in comp),
                                    mean_acc=sum(c["acc"] for c in comp) / len(comp))
    # --- Design L contrast: additive operations are not compilable (doc 53 §53.2)
    try:
        C.build_L(C.OPS["M+A"]); res["design_L_M+A"] = "compiled (unexpected)"
    except AssertionError as ex:
        res["design_L_M+A"] = f"refused: {ex}"
    (OUT / "instructions.json").write_text(json.dumps(res, indent=1))
    for k, v in res["single"].items():
        print(k, {a: b for a, b in v.items() if a != "changed"}, sorted(v["changed"]))
    print("unit edits exact:", res["unit_edits_exact"], "| composites:", res["composite_summary"])
    print("reversibility:", res["reversibility"])
    print("blocks:", res["independence"]["blocks_by_component"])
    print("Design L M+A:", res["design_L_M+A"])


if __name__ == "__main__":
    main()
