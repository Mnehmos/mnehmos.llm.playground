"""Doc 51 D7: exact edits of the decoder's own parameters (no token remapping, no hooks).

Output operator: permute the rows of W_U so that the network outputs g(F) where F is what it computed:
  W_U'[g(F)] = W_U[F] for a bijection g on Z_13 (e.g. F+1, 2F, F^-1 with 0->0). Exact by construction (a relabelling of
  the output layer). This edits the DECODER's parameters, not the input, but it acts after the core computation
  (post-composition), so it does not change the algorithm itself. Composed with token-level programs (Phase IV).
Reported: exactness over the full domain (familiar / unseen renderings), reversibility, composition benchmark.
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

from nri import dimtask as T  # noqa: E402
from sci import algorithm as A  # noqa: E402
from sci import compile_edit as CE  # noqa: E402
from sci import rescue as R  # noqa: E402

P = T.P
OPS = {"F+1": lambda f: (f + 1) % P, "2F": lambda f: 2 * f % P, "F^-1": lambda f: pow(f, -1, P) if f else 0,
       "-F": lambda f: (-f) % P, "identity": lambda f: f}


def output_op(m, g):
    e = copy.deepcopy(m)
    W = m.W_U.weight.data
    for f in range(P):
        e.W_U.weight.data[g(f)] = W[f]
    return e


@torch.no_grad()
def eval_full(m, target_fn):
    L = A.domain()
    b = T.build(L)
    pred = m(b.tokens, torch.zeros_like(b.tokens)).argmax(-1)
    want = target_fn(L)
    unseen = ((L["mu"] == 2) & (L["m"] >= 9)) | ((L["au"] == 2) & (L["a"] >= 9))
    return dict(familiar=float((pred[~unseen] == want[~unseen]).float().mean()), unseen=float((pred[unseen] == want[unseen]).float().mean()))


def suite(name, n_seq=30, seed=0):
    m = R.load(name)
    base_target = lambda L: T.build(L).y
    out = {"single": {}, "composed": [], "reversible": {}}
    for k, g in OPS.items():
        e = output_op(m, g)
        out["single"][k] = eval_full(e, lambda L, g=g: torch.tensor([g(int(y)) for y in T.build(L).y]))
        if k != "identity":
            ginv = {g(f): f for f in range(P)}
            e2 = output_op(e, lambda f: ginv[f])
            out["reversible"][k] = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), e2.state_dict().values()))
    rng = random.Random(seed)
    tok_keys = [k for k in CE.INSTR if k not in ("identity", "const_value_5")]
    for _ in range(n_seq):
        seq = [rng.choice(tok_keys) for _ in range(rng.choice((1, 2)))]
        progs = [CE.INSTR[k] for k in seq]
        op = rng.choice([k for k in OPS if k != "identity"])
        e = output_op(CE.compiled_seq(m, progs), OPS[op])
        comp = CE.Program()
        for p_ in progs:
            comp = comp.then(p_)
        r = eval_full(e, lambda L, comp=comp, op=op: torch.tensor([OPS[op](int(y)) for y in comp.target(L)[0]]))
        out["composed"].append(dict(tokens=seq, output_op=op, **r))
    return out


def main():
    torch.set_num_threads(4)
    res = {}
    import statistics as st
    for s in (307, 311, 300, 304):
        n = f"mass_only_s{s}"
        res[n] = suite(n)
        c = res[n]["composed"]
        print(n, "single:", {k: (round(v["familiar"], 3), round(v["unseen"], 3)) for k, v in res[n]["single"].items()},
              "| composed familiar/unseen:", round(st.mean(x["familiar"] for x in c), 3), round(st.mean(x["unseen"] for x in c), 3),
              "| reversible:", all(res[n]["reversible"].values()), flush=True)
    (R.OUT.parent / "phase5" / "operator_edit.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
