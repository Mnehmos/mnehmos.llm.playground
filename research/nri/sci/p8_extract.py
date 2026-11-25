"""Doc 59 MR (interpret + formalise): extract a candidate specification from a frozen learned module WITHOUT access to
the generating rule. This file must not import sci.p8_rules or sci.p8_q (checked at runtime).

Interpret : query the module through the network on all 13 x 13 x 3 canonical (M, A, S) states (scale-1 renderings).
Formalise : search the compiler's hypothesis space
              lin : (a M + b A + c S + d) mod 13,  a,b,c,d in 0..12
              mul : 2^(alpha dlog M + beta dlog A + gamma dlog S) with zero gate,  alpha,beta,gamma in 0..11
            accept iff best agreement >= 0.99 AND unique best.
Negative control: the OPR module (random lookup table) must yield "no candidate".

    python -m sci.p8_extract     # -> runs/phase8/extraction.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import construct as C  # noqa: E402
from sci import modular as Q  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase8"
MODS = OUT / "q_modules"
BASE_NODES = [("OUT1", "mas", None), ("OUTadd", "add", None), ("X", "mul", None), ("Y", "add", None)]


def query(net, mod, op):
    """I/O table of a learned module on canonical states, through the network's own forward pass."""
    rows = list(itertools.product(range(P), range(P), range(3)))
    L = dict(m=torch.tensor([r[0] for r in rows]), mu=torch.zeros(len(rows), dtype=torch.long),
             a=torch.tensor([r[1] for r in rows]), au=torch.zeros(len(rows), dtype=torch.long),
             fu=torch.tensor([r[2] for r in rows]), order=torch.zeros(len(rows), dtype=torch.long))
    b = T.build(L)
    tok = b.tokens.clone(); tok[:, 5] = Q.OPTOK[op]
    h = net.features(tok)
    read = Q.read_dims(net, ("X", "Y"))
    with torch.no_grad():
        out = mod(h[:, read]).argmax(-1)
    M = L["m"]; A = L["a"]; S = torch.tensor(C.SINV)[L["fu"]]
    return M, A, S, out


def search(M, A, S, out, mask=None):
    mask = torch.ones(len(out), dtype=torch.bool) if mask is None else mask
    best = []
    co = torch.tensor(list(itertools.product(range(P), repeat=4)))                      # lin: a, b, c, d
    pred = (co[:, 0:1] * M + co[:, 1:2] * A + co[:, 2:3] * S + co[:, 3:4]) % P          # 28561 x 507
    agree = (pred[:, mask] == out[mask]).float().mean(1)
    for i in torch.argsort(-agree)[:3].tolist():
        best.append((float(agree[i]), "lin", tuple(co[i].tolist())))
    dl = lambda x: torch.tensor([C.DLOG.get(int(v), 0) for v in x])
    lM, lA, lS = dl(M), dl(A), dl(S)
    cm = torch.tensor(list(itertools.product(range(12), repeat=3)))
    e = (cm[:, 0:1] * lM + cm[:, 1:2] * lA + cm[:, 2:3] * lS) % 12
    pw = torch.tensor([pow(2, k, P) for k in range(12)])
    predm = torch.where(((M == 0) | (A == 0))[None, :], torch.zeros_like(e), pw[e])
    agm = (predm[:, mask] == out[mask]).float().mean(1)
    for i in torch.argsort(-agm)[:3].tolist():
        best.append((float(agm[i]), "mul", tuple(cm[i].tolist())))
    best.sort(key=lambda x: -x[0])
    top, second = best[0], best[1]
    accepted = top[0] >= 0.99 and top[0] > second[0]
    return dict(best=top, second=second, accepted=accepted, top5=best[:5])


def main():
    assert "sci.p8_rules" not in sys.modules and "sci.p8_q" not in sys.modules
    torch.set_num_threads(1)
    net = Q.ModularNet(BASE_NODES)
    heldout = Q.heldout_pairs()
    res = {}
    for seed in (0, 1, 2):
        for op in ("OP4", "OPR"):
            mod = torch.nn.Sequential(torch.nn.Linear(len(Q.read_dims(net, ("X", "Y"))), 64), torch.nn.ReLU(), torch.nn.Linear(64, P))
            mod.load_state_dict(torch.load(MODS / f"reg_s{seed}_{op}.pt"))
            M, A, S, out = query(net, mod, op)
            trained = ~heldout[M * P + A]                     # split membership (not the answer)
            r = dict(all_states=search(M, A, S, out), trained_pairs_only=search(M, A, S, out, trained))
            res[f"{op}_s{seed}"] = r
            print(op, seed, "all:", r["all_states"]["best"], "accepted", r["all_states"]["accepted"],
                  "| trained-pairs:", r["trained_pairs_only"]["best"], "accepted", r["trained_pairs_only"]["accepted"], flush=True)
    assert "sci.p8_rules" not in sys.modules
    (OUT / "extraction.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
