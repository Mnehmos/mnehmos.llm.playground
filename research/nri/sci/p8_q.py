"""Doc 59 Q (progressive acquisition), E-modular (learning curves for the modular architecture) and the training side of
MR (OP4 / OPR modules, OP5 with and without the recompiled register). Labels come from the task oracle (p8_rules).

    python -m sci.p8_q q          # -> runs/phase8/q.json, modules saved under runs/phase8/q_modules/
    python -m sci.p8_q emod       # -> runs/phase8/e_modular.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import construct as C  # noqa: E402
from sci import modular as Q  # noqa: E402
from sci import p8_rules as RU  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase8"
MODS = OUT / "q_modules"
BASE_NODES = [("OUT1", "mas", None), ("OUTadd", "add", None), ("X", "mul", None), ("Y", "add", None)]


def targets(op, M, A, S):
    if op == "OP1":
        return M * A * S % P
    if op == "OPadd":
        return (M + A) % P
    if op == "OP2":
        return (M * A + S) % P
    if op == "OP3":
        return (M + A) * S % P
    if op == "OP4":
        return RU.op4(M, A, S)
    if op == "OP5":
        return RU.op4(M, A, S) * M % P
    if op == "OPR":
        return RU.opr(M, A, S)
    raise ValueError(op)


def setup(nodes=BASE_NODES):
    net = Q.ModularNet(nodes)
    net.owner = {"OP1": ("compiled", "OUT1"), "OPadd": ("compiled", "OUTadd")}
    L, tok, M, A, S = Q.domain_ops(Q.OPS)
    h = net.features(tok)
    y = torch.zeros(len(M), dtype=torch.long)
    for i, op in enumerate(Q.OPS):
        s = L["op"] == i
        y[s] = targets(op, M[s], A[s], S[s])
    return net, L, tok, M, A, S, h, y


def masks(L, M, A, op, heldout):
    i = Q.OPS.index(op)
    sel = L["op"] == i
    un = C.unseen_mask(L)
    ho = heldout[M * P + A]
    return dict(train=torch.nonzero(sel & ~un & ~ho).flatten(), familiar=torch.nonzero(sel & ~un & ~ho).flatten(),
                unseen=torch.nonzero(sel & un & ~ho).flatten(), heldout_pairs=torch.nonzero(sel & ho).flatten())


def competence(hist, keys=("familiar", "unseen")):
    for e in hist:
        if e["familiar"] >= 0.99 and e["unseen"] >= 0.95:
            return e["step"]
    return None


def run_q():
    torch.set_num_threads(1)
    MODS.mkdir(parents=True, exist_ok=True)
    heldout = Q.heldout_pairs()
    res = {}
    for cond in ("reg", "base"):
        for seed in (0, 1, 2):
            net, L, tok, M, A, S, h, y = setup()
            read = Q.read_dims(net, ("X", "Y") if cond == "reg" else ())
            stages = {}
            prev_logits = None
            for op in ("OP2", "OP3", "OP4"):
                before = net.logits(h).clone()
                mk = masks(L, M, A, op, heldout)
                mod, info = Q.train_module(h, y, read, mk["train"], {k: v for k, v in mk.items() if k != "train"}, seed)
                for p in mod.parameters():
                    p.requires_grad_(False)
                name = f"{op}"
                net.modules_[name] = mod; net.reads[name] = read; net.owner[op] = ("module", name)
                after = net.logits(h)
                # retention: every previously owned op's logits bitwise identical
                ret = {}
                for j, o in enumerate(Q.OPS):
                    if o != op and o in net.owner:
                        s = L["op"] == j
                        ret[o] = bool(torch.equal(before[s], after[s]))
                torch.save(mod.state_dict(), MODS / f"{cond}_s{seed}_{op}.pt")
                stages[op] = dict(final=info["final"], steps_to_competence=competence(info["history"]), params=info["params"],
                                  retention_bitwise=ret, history=info["history"], read_dims=len(read))
                print(cond, seed, op, {k: round(v, 3) for k, v in info["final"].items() if k != "step"}, "comp", stages[op]["steps_to_competence"],
                      "retained", all(ret.values()), flush=True)
            # final: every op over the full domain
            lg = net.logits(h)
            pr = lg.argmax(-1)
            stages["final_all_ops"] = {o: float((pr[L["op"] == j] == y[L["op"] == j]).float().mean()) for j, o in enumerate(Q.OPS) if o in net.owner}
            # OPR module for the extraction negative control (reg condition only)
            if cond == "reg":
                mk = masks(L, M, A, "OPR", heldout)
                mod, info = Q.train_module(h, y, read, mk["train"], {k: v for k, v in mk.items() if k != "train"}, seed)
                torch.save(mod.state_dict(), MODS / f"{cond}_s{seed}_OPR.pt")
                stages["OPR"] = dict(final=info["final"])
            res[f"{cond}_s{seed}"] = stages
            (OUT / "q.json").write_text(json.dumps(res, indent=1))


def run_emod():
    """learning curves for the modular architecture on OP2 (OP1 compiled), registers on/off, random vs structured sampling."""
    torch.set_num_threads(1)
    net, L, tok, M, A, S, h, y = setup()
    i2 = Q.OPS.index("OP2")
    un = C.unseen_mask(L)
    pool = torch.nonzero((L["op"] == i2) & ~un).flatten()
    eval_sets = dict(familiar=pool, unseen=torch.nonzero((L["op"] == i2) & un).flatten())
    res = []
    for cond in ("reg", "base"):
        read = Q.read_dims(net, ("X", "Y") if cond == "reg" else ())
        for frac in (0.25, 0.40, 0.55, 0.70, 0.85, 1.0):
            for sampling in (("random", "structured") if frac in (0.25, 0.40) else ("random",)):
                for seed in (0, 1):
                    idx = select(pool, M, A, L, frac, sampling, seed)
                    mod, info = Q.train_module(h, y, read, idx, eval_sets, seed)
                    cov_pairs = len(set((M[idx] * P + A[idx]).tolist())) / (P * P)
                    cov_trip = len(set((M[idx] * 100 + A[idx] * 10 + L["fu"][idx]).tolist())) / (P * P * 3)
                    r = dict(cond=cond, frac=frac, sampling=sampling, seed=seed, final=info["final"], steps_to_competence=competence(info["history"]),
                             coverage_MA_pairs=cov_pairs, coverage_MAS_triples=cov_trip, params=info["params"],
                             module_flops_per_step=info["module_flops_per_step"])
                    res.append(r)
                    print(cond, frac, sampling, seed, {k: round(v, 3) for k, v in info["final"].items() if k != "step"}, "comp", r["steps_to_competence"],
                          "cov", round(cov_pairs, 3), round(cov_trip, 3), flush=True)
                    (OUT / "e_modular.json").write_text(json.dumps(res, indent=1))


def select(pool, M, A, L, frac, sampling, seed):
    g = torch.Generator().manual_seed(7000 + seed + int(frac * 100))
    n = int(len(pool) * frac)
    if sampling == "random":
        return pool[torch.randperm(len(pool), generator=g)[:n]]
    key = (M[pool] * 1000 + A[pool] * 10 + L["fu"][pool]).tolist()
    cells = {}
    for i in torch.randperm(len(pool), generator=g).tolist():
        cells.setdefault(key[i], []).append(int(pool[i]))
    out, k, lists = [], 0, list(cells.values())
    while len(out) < n:
        for lst in lists:
            if k < len(lst) and len(out) < n:
                out.append(lst[k])
        k += 1
    return torch.tensor(out)


if __name__ == "__main__":
    {"q": run_q, "emod": run_emod}[sys.argv[1]]()
