"""Doc 59 MR (compile + verify + protect + extend). Uses the candidate produced by sci.p8_extract. The registered acceptance
(agreement >= 0.99 on ALL states) FAILED; the candidate used here is the supplementary trained-pairs candidate, so every
result below is labelled post-hoc. The oracle (p8_rules) is imported ONLY here, for verification and OP5 labels.

    python -m sci.p8_mr_verify      # -> runs/phase8/mr.json
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
from sci import p8_q as PQ  # noqa: E402   (imports the oracle for labels)
from sci import p8_rules as RU  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase8"


def main():
    torch.set_num_threads(1)
    ext = json.loads((OUT / "extraction.json").read_text())
    cand = ext["OP4_s0"]["trained_pairs_only"]["best"]          # (agreement, kind, coef)
    assert all(ext[f"OP4_s{s}"]["trained_pairs_only"]["best"][2] == cand[2] for s in (0, 1, 2))
    coef = tuple(cand[2])
    res = dict(candidate=coef, candidate_kind=cand[1], registered_acceptance=[ext[f"OP4_s{s}"]["all_states"]["accepted"] for s in (0, 1, 2)],
               oracle_rule=RU.RULE4, candidate_equals_oracle=tuple(RU.RULE4) == coef)
    # compile the recovered spec as a protected node R4 and give OP4 ownership to it
    nodes = PQ.BASE_NODES + [("R4", "lin", coef)]
    net = Q.ModularNet(nodes)
    net.owner = {"OP1": ("compiled", "OUT1"), "OPadd": ("compiled", "OUTadd"), "OP4": ("compiled", "R4")}
    L, tok, M, A, S = Q.domain_ops(Q.OPS)
    h = net.features(tok)
    i4 = Q.OPS.index("OP4")
    s4 = L["op"] == i4
    y4 = RU.op4(M[s4], A[s4], S[s4])
    comp = net.logits(h)[s4]
    res["compiled_exact_vs_oracle"] = bool((comp == torch.nn.functional.one_hot(y4, P).float()).all())
    # behavioural agreement with the learned module (seed 0..2)
    heldout = Q.heldout_pairs()
    ho = heldout[M[s4] * P + A[s4]]
    read = Q.read_dims(net, ("X", "Y"))
    res["vs_learned"] = {}
    for seed in (0, 1, 2):
        mod = torch.nn.Sequential(torch.nn.Linear(len(read), 64), torch.nn.ReLU(), torch.nn.Linear(64, P))
        mod.load_state_dict(torch.load(OUT / "q_modules" / f"reg_s{seed}_OP4.pt"))
        with torch.no_grad():
            lp = mod(h[s4][:, read]).argmax(-1)
        cp = comp.argmax(-1)
        # causal: interchange the M register (Q dims) with every value on a sample of trained-pair prompts
        idx = torch.nonzero(~ho).flatten()[::50]
        hh = h[s4][idx]
        agree = tot = 0
        for v in range(P):
            h2 = hh.clone(); h2[:, C.DIM["Q"]] = 0; h2[:, C.DIM["Q"][v]] = 1
            h2r = h2.clone()
            pre = h2[:, :80]
            h2r = torch.cat([pre, torch.zeros(len(pre), net.D - 80)], 1)
            h2r = h2r + net.W_out(torch.relu(net.W_in(h2r)))           # recompute the compiled stage on the edited register
            h2r[:, Q.OPG[i4]] = 1
            with torch.no_grad():
                a1 = mod(h2r[:, read]).argmax(-1)
            a2 = h2r[:, net.regions["R4"]].argmax(-1)
            Mv = torch.full_like(M[s4][idx], v)
            ok_rows = ~heldout[Mv * P + A[s4][idx]]                     # compare where the learned module is in-distribution
            agree += int((a1[ok_rows] == a2[ok_rows]).sum()); tot += int(ok_rows.sum())
        res["vs_learned"][seed] = dict(agree_trained_pairs=float((lp[~ho] == cp[~ho]).float().mean()),
                                       agree_heldout_pairs=float((lp[ho] == cp[ho]).float().mean()),
                                       learned_vs_oracle_heldout=float((lp[ho] == y4[ho]).float().mean()),
                                       compiled_vs_oracle_heldout=float((cp[ho] == y4[ho]).float().mean()),
                                       interchange_M_agreement_in_distribution=agree / max(tot, 1))
    # protect & extend: OP5 = OP4 * M, with the compiled R4 register exposed vs not
    y = torch.zeros(len(M), dtype=torch.long)
    for j, o in enumerate(Q.OPS):
        s = L["op"] == j
        y[s] = PQ.targets(o, M[s], A[s], S[s])
    mk = PQ.masks(L, M, A, "OP5", heldout)
    res["OP5"] = {}
    for cond, regs in (("with_R4", ("X", "Y", "R4")), ("without_R4", ("X", "Y"))):
        rd = Q.read_dims(net, regs)
        for seed in (0, 1, 2):
            mod5, info = Q.train_module(h, y, rd, mk["train"], {k: v for k, v in mk.items() if k != "train"}, seed)
            res["OP5"][f"{cond}_s{seed}"] = dict(final=info["final"], steps_to_competence=PQ.competence(info["history"]))
            print("OP5", cond, seed, {k: round(v, 3) for k, v in info["final"].items() if k != "step"}, res["OP5"][f"{cond}_s{seed}"]["steps_to_competence"], flush=True)
    (OUT / "mr.json").write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k != "OP5"})


if __name__ == "__main__":
    main()
