"""Doc 60 E diagnosis (post hoc): coverage of the OP2 training set for every E run, recomputed from the
deterministic selector (no retraining). Reports canonical (M, A, S) triple and (M, A) pair coverage, the number of
OP2 training prompts, and the structured sampler's cell-key collision (op*10000 vs M*1000 with M <= 12).
    python -m sci.p8_ecov   # -> runs/phase8/e_coverage.json
"""
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sci import construct as C, hybrid as H, p8  # noqa: E402
T = C.T

def main():
    L = H.domain2(); un = H.unseen2(L); pool0 = torch.nonzero(~un).flatten()
    M, A, S = C.canon_scales(L, dict(MU=T.MASS_S, AU=T.ACC_S))
    op2 = L["op"] == 1
    allt = {(int(M[i]), int(A[i]), int(L["fu"][i])) for i in torch.nonzero(op2 & ~un).flatten().tolist()}
    allp = {(m, a) for m, a, _ in allt}
    key = (L["op"] * 10000 + M * 1000 + A * 10 + L["fu"])
    true_cells = len({(int(L["op"][i]), int(M[i]), int(A[i]), int(L["fu"][i])) for i in pool0.tolist()})
    coll_cells = len(set(key[pool0].tolist()))
    out = dict(cells_true=true_cells, cells_with_collided_key=coll_cells, runs={})
    for frac in (0.25, 0.4, 0.55, 0.7, 0.85):
        for smp in (("random", "structured") if frac in (0.25, 0.4) else ("random",)):
            for seed in (0, 1):
                pool = p8.select_pool(L, pool0, frac, smp, seed)
                p2 = pool[op2[pool]]
                tr = {(int(M[i]), int(A[i]), int(L["fu"][i])) for i in p2.tolist()}
                out["runs"][f"{frac}_{smp}_s{seed}"] = dict(n_op2=len(p2), triple_cov=len(tr) / len(allt),
                                                          pair_cov=len({(m, a) for m, a, _ in tr}) / len(allp))
                print(frac, smp, seed, out["runs"][f"{frac}_{smp}_s{seed}"], flush=True)
    print("cells true", true_cells, "with collided key", coll_cells)
    (ROOT / "runs/phase8/e_coverage.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
