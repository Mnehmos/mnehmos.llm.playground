import sys, time
sys.path.insert(0, "/home/user/mnehmos.llm.playground/research/nri")
import torch
import analyse_unitcalc as A, run_unitcalc as R
from nri import unitcalc as U
m, cfg = R.load("/home/user/mnehmos.llm.playground/research/nri/runs/zoo_unitcalc_pilot/sweep/L4_d64_30k_all/natural_s951")
fit, ev = A.draw(A.N, A.FIT_S, "fit"), A.draw(A.N, A.EVAL_S, "eval")
for mode, slot in (("mul", "u1"), ("msum", "u1")):
    mi = U.MODES.index(mode)
    bf, be = A.subset(fit, fit["mode"] == mi), A.subset(ev, ev["mode"] == mi)
    sf, se = A.pairs(bf, 41), A.pairs(be, 42)
    for site in ("L1:query", "L2:query"):
        for k in (16, 32, 48):
            A.K_DAS = k
            t = time.time()
            r = {v: A.das_iia(m, bf, sf, be, se, slot, site, v, seed=5) for v in ("canon", "value", "label")}
            print(mode, site, "k", k, {v: round(x["iia"], 2) for v, x in r.items()}, "rand", round(r["value"]["random_iia"], 2), f"{time.time()-t:.0f}s", flush=True)
