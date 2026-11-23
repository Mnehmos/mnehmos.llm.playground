import sys
sys.path.insert(0, "/home/user/mnehmos.llm.playground/research/nri")
import torch, torch.nn.functional as F
import analyse_unitcalc as A, run_unitcalc as R
from nri import unitcalc as U
m, cfg = R.load("/home/user/mnehmos.llm.playground/research/nri/runs/zoo_unitcalc_pilot/sweep/L4_d64_30k_all/natural_s951")
fit, ev = A.draw(A.N, A.FIT_S, "fit"), A.draw(A.N, A.EVAL_S, "eval")
mi = 0
bf, be = A.subset(fit, fit["mode"] == mi), A.subset(ev, ev["mode"] == mi)
sf, se = A.pairs(bf, 41), A.pairs(be, 42)
orig = A.counterfactual
# sanity target: the SOURCE's own answer (swap everything) and k=64 for canon
A.counterfactual = lambda b, s, slot, var: U.build(s).y if var == "srcans" else orig(b, s, slot, var)
for k, var in ((16, "srcans"), (32, "srcans"), (64, "canon"), (64, "srcans")):
    A.K_DAS = k
    r = A.das_iia(m, bf, sf, be, se, "u1", "L2:query", var, seed=5)
    print("h", k, var, round(r["iia"], 3), "random", round(r["random_iia"], 3), flush=True)
