"""Interleaved in-process ranking of encoder candidates: one full IIT training-step compute (fwd+bwd, no optimizer)."""
import sys, time, json, torch, torch.nn.functional as F
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import run_swap as S, cand
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
D, T = S.D, S.T
g = torch.Generator().manual_seed(0)
lb, ls = T.sample_latents(512, g), T.sample_latents(512, g); bb, sb = T.build(lb), T.build(ls)
cf = D.counterfactual(lb, ls, "M"); mk = D.mask(0, 16)
torch.manual_seed(307); m = S.TinyTyped("mass_only", **S.MODEL_KW)
V = sys.argv[3].split(",")
def step(v, iit):
    m.zero_grad(set_to_none=True)
    hb, hs = cand.pair(m, D, bb.tokens, sb.tokens if iit else None, v)
    loss = F.cross_entropy(m.head_from_resid(hb), bb.y)
    if iit: loss = loss + 3 * F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk), cf)
    loss.backward()
res = {}
for iit in (True, False):
    t = {v: [] for v in V}
    for v in V:
        for _ in range(3): step(v, iit)
    for r in range(7):
        for v in (V if r % 2 == 0 else V[::-1]):
            t0 = time.perf_counter()
            for _ in range(5): step(v, iit)
            t[v].append((time.perf_counter() - t0) / 5 * 1e3)
    med = {v: sorted(x)[3] for v, x in t.items()}
    res["mass_only" if iit else "neither"] = med
    print("mass_only" if iit else "neither", {v: round(x, 1) for v, x in med.items()}, {v: round(med["baseline"] / x, 2) for v, x in med.items()}, flush=True)
