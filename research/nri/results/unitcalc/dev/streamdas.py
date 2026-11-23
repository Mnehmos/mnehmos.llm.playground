import sys
sys.path.insert(0, "/home/user/mnehmos.llm.playground/research/nri")
import torch, torch.nn.functional as F
import analyse_unitcalc as A, run_unitcalc as R
from nri import unitcalc as U
m, cfg = R.load("/home/user/mnehmos.llm.playground/research/nri/runs/zoo_unitcalc_pilot/sweep/L4_d64_30k_all/natural_s951")
fit, ev = A.draw(A.N, A.FIT_S, "fit"), A.draw(A.N, A.EVAL_S, "eval")

def stream(m, tok, L):
    x0 = m.embed(tok, torch.zeros_like(tok))
    return x0 if L == 0 else x0 + m.attn1(x0)

def run_from(m, x, L):
    if L == 0:
        x = x + m.attn1(x)
    x = x + m.attn2(x)
    return m.head_from_resid(x[:, -1])

def das_stream(mode, slot, var, L, k=16, steps=400):
    mi = U.MODES.index(mode)
    bf, be = A.subset(fit, fit["mode"] == mi), A.subset(ev, ev["mode"] == mi)
    sf, se = A.pairs(bf, 41), A.pairs(be, 42)
    with torch.no_grad():
        xbf, xsf = stream(m, U.build(bf).tokens, L).flatten(1), stream(m, U.build(sf).tokens, L).flatten(1)
        xbe, xse = stream(m, U.build(be).tokens, L).flatten(1), stream(m, U.build(se).tokens, L).flatten(1)
    yf, ye = A.counterfactual(bf, sf, slot, var), A.counterfactual(be, se, slot, var)
    y0 = U.build(be).y
    torch.manual_seed(0)
    rot = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(512, k, bias=False))
    opt = torch.optim.Adam(rot.parameters(), lr=1e-2)
    for p in m.parameters(): p.requires_grad_(False)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        i = torch.randint(0, len(yf), (1024,), generator=g)
        Q = rot.weight.T
        x = xbf[i] + (xsf[i] - xbf[i]) @ Q @ Q.T
        loss = F.cross_entropy(run_from(m, x.view(-1, 8, 64), L), yf[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Q = rot.weight.T
        p = run_from(m, (xbe + (xse - xbe) @ Q @ Q.T).view(-1, 8, 64), L).argmax(-1)
    el = ye != y0
    return round((p == ye)[el].float().mean().item(), 3)

for L in (1,):
    for mode in ("mul", "msum", "tsum"):
        print(mode, "L%d stream" % L, {v: das_stream(mode, "u1", v, L) for v in ("canon", "value", "label")}, flush=True)
