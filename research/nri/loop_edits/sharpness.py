"""Agent's bespoke stability diagnostic: top Hessian eigenvalue (power iteration on Hessian-vector
products) of the TRAINING objective (task + lambda * IIT) on a fixed FIT batch with fixed IIT sources,
and its Adam-preconditioned version (H scaled by 1/sqrt(v) from the optimizer state), which is what
sets the edge of stability for Adam."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_correction as C  # noqa: E402
import run_dimensional as D  # noqa: E402
import run_swap as S  # noqa: E402
from nri import atlas as AT  # noqa: E402
from nri import dimtask as T  # noqa: E402


def objective(m, step):
    d = AT.partition()
    fit, src = d["FIT"], d["FIT_src"]
    tb, ts = T.build(fit).tokens, T.build(src).tokens
    _, hb = D.encode(m, tb)
    _, hs = D.encode(m, ts)
    mk = D.mask(*S.BLOCK)
    task = F.cross_entropy(m.head_from_resid(hb), T.build(fit).y)
    iit = F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk), D.counterfactual(fit, src, "M"))
    return task + S.lam_at(step, "warmup") * iit


def sharpness(st, iters=30, precond=False):
    m, opt = st["model"], st["opt"]
    params = [p for p in m.parameters()]
    if precond:
        pre = []
        for p in params:
            s_ = opt.state.get(p, {})
            v = s_.get("exp_avg_sq")
            pre.append((1.0 / (v.sqrt() + 1e-8)) if v is not None else torch.zeros_like(p))
    g = torch.Generator().manual_seed(0)
    vec = [torch.randn(p.shape, generator=g) for p in params]
    lam = 0.0
    for _ in range(iters):
        if precond:
            vec = [v * q.sqrt() for v, q in zip(vec, pre)]
        nrm = torch.sqrt(sum((v ** 2).sum() for v in vec))
        vec = [v / nrm for v in vec]
        loss = objective(m, st["step"])
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grads = [gg if gg is not None else torch.zeros_like(p) for gg, p in zip(grads, params)]
        hv = torch.autograd.grad(sum((gg * v).sum() for gg, v in zip(grads, vec)), params, allow_unused=True)
        hv = [h if h is not None else torch.zeros_like(p) for h, p in zip(hv, params)]
        if precond:
            hv = [h * q.sqrt() for h, q in zip(hv, pre)]
        lam = sum((h * v).sum() for h, v in zip(hv, vec)).item()
        vec = [h.detach() for h in hv]
    return lam


if __name__ == "__main__":
    for path in sys.argv[1:]:
        st = C.load_state(path)
        print(path.replace("runs/", ""), "step", st["step"], "lr", round(st["sched"].get_last_lr()[0], 5),
              "sharpness", round(sharpness(st), 2), "Adam-preconditioned", round(sharpness(st, precond=True), 1),
              "(stability edge ~ 2/lr =", round(2 / st["sched"].get_last_lr()[0]), ")", flush=True)
