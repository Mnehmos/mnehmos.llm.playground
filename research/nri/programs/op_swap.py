"""NRI weight program: make categorical variable value x behave like value y.

Generic over hosts: it reads only the NRI document's variable binding and the host's
activations, so it can be re-run on any compatible network (it re-derives the edit
from that host's own statistics instead of copying a numeric delta).

    h -> h + (mu_y - mu_x) * g(h),  g affine on the variable's class-mean subspace,
    g(mu_x) = 1, g(mu_j) = 0 for j != x;  folded into W_in as a rank-one update.
"""
import torch


def program(host, doc, var, x, y, site="resid_pre_mlp", n_fit=8192, seed=4242):
    from nri.nri_exec import _vars, _subspace
    g = torch.Generator().manual_seed(seed)
    tok, typ = host.sampler(n_fit, g)
    V = _vars(doc, tok)[var]
    h = host.site(site, tok, typ)
    vals = V.unique().tolist()
    mu = torch.stack([h[V == c].mean(0) for c in vals])
    B = _subspace(h, V)
    A = torch.cat([mu @ B, torch.ones(len(vals), 1)], 1)
    t = torch.tensor([1.0 if c == x else 0.0 for c in vals]).unsqueeze(1)
    sol = torch.linalg.lstsq(A, t).solution.squeeze(1)
    w, c = B @ sol[:-1], sol[-1]
    delta = mu[vals.index(y)] - mu[vals.index(x)]
    Wd = host.model.W_in.weight @ delta
    return torch.outer(Wd, w), Wd * c
