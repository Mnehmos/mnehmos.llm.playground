"""Closed-form edit of the MLP input weights (no gradient descent).

Network (as reverse-engineered): output = (a OP b) mod 11 where OP is the single token in
{11:'+', 12:'-', 13:'*'}, a is the token just before it and b the token just after.
Edit goal: inputs containing token 11 should produce what they would with token 12, i.e.
(a - b) mod 11 instead of (a + b) mod 11; all other inputs unchanged.

Method (pure linear algebra on activation statistics):
 1. Sample inputs, record resid_pre_mlp h and the counterfactual h_cf (token 11 -> 12).
 2. Desired MLP pre-activations z* = W h_cf + b on scope rows (token 11 present),
    z* = W h + b elsewhere.
 3. Per hidden unit j, fit new weights (w'_j, b'_j) by ReLU-aware weighted least squares:
    where z*_ij > 0 the pre-activation must equal z*_ij, where z*_ij <= 0 it only has to stay
    <= 0 (alternating projection: target <- min(prediction, 0)). Row weights:
    w_scope on scope rows, times (importance of unit j for row i's decision)^gpow, where
    importance = rms_k (U[y_i, j] - U[k, j]), U = W_U W_out. A few boosting rounds double the
    weight of rows whose edited prediction is still wrong (capped at max_boost x).
 Returns (dW_in, db_in) = (W' - W, b' - b).
"""
import torch
import torch.nn.functional as F


def program(host, doc=None, n=16000, seed=0, w_scope=5.0, gpow=3.0, iters=30,
            boost_rounds=6, max_boost=16.0, ridge=1e-4, **args):
    net = host.model
    g = torch.Generator().manual_seed(seed)
    tokens, ch2 = host.sampler(n, g)
    W = net.W_in.weight.detach().float()
    b = net.W_in.bias.detach().float()
    U = (net.W_U.weight @ net.W_out.weight).detach().float()          # [11, 256]
    with torch.no_grad():
        h = host.site("resid_pre_mlp", tokens, ch2).detach().float()
        has = (tokens == 11).any(1)
        t_cf = tokens.clone(); t_cf[t_cf == 11] = 12
        h_cf = host.site("resid_pre_mlp", t_cf, ch2).detach().float()
        y_orig = net.head_from_resid(h).argmax(-1)
        y_cf = net.head_from_resid(h_cf).argmax(-1)
        want = torch.where(has, y_cf, y_orig)
        Z = h @ W.T + b
        Z[has] = h_cf[has] @ W.T + b
        pos = Z > 0
        X = torch.cat([h, torch.ones(len(h), 1)], 1)                     # [n, 65]
        imp = (U[want][:, None, :] - U[None, :, :]).pow(2).mean(1)        # [n, 256]
        imp = (imp / imp.mean(0, keepdim=True).clamp_min(1e-8)) ** gpow
        rw0 = torch.ones(len(h)); rw0[has] = w_scope
        rw = rw0.clone()
        S = None
        for _ in range(boost_rounds + 1):
            Wt = rw[:, None] * imp                                       # [n, 256]
            G = torch.einsum('nj,np,nq->jpq', Wt, X, X)
            G = G + ridge * Wt.sum(0)[:, None, None] * torch.eye(65)
            Gi = torch.linalg.inv(G)
            tg = Z.clone()
            for _ in range(iters):
                rhs = torch.einsum('nj,np->jp', Wt * tg, X)
                S = torch.einsum('jpq,jq->jp', Gi, rhs)                  # [256, 65]
                z = X @ S.T
                tg = torch.where(pos, Z, z.clamp(max=0.0))
            y_new = net.head_from_resid(h, S[:, :64], S[:, 64]).argmax(-1)
            rw = torch.minimum(rw * torch.where(y_new != want, 2.0, 1.0), rw0 * max_boost)
        dW = (S[:, :64] - W).contiguous()
        db = (S[:, 64] - b).contiguous()
    return dW, db
