"""Shared tools for the agent's bespoke edits (docs/26). Each step module imports what it needs.

mean_translation_edit: a constrained-optimization edit. On FIT minibatches, it adjusts the chosen
parameters so that translating the block of h by (mu[pi(M)] - mu[M]) -- the class-mean difference of
the block code, for random class permutations pi -- yields the answer with mass pi(M). If that works
for arbitrary pi, the block code is ADDITIVE in mass: route (unit, position) information may remain,
but it no longer interacts with mass. A KL term keeps the unedited predictions, and a CE term keeps
the task. The permutations never equal the inversion or add-one maps (the gate's target and the
held-out final edit). The result is returned as a delta, scaled to a relative budget.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

import run_dimensional as D
from nri import atlas as AT
from nri import dimtask as T

INV = [0] + [pow(x, -1, T.P) for x in range(1, T.P)]
ADD1 = [(x + 1) % T.P for x in range(T.P)]


def forward_h(m, tok):
    _, h = D.encode(m, tok)
    return h


def mean_translation_edit(m, ctx, params, steps=200, lr=1e-3, kl=1.0, task=1.0, batch=512, seed=0,
                          max_rel_model=0.08, max_rel_tensor=0.45, block=AT.BLK):
    mm = copy.deepcopy(m)
    P = dict(mm.named_parameters())
    for k, p in P.items():
        p.requires_grad_(k in params)
    opt = torch.optim.Adam([P[k] for k in params], lr=lr)
    fit = ctx["FIT"]
    tok_all = T.build(fit).tokens
    y_all = T.build(fit).y
    M_all, A_all, S_all = T.canon(fit)
    with torch.no_grad():
        p0 = torch.log_softmax(m(tok_all, torch.zeros_like(tok_all)), -1)
    g = torch.Generator().manual_seed(seed)
    n = len(y_all)
    for _ in range(steps):
        i = torch.randint(0, n, (batch,), generator=g)
        tok, y, M, A, S_ = tok_all[i], y_all[i], M_all[i], A_all[i], S_all[i]
        hf = forward_h(mm, tok_all)                      # class means from all FIT (with grad)
        mu = torch.stack([hf[M_all == c][:, block].mean(0) for c in range(T.P)])
        h = hf[i]
        while True:
            pi = torch.randperm(T.P, generator=g)
            if pi.tolist() not in (INV, ADD1, list(range(T.P))):
                break
        hp = h.clone()
        hp[:, block] = h[:, block] + mu[pi[M]] - mu[M]
        tgt = T.target(pi[M], A, S_)
        loss = F.cross_entropy(mm.head_from_resid(hp), tgt)
        logits = mm.head_from_resid(h)
        loss = loss + task * F.cross_entropy(logits, y)
        loss = loss + kl * F.kl_div(torch.log_softmax(logits, -1), p0[i], log_target=True, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
    W0 = dict(m.named_parameters())
    delta = {k: (P[k].detach() - W0[k].detach()).clone() for k in params}
    return scale(m, delta, max_rel_model, max_rel_tensor)


def scale(m, delta, max_rel_model, max_rel_tensor):
    W0 = dict(m.named_parameters())
    mn = torch.cat([p.detach().flatten() for p in m.parameters()]).norm()
    s = 1.0
    tot = torch.cat([v.flatten() for v in delta.values()]).norm()
    s = min(s, (max_rel_model * mn / tot.clamp_min(1e-12)).item())
    for k, v in delta.items():
        s = min(s, (max_rel_tensor * W0[k].detach().norm() / v.norm().clamp_min(1e-12)).item())
    return {k: v * s for k, v in delta.items()}


def describe(m, delta):
    W0 = dict(m.named_parameters())
    return {k: round((v.norm() / W0[k].detach().norm()).item(), 4) for k, v in delta.items()}


# ------------------------------------------------------------------ head-level canonical writer
def v_rows(head):
    """Rows of attn2.qkv holding the VALUE projection of `head` (view(B, L, 3, h, dh) ordering)."""
    return list(range(128 + 16 * head, 128 + 16 * (head + 1)))


@torch.no_grad()
def head_inputs(m, lat):
    c = AT.capture(m, lat)
    return c, c["pat2"][:, :, -1]          # (n, 4, 8): attention from '=' per head


@torch.no_grad()
def canonical_head_edit(m, ctx, heads=(0, 2), mode="canonical", alpha=1.0, ridge=1e-2, canon_units=(0, 2)):
    """Re-fit the VALUE weights of the mass-writing layer-2 heads so that their output at '=' is the
    same mass code for every unit.
      mode 'canonical': target = class mean (over examples whose mass unit is in canon_units) of the
                        head output for the example's mass + the example's own deviation from its
                        (mass, route) cell mean (within-cell variation kept); route main effect is
                        re-added from the head's own route means.
      mode 'interaction': target = head output minus its (mass x route) interaction.
    Ridge least squares of dV, db on FIT from the attention-weighted layer-1 input a_h = sum_j pat_j x1_j,
    then alpha-scaled. Returns {'attn2.qkv.weight': d, 'attn2.qkv.bias': d}."""
    fit = ctx["FIT"]
    c, pat = head_inputs(m, fit)
    fo = AT.factors(fit)
    dW = torch.zeros_like(m.attn2.qkv.weight)
    db = torch.zeros_like(m.attn2.qkv.bias)
    for hd in heads:
        a = (pat[:, hd, :, None] * c["x1"]).sum(1)                    # (n, 64)
        out = c["pre2"][:, 16 * hd:16 * (hd + 1)]                     # (n, 16)
        dec, table = AT.interaction_decomp(out, fo["M"], fo["route"])
        if mode == "interaction":
            target = out - table[fo["M"], fo["route"]]
        else:
            sel = torch.zeros(len(out), dtype=torch.bool)
            for u in canon_units:
                sel |= fo["mu"] == u
            canon = AT.class_means(out[sel], fo["M"][sel], AT.P)
            cell = torch.zeros(AT.P, 6, out.shape[1])
            for k in range(AT.P):
                for q in range(6):
                    s = (fo["M"] == k) & (fo["route"] == q)
                    if s.any():
                        cell[k, q] = out[s].mean(0)
            rmain = torch.stack([out[fo["route"] == q].mean(0) - out.mean(0) for q in range(6)])
            target = canon[fo["M"]] + rmain[fo["route"]] + (out - cell[fo["M"], fo["route"]])
        resid = target - out
        U = torch.cat([a, torch.ones(len(a), 1)], 1).double()
        G = U.T @ U + ridge * len(U) * torch.eye(65, dtype=torch.float64)
        Th = torch.linalg.solve(G, U.T @ resid.double()).float()
        # the head output is pat-weighted V x + b (attention weights sum to 1), so db adds directly
        dW[v_rows(hd)] = alpha * Th[:-1].T
        db[v_rows(hd)] = alpha * Th[-1]
    return {"attn2.qkv.weight": dW, "attn2.qkv.bias": db}


def rewire_edit(m, ctx, upstream_delta, post=("W_in.weight", "W_in.bias", "W_out.weight", "W_out.bias"),
                steps=400, lr=2e-3, batch=512, seed=1, translate=0.5, also_upstream=False, kl=1.0, iit=0.0,
                l2=0.0, task=0.0):
    """Install `upstream_delta` (e.g. a canonical-head edit), then re-fit the post-site weights (and
    optionally keep optimizing the upstream ones) so the NEW network reproduces the ORIGINAL network's
    predictions on FIT (KL), plus a mean-translation term that rewards an additive block code.
    Returns the total delta (upstream + post)."""
    mm = copy.deepcopy(m)
    with torch.no_grad():
        P = dict(mm.named_parameters())
        for k, v in upstream_delta.items():
            P[k].add_(v)
    fit = ctx["FIT"]
    b = T.build(fit)
    tok, y = b.tokens, b.y
    M_all, A_all, S_all = T.canon(fit)
    with torch.no_grad():
        p0 = torch.log_softmax(m(tok, torch.zeros_like(tok)), -1)
    train_keys = list(post) + (list(upstream_delta) if also_upstream else [])
    W0 = dict(m.named_parameters())
    for k, p in P.items():
        p.requires_grad_(k in train_keys)
    opt = torch.optim.Adam([P[k] for k in train_keys], lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    for _ in range(steps):
        i = torch.randint(0, n, (batch,), generator=g)
        hf = forward_h(mm, tok)
        logits = mm.head_from_resid(hf[i])
        loss = kl * F.kl_div(torch.log_softmax(logits, -1), p0[i], log_target=True, reduction="batchmean")
        if task:
            loss = loss + task * F.cross_entropy(logits, y[i])
        if iit:
            j = torch.randint(0, n, (batch,), generator=g)
            hb = hf[i].clone()
            hb[:, AT.BLK] = hf[j][:, AT.BLK]
            src = {kk: vv[j] for kk, vv in fit.items()}
            base = {kk: vv[i] for kk, vv in fit.items()}
            loss = loss + iit * F.cross_entropy(mm.head_from_resid(hb), D.counterfactual(base, src, "M"))
        if l2:
            loss = loss + l2 * sum(((P[k] - W0[k].detach()) ** 2).sum() / (W0[k].detach() ** 2).sum()
                                   for k in train_keys)
        if translate:
            mu = torch.stack([hf[M_all == c][:, AT.BLK].mean(0) for c in range(T.P)])
            while True:
                pi = torch.randperm(T.P, generator=g)
                if pi.tolist() not in (INV, ADD1, list(range(T.P))):
                    break
            hp = hf[i].clone()
            hp[:, AT.BLK] = hf[i][:, AT.BLK] + mu[pi[M_all[i]]] - mu[M_all[i]]
            loss = loss + translate * F.cross_entropy(mm.head_from_resid(hp), T.target(pi[M_all[i]], A_all[i], S_all[i]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: (P[k].detach() - W0[k].detach()).clone() for k in set(train_keys) | set(upstream_delta)}


# ------------------------------------------------------------------ differentiable interaction penalty
def _design(M, R, nr=6):
    cells = sorted({(int(a), int(b)) for a, b in zip(M.tolist(), R.tolist())})
    X = torch.zeros(len(cells), 1 + T.P + nr)
    for i, (c, q) in enumerate(cells):
        X[i, 0], X[i, 1 + c], X[i, 1 + T.P + q] = 1, 1, 1
    idx = {cq: i for i, cq in enumerate(cells)}
    cell_of = torch.tensor([idx[(int(a), int(b))] for a, b in zip(M.tolist(), R.tolist())])
    Pr = torch.eye(len(cells)) - X @ torch.linalg.pinv(X)          # residual-maker: interaction part
    return cell_of, Pr, len(cells)


def interaction_loss(Z, cell_of, Pr, ncell, M):
    """Differentiable share of within-mass variance carried by the mass x route interaction."""
    cnt = torch.bincount(cell_of, minlength=ncell).float()
    cm = torch.zeros(ncell, Z.shape[1]).index_add(0, cell_of, Z) / cnt[:, None]
    inter = Pr @ cm
    mu = torch.zeros(T.P, Z.shape[1]).index_add(0, M, Z) / torch.bincount(M, minlength=T.P).float().clamp_min(1)[:, None]
    within = ((Z - mu[M]) ** 2).sum()
    return (cnt[:, None] * inter ** 2).sum() / within


def organize_edit(m, ctx, params, steps=300, lr=1e-3, w_inter=1.0, kl=10.0, iit=1.0, l2=3.0, seed=2, keep=0.0):
    """Optimize `params` to shrink the block's mass x route interaction on FIT while keeping the
    original predictions (KL) and the block's causal sufficiency (interchange CE), with an L2 leash."""
    mm = copy.deepcopy(m)
    P = dict(mm.named_parameters())
    W0 = {k: v.detach().clone() for k, v in m.named_parameters()}
    for k, p in P.items():
        p.requires_grad_(k in params)
    opt = torch.optim.Adam([P[k] for k in params], lr=lr)
    fit = ctx["FIT"]
    b = T.build(fit)
    tok = b.tokens
    fo = AT.factors(fit)
    cell_of, Pr, nc = _design(fo["M"], fo["route"])
    with torch.no_grad():
        p0 = torch.log_softmax(m(tok, torch.zeros_like(tok)), -1)
    ok = p0.argmax(-1) == b.y                     # originally correct FIT answers
    g = torch.Generator().manual_seed(seed)
    n = len(tok)
    for _ in range(steps):
        hf = forward_h(mm, tok)
        li = interaction_loss(hf[:, AT.BLK], cell_of, Pr, nc, fo["M"])
        logits = mm.head_from_resid(hf)
        loss = w_inter * li + kl * F.kl_div(torch.log_softmax(logits, -1), p0, log_target=True, reduction="batchmean")
        if keep and ok.any():
            loss = loss + keep * F.cross_entropy(logits[ok], b.y[ok])
        if iit:
            j = torch.randperm(n, generator=g)
            hb = hf.clone()
            hb[:, AT.BLK] = hf[j][:, AT.BLK]
            src = {kk: vv[j] for kk, vv in fit.items()}
            loss = loss + iit * F.cross_entropy(mm.head_from_resid(hb), D.counterfactual(fit, src, "M"))
        if l2:
            loss = loss + l2 * sum(((P[k] - W0[k]) ** 2).sum() / (W0[k] ** 2).sum() for k in params)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: (P[k].detach() - W0[k]).clone() for k in params}
