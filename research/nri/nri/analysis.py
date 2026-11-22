"""LLM-free, deterministic causal measurements of interpretability and editability.

Everything here uses only (i) the network, (ii) inputs, (iii) task-level variable values
(a, b, op). None of it reads the type labels, so any advantage measured for the
typed conditions is not because the analyst was handed the labels.

Measurements
------------
iia_curve        Interchange-intervention accuracy (causal abstraction; Geiger et al.)
                 for variable V at site resid_pre_mlp, patching the top-k directions
                 of the V-class-mean subspace. High IIA at small k => V is linearly
                 localized in a low-dim subspace.
neuron_iia       Same, but patching the m most V-selective MLP pre-activations
                 (privileged-basis / human-readable localization).
edit_program     A closed-form rank-one *weight program*: re-route op X -> op Y by
                 folding  h -> h + (mu_Y - mu_X) g(h)  into W_in, with g affine on the
                 op subspace, g(mu_X)=1, g(mu_Y)=g(mu_Z)=0. No gradients.
transfer         rerun the program on another host vs copy dW vs Procrustes-aligned dW.
"""
from __future__ import annotations

import itertools

import torch

from . import tasks

FIT_SEED, TEST_SEED, PAIR_SEED = 4242, 7777, 31337
VARS = {"op": len(tasks.OPS), "a": tasks.P, "b": tasks.P}


@torch.no_grad()
def collect(model, lat, condition):
    b = tasks.build(lat, condition)
    h = model.resid_pre_mlp(b.tokens, b.types)
    return b, h


def latents(n, seed):
    return tasks.sample_latents(n, torch.Generator().manual_seed(seed))


def class_means(h, labels, n_classes):
    return torch.stack([h[labels == c].mean(0) for c in range(n_classes)])


def mean_subspace(h, labels, n_classes):
    """Orthonormal basis (d x r) of the centered class-mean span, by singular value."""
    mu = class_means(h, labels, n_classes)
    U, S, Vh = torch.linalg.svd(mu - mu.mean(0), full_matrices=False)
    r = int((S > 1e-6 * S[0]).sum())
    return Vh[:r].T, mu


def counterfactual(base_lat, src_lat, var):
    lat = tasks.with_latent(base_lat, **{var: src_lat[var]})
    return tasks.apply_op(lat["op"], lat["a"], lat["b"])


@torch.no_grad()
def iia_curve(model, condition, var, n_fit=8192, n_pairs=4096):
    lat_fit = latents(n_fit, FIT_SEED)
    _, h_fit = collect(model, lat_fit, condition)
    basis, _ = mean_subspace(h_fit, lat_fit[var], VARS[var])
    base, src = latents(n_pairs, PAIR_SEED), latents(n_pairs, PAIR_SEED + 1)
    _, hb = collect(model, base, condition)
    _, hs = collect(model, src, condition)
    y_cf = counterfactual(base, src, var)
    curve = []
    for k in range(0, basis.shape[1] + 1):
        B = basis[:, :k]
        hp = hb + (hs - hb) @ B @ B.T
        curve.append((model.head_from_resid(hp).argmax(-1) == y_cf).float().mean().item())
    # full-residual patch upper bound (replaces everything at the site)
    full = (model.head_from_resid(hs).argmax(-1) == y_cf).float().mean().item()
    return dict(curve=curve, rank=basis.shape[1], full_patch=full,
                k90=next((k for k, v in enumerate(curve) if v >= 0.9), None))


@torch.no_grad()
def neuron_iia(model, condition, var, n_fit=8192, n_pairs=4096, ms=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    lat_fit = latents(n_fit, FIT_SEED)
    _, h_fit = collect(model, lat_fit, condition)
    z = h_fit @ model.W_in.weight.T + model.W_in.bias
    # one-way ANOVA F statistic per neuron for the variable
    lab = lat_fit[var]
    mu = class_means(z, lab, VARS[var])
    grand = z.mean(0)
    between = ((mu - grand) ** 2).mean(0)
    within = torch.stack([((z[lab == c] - mu[c]) ** 2).mean(0) for c in range(VARS[var])]).mean(0)
    order = (between / (within + 1e-8)).argsort(descending=True)
    base, src = latents(n_pairs, PAIR_SEED), latents(n_pairs, PAIR_SEED + 1)
    _, hb = collect(model, base, condition)
    _, hs = collect(model, src, condition)
    zb = hb @ model.W_in.weight.T + model.W_in.bias
    zs = hs @ model.W_in.weight.T + model.W_in.bias
    y_cf = counterfactual(base, src, var)
    out = {}
    for m in ms:
        idx = order[:m]
        zp = zb.clone()
        zp[:, idx] = zs[:, idx]
        r = hb + model.W_out(torch.relu(zp))
        out[m] = (model.W_U(r).argmax(-1) == y_cf).float().mean().item()
    m90 = next((m for m in ms if out[m] >= 0.9), None)
    return dict(curve=out, m90=m90)


# ---------------------------------------------------------------- component level (heads)
def _heads(attn, x):
    """Per-head contributions of an Attn module: ([B, H, L, d], bias, pattern [B, H, L, L])."""
    B, L, d = x.shape
    q, k, v = attn.qkv(x).view(B, L, 3, attn.h, attn.dh).unbind(2)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    pat = ((q @ k.transpose(-1, -2)) / attn.dh ** 0.5).softmax(-1)
    z = pat @ v  # B, H, L, dh
    Wo = attn.o.weight.view(d, attn.h, attn.dh)  # d x H x dh
    contrib = torch.einsum("bhld,ehd->bhle", z, Wo)
    return contrib, attn.o.bias, pat


@torch.no_grad()
def h_with_head_patch(model, bb, bs, layer, heads):
    """resid_pre_mlp of base batch bb, with the given heads' outputs at the final position
    replaced by those computed on source batch bs."""
    x0b = model.embed(bb.tokens, bb.types)
    x0s = model.embed(bs.tokens, bs.types)
    cb1, bias1, _ = _heads(model.attn1, x0b)
    cs1, _, _ = _heads(model.attn1, x0s)
    if layer == 1:
        cb1 = cb1.clone()
        cb1[:, heads, -1] = cs1[:, heads, -1]
    x1b = x0b + cb1.sum(1) + bias1
    x1s = x0s + cs1.sum(1) + bias1
    cb2, bias2, _ = _heads(model.attn2, x1b)
    if layer == 2:
        cs2, _, _ = _heads(model.attn2, x1s)
        cb2 = cb2.clone()
        cb2[:, heads, -1] = cs2[:, heads, -1]
    return (x1b + cb2.sum(1) + bias2)[:, -1]


@torch.no_grad()
def head_iia(model, condition, var, n_pairs=4096):
    base, src = latents(n_pairs, PAIR_SEED), latents(n_pairs, PAIR_SEED + 1)
    bb, bs = tasks.build(base, condition), tasks.build(src, condition)
    y_cf = counterfactual(base, src, var)
    H = model.attn1.h
    res = {}
    for layer in (1, 2):
        for hs in [[i] for i in range(H)] + [list(p) for p in itertools.combinations(range(H), 2)]:
            h = h_with_head_patch(model, bb, bs, layer, hs)
            res[f"L{layer}H{''.join(map(str, hs))}"] = (
                model.head_from_resid(h).argmax(-1) == y_cf).float().mean().item()
    single = {k: v for k, v in res.items() if len(k) == 4}
    best1 = max(single, key=single.get)
    best2 = max(res, key=res.get)
    return dict(all=res, best_single=best1, best_single_iia=single[best1], best_any=best2,
                best_any_iia=res[best2])


@torch.no_grad()
def role_purity(model, condition, n=4096):
    """Layer-2 (and 1) attention from the final '=' position, by contextual ROLE of the
    attended position. purity = max role mass. Human-legible routing metric."""
    lat = latents(n, TEST_SEED)
    b = tasks.build(lat, condition)
    # roles by position (independent of whether the model was given types)
    roles = torch.full((n, tasks.SEQ_LEN), tasks.T_DISTRACTOR)
    if condition == "canonical":
        roles[:, 0], roles[:, 1], roles[:, 2] = tasks.T_LEFT, tasks.T_OP, tasks.T_RIGHT
    else:
        i = torch.arange(n)
        roles[i, lat["op_pos"] - 1] = tasks.T_LEFT
        roles[i, lat["op_pos"]] = tasks.T_OP
        roles[i, lat["op_pos"] + 1] = tasks.T_RIGHT
    roles[:, -1] = tasks.T_QUERY
    x0 = model.embed(b.tokens, b.types)
    c1, b1, p1 = _heads(model.attn1, x0)
    x1 = x0 + c1.sum(1) + b1
    _, _, p2 = _heads(model.attn2, x1)
    out = {}
    for layer, pat in ((1, p1), (2, p2)):
        last = pat[:, :, -1, :]  # B, H, L
        for hd in range(pat.shape[1]):
            mass = {tasks.TYPE_NAMES[r]: (last[:, hd] * (roles == r)).sum(1).mean().item()
                    for r in range(tasks.N_TYPES)}
            top = max(mass, key=mass.get)
            out[f"L{layer}H{hd}"] = dict(role=top, purity=mass[top], mass=mass)
    n_pure = sum(v["purity"] >= 0.8 and v["role"] in ("LEFT", "RIGHT", "OP") for v in out.values())
    return dict(heads=out, n_pure_role_heads=n_pure)


# ---------------------------------------------------------------- decodability & DAS
def probe_acc(model, condition, var, n_fit=8192, n_test=4096, steps=300):
    """Linear decodability of `var` from the site (information present, not causal)."""
    _, h = collect(model, latents(n_fit, FIT_SEED), condition)
    lat_t = latents(n_test, TEST_SEED)
    _, ht = collect(model, lat_t, condition)
    y = latents(n_fit, FIT_SEED)[var]
    mu, sd = h.mean(0), h.std(0) + 1e-6
    torch.manual_seed(0)
    lin = torch.nn.Linear(h.shape[1], VARS[var])
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin((h - mu) / sd), y) + 1e-4 * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return (lin((ht - mu) / sd).argmax(-1) == lat_t[var]).float().mean().item()


def das(model, condition, var, k, steps=400, n_pairs=8192, n_test=4096, lr=1e-2, seed=0):
    """Distributed Alignment Search (Geiger et al. 2023): learn an orthonormal d x k basis Q
    maximizing interchange accuracy for `var`. The network stays frozen. Returns (test IIA, Q)."""
    torch.manual_seed(seed)
    base, src = latents(n_pairs, PAIR_SEED + 10), latents(n_pairs, PAIR_SEED + 11)
    _, hb = collect(model, base, condition)
    _, hs = collect(model, src, condition)
    y_cf = counterfactual(base, src, var)
    rot = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(hb.shape[1], k, bias=False))
    opt = torch.optim.Adam(rot.parameters(), lr=lr)
    for p in model.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, n_pairs, (1024,), generator=g)
        Q = rot.weight.T  # d x k, orthonormal columns
        hp = hb[idx] + (hs[idx] - hb[idx]) @ Q @ Q.T
        loss = torch.nn.functional.cross_entropy(model.head_from_resid(hp), y_cf[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        Q = rot.weight.T.detach().clone()
        tb, ts = latents(n_test, PAIR_SEED), latents(n_test, PAIR_SEED + 1)
        _, hb = collect(model, tb, condition)
        _, hs = collect(model, ts, condition)
        hp = hb + (hs - hb) @ Q @ Q.T
        acc = (model.head_from_resid(hp).argmax(-1) == counterfactual(tb, ts, var)).float().mean().item()
    return acc, Q


def das_curve(model, condition, var, ks=(1, 2, 4, 8, 16, 32)):
    out, Qs = {}, {}
    for k in ks:
        out[k], Qs[k] = das(model, condition, var, k)
    k90 = next((k for k in ks if out[k] >= 0.9), None)
    return dict(curve=out, k90=k90), Qs


def das_swap_program(model, condition, x, y, Q, n_fit=8192):
    """Edit that moves activations only INSIDE the DAS subspace Q of the op variable:
    h -> h + Q Q^T (mu_y - mu_x) g(h), g = least-squares indicator[op == x] on Q^T h.
    If Q is the causal locus of op, this should re-route x -> y with low collateral."""
    lat = latents(n_fit, FIT_SEED)
    _, h = collect(model, lat, condition)
    mu = class_means(h, lat["op"], VARS["op"])
    z = h @ Q
    X = torch.cat([z, torch.ones(len(z), 1)], 1)
    t = (lat["op"] == x).float().unsqueeze(1)
    sol = torch.linalg.lstsq(X, t).solution.squeeze(1)
    w, c = Q @ sol[:-1], sol[-1]
    delta = Q @ Q.T @ (mu[y] - mu[x])
    Wd = model.W_in.weight @ delta
    return torch.outer(Wd, w), Wd * c


# ---------------------------------------------------------------- weight program
def op_swap_program(model, condition, x: int, y: int, n_fit=8192):
    """Returns (dW_in, db_in, spec) implementing op x -> op y. Pure closed form."""
    lat = latents(n_fit, FIT_SEED)
    _, h = collect(model, lat, condition)
    basis, mu = mean_subspace(h, lat["op"], VARS["op"])  # basis d x r (r = 2)
    # g(h) = w.h + c with w = basis @ alpha; solve g(mu_j) = [j == x]
    A = torch.cat([mu @ basis, torch.ones(mu.shape[0], 1)], dim=1)  # 3 x (r+1)
    t = torch.tensor([1.0 if j == x else 0.0 for j in range(mu.shape[0])])
    sol = torch.linalg.lstsq(A, t.unsqueeze(1)).solution.squeeze(1)
    w, c = basis @ sol[:-1], sol[-1]
    delta = mu[y] - mu[x]
    Wd = model.W_in.weight @ delta  # d_mlp
    dW = torch.outer(Wd, w)
    db = Wd * c
    return dW, db, dict(w=w, c=c, delta=delta, mu=mu, basis=basis)


@torch.no_grad()
def eval_edit(model, condition, x, y, dW, db, n=8192):
    lat = latents(n, TEST_SEED)
    b, h = collect(model, lat, condition)
    pred = model.head_from_resid(h, model.W_in.weight + dW, model.W_in.bias + db).argmax(-1)
    is_x = b.op == x
    target_y = tasks.apply_op(torch.full_like(b.op, y), b.a, b.b)
    success = (pred[is_x] == target_y[is_x]).float().mean().item()
    preserve = (pred[~is_x] == b.y[~is_x]).float().mean().item()
    base_pred = model.head_from_resid(h).argmax(-1)
    base_pres = (base_pred[~is_x] == b.y[~is_x]).float().mean().item()
    return dict(success=success, preserve=preserve, collateral=base_pres - preserve,
                rel_norm=(dW.norm() / model.W_in.weight.norm()).item())


def all_swaps(model, condition):
    res = {}
    for x, y in itertools.permutations(range(len(tasks.OPS)), 2):
        dW, db, _ = op_swap_program(model, condition, x, y)
        res[f"{tasks.OPS[x]}->{tasks.OPS[y]}"] = eval_edit(model, condition, x, y, dW, db)
    return res


# ---------------------------------------------------------------- transfer
@torch.no_grad()
def procrustes(hA, hB):
    """Orthogonal R with (hB - mB) R ~= (hA - mA)."""
    mA, mB = hA.mean(0), hB.mean(0)
    U, _, Vh = torch.linalg.svd((hB - mB).T @ (hA - mA))
    return U @ Vh, mA, mB


@torch.no_grad()
def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = (X.T @ Y).norm() ** 2
    return (hsic / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()


@torch.no_grad()
def transfer(model_A, model_B, condition, x, y, n_fit=8192):
    lat = latents(n_fit, FIT_SEED)
    _, hA = collect(model_A, lat, condition)
    _, hB = collect(model_B, lat, condition)
    dWA, dbA, spec = op_swap_program(model_A, condition, x, y)
    out = {}
    # 1. rerun program on B (program re-derives statistics from host B)
    dWB, dbB, _ = op_swap_program(model_B, condition, x, y)
    out["rerun_program"] = eval_edit(model_B, condition, x, y, dWB, dbB)
    # 2. copy A's numeric delta
    out["copy_delta"] = eval_edit(model_B, condition, x, y, dWA, dbA)
    # 3. Procrustes-aligned delta (alignment fitted on shared inputs)
    R, mA, mB = procrustes(hA, hB)
    w_B = R @ spec["w"]
    c_B = spec["c"] + spec["w"] @ mA - w_B @ mB
    delta_B = R @ spec["delta"]
    Wd = model_B.W_in.weight @ delta_B
    out["aligned_delta"] = eval_edit(model_B, condition, x, y, torch.outer(Wd, w_B), Wd * c_B)
    resid = ((hB - mB) @ R - (hA - mA)).norm() / (hA - mA).norm()
    out["procrustes_rel_residual"] = resid.item()
    out["cka"] = linear_cka(hA, hB)
    return out
