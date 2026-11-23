"""Concept-formation atlas instrumentation and the upstream organization edit (Issue #22; docs/24).

Reusable, read-only measurements on a TinyTyped `mass_only` network for the dimensional task
(nri/dimtask.py). Nothing here trains a network or touches an optimizer: the trainer lives in
run_atlas.py. Every function is deterministic given its inputs.

Contents
  partition()          frozen question partition: FIT / VAL (policy calibration), OBS (atlas panel),
                       FINAL (sealed, run_perm reserved set), OOD_DEV / OOD_FINAL (held-out combos)
  capture()            one forward pass returning every registered site
  observe()            the cheap per-checkpoint atlas card (decodability, transfer, geometry, causal
                       interchange, per-head / per-position causal tests, neuron signatures)
  weight_dynamics()    per-tensor weight-change metrics between snapshots
  attribution()        first-order path attribution of loss changes to tensors / heads / neurons
  interaction_decomp() M x route two-way decomposition of the declared block
  decide()             the registered upstream edit policy: attn2.o rows 0-15 (+ bias), shrinking the
                       M x route interaction of the block while keeping both main effects
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import run_dimensional as D
import run_perm as RP
import run_swap as S
from nri import dimtask as T

BLK = list(range(*S.BLOCK))
P = T.P
LOG = {}
_x = 1
for _k in range(12):            # discrete log base 2 (2 is a primitive root mod 13)
    LOG[_x] = _k
    _x = _x * 2 % P

PART_S = 24_100                  # partition seed (fresh)
PAIR_S = 24_101                  # interchange-pair permutation seed (fresh)
SIG_FACTORS = ("M", "A", "F", "MA", "mu", "au", "fu", "order")
ETA_FACTORS = ("M", "A", "F", "MA", "m", "a", "mu", "au", "fu", "order", "route")
PROBE_FACTORS = ("M", "m", "mu", "A", "a", "au", "fu", "order", "F", "MA", "route")
FAMILY_ETA = 0.25                # neuron family threshold on eta^2 (registered)
ACTIVE_FRAC = 0.01               # a neuron is 'dead' if active on < 1% of the panel


# ------------------------------------------------------------------ data partition
def all_ood_questions():
    rows = [(m, mu, a, au, fu, o)
            for m in range(P) for mu in range(3) for a in range(P) for au in range(3)
            for fu in range(3) for o in range(2)
            if (mu == T.OOD_UNIT and m >= T.OOD_VALUES[0]) or (au == T.OOD_UNIT and a >= T.OOD_VALUES[0])]
    return rows


def _ood_vals(q):
    m, mu, a, au, fu, o = q
    v = []
    if mu == T.OOD_UNIT and m >= 9:
        v.append(m)
    if au == T.OOD_UNIT and a >= 9:
        v.append(a)
    return v


_PART = {}


def partition():
    """FINAL = run_perm reserved 30% of ID questions (sealed). The remaining 70% are split once
    (seed PART_S) into FIT 50% / VAL 25% / OBS 25%. OOD questions are split by held-out
    combination: OOD_DEV uses only held-out values {9, 10}, OOD_FINAL only {11, 12};
    questions mixing both are excluded."""
    if not _PART:
        res = RP.reserved_set()
        nonres = sorted(set(RP.all_questions()) - res)
        g = torch.Generator().manual_seed(PART_S)
        perm = torch.randperm(len(nonres), generator=g).tolist()
        n = len(nonres)
        a, b = n // 2, n // 2 + n // 4
        fit = [nonres[i] for i in perm[:a]]
        val = [nonres[i] for i in perm[a:b]]
        obs = [nonres[i] for i in perm[b:]]
        ood = all_ood_questions()
        dev = [q for q in ood if set(_ood_vals(q)) <= {9, 10}]
        fin = [q for q in ood if set(_ood_vals(q)) <= {11, 12}]
        _PART.update(FIT=RP.as_lat(sorted(fit)), VAL=RP.as_lat(sorted(val)), OBS=RP.as_lat(sorted(obs)),
                     FINAL=RP.eval_lat(res), OOD_DEV=RP.as_lat(sorted(dev)), OOD_FINAL=RP.as_lat(sorted(fin)))
        for k in ("FIT", "VAL", "OBS", "FINAL", "OOD_DEV", "OOD_FINAL"):
            lat = _PART[k]
            g2 = torch.Generator().manual_seed(PAIR_S + len(k))
            _PART[k + "_src"] = {kk: v[torch.randperm(len(lat["m"]), generator=g2)] for kk, v in lat.items()}
            # same-order sources (layer-1 positional interchange needs aligned positions)
            src = {kk: v.clone() for kk, v in lat.items()}
            for o in (0, 1):
                idx = (lat["order"] == o).nonzero().squeeze(1)
                p = idx[torch.randperm(len(idx), generator=g2)]
                for kk in src:
                    src[kk][idx] = lat[kk][p]
            _PART[k + "_src_same_order"] = src
    return _PART


def factors(lat):
    M, A, S_ = T.canon(lat)
    Fv = T.target(M, A, S_)
    return dict(M=M, A=A, F=Fv, MA=M * A % P, m=lat["m"], a=lat["a"], mu=lat["mu"], au=lat["au"],
                fu=lat["fu"], order=lat["order"], route=lat["mu"] * 2 + lat["order"],
                aroute=lat["au"] * 2 + lat["order"])


# ------------------------------------------------------------------ forward capture
def _attn_parts(attn, x):
    B, L, d = x.shape
    q, k, v = attn.qkv(x).view(B, L, 3, attn.h, attn.dh).unbind(2)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    pat = ((q @ k.transpose(-1, -2)) / math.sqrt(attn.dh)).softmax(-1)
    pre = (pat @ v).transpose(1, 2).reshape(B, L, d)
    return attn.o(pre), pat, pre


@torch.no_grad()
def capture(m, lat, patch=None):
    """All registered sites for the examples `lat`. patch: optional dict site->callable that
    rewrites a site before the rest of the forward pass ('x1', 'pre2', 'h')."""
    tok = T.build(lat).tokens
    x0 = m.embed(tok, torch.zeros_like(tok))
    a1, pat1, _ = _attn_parts(m.attn1, x0)
    x1 = x0 + a1
    if patch and "x1" in patch:
        x1 = patch["x1"](x1)
    a2, pat2, pre2 = _attn_parts(m.attn2, x1)
    pre2_last = pre2[:, -1]
    if patch and "pre2" in patch:
        pre2_last = patch["pre2"](pre2_last)
    a2_last = m.attn2.o(pre2_last)
    h = x1[:, -1] + a2_last
    if patch and "h" in patch:
        h = patch["h"](h)
    z = h @ m.W_in.weight.T + m.W_in.bias
    act = F.relu(z)
    r = h + m.W_out(act)
    return dict(x0=x0, x1=x1, pat1=pat1, pat2=pat2, pre2=pre2_last, a2=a2_last, h=h, z=z, act=act,
                logits=m.W_U(r))


def role_pos(lat):
    o = lat["order"]
    return dict(mval=torch.where(o == 0, 0, 3), munit=torch.where(o == 0, 1, 4),
                aval=torch.where(o == 0, 3, 0), aunit=torch.where(o == 0, 4, 1))


def probe_sites(c, lat):
    idx = torch.arange(len(lat["m"]))
    rp = role_pos(lat)
    s = {f"L1@{k}": c["x1"][idx, v] for k, v in rp.items()}
    s["L1@last"] = c["x1"][:, -1]
    s["A2@last"] = c["a2"]
    for hd in range(4):
        s[f"A2h{hd}"] = c["pre2"][:, 16 * hd:16 * (hd + 1)]
    s["h"] = c["h"]
    s["h_block"] = c["h"][:, BLK]
    s["h_rest"] = c["h"][:, 16:]
    s["mlp"] = c["act"]
    return s


# ------------------------------------------------------------------ probes (descriptive only)
def ridge_probe(Xf, yf, Xe, ye, ncls, lam=1e-2):
    """Closed-form ridge one-hot probe on standardized features; accuracy on (Xe, ye)."""
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    A = torch.cat([(Xf - mu) / sd, torch.ones(len(Xf), 1)], 1).double()
    E = torch.cat([(Xe - mu) / sd, torch.ones(len(Xe), 1)], 1).double()
    Y = F.one_hot(yf, ncls).double()
    G = A.T @ A + lam * len(A) * torch.eye(A.shape[1], dtype=torch.float64)
    W = torch.linalg.solve(G, A.T @ Y)
    return ((E @ W).argmax(1) == ye).float().mean().item()


NCLS = dict(M=P, m=P, A=P, a=P, F=P, MA=P, mu=3, au=3, fu=3, order=2, route=6, aroute=6)


def chance(y, ncls):
    return torch.bincount(y, minlength=ncls).max().item() / len(y)


# ------------------------------------------------------------------ geometry
def class_means(Z, y, k):
    out = torch.zeros(k, Z.shape[1])
    for c in range(k):
        s = y == c
        if s.any():
            out[c] = Z[s].mean(0)
    return out


def interaction_decomp(Z, y, r, nr=6):
    """Two-way decomposition of the block over (M class y, route r) cells.
    Cell means mu_cq = mu + a_c + b_q + I_cq (least squares over PRESENT cells, unweighted).
    Returns shares of the within-M variance (sum over examples of ||z - class mean||^2):
    route main effect, interaction, within-cell residual; and the (P, nr, d) interaction table."""
    d = Z.shape[1]
    cells, cm, cnt = [], [], []
    for c in range(P):
        for q in range(nr):
            s = (y == c) & (r == q)
            if s.any():
                cells.append((c, q))
                cm.append(Z[s].mean(0))
                cnt.append(int(s.sum()))
    cm = torch.stack(cm).double()
    X = torch.zeros(len(cells), 1 + P + nr, dtype=torch.float64)
    for i, (c, q) in enumerate(cells):
        X[i, 0], X[i, 1 + c], X[i, 1 + P + q] = 1, 1, 1
    W = torch.linalg.lstsq(X, cm, driver="gelsd").solution
    add = X @ W
    inter = (cm - add).float()
    table = torch.zeros(P, nr, d)
    for i, (c, q) in enumerate(cells):
        table[c, q] = inter[i]
    # variance accounting on examples
    mu_c = class_means(Z, y, P)
    within = ((Z - mu_c[y]) ** 2).sum().item()
    resid = 0.0
    inter_ss, route_ss = 0.0, 0.0
    for i, (c, q) in enumerate(cells):
        s = (y == c) & (r == q)
        resid += ((Z[s] - cm[i].float()) ** 2).sum().item()
        inter_ss += cnt[i] * (inter[i] ** 2).sum().item()
        route_ss += cnt[i] * ((cm[i].float() - mu_c[c]) ** 2).sum().item()
    w = max(within, 1e-12)
    return dict(route_share=route_ss / w, interaction_share=inter_ss / w,
                route_main_share=(route_ss - inter_ss) / w, residual_share=resid / w, within_ss=within), table


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = (X.T @ Y).norm() ** 2
    den = (X.T @ X).norm() * (Y.T @ Y).norm()
    return (num / den.clamp_min(1e-12)).item()


def per_unit_cka(Z, y, unit):
    """Coalescence index: mean pairwise linear CKA between the per-unit class-mean configurations
    (over classes present for both units). 1 = the same geometry for every unit."""
    vals = []
    for u1 in range(3):
        for u2 in range(u1 + 1, 3):
            cs = [c for c in range(P) if ((y == c) & (unit == u1)).any() and ((y == c) & (unit == u2)).any()]
            if len(cs) < 3:
                continue
            A = torch.stack([Z[(y == c) & (unit == u1)].mean(0) for c in cs])
            B = torch.stack([Z[(y == c) & (unit == u2)].mean(0) for c in cs])
            vals.append(linear_cka(A, B))
    return sum(vals) / len(vals) if vals else float("nan")


def char_concentration(means):
    """Max share of class-mean variance (classes 1..12) on one multiplicative (discrete-log)
    frequency vs one additive frequency (doc 20 measure)."""
    xs = list(range(1, 13))
    Mu = means[xs] - means[xs].mean(0)
    tot = (Mu ** 2).sum().item()
    if tot < 1e-12:
        return dict(mult=float("nan"), add=float("nan"))
    out = {}
    for kind in ("mult", "add"):
        best = 0.0
        for f in range(1, 7):
            ang = torch.tensor([2 * math.pi * f * (LOG[x] / 12 if kind == "mult" else x / 13) for x in xs])
            Bm = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
            Q, _ = torch.linalg.qr(Bm - Bm.mean(0))
            best = max(best, ((Q.T @ Mu) ** 2).sum().item() / tot)
        out[kind] = best
    return out


def mean_subspace(Z, y, k, var_frac=0.9):
    mu = class_means(Z, y, k)
    mu = mu - mu.mean(0)
    U, Sv, Vh = torch.linalg.svd(mu, full_matrices=False)
    e = (Sv ** 2).cumsum(0) / (Sv ** 2).sum().clamp_min(1e-12)
    r = int((e < var_frac).sum().item()) + 1
    return Vh[:r].T, mu


def subspace_overlap(Q1, Q2):
    """Mean squared cosine of principal angles (1 = same subspace)."""
    if Q1 is None or Q2 is None:
        return float("nan")
    s = torch.linalg.svdvals(Q1.T @ Q2)
    return ((s ** 2).sum() / min(Q1.shape[1], Q2.shape[1])).item()


def eta2(Zc, y, k):
    """Per-column share of variance explained by factor y (one-way ANOVA)."""
    tot = ((Zc - Zc.mean(0)) ** 2).sum(0)
    mu = class_means(Zc, y, k)
    cnt = torch.bincount(y, minlength=k).float()
    btw = (cnt[:, None] * (mu - Zc.mean(0)) ** 2).sum(0)
    return btw / tot.clamp_min(1e-12)


# ------------------------------------------------------------------ causal (activation-level)
def counterfactual(base, src, var):
    """D.counterfactual plus unit-only counterfactuals: 'mu' (the mass unit comes from the source,
    the raw value from the base) and 'au' (same for acceleration)."""
    if var in ("mu", "au"):
        M, A, S_ = T.canon(base)
        if var == "mu":
            M = base["m"] * torch.tensor(T.MASS_S)[src["mu"]] % P
        else:
            A = base["a"] * torch.tensor(T.ACC_S)[src["au"]] % P
        return T.target(M, A, S_)
    return D.counterfactual(base, src, var)


@torch.no_grad()
def iia(pred, base, src, var):
    """Interchange-intervention accuracy on pairs whose counterfactual differs from the base answer."""
    cf = counterfactual(base, src, var)
    el = cf != T.build(base).y
    return (pred == cf)[el].float().mean().item()


@torch.no_grad()
def swap_h(m, base, src, dims, var, cb=None, cs=None):
    cb = capture(m, base) if cb is None else cb
    cs = capture(m, src) if cs is None else cs
    hp = cb["h"].clone()
    hp[:, dims] = cs["h"][:, dims]
    return iia(m.head_from_resid(hp).argmax(-1), base, src, var)


@torch.no_grad()
def swap_head(m, base, src, head, var, cs):
    sl = slice(16 * head, 16 * (head + 1))

    def f(p):
        p = p.clone()
        p[:, sl] = cs["pre2"][:, sl]
        return p
    return iia(capture(m, base, {"pre2": f})["logits"].argmax(-1), base, src, var)


@torch.no_grad()
def swap_l1(m, base, src, roles, var, cs):
    """Layer-1 positional interchange (base and source share the token order)."""
    rp = role_pos(base)
    idx = torch.arange(len(base["m"]))

    def f(x1):
        x1 = x1.clone()
        for r_ in roles:
            x1[idx, rp[r_]] = cs["x1"][idx, rp[r_]]
        return x1
    return iia(capture(m, base, {"x1": f})["logits"].argmax(-1), base, src, var)


@torch.no_grad()
def pooled_edit(m, fit_lat, eval_lat, var="M", kind="inv", basis=None, h_fit=None, h_eval=None):
    """The doc-15 pooled editor (an affine map in span(basis) fitted on class means of fit_lat,
    folded into W_in: a PERSISTENT weight change), scored with the frozen run_swap scorer."""
    B = torch.eye(64)[:, BLK] if basis is None else basis
    h = capture(m, fit_lat)["h"] if h_fit is None else h_fit
    y = D.label(fit_lat, var)
    mu = D.class_means(h, y, P)
    move = mu[S.EDITS[kind](y)] - mu[y]
    m0 = h.mean(0)
    z = (h - m0) @ B
    X = torch.cat([z, torch.ones(len(z), 1)], 1)
    W = torch.linalg.lstsq(X, move @ B).solution
    Dm = B @ W[:-1].T @ B.T
    dW, db = m.W_in.weight @ Dm, m.W_in.weight @ (B @ W[-1] - Dm @ m0)
    he = capture(m, eval_lat)["h"] if h_eval is None else h_eval
    pred = m.head_from_resid(he, m.W_in.weight + dW, m.W_in.bias + db).argmax(-1)
    return S.score(pred, eval_lat, var, kind)


@torch.no_grad()
def copy_block(m, lat, var="M", kind="inv", cb=None):
    src = S.source_for(lat, var, kind)
    cb = capture(m, lat) if cb is None else cb
    hs = capture(m, src)["h"]
    hp = cb["h"].clone()
    hp[:, BLK] = hs[:, BLK]
    return S.score(m.head_from_resid(hp).argmax(-1), lat, var, kind)


@torch.no_grad()
def accuracy(m, lat, c=None):
    c = capture(m, lat) if c is None else c
    return (c["logits"].argmax(-1) == T.build(lat).y).float().mean().item()


# ------------------------------------------------------------------ neuron signatures / families
def signatures(Z, fac):
    """Response signature per column: concatenated factor-conditional means (z-scored per column)
    for SIG_FACTORS. Returns (ncol, nsig)."""
    sd = Z.std(0).clamp_min(1e-6)
    Zs = (Z - Z.mean(0)) / sd
    parts = [class_means(Zs, fac[f], NCLS[f]) for f in SIG_FACTORS]
    return torch.cat(parts, 0).T


def families(Z, fac):
    """Family label per column: argmax eta^2 factor if >= FAMILY_ETA, 'mixed' otherwise,
    'dead' if active (> 0) on < ACTIVE_FRAC of the panel or constant."""
    etas = torch.stack([eta2(Z, fac[f], NCLS[f]) for f in ETA_FACTORS], 1)   # (ncol, nf)
    best, arg = etas.max(1)
    active = ((Z.abs() > 1e-6).float().mean(0) >= ACTIVE_FRAC) & (Z.std(0) > 1e-6)
    lab = []
    for j in range(Z.shape[1]):
        if not active[j]:
            lab.append("dead")
        elif best[j] >= FAMILY_ETA:
            lab.append(ETA_FACTORS[arg[j]])
        else:
            lab.append("mixed")
    return lab, etas


def match_features(sig_prev, sig_now):
    """Hungarian matching of response signatures between consecutive checkpoints (cosine).
    Returns per-current-column: matched previous index, cosine, self-cosine, status in
    {stable, relocated, uncertain, new}."""
    from scipy.optimize import linear_sum_assignment
    a = F.normalize(sig_prev, dim=1)
    b = F.normalize(sig_now, dim=1)
    C = (b @ a.T).nan_to_num(0.0)
    r, c = linear_sum_assignment(-C.numpy())
    out = []
    for i, j in zip(r.tolist(), c.tolist()):
        cs, ss = C[i, j].item(), C[i, i].item()
        if cs >= 0.8 and j == i:
            st = "stable"
        elif cs >= 0.8 and ss < 0.8:
            st = "relocated"
        elif cs >= 0.5 or ss >= 0.5:
            st = "uncertain"
        else:
            st = "new"
        out.append((j, round(cs, 3), round(ss, 3), st))
    return out


# ------------------------------------------------------------------ the cheap atlas card
@torch.no_grad()
def observe(m, prev=None):
    """Every-100-step atlas observation on the OBS panel (descriptive decodability, geometry,
    signatures) with activation-level causal tests on OBS pairs. Never reads VAL/FINAL.
    `prev`: the previous call's `carry` dict (for alignment across checkpoints).
    Returns (card, carry, compact_record)."""
    d = partition()
    fit, obs, src, src_o = d["FIT"], d["OBS"], d["OBS_src"], d["OBS_src_same_order"]
    cf_fit, cf_obs = capture(m, fit), capture(m, obs)
    cs, cso = capture(m, src), capture(m, src_o)
    ff, fo = factors(fit), factors(obs)
    card = dict(acc_obs=accuracy(m, obs, cf_obs), acc_ood_dev=accuracy(m, d["OOD_DEV"]))
    # decodability: probes fit on FIT, scored on OBS, chance-adjusted
    sf, so = probe_sites(cf_fit, fit), probe_sites(cf_obs, obs)
    dec = {}
    for site in sf:
        for fac in PROBE_FACTORS:
            acc = ridge_probe(sf[site], ff[fac], so[site], fo[fac], NCLS[fac])
            ch = chance(fo[fac], NCLS[fac])
            dec[f"{fac}@{site}"] = round((acc - ch) / (1 - ch), 4)
    card["decode_kappa"] = dec
    # equivalence: cross-unit transfer probes (train on two units in FIT, test on the third in OBS)
    tr = {}
    for fac, unit in (("M", "mu"), ("A", "au")):
        for site in ("L1@mval" if fac == "M" else "L1@aval", "A2@last", "h", "h_block", "mlp"):
            accs, wins = [], []
            for u in range(3):
                trn, tst = ff[unit] != u, fo[unit] == u
                if tst.sum() < 10:
                    continue
                accs.append(ridge_probe(sf[site][trn], ff[fac][trn], so[site][tst], fo[fac][tst], P))
                wins.append(ridge_probe(sf[site], ff[fac], so[site][tst], fo[fac][tst], P))
            tr[f"{fac}@{site}"] = dict(transfer=round(sum(accs) / 3, 4), within=round(sum(wins) / 3, 4))
    card["unit_transfer"] = tr
    # geometry of mass (and controls) at the block and full h
    geo = {}
    for name, Z in (("block", cf_obs["h"][:, BLK]), ("h", cf_obs["h"]), ("mlp", cf_obs["act"])):
        dm, _ = interaction_decomp(Z, fo["M"], fo["route"])
        da, _ = interaction_decomp(Z, fo["A"], fo["aroute"])
        geo[name] = dict(M_route_share=round(dm["route_share"], 4), M_interaction_share=round(dm["interaction_share"], 4),
                         M_route_main_share=round(dm["route_main_share"], 4),
                         M_residual_share=round(dm["residual_share"], 4),
                         A_route_share=round(da["route_share"], 4), A_interaction_share=round(da["interaction_share"], 4),
                         M_unit_cka=round(per_unit_cka(Z, fo["M"], fo["mu"]), 4),
                         A_unit_cka=round(per_unit_cka(Z, fo["A"], fo["au"]), 4),
                         M_char=char_concentration(class_means(Z, fo["M"], P)),
                         F_char=char_concentration(class_means(Z, fo["F"], P)))
    card["geometry"] = geo
    # distributed subspaces at h: where does each concept's class-mean variance live?
    sub, carry = {}, {}
    for fac in ("M", "A", "F", "route", "fu"):
        Q, mu = mean_subspace(cf_obs["h"], fo[fac], NCLS[fac])
        tot = (mu ** 2).sum().clamp_min(1e-12)
        sub[fac] = dict(rank90=Q.shape[1], block_share=round(((mu[:, BLK] ** 2).sum() / tot).item(), 4),
                        stability=round(subspace_overlap(prev["Q"][fac], Q), 4) if prev else None)
        carry.setdefault("Q", {})[fac] = Q
    sub["M_vs_A_overlap"] = round(subspace_overlap(carry["Q"]["M"], carry["Q"]["A"]), 4)
    card["subspaces"] = sub
    card["cka_h_prev"] = round(linear_cka(prev["h"], cf_obs["h"]), 4) if prev else None
    carry["h"] = cf_obs["h"]
    # causal (activation-level interchange on OBS pairs)
    rest = list(range(16, 64))
    cz = dict(iia_M_block=swap_h(m, obs, src, BLK, "M", cf_obs, cs),
              iia_M_rest=swap_h(m, obs, src, rest, "M", cf_obs, cs),
              iia_A_block=swap_h(m, obs, src, BLK, "A", cf_obs, cs),
              iia_A_rest=swap_h(m, obs, src, rest, "A", cf_obs, cs),
              iia_S_rest=swap_h(m, obs, src, rest, "S", cf_obs, cs))
    for hd in range(4):
        cz[f"iia_M_head{hd}"] = swap_head(m, obs, src, hd, "M", cs)
        cz[f"iia_A_head{hd}"] = swap_head(m, obs, src, hd, "A", cs)
    # layer-1 positional interchange: the counterfactual matches what was swapped
    for tag, roles, var in (("M_L1_val+unit", ("mval", "munit"), "M"), ("m_L1_val", ("mval",), "m"),
                            ("mu_L1_unit", ("munit",), "mu"), ("A_L1_val+unit", ("aval", "aunit"), "A"),
                            ("a_L1_val", ("aval",), "a"), ("au_L1_unit", ("aunit",), "au")):
        cz[f"iia_{tag}"] = swap_l1(m, obs, src_o, roles, var, cso)
    pe = pooled_edit(m, fit, obs, h_fit=cf_fit["h"], h_eval=cf_obs["h"])
    cz["edit_block_intended"], cz["edit_block_wrong"] = pe["intended"], pe["wrong"]
    cz["copy_block_intended"] = copy_block(m, obs, cb=cf_obs)["intended"]
    # MLP neuron mean-ablation importance (loss increase on OBS)
    y = T.build(obs).y
    base_loss = F.cross_entropy(cf_obs["logits"], y).item()
    act = cf_obs["act"]
    mean_act = act.mean(0)
    contrib = (act - mean_act)[:, :, None] * m.W_out.weight.T[None]        # (n, 256, 64)
    r0 = cf_obs["h"] + m.W_out(act)
    imp = []
    for j in range(act.shape[1]):
        lg = m.W_U(r0 - contrib[:, j])
        imp.append(F.cross_entropy(lg, y).item() - base_loss)
    imp = torch.tensor(imp)
    card["causal"] = cz
    # neuron families and alignment (MLP neurons; attn2 head dims; h dims)
    fam, sig = {}, {}
    for name, Z in (("mlp", act), ("attn2", cf_obs["pre2"]), ("h", cf_obs["h"])):
        lab, et = families(Z, fo)
        fam[name] = lab
        sig[name] = signatures(Z, fo)
    carry["sig"] = sig
    carry["fam"] = fam
    lineage = {}
    for name in sig:
        cnt = {}
        for l in fam[name]:
            cnt[l] = cnt.get(l, 0) + 1
        ent = dict(counts=cnt)
        if prev:
            mt = match_features(prev["sig"][name], sig[name])
            st_cnt = {}
            for (_, _, _, s) in mt:
                st_cnt[s] = st_cnt.get(s, 0) + 1
            trans = {}
            for j, l in enumerate(fam[name]):
                k = f"{prev['fam'][name][j]}->{l}"
                if prev["fam"][name][j] != l:
                    trans[k] = trans.get(k, 0) + 1
            ent.update(match_status=st_cnt, family_changes=trans,
                       mean_self_cos=round(sum(x[2] for x in mt) / len(mt), 4))
        lineage[name] = ent
    card["features"] = lineage
    card["mlp_top_importance"] = [[int(i), round(imp[i].item(), 4), fam["mlp"][i]] for i in imp.argsort(descending=True)[:12]]
    card["mlp_importance_by_family"] = {}
    for l in set(fam["mlp"]):
        sel = torch.tensor([x == l for x in fam["mlp"]])
        card["mlp_importance_by_family"][l] = round(imp[sel].sum().item(), 4)
    rec = dict(families=fam, mlp_importance=imp.half(),
               sample=dict(h=cf_obs["h"][:128].clone().half(), pre2=cf_obs["pre2"][:128].clone().half(),
                           act=cf_obs["act"][:128].clone().half(), x1_last=cf_obs["x1"][:128, -1].clone().half(),
                           pat1=cf_obs["pat1"][:128, :, -1].clone().half(), pat2=cf_obs["pat2"][:128, :, -1].clone().half()),
               sig={k: v.clone().half() for k, v in sig.items()},
               eta_mlp=families(act, fo)[1].half())
    return card, carry, rec


# ------------------------------------------------------------------ weights
def weight_dynamics(sd, sd_prev, sd_init, upd_prev=None):
    out, upd = {}, {}
    for k, w in sd.items():
        if not torch.is_floating_point(w) or k.startswith("mask"):
            continue
        w0, wp = sd_init[k], sd_prev[k]
        dl = (w - wp).flatten()
        upd[k] = dl
        n = w.norm().item()
        e = dict(norm=round(n, 5),
                 rel_from_init=round(((w - w0).norm() / w0.norm().clamp_min(1e-12)).item(), 5),
                 rel_from_prev=round((dl.norm() / wp.norm().clamp_min(1e-12)).item(), 6),
                 update_sparsity=round(((dl.abs() < 1e-3 * dl.abs().max().clamp_min(1e-12)) ).float().mean().item(), 4)
                 if dl.abs().max() > 0 else 1.0)
        if upd_prev is not None and k in upd_prev and upd_prev[k].norm() > 0 and dl.norm() > 0:
            e["update_cos_prev"] = round(F.cosine_similarity(dl, upd_prev[k], dim=0).item(), 4)
        if w.dim() == 2:
            sv = torch.linalg.svdvals(w)
            e["top_sv"] = round(sv[0].item(), 4)
            e["eff_rank"] = round(((sv.sum() ** 2) / (sv ** 2).sum()).item(), 3)
            dsv = torch.linalg.svdvals((w - wp))
            e["update_top1_share"] = round(((dsv[0] ** 2) / (dsv ** 2).sum().clamp_min(1e-12)).item(), 4)
            U1 = torch.linalg.svd(w, full_matrices=False)[0][:, :3]
            U0 = torch.linalg.svd(wp, full_matrices=False)[0][:, :3]
            e["top3_left_overlap_prev"] = round(subspace_overlap(U0, U1), 4)
        out[k] = e
    return out, upd


def loss_terms(m, fit_lat, src_lat, lam):
    """Differentiable training-objective terms on a fixed FIT batch: task CE, and the M-interchange
    CE at the block (the IIT term)."""
    tb, ts = T.build(fit_lat).tokens, T.build(src_lat).tokens
    _, hb = D.encode(m, tb)
    _, hs = D.encode(m, ts)
    task = F.cross_entropy(m.head_from_resid(hb), T.build(fit_lat).y)
    mk = D.mask(*S.BLOCK)
    iit = F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk), D.counterfactual(fit_lat, src_lat, "M"))
    return task, iit


def grads(m, lam):
    d = partition()
    out = {}
    task, iit = loss_terms(m, d["FIT"], d["FIT_src"], lam)
    for name, L in (("task", task), ("iit", iit)):
        g = torch.autograd.grad(L, list(m.parameters()), allow_unused=True, retain_graph=(name == "task"))
        out[name] = {k: (gi.detach().clone() if gi is not None else torch.zeros_like(p))
                     for (k, p), gi in zip(m.named_parameters(), g)}
        out[name + "_loss"] = L.item()
    return out


def attribution(g_now, g_prev, sd_now, sd_prev):
    """Trapezoid path attribution of the change in each loss term over the interval to tensors,
    attention heads and MLP neurons: sum 0.5 (g_prev + g_now) * (w_now - w_prev). Negative = the
    parameters' change reduced the loss (helped)."""
    out = {}
    for term in ("task", "iit"):
        per_t, per_head, per_neuron = {}, {}, torch.zeros(256)
        tot = 0.0
        for k in g_now[term]:
            c = 0.5 * (g_now[term][k] + g_prev[term][k]) * (sd_now[k] - sd_prev[k])
            v = c.sum().item()
            per_t[k] = round(v, 6)
            tot += v
            for L in ("attn1", "attn2"):
                if k == f"{L}.o.weight":
                    for hd in range(4):
                        per_head[f"{L}.h{hd}"] = per_head.get(f"{L}.h{hd}", 0) + c[:, 16 * hd:16 * (hd + 1)].sum().item()
                if k in (f"{L}.qkv.weight", f"{L}.qkv.bias"):
                    for hd in range(4):
                        rows = [j for part in range(3) for j in range(part * 64 + 16 * hd, part * 64 + 16 * (hd + 1))]
                        per_head[f"{L}.h{hd}"] = per_head.get(f"{L}.h{hd}", 0) + c[rows].sum().item()
            if k in ("W_in.weight", "W_in.bias"):
                per_neuron += c.sum(1) if c.dim() == 2 else c
            if k == "W_out.weight":
                per_neuron += c.sum(0)
        out[term] = dict(total_first_order=round(tot, 6),
                         actual_change=round(g_now[term + "_loss"] - g_prev[term + "_loss"], 6),
                         per_tensor=per_t, per_head={k: round(v, 6) for k, v in per_head.items()},
                         top_neurons=[[int(i), round(per_neuron[i].item(), 6)] for i in per_neuron.argsort()[:8]])
    return out


# ------------------------------------------------------------------ the registered upstream edit
EDIT_TENSORS = ("attn2.o.weight", "attn2.o.bias")
STEPS_REL = (0.25, 0.125, 0.0625)   # candidate magnitudes: ||[dW, db]|| / ||[W_o, b_o] block rows||
RHO = STEPS_REL[0]                  # per-edit budget (the largest candidate)
RHO_CUM = 1.5                       # cumulative budget: sum of accepted relative norms
RIDGE = 1e-3
G_IIA, G_INTER = 0.5, 0.05                    # readiness gates
E_EDIT, E_IIA, E_ACC, E_AGREE, E_OOD, E_ROUTE, E_WRONG = 0.01, 0.01, 0.005, 0.99, 0.01, 0.02, 0.02
E_SHAM, N_SHAM, SHAM_S = 0.005, 3, 7_700    # E9: gain must beat equal-norm random directions


@dataclass
class Proposal:
    step_rel: float
    dW: torch.Tensor       # (16, 64) added to attn2.o.weight[0:16]
    db: torch.Tensor       # (16,)    added to attn2.o.bias[0:16]
    alpha_equiv: float     # fraction of the full interaction-removal solution applied
    full_rel_norm: float   # relative norm of the full (alpha = 1) solution


@torch.no_grad()
def interaction_direction(m, fit_lat=None):
    """Ridge least-squares change of attn2.o rows 0-15 and bias that, on FIT, moves the block of h by
    -I_{M,route}: removes the M x route INTERACTION of the block while keeping the mass main
    effect, the route (unit, position) main effect and the within-cell variation."""
    fit_lat = partition()["FIT"] if fit_lat is None else fit_lat
    c = capture(m, fit_lat)
    fo = factors(fit_lat)
    _, table = interaction_decomp(c["h"][:, BLK], fo["M"], fo["route"])
    target = -table[fo["M"], fo["route"]]
    U = torch.cat([c["pre2"], torch.ones(len(target), 1)], 1).double()
    G = U.T @ U + RIDGE * len(U) * torch.eye(65, dtype=torch.float64)
    Th = torch.linalg.solve(G, U.T @ target.double()).float()
    return Th[:-1].T.contiguous(), Th[-1].contiguous()


def block_row_norm(m):
    return torch.cat([m.attn2.o.weight[BLK].flatten(), m.attn2.o.bias[BLK]]).norm().item()


@torch.no_grad()
def make_proposal(m, step_rel, direction=None):
    dW, db = interaction_direction(m) if direction is None else direction
    full = torch.cat([dW.flatten(), db]).norm().item()
    base = block_row_norm(m)
    s = min(1.0, step_rel * base / max(full, 1e-12))
    return Proposal(step_rel, dW * s, db * s, s, full / base)


@torch.no_grad()
def apply_delta(m, dW, db, sign=1.0):
    m.attn2.o.weight[BLK] += sign * dW
    m.attn2.o.bias[BLK] += sign * db


@torch.no_grad()
def policy_metrics(m):
    """The policy's calibration measurements (FIT for fitting, VAL for scoring, OOD_DEV for the
    development composition check). Never touches OBS, FINAL or OOD_FINAL."""
    d = partition()
    fit, val, vs = d["FIT"], d["VAL"], d["VAL_src"]
    cf, cv, cvs = capture(m, fit), capture(m, val), capture(m, vs)
    fv = factors(val)
    pe = pooled_edit(m, fit, val, h_fit=cf["h"], h_eval=cv["h"])
    dec, _ = interaction_decomp(cv["h"][:, BLK], fv["M"], fv["route"])
    ff = factors(fit)
    route_probe = ridge_probe(cf["h"][:, BLK], ff["route"], cv["h"][:, BLK], fv["route"], 6)
    unit_probe_h = ridge_probe(cf["h"], ff["mu"], cv["h"], fv["mu"], 3)
    return dict(edit=pe["intended"], wrong=pe["wrong"], iia=swap_h(m, val, vs, BLK, "M", cv, cvs),
                acc=accuracy(m, val, cv), pred=cv["logits"].argmax(-1),
                ood_dev=accuracy(m, d["OOD_DEV"]), interaction=dec["interaction_share"],
                route_main=dec["route_share"], route_probe_block=route_probe, unit_probe_h=unit_probe_h)


def checks(before, after, prop, m, cum_used, cum_cap):
    v = dict(step_rel=prop.step_rel, alpha_equiv=round(prop.alpha_equiv, 4), full_rel_norm=round(prop.full_rel_norm, 4),
             rel_norm=round(torch.cat([prop.dW.flatten(), prop.db]).norm().item() / block_row_norm(m), 5),
             E1_edit_gain=after["edit"] - before["edit"],
             E2_iia_change=after["iia"] - before["iia"],
             E3_acc_change=after["acc"] - before["acc"],
             E4_agreement=(after["pred"] == before["pred"]).float().mean().item(),
             E5_ood_dev_change=after["ood_dev"] - before["ood_dev"],
             E6_route_info_change=after["route_probe_block"] - before["route_probe_block"],
             E6b_unit_info_change=after["unit_probe_h"] - before["unit_probe_h"],
             E7_wrong_change=after["wrong"] - before["wrong"],
             interaction_before=before["interaction"], interaction_after=after["interaction"],
             edit_before=before["edit"], edit_after=after["edit"], iia_before=before["iia"],
             acc_before=before["acc"], ood_dev_before=before["ood_dev"])
    v["E1"] = v["E1_edit_gain"] >= E_EDIT
    v["E2"] = v["E2_iia_change"] >= -E_IIA
    v["E3"] = v["E3_acc_change"] >= -E_ACC
    v["E4"] = v["E4_agreement"] >= E_AGREE
    v["E5"] = v["E5_ood_dev_change"] >= -E_OOD
    v["E6"] = v["E6_route_info_change"] >= -E_ROUTE and v["E6b_unit_info_change"] >= -E_ROUTE
    v["E7"] = v["E7_wrong_change"] <= E_WRONG
    v["E8"] = bool(torch.isfinite(prop.dW).all() and torch.isfinite(prop.db).all()
                   and v["rel_norm"] <= RHO + 1e-6 and (cum_cap is None or cum_used + v["rel_norm"] <= cum_cap + 1e-9))
    v["passed"] = all(v[k] for k in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"))
    return v


def random_directions(n=N_SHAM):
    g = torch.Generator().manual_seed(SHAM_S)
    return [(torch.randn(16, 64, generator=g), torch.randn(16, generator=g)) for _ in range(n)]


@torch.no_grad()
def sham_gain(m, prop, before):
    """E9: mean pooled-edit gain of N_SHAM fixed random directions with the candidate's norm."""
    nrm = torch.cat([prop.dW.flatten(), prop.db]).norm()
    gains = []
    for rW, rb in random_directions():
        rn = torch.cat([rW.flatten(), rb]).norm()
        aW, ab = rW * nrm / rn, rb * nrm / rn
        apply_delta(m, aW, ab)
        gains.append(policy_metrics(m)["edit"] - before["edit"])
        apply_delta(m, aW, ab, -1.0)
    return sum(gains) / len(gains), gains


def _comparators(m, prop, before):
    """Evaluation-only comparators of equal norm (never applied): (a) the IIT-gradient direction on
    the same rows (what 'more IIT' would do), (b) a fixed random direction."""
    out = dict(step_rel=prop.step_rel)
    d = partition()
    nrm = torch.cat([prop.dW.flatten(), prop.db]).norm()
    mm = copy.deepcopy(m)
    with torch.enable_grad():
        _, iit = loss_terms(mm, d["FIT"], d["FIT_src"], 0)
        gW, gb = torch.autograd.grad(iit, [mm.attn2.o.weight, mm.attn2.o.bias])
    gW, gb = -gW[BLK], -gb[BLK]
    gn = torch.cat([gW.flatten(), gb]).norm().clamp_min(1e-12)
    g = torch.Generator().manual_seed(77)
    rW, rb = torch.randn(16, 64, generator=g), torch.randn(16, generator=g)
    rn = torch.cat([rW.flatten(), rb]).norm()
    for name, (aW, ab) in (("iit_grad", (gW * nrm / gn, gb * nrm / gn)), ("random", (rW * nrm / rn, rb * nrm / rn))):
        apply_delta(mm, aW, ab)
        a = policy_metrics(mm)
        apply_delta(mm, aW, ab, -1.0)
        out[name] = dict(edit_gain=round(a["edit"] - before["edit"], 4),
                         interaction_change=round(a["interaction"] - before["interaction"], 4),
                         acc_change=round(a["acc"] - before["acc"], 4),
                         agreement=round((a["pred"] == before["pred"]).float().mean().item(), 4))
    return out


def decide(m, cum_used=0.0, cum_cap=None, comparators=True):
    """The registered policy at one checkpoint. Works on a COPY of the model; never mutates `m`.
    Returns a decision record and the accepted (dW, db) or None."""
    mm = copy.deepcopy(m)
    before = policy_metrics(mm)
    rec = dict(before={k: v for k, v in before.items() if k != "pred"})
    if before["iia"] < G_IIA:
        rec.update(status="skipped", reason=f"no representation yet: VAL block IIA(M) {before['iia']:.3f} < {G_IIA}")
        return rec, None
    if before["interaction"] < G_INTER:
        rec.update(status="skipped", reason=f"nothing to organize: interaction share {before['interaction']:.3f} < {G_INTER}")
        return rec, None
    if cum_cap is not None and cum_used >= cum_cap - 1e-9:
        rec.update(status="skipped", reason="cumulative edit budget exhausted")
        return rec, None
    tried, chosen = [], None
    direction = interaction_direction(mm)
    for a in STEPS_REL:
        prop = make_proposal(mm, a, direction)
        apply_delta(mm, prop.dW, prop.db)
        after = policy_metrics(mm)
        apply_delta(mm, prop.dW, prop.db, -1.0)
        v = checks(before, after, prop, mm, cum_used, cum_cap)
        if v["passed"]:          # E9 is only evaluated when E1-E8 pass (it is the costliest check)
            sg, gl = sham_gain(mm, prop, before)
            v.update(E9_sham_mean_gain=sg, E9_sham_gains=gl, E9_margin=v["E1_edit_gain"] - sg)
            v["E9"] = v["E9_margin"] >= E_SHAM
            v["passed"] = v["E9"]
        tried.append(v)
        if v["passed"]:
            chosen = prop
            break
    rec["candidates"] = tried
    if comparators:
        rec["comparators"] = _comparators(mm, chosen or prop, before)
    if chosen is None:
        fails = sorted({k for v in tried for k in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9") if not v.get(k, True)})
        rec.update(status="rejected", reason="no candidate magnitude passed; failing checks: " + ",".join(fails))
        return rec, None
    rec.update(status="accepted", step_rel=chosen.step_rel, alpha_equiv=chosen.alpha_equiv, rel_norm=tried[-1]["rel_norm"])
    return rec, (chosen.dW.clone(), chosen.db.clone())


# ------------------------------------------------------------------ registered atlas criteria (docs/24 §24.7)
CONCEPTS = {
    "mass (canonical M)": ("M", ["iia_M_block", "iia_M_rest", "iia_M_head0", "iia_M_head1", "iia_M_head2",
                                 "iia_M_head3", "iia_M_L1_val+unit"]),
    "acceleration (canonical A)": ("A", ["iia_A_block", "iia_A_rest", "iia_A_head0", "iia_A_head1", "iia_A_head2",
                                         "iia_A_head3", "iia_A_L1_val+unit"]),
    "raw mass value m": ("m", ["iia_m_L1_val"]),
    "raw acceleration value a": ("a", ["iia_a_L1_val"]),
    "mass unit": ("mu", ["iia_mu_L1_unit"]),
    "acceleration unit": ("au", ["iia_au_L1_unit"]),
    "output unit / factor S": ("fu", ["iia_S_rest"]),
    "mass route (unit x position)": ("route", []),
    "product M*A (intermediate)": ("MA", []),
    "answer F": ("F", []),
    "token order": ("order", []),
}
K_DECODE, K_WEAK, IIA_CAUSAL, IIA_VERIFIED = 0.5, 0.2, 0.5, 0.9
SUSTAIN = 3            # a criterion must hold at a checkpoint and the next SUSTAIN-1 checkpoints
COAL_RATIO, COAL_SUSTAIN = 0.8, 5


def concept_state(card, concept):
    fac, tests = CONCEPTS[concept]
    dec = {k.split("@", 1)[1]: v for k, v in card["decode_kappa"].items() if k.split("@", 1)[0] == fac}
    best_site = max(dec, key=dec.get)
    cz = {t: card["causal"][t] for t in tests}
    best_c = max(cz, key=cz.get) if cz else None
    kc = dec[best_site]
    cv = cz[best_c] if best_c else None
    if cv is not None and cv >= IIA_VERIFIED:
        status = "causal (verified, IIA >= 0.9)"
    elif cv is not None and cv >= IIA_CAUSAL:
        status = "causal (IIA >= 0.5)"
    elif kc >= K_DECODE:
        status = "decodable only" + ("" if tests else " (no causal test registered)")
    elif kc >= K_WEAK:
        status = "weakly decodable"
    else:
        status = "absent"
    return dict(status=status, best_decode_site=best_site, best_kappa=round(kc, 3),
                decodable_sites=sorted(s for s, v in dec.items() if v >= K_DECODE),
                best_causal_test=best_c, best_iia=None if cv is None else round(cv, 3),
                causal_sites=sorted(t for t, v in cz.items() if v >= IIA_CAUSAL))


def first_sustained(steps, flags, n=SUSTAIN):
    for i in range(len(flags)):
        win = flags[i:i + n]
        if win and all(win) and (len(win) == n or i + len(win) == len(flags)):
            return steps[i]
    return None


def registered_events(cards):
    """Emergence / identification / coalescence timing (registered definitions) from the ordered
    per-checkpoint observation cards of one trajectory."""
    steps = [c["step"] for c in cards]
    ob = [c["observe"] for c in cards]
    ev = {}
    for concept, (fac, tests) in CONCEPTS.items():
        ks = [max(v for k, v in o["decode_kappa"].items() if k.split("@", 1)[0] == fac) for o in ob]
        ev[f"{concept}: first decodable (kappa >= 0.5, sustained)"] = first_sustained(steps, [k >= K_DECODE for k in ks])
        if tests:
            cs = [max(o["causal"][t] for t in tests) for o in ob]
            ev[f"{concept}: first causal (IIA >= 0.5, sustained)"] = first_sustained(steps, [c >= IIA_CAUSAL for c in cs])
            ev[f"{concept}: first causally verified (IIA >= 0.9, sustained)"] = first_sustained(steps, [c >= IIA_VERIFIED for c in cs])
    ev["MASS emergence (acc >= 0.5 and block IIA(M) >= 0.5, sustained)"] = first_sustained(
        steps, [o["acc_obs"] >= 0.5 and o["causal"]["iia_M_block"] >= 0.5 for o in ob])
    ev["MASS identified, C1 (block IIA(M) >= 0.9 and acc >= 0.95, sustained)"] = first_sustained(
        steps, [o["acc_obs"] >= 0.95 and o["causal"]["iia_M_block"] >= 0.9 for o in ob])
    ev["MASS outside the block causal (IIA(M) on dims 16-63 >= 0.5, sustained)"] = first_sustained(
        steps, [o["causal"]["iia_M_rest"] >= 0.5 for o in ob])
    for site in ("h_block", "h"):
        flags = []
        for o in ob:
            t = o["unit_transfer"][f"M@{site}"]
            ratio = t["transfer"] / max(t["within"], 1e-9)
            flags.append(ratio >= COAL_RATIO and o["causal"]["iia_M_block"] >= IIA_VERIFIED)
        ev[f"MASS coalescence at {site} (unit-transfer ratio >= 0.8 with block IIA >= 0.9, 5 checkpoints)"] = \
            first_sustained(steps, flags, COAL_SUSTAIN)
    return ev


def equivalence_regime(card, site="h_block"):
    t = card["unit_transfer"][f"M@{site}"]
    r = t["transfer"] / max(t["within"], 1e-9)
    return ("shared" if r >= COAL_RATIO else "partially shared" if r >= 0.5 else "unit-dependent"), round(r, 3)
