"""Unit-calculus analysis (registration: docs/21). Frozen before the main networks are analysed.

Per network:
  A1 readability   linear probes (fit questions -> reserved questions) for every quantity at every
                   role-aligned site, per mode
  A2 geometry      character spectra (multiplicative = discrete-log circles over the 12 nonzero
                   classes; additive = value-order circles over all 13) of the calculator
                   intermediate and of the ANSWER, at the query position before and after the MLP
  A3 algorithm     CORE. Causal comparison of two candidate algorithms at each site feeding the
                   calculator, with interchange interventions on a learned 16-dim subspace (DAS):
                     convert-first: the converted quantity (value x label scale / + offset) exists
                     route:         raw value and raw label are separately swappable
                   For each operand slot, three counterfactuals: canonical quantity from the
                   source; raw value only from the source (label from the base); label only.
                   route index = min(IIA value-only, IIA label-only)
                   Built-in controls: L0 embeddings (route by construction); the declared arm's
                   16-dim block in mul mode (canonical mass by construction).
  A4 composition   accuracy on held-out label pairs per mode
  A5 edits         one-row embedding edit giving a label a NEW meaning (a parameter no label in
                   its dimension has); benchmark = copying another label's row for an EXISTING
                   meaning. Success: intended >= 0.9 x benchmark.

    python analyse_unitcalc.py --zoo runs/zoo_unitcalc --out results/unitcalc/metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

import run_unitcalc as R
from nri import unitcalc as U
from nri.train import set_determinism

P = U.P
FIT_S, EVAL_S, HELD_S, EDIT_S = 31_001, 32_002, 33_003, 34_004
N = 16_384
RESERVE_MOD, RESERVE_LT = 10, 3          # 30% of questions reserved from every fitted component


# ------------------------------------------------------------------ data
def qkey(lat):
    """Deterministic integer key of each question (for the reserved split)."""
    k = torch.zeros_like(lat["mode"])
    for name, base in (("mode", 4), ("x1", 13), ("u1", 6), ("x2", 13), ("u2", 6), ("uo", 6), ("order", 2)):
        k = k * base + lat[name]
    return k


def reserved(lat):
    return (qkey(lat) * 2654435761 % 1_000_003) % RESERVE_MOD < RESERVE_LT


def draw(n, seed, kind, modes=U.MODES):
    """kind: 'fit' (train-split, non-reserved), 'eval' (train-split, reserved), 'held'."""
    g = torch.Generator().manual_seed(seed)
    parts, have = [], 0
    while have < n:
        lat = U.sample(4 * n, g, "held" if kind == "held" else "train", modes)
        keep = torch.ones_like(lat["mode"], dtype=torch.bool) if kind == "held" else \
            (reserved(lat) if kind == "eval" else ~reserved(lat))
        parts.append({k: v[keep] for k, v in lat.items()})
        have += int(keep.sum())
    return {k: torch.cat([p[k] for p in parts])[:n] for k in parts[0]}


def subset(lat, sel):
    return {k: v[sel] for k, v in lat.items()}


@torch.no_grad()
def sites(m, tok):
    z = torch.zeros_like(tok)
    x0 = m.embed(tok, z)
    x1 = x0 + m.attn1(x0)
    x2 = x1 + m.attn2(x1)
    h = x2[:, -1]
    r = h + m.W_out(torch.relu(m.W_in(h)))
    return dict(L0=x0, L1=x1, L2=x2, h=h, r=r)


def role_pos(lat):
    """Role-aligned positions: op1 = mass (mul) / first operand (others); op2 = acc / second."""
    swap = (lat["mode"] == 0) & (lat["order"] == 1)
    one = torch.ones_like(lat["mode"])
    return dict(op1_val=torch.where(swap, 3 * one, 0 * one), op1_lab=torch.where(swap, 4 * one, 1 * one),
                op2_val=torch.where(swap, 0 * one, 3 * one), op2_lab=torch.where(swap, 1 * one, 4 * one),
                out_lab=6 * one)


def site_vectors(S, lat):
    pos = role_pos(lat)
    ar = torch.arange(len(lat["mode"]))
    out = {}
    for L in ("L1", "L2"):
        for role, p in pos.items():
            out[f"{L}:{role}"] = S[L][ar, p]
        out[f"{L}:query"] = S[L][:, -1]
    out["L3:query"] = S["r"]
    return out


def quantities(lat):
    q = U.quantities(lat)
    mode = lat["mode"]
    op1_c = torch.stack([q["M_mul"], q["F_div"], q["M1"], q["T1"]], 1)[torch.arange(len(mode)), mode]
    op2_c = torch.stack([q["A_mul"], q["M_div"], q["M2"], q["T2"]], 1)[torch.arange(len(mode)), mode]
    calc = torch.stack([q["M_mul"] * q["A_mul"] % P, q["F_div"] * U.INV[q["M_div"]] % P,
                        (q["M1"] + q["M2"]) % P, (q["T1"] + q["T2"]) % P], 1)[torch.arange(len(mode)), mode]
    return {"op1 value": lat["x1"], "op1 label": lat["u1"], "op1 canonical": op1_c,
            "op2 value": lat["x2"], "op2 label": lat["u2"], "op2 canonical": op2_c,
            "calculator": calc, "answer": q["y"]}


# ------------------------------------------------------------------ A1 probes
def probe(Xf, yf, Xe, ye):
    A = torch.cat([Xf, torch.ones(len(Xf), 1)], 1).double()
    Y = F.one_hot(yf, P).double()
    W = torch.linalg.lstsq(A, Y, driver="gelsd").solution
    return ((torch.cat([Xe, torch.ones(len(Xe), 1)], 1).double() @ W).argmax(1) == ye).float().mean().item()


# ------------------------------------------------------------------ A2 character geometry
def mult_basis():
    """Orthonormal real basis over the 12 nonzero classes, grouped by multiplicative frequency."""
    xs = list(range(1, P))
    k = torch.tensor([U.LOG[x] for x in xs], dtype=torch.double)
    groups = {}
    for f in range(1, 7):
        cols = [torch.cos(2 * math.pi * f * k / 12)] + ([torch.sin(2 * math.pi * f * k / 12)] if f < 6 else [])
        groups[f] = torch.stack(cols, 1) / math.sqrt(6 if f < 6 else 12)
    return xs, groups


def add_basis():
    xs = list(range(P))
    k = torch.arange(P, dtype=torch.double)
    return xs, {f: torch.stack([torch.cos(2 * math.pi * f * k / P), torch.sin(2 * math.pi * f * k / P)], 1)
                / math.sqrt(P / 2) for f in range(1, 7)}


def spectrum(means, kind):
    xs, groups = mult_basis() if kind == "mult" else add_basis()
    Mu = means[xs].double()
    Mu = Mu - Mu.mean(0)
    tot = (Mu ** 2).sum()
    return {f: ((B.T @ Mu) ** 2).sum().item() / max(tot.item(), 1e-12) for f, B in groups.items()}


def class_means(X, y):
    return torch.stack([X[y == c].mean(0) if (y == c).any() else torch.zeros(X.shape[1]) for c in range(P)])


def top2(spec):
    return sum(sorted(spec.values(), reverse=True)[:2])


# ------------------------------------------------------------------ A3 label-meaning recovery
SLOTS = {  # (mode, slot) -> (label dimension, expected sign of the label's step in its operand)
    ("mul", "u1"): ("mass", +1), ("mul", "u2"): ("acc", +1),
    ("div", "u1"): ("force", +1), ("div", "u2"): ("mass", +1),
    ("msum", "u1"): ("mass", +1), ("msum", "u2"): ("mass", +1),
    ("tsum", "u1"): ("temp", +1), ("tsum", "u2"): ("temp", +1)}
VALUE_OF = {"u1": "x1", "u2": "x2"}


def true_table():
    """Label parameters of the ACTIVE label set (set by run_unitcalc.load)."""
    return {"mass": U.MASS_LOG, "acc": U.ACC_LOG, "force": U.FORCE_LOG,
            "acc_out": U.ACC_LOG[:U.N_ACC_OUT], "temp": U.TEMP_OFF}
def kind_of(dim):
    return "add" if dim == "temp" else "mult"


# ------------------------------------------------------------------ A3 algorithm comparison (core)
K_DAS, DAS_STEPS, DAS_LR, N_PAIRS = 16, 400, 1e-2, 8192
ALGO_SITES = ("L0:val", "L1:val", "L1:lab", "L1:query", "L2:query")
VARS = ("canon", "value", "label")


def pairs(lat, seed):
    """Source for each base question: a random question of the same mode and word order."""
    g = torch.Generator().manual_seed(seed)
    src = torch.empty_like(lat["mode"])
    key = lat["mode"] * 2 + lat["order"]
    for k_ in key.unique():
        idx = (key == k_).nonzero().squeeze(1)
        src[idx] = idx[torch.randperm(len(idx), generator=g)]
    return subset(lat, src)


def counterfactual(base, src, slot, var):
    """Answer of the base question when, for operand `slot`, only `var` comes from the source:
    canon = the converted quantity (value under its label); value = raw value (base label);
    label = label (base value)."""
    if var in ("value", "label"):
        cf = dict(base)
        key = VALUE_OF[slot] if var == "value" else slot
        cf[key] = src[key]
        return U.build(cf).y
    qb, qs = quantities(base), quantities(src)
    op1 = qs["op1 canonical"] if slot == "u1" else qb["op1 canonical"]
    op2 = qs["op2 canonical"] if slot == "u2" else qb["op2 canonical"]
    p_ = U._params(False)
    mode = base["mode"]
    y = torch.zeros_like(mode)
    y = torch.where(mode == 0, op1 * op2 * U.INV[p_["force"][base["uo"]]] % P, y)
    y = torch.where(mode == 1, op1 * U.INV[op2] * U.INV[p_["acc"][base["uo"]]] % P, y)
    y = torch.where(mode == 2, (op1 + op2) * U.INV[p_["mass"][base["uo"]]] % P, y)
    y = torch.where(mode == 3, (op1 + op2 - p_["temp"][base["uo"]]) % P, y)
    return y


def site_position(lat, slot, site):
    pos = role_pos(lat)
    if site.endswith(":val"):
        return pos["op1_val" if slot == "u1" else "op2_val"]
    if site.endswith(":lab"):
        return pos["op1_lab" if slot == "u1" else "op2_lab"]
    return torch.full_like(lat["mode"], 7)


def patched_logits(m, xb, xs, site, posb, Q, block=None):
    """Swap span(Q) of the source activation into the base at `site`, then run to the output.
    xb/xs: the stage activations (L0 or L1 residual streams, or h for L2:query)."""
    ar = torch.arange(len(posb))
    if site == "L2:query":
        hb = xb + (xs - xb) @ Q @ Q.T
        return m.head_from_resid(hb)
    x = xb.clone()
    vb, vs = xb[ar, posb], xs[ar, posb]
    x[ar, posb] = vb + (vs - vb) @ Q @ Q.T
    if site.startswith("L0"):
        x = x + m.attn1(x)
    x = x + m.attn2(x)
    return m.head_from_resid(x[:, -1])


def stage(m, tok, site):
    z = torch.zeros_like(tok)
    x0 = m.embed(tok, z)
    if site.startswith("L0"):
        return x0
    x1 = x0 + m.attn1(x0)
    if site.startswith("L1"):
        return x1
    return (x1 + m.attn2(x1))[:, -1]


def das_iia(m, base_f, src_f, base_e, src_e, slot, site, var, seed, dims=None):
    """Learn a K_DAS-dim orthogonal subspace (inside `dims` if given) on fit pairs; IIA on
    reserved pairs where the counterfactual differs from the base answer."""
    d = m.d
    with torch.no_grad():
        xb_f, xs_f = stage(m, U.build(base_f).tokens, site), stage(m, U.build(src_f).tokens, site)
        xb_e, xs_e = stage(m, U.build(base_e).tokens, site), stage(m, U.build(src_e).tokens, site)
    pf, pe = site_position(base_f, slot, site), site_position(base_e, slot, site)
    yf, ye = counterfactual(base_f, src_f, slot, var), counterfactual(base_e, src_e, slot, var)
    y0e = U.build(base_e).y
    torch.manual_seed(seed)
    width = d if dims is None else len(dims)
    rot = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(width, K_DAS, bias=False))
    embed_dims = torch.eye(d)[:, dims] if dims is not None else torch.eye(d)
    opt = torch.optim.Adam(rot.parameters(), lr=DAS_LR)
    for p_ in m.parameters():
        p_.requires_grad_(False)
    g = torch.Generator().manual_seed(seed)
    for _ in range(DAS_STEPS):
        i = torch.randint(0, len(yf), (1024,), generator=g)
        Q = embed_dims @ rot.weight.T
        loss = F.cross_entropy(patched_logits(m, xb_f[i], xs_f[i], site, pf[i], Q), yf[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p_ in m.parameters():
        p_.requires_grad_(True)
    with torch.no_grad():
        Q = embed_dims @ rot.weight.T
        pred = patched_logits(m, xb_e, xs_e, site, pe, Q).argmax(-1)
        elig = ye != y0e
        # random-subspace baseline of the same size
        Rq, _ = torch.linalg.qr(torch.randn(width, K_DAS, generator=torch.Generator().manual_seed(seed + 1)))
        pred_r = patched_logits(m, xb_e, xs_e, site, pe, embed_dims @ Rq).argmax(-1)
    return dict(iia=(pred == ye)[elig].float().mean().item(),
                unchanged=(pred == y0e)[elig].float().mean().item(),
                random_iia=(pred_r == ye)[elig].float().mean().item(), n=int(elig.sum()))


def algorithm(m, fit, ev, arm):
    """A3 for every mode x operand slot x site x variable."""
    out = {}
    for mi, mode in enumerate(U.MODES):
        bf = subset(fit, fit["mode"] == mi)
        be = subset(ev, ev["mode"] == mi)
        bf, be = subset(bf, torch.arange(min(N_PAIRS, len(bf["mode"])))), subset(be, torch.arange(min(N_PAIRS, len(be["mode"]))))
        sf, se = pairs(bf, 41), pairs(be, 42)
        for slot in ("u1", "u2"):
            for site in ALGO_SITES:
                r = {var: das_iia(m, bf, sf, be, se, slot, site, var, seed=100 + mi * 10 + (slot == "u2"))
                     for var in VARS}
                r["route_index"] = min(r["value"]["iia"], r["label"]["iia"])
                out[f"{mode}:{slot}:{site}"] = r
    if arm == "declared":   # positive control: the declared canonical-mass block (mul mode)
        mi = 0
        bf, be = subset(fit, fit["mode"] == mi), subset(ev, ev["mode"] == mi)
        bf, be = subset(bf, torch.arange(min(N_PAIRS, len(bf["mode"])))), subset(be, torch.arange(min(N_PAIRS, len(be["mode"]))))
        sf, se = pairs(bf, 41), pairs(be, 42)
        blk = list(range(*R.BLOCKS["M"]))
        r = {var: das_iia(m, bf, sf, be, se, "u1", "L2:query", var, seed=7, dims=blk) for var in VARS}
        r["route_index"] = min(r["value"]["iia"], r["label"]["iia"])
        out["control:declared_block_M"] = r
    return out


def row_edit(m, mode, slot, u, target_step, fit, ev, steps=300, lr=0.05):
    """Give label `u` (in `slot` of `mode`) the parameter true[u] + target_step by changing ONLY
    its embedding row. Fit on fit questions; score on reserved questions with that label."""
    mi = U.MODES.index(mode)
    dim, _ = SLOTS[(mode, slot)]
    tok_id = {"mass": U.MASS_U, "acc": U.ACC_U, "force": U.FORCE_U, "temp": U.TEMP_U}[dim][u]

    def answers(lat, new_param):
        with with_param(dim, u, new_param):
            return U.build(lat).y

    true = true_table()[dim][u]
    order = 12 if dim != "temp" else P
    new = (true + target_step) % order
    f_lat = subset(fit, (fit["mode"] == mi) & (fit[slot] == u))
    e_lat = subset(ev, (ev["mode"] == mi) & (ev[slot] == u))
    y_new_f = answers(f_lat, new)
    E = m.E_c.weight
    row = E[tok_id].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([row], lr=lr)
    tok_f = U.build(f_lat).tokens
    for p in m.parameters():
        p.requires_grad_(False)
    for _ in range(steps):
        with torch.no_grad():
            pass
        W = E.detach().clone()
        Wn = torch.cat([W[:tok_id], row[None], W[tok_id + 1:]])
        x = F.embedding(tok_f, Wn)
        # replicate TinyTyped.embed with the substituted content table
        x = x + m.E_t(torch.zeros_like(tok_f)) + m.E_p.weight[: tok_f.shape[1]]
        x = x + m.attn1(x)
        x = x + m.attn2(x)
        loss = F.cross_entropy(m.head_from_resid(x[:, -1]), y_new_f)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in m.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        saved = E[tok_id].clone()
        E[tok_id] = row.detach()
        res = row_edit_score(m, mode, slot, u, new, ev)
        E[tok_id] = saved
    res["rel_change"] = ((row.detach() - saved).norm() / saved.norm()).item()
    return res


def with_param(dim, u, value):
    """Context: label u of `dim` temporarily has parameter `value` (token-level meaning)."""
    import contextlib

    @contextlib.contextmanager
    def ctx():
        name = {"mass": "MASS_LOG", "acc": "ACC_LOG", "force": "FORCE_LOG", "temp": "TEMP_OFF"}[dim]
        saved = getattr(U, name)
        setattr(U, name, tuple(value if i == u else x for i, x in enumerate(saved)))
        try:
            yield
        finally:
            setattr(U, name, saved)
    return ctx()


@torch.no_grad()
def row_edit_score(m, mode, slot, u, new_param, ev):
    """Score the CURRENT model on reserved questions of `mode` that contain label u in any
    slot of its dimension: intended = answer if u had new_param everywhere."""
    mi = U.MODES.index(mode)
    dim, _ = SLOTS[(mode, slot)]
    e_lat = subset(ev, (ev["mode"] == mi) & (ev[slot] == u))
    y_old = U.build(e_lat).y
    with with_param(dim, u, new_param):
        y_new = U.build(e_lat).y
    pred = m.head_from_resid(R.encode(m, U.build(e_lat).tokens)).argmax(-1)
    elig = y_new != y_old
    return dict(n=int(elig.sum()), intended=(pred == y_new)[elig].float().mean().item(),
                unchanged=(pred == y_old)[elig].float().mean().item())


def check_embed_matches(m):
    tok = U.build(U.sample(8, torch.Generator().manual_seed(1))).tokens
    a = m.embed(tok, torch.zeros_like(tok))
    b = F.embedding(tok, m.E_c.weight) + m.E_t(torch.zeros_like(tok)) + m.E_p.weight[: tok.shape[1]]
    return torch.allclose(a, b, atol=1e-6)


# ------------------------------------------------------------------ one network
def analyse(run):
    set_determinism()
    m, cfg = R.load(run)
    assert check_embed_matches(m), "embedding replica does not match the model"
    fit, ev, held = draw(N, FIT_S, "fit"), draw(N, EVAL_S, "eval"), draw(N // 2, HELD_S, "held")
    out = dict(run=Path(run).name, arm=cfg["arm"], seed=cfg["seed"], acc=cfg["acc"])
    with torch.no_grad():
        Sf, Se = sites(m, U.build(fit).tokens), sites(m, U.build(ev).tokens)
        Vf, Ve = site_vectors(Sf, fit), site_vectors(Se, ev)
        Qf, Qe = quantities(fit), quantities(ev)
        # A1 readability
        out["probe"] = {}
        for mi, mode in enumerate(U.MODES):
            sf, se = fit["mode"] == mi, ev["mode"] == mi
            out["probe"][mode] = {s: {q: probe(Vf[s][sf], Qf[q][sf], Ve[s][se], Qe[q][se]) for q in Qf}
                                  for s in Vf}
        # A2 geometry: calculator intermediate and answer, before and after the MLP
        out["geometry"] = {}
        for mi, mode in enumerate(U.MODES):
            sf = fit["mode"] == mi
            g = {}
            for site in ("L1:query", "L2:query", "L3:query"):
                X = Vf[site][sf]
                for q in ("calculator", "answer"):
                    sm = spectrum(class_means(X, Qf[q][sf]), "mult")
                    sa = spectrum(class_means(X, Qf[q][sf]), "add")
                    g[f"{site}|{q}"] = dict(mult=sm, add=sa, mult_top2=top2(sm), add_top2=top2(sa))
            out["geometry"][mode] = g
        # A4 composition: held-out label pairs
        Sh = sites(m, U.build(held).tokens)
        pred_h = m.head_from_resid(Sh["h"]).argmax(-1)
        y_h = U.build(held).y
        out["held"] = {mode: (pred_h == y_h)[held["mode"] == mi].float().mean().item()
                       for mi, mode in enumerate(U.MODES)}
    # A3 algorithm comparison (core)
    out["algorithm"] = algorithm(m, fit, ev, cfg["arm"])
    # A5 one-row embedding edits: a NEW meaning (a parameter no label in the dimension has)
    out["row_edit"] = {}
    if cfg["arm"] != "meaningless":
        for mode, slot, u in (("mul", "u1", 1), ("mul", "u2", 1), ("tsum", "u1", 1)):
            dim = SLOTS[(mode, slot)][0]
            order = 12 if dim != "temp" else P
            T_ = true_table()[dim]
            used = set(T_)
            novel = next(s for s in range(1, order) if (T_[u] + s) % order not in used)
            existing = next(s for s in range(1, order) if (T_[u] + s) % order in used
                            and (T_[u] + s) % order != T_[u])
            out["row_edit"][f"{mode}:{slot}:{u}"] = dict(
                novel=row_edit(m, mode, slot, u, novel, fit, ev),
                copy_existing=copy_row(m, mode, slot, u, existing, fit, ev))
    return out


@torch.no_grad()
def copy_row(m, mode, slot, u, step, fit, ev):
    """Copy benchmark for row edits: give label u an EXISTING meaning (that of label v) by
    copying v's embedding row. Scored like row_edit: the token's meaning changes everywhere."""
    dim, _ = SLOTS[(mode, slot)]
    order = 12 if dim != "temp" else P
    T_ = true_table()[dim]
    v = T_.index((T_[u] + step) % order)
    ids = {"mass": U.MASS_U, "acc": U.ACC_U, "force": U.FORCE_U, "temp": U.TEMP_U}[dim]
    E = m.E_c.weight
    saved = E[ids[u]].clone()
    E[ids[u]] = E[ids[v]]
    try:
        return row_edit_score(m, mode, slot, u, T_[v], ev)
    finally:
        E[ids[u]] = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    runs = a.runs or sorted(str(p) for p in Path(a.zoo).glob("*") if (p / "model.pt").exists())
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(analyse, runs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"analysed {len(res)} networks -> {a.out}")


if __name__ == "__main__":
    main()
