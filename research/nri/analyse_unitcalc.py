"""Unit-calculus analysis (registration: docs/21). Frozen before the main networks are analysed.

Per network:
  A1 readability   linear probes (fit questions -> reserved questions) for every quantity at every
                   role-aligned site, per mode
  A2 geometry      class-mean character spectra of each mode's calculator intermediate at the
                   query site before the MLP (h): multiplicative (discrete-log circles, nonzero
                   classes) vs additive (value-order circles, all 13 classes)
  A3 label table   recover each label's meaning from the network: swap ONLY the label token,
                   measure the rotation of h in the calculator's character plane (plane fitted on
                   the network's own predicted answers, never on the true table), convert to a
                   log-scale / offset step. Direction (inverse labels, divisors, output labels)
                   is recovered as the sign / value-dependence of the rotation.
  A4 composition   accuracy on held-out label pairs; agreement of the network's held-out answers
                   with the answers implied by the recovered table
  A5 convert-first canonical-operand readability at the operand's label site, msum vs mul
  A6 edits         (a) map-guided relabel: rotate h in the character planes by the recovered
                   step; copy benchmark = the real relabelled question's h
                   (b) one-row embedding edit giving a label a NEW meaning (a log-scale no label
                   in its dimension has); copy benchmark = copying another label's row for an
                   existing meaning

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
def plane(X, y, kind, f):
    """Complex coordinate map z(x) for character frequency f, fitted from class means of X by
    the class labels y (here: the network's own predicted answers). Returns (W, mean) with
    z = (x - mean) @ W[:,0] + i (x - mean) @ W[:,1]."""
    mu = class_means(X, y).double()
    if kind == "mult":
        xs = list(range(1, P))
        ang = torch.tensor([2 * math.pi * f * U.LOG[x] / 12 for x in xs], dtype=torch.double)
    else:
        xs = list(range(P))
        ang = torch.tensor([2 * math.pi * f * x / P for x in xs], dtype=torch.double)
    target = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    Mx = mu[xs] - mu[xs].mean(0)
    W = torch.linalg.lstsq(Mx, target - target.mean(0), driver="gelsd").solution
    return W, mu[xs].mean(0)


def zc(X, W, m0):
    v = (X.double() - m0) @ W
    return torch.complex(v[:, 0], v[:, 1])


def best_freq(spec, kind):
    """Strongest frequency whose rotation is invertible: coprime to 12 (mult) or any (add)."""
    ok = [f for f in spec if (math.gcd(f, 12) == 1 if kind == "mult" else True)]
    return max(ok, key=lambda f: spec[f])


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
GROUP_ORDER = {"mult": 12, "add": P}
LABEL_SITES = ("L1:val", "L2:val", "L2:query", "L3:query")
PRIMARY_SITE = "L2:query"      # the MLP input: before the calculator


def kind_of(dim):
    return "add" if dim == "temp" else "mult"


def slot_vectors(S, lat, slot, site):
    """Activation at `site` for the operand that `slot` labels (role-aligned)."""
    pos = role_pos(lat)
    ar = torch.arange(len(lat["mode"]))
    if site == "L2:query":
        return S["h"]
    if site == "L3:query":
        return S["r"]
    L, _ = site.split(":")
    return S[L][ar, pos["op1_val" if slot == "u1" else "op2_val"]]


def value_plane(m, fit, mode, slot, site, kind):
    """Plane in which the operand's RAW VALUE class means lie on a character circle (value
    angles come from number theory: discrete log or the value itself; no unit table used).
    Frequency: the strongest invertible one in the value spectrum at that site."""
    mi = U.MODES.index(mode)
    sel = fit["mode"] == mi
    if mode == "mul" and slot == "u2":
        sel = sel & ~torch.tensor(U.ACC_INVERSE)[fit["u2"]]       # direct labels define the circle
    lat = subset(fit, sel)
    X = slot_vectors(sites(m, U.build(lat).tokens), lat, slot, site)
    y = lat[VALUE_OF[slot]]
    spec = spectrum(class_means(X, y), kind)
    f = best_freq(spec, kind)
    return f, spec[f], plane(X, y, kind, f)


@torch.no_grad()
def recover_slot(m, mode, slot, site, fit, ev, kind):
    """Swap ONLY the label in `slot` from the reference label 0 to each other label and
    measure the rotation of the operand's value-plane coordinate at `site`."""
    mi = U.MODES.index(mode)
    f, conc, (W, m0) = value_plane(m, fit, mode, slot, site, kind)
    order = GROUP_ORDER[kind]
    base = subset(ev, (ev["mode"] == mi) & (ev[slot] == 0))
    if kind == "mult":
        base = subset(base, base[VALUE_OF[slot]] != 0)
    z0 = zc(slot_vectors(sites(m, U.build(base).tokens), base, slot, site), W, m0)
    out = {}
    for u in range(1, U.NL):
        sw = dict(base)
        sw[slot] = torch.full_like(base[slot], u)
        keep = ~U.is_held(sw)
        if keep.sum() < 20:
            continue
        swk = subset(sw, keep)
        z1 = zc(slot_vectors(sites(m, U.build(swk).tokens), swk, slot, site), W, m0)
        ratio = z1 / z0[keep]
        ratio = ratio / ratio.abs().clamp_min(1e-9)
        ks = (0, -2) if (mode == "mul" and slot == "u2") else (0,)
        lv = torch.tensor([U.LOG.get(int(x), 0) for x in base[VALUE_OF[slot]][keep]], dtype=torch.double)
        best = None
        for k in ks:   # inverse labels: the value's own phase flips -> value-dependent rotation
            c = (ratio * torch.exp(-1j * 2 * math.pi * f * k * lv / 12)).mean()
            if best is None or c.abs() > best[1].abs():
                best = (k, c)
        k_hat, c = best
        steps = torch.angle(c).item() * order / (2 * math.pi)
        step = round(steps) * pow(f, -1, order) % order
        out[u] = dict(step=int(step), raw=steps, coherence=c.abs().item(), inverse=k_hat == -2)
    return dict(freq=f, value_conc=conc, rec=out)


def score_table(rec, dim, sign):
    true = true_table()[dim]
    order = 12 if dim != "temp" else P
    res = []
    for u, r in rec.items():
        t = (sign * (true[u] - true[0])) % order
        if dim == "acc" and U.ACC_INVERSE[u]:
            t = (sign * (true[u] - true[0])) % order    # constant part; direction scored separately
        err = min((r["step"] - t) % order, (t - r["step"]) % order)
        res.append(dict(label=u, true=int(t), recovered=r["step"], exact=err == 0, circ_err=err,
                        coherence=r["coherence"], inverse_recovered=r["inverse"],
                        inverse_true=bool(dim == "acc" and U.ACC_INVERSE[u])))
    return res


# ------------------------------------------------------------------ A6b embedding-row edit
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
    y_new_f, y_new_e = answers(f_lat, new), answers(e_lat, new)
    y_old_e = U.build(e_lat).y
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
        pred_f = m.head_from_resid(Sf["h"]).argmax(-1)
        # A1 readability
        out["probe"] = {}
        for mi, mode in enumerate(U.MODES):
            sf, se = fit["mode"] == mi, ev["mode"] == mi
            out["probe"][mode] = {s: {q: probe(Vf[s][sf], Qf[q][sf], Ve[s][se], Qe[q][se]) for q in Qf}
                                  for s in Vf}
        # A2 geometry (calculator intermediate, true classes) and fitted planes (predicted answers)
        out["geometry"] = {}
        for mi, mode in enumerate(U.MODES):
            sf = fit["mode"] == mi
            g = {}
            for site in ("L1:query", "L2:query", "L3:query"):
                X = Vf[site][sf]
                spec_m = spectrum(class_means(X, Qf["calculator"][sf]), "mult")
                spec_a = spectrum(class_means(X, Qf["calculator"][sf]), "add")
                g[site] = dict(mult=spec_m, add=spec_a, mult_top2=top2(spec_m), add_top2=top2(spec_a))
            out["geometry"][mode] = g
        # A3 label tables: rotation of each operand's value circle when its label is swapped
        out["labels"] = {}
        for (mode, slot), (dim, sign) in SLOTS.items():
            kind = kind_of(dim)
            out["labels"][f"{mode}:{slot}"] = {}
            for site in LABEL_SITES:
                r = recover_slot(m, mode, slot, site, fit, ev, kind)
                out["labels"][f"{mode}:{slot}"][site] = dict(
                    dim=dim, sign=sign, kind=kind, freq=r["freq"], value_conc=r["value_conc"],
                    rows=score_table(r["rec"], dim, sign))
        # A4 composition: held-out label pairs; agreement with the recovered-table answers
        Sh = sites(m, U.build(held).tokens)
        pred_h = m.head_from_resid(Sh["h"]).argmax(-1)
        y_h = U.build(held).y
        out["held"] = {mode: dict(acc=(pred_h == y_h)[held["mode"] == mi].float().mean().item())
                       for mi, mode in enumerate(U.MODES)}
        tables = recovered_tables({k: v[PRIMARY_SITE] for k, v in out["labels"].items()})
        for mi, mode in enumerate(U.MODES):
            if mode == "msum":
                continue
            sel = held["mode"] == mi
            y_tab = table_answers(subset(held, sel), mode, tables)
            if y_tab is not None:
                out["held"][mode]["agree_recovered_table"] = (pred_h[sel] == y_tab).float().mean().item()
                out["held"][mode]["recovered_table_correct"] = (y_tab == y_h[sel]).float().mean().item()
        # A5 convert-first: canonical operand readability at the operand label / value sites
        out["convert_first"] = {mode: {s: out["probe"][mode][s]["op1 canonical"]
                                       for s in ("L1:op1_lab", "L2:op1_lab", "L1:op1_val", "L2:op1_val", "L2:query")}
                                for mode in ("mul", "msum")}
        # A6a map-guided relabel at h (mul mass labels; tsum temp labels)
        out["relabel"] = {}
        for mode, slot in (("mul", "u1"), ("mul", "u2"), ("div", "u1"), ("tsum", "u1")):
            out["relabel"][f"{mode}:{slot}"] = relabel(m, mode, slot, fit, ev,
                                                       out["labels"][f"{mode}:{slot}"][PRIMARY_SITE])
    # A6b one-row embedding edits: a NEW meaning (a parameter no label in the dimension has)
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


def recovered_tables(labels):
    t = {}
    for key, v in labels.items():
        t[key] = {0: 0, **{r["label"]: r["recovered"] for r in v["rows"]}}
        t[key + ":inv"] = {0: False, **{r["label"]: r["inverse_recovered"] for r in v["rows"]}}
    return t


def table_answers(lat, mode, T):
    """Answers implied by the recovered steps (reference label 0 has the true parameter)."""
    if mode == "mul":
        lm = torch.tensor([T["mul:u1"][int(u)] for u in lat["u1"]]) + U.MASS_LOG[0]
        la = torch.tensor([T["mul:u2"][int(u)] for u in lat["u2"]]) + U.ACC_LOG[0]
        lo = -torch.tensor([U.FORCE_LOG[int(u)] - U.FORCE_LOG[0] for u in lat["uo"]])  # output: true
        inv = torch.tensor([T["mul:u2:inv"][int(u)] for u in lat["u2"]])
        M = lat["x1"] * U.scale(lm.tolist()) % P
        A = torch.where(inv, U.scale(la.tolist()) * U.INV[lat["x2"]], lat["x2"] * U.scale(la.tolist())) % P
        return M * A * U.scale(lo.tolist()) * pow(pow(2, U.FORCE_LOG[0], P), -1, P) % P
    if mode == "div":
        lf = torch.tensor([T["div:u1"][int(u)] for u in lat["u1"]]) + U.FORCE_LOG[0]
        lmn = -torch.tensor([T["div:u2"][int(u)] for u in lat["u2"]])    # divisor: opposite sign
        lo = -torch.tensor([U.ACC_LOG[int(u)] - U.ACC_LOG[0] for u in lat["uo"]])     # output: true
        return (lat["x1"] * U.scale(lf.tolist()) * U.INV[lat["x2"]] * U.scale(lmn.tolist())
                * pow(pow(2, U.MASS_LOG[0], P), -1, P) * U.scale(lo.tolist())
                * pow(pow(2, U.ACC_LOG[0], P), -1, P)) % P
    if mode == "tsum":
        c1 = torch.tensor([T["tsum:u1"][int(u)] for u in lat["u1"]]) + U.TEMP_OFF[0]
        c2 = torch.tensor([T["tsum:u2"][int(u)] for u in lat["u2"]]) + U.TEMP_OFF[0]
        co = -torch.tensor([U.TEMP_OFF[int(u)] for u in lat["uo"]])                    # output: true
        return (lat["x1"] + c1 + lat["x2"] + c2 + co) % P
    return None


@torch.no_grad()
def relabel(m, mode, slot, fit, ev, lab):
    """Map-guided relabel at the MLP input: for questions with reference label 0 in `slot`,
    rotate the operand's value-circle coordinates (every frequency of that character family)
    by the RECOVERED step of label u; compare with the real relabelled question."""
    mi = U.MODES.index(mode)
    dim, _ = SLOTS[(mode, slot)]
    kind = kind_of(dim)
    order = GROUP_ORDER[kind]
    base = subset(ev, (ev["mode"] == mi) & (ev[slot] == 0))
    if kind == "mult":
        base = subset(base, base[VALUE_OF[slot]] != 0)
    hb = R.encode(m, U.build(base).tokens)
    planes = {}
    for ff in range(1, order // 2 + 1):
        if kind == "mult" and ff == 6:
            continue
        lat_f = subset(fit, fit["mode"] == mi)
        X = R.encode(m, U.build(lat_f).tokens)
        planes[ff] = plane(X, lat_f[VALUE_OF[slot]], kind, ff)
    res = {}
    for r in lab["rows"]:
        u = r["label"]
        if r["inverse_true"]:
            continue
        sw = dict(base)
        sw[slot] = torch.full_like(base[slot], u)
        keep = ~U.is_held(sw)
        if keep.sum() < 20:
            continue
        sw = subset(sw, keep)
        hk = hb[keep]
        y_new, y_old = U.build(sw).y, U.build(subset(base, keep)).y
        elig = y_new != y_old
        hp = hk.double()
        for ff, (Wf, mf) in planes.items():
            z = zc(hk, Wf, mf)
            ang = 2 * math.pi * ff * r["recovered"] / order
            dz = z * (complex(math.cos(ang), math.sin(ang)) - 1)
            hp = hp + torch.stack([dz.real, dz.imag], 1) @ torch.linalg.pinv(Wf)
        pred_rot = m.head_from_resid(hp.float()).argmax(-1)
        pred_copy = m.head_from_resid(R.encode(m, U.build(sw).tokens)).argmax(-1)
        res[u] = dict(n=int(elig.sum()),
                      rotate=dict(intended=(pred_rot == y_new)[elig].float().mean().item(),
                                  unchanged=(pred_rot == y_old)[elig].float().mean().item()),
                      copy=dict(intended=(pred_copy == y_new)[elig].float().mean().item()))
    return res


_PLANES = {}


def plane_cache(m, mode, kind, f):
    key = (id(m), mode, kind, f)
    if key not in _PLANES:
        fit = draw(N, FIT_S, "fit")
        sel = fit["mode"] == U.MODES.index(mode)
        h = R.encode(m, U.build(subset(fit, sel)).tokens)
        pred = m.head_from_resid(h).argmax(-1)
        _PLANES[key] = plane(h, pred, kind, f)
    return _PLANES[key]


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
