"""Doc 57 experiment I: corrected interpretability instruments, validated on constructed networks first.

I1  interventional S-lens: S_s(x) = v.h(x) - mean_{t' in type(s)} v.h(x with token s := t')   (exact, finite)
I1b integrated gradients from the type-conditional mean embedding (64 steps)
I2  gate-aware lens: A_ctx = E_x[J(x)^T W_U^T W_U J(x)] (second moment of exact per-prompt Jacobians)
I3  interventional sectors: PCA directions of h ranked by their exact single-direction ablation effect
I4  averaged-J vs exact per-prompt J vs gate-aware predictions of random subspace ablations, against actual effects

    python -m sci.instruments2 validate | learned     # -> runs/phase7/instruments_{validation,learned}.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import construct as C  # noqa: E402
from sci import neuron_lens as N  # noqa: E402
from sci import rescue as R  # noqa: E402
from sci import residual as RS  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase7"


# ------------------------------------------------------------------ helpers
@torch.no_grad()
def h_of(m, tokens):
    return m.resid_pre_mlp(tokens, torch.zeros_like(tokens))


def token_types():
    """type set for each token id (tokens that may legally replace it)."""
    types = {}
    for v in T.VAL:
        types[v] = list(T.VAL)
    for grp in (T.MU, T.AU, T.FU):
        for t in grp:
            types[t] = list(grp)
    for t in (T.TIMES, T.ARROW, T.EQ):
        types[t] = [t]
    return types


@torch.no_grad()
def s_lens_interventional(m, tokens, v):
    """S_s(x) for every prompt and position: exact finite interventions over type-legal replacements."""
    types = token_types()
    base = h_of(m, tokens) @ v
    N_, Lp = tokens.shape
    S = torch.zeros(N_, Lp, dtype=torch.float64)
    for s in range(Lp):
        col = tokens[:, s]
        acc = torch.zeros(N_, dtype=torch.float64)
        cnt = torch.zeros(N_, dtype=torch.float64)
        alts = sorted({t for tok in col.unique().tolist() for t in types[tok]})
        for t2 in alts:
            legal = torch.tensor([t2 in types[int(t)] for t in col])
            if not legal.any():
                continue
            tk = tokens[legal].clone(); tk[:, s] = t2
            acc[legal] += (h_of(m, tk) @ v).double()
            cnt[legal] += 1
        S[:, s] = base.double() - acc / cnt.clamp_min(1)
    return S


def s_lens_ig(m, tokens, v, steps=64):
    types = token_types()
    E = m.E_c.weight.detach()
    base_emb = torch.stack([E[torch.tensor(types[int(t)])].mean(0) for t in range(E.shape[0])])   # type-conditional mean
    x_tok = m.embed(tokens, torch.zeros_like(tokens)).detach()
    x_base = x_tok - E[tokens] + base_emb[tokens]
    tot = torch.zeros_like(x_tok)
    for k in range(1, steps + 1):
        x = (x_base + (k / steps) * (x_tok - x_base)).requires_grad_(True)
        x1 = x + m.attn1(x)
        h = (x1 + m.attn2(x1))[:, -1]
        g, = torch.autograd.grad((h @ v).sum(), x)
        tot += g
    return ((x_tok - x_base) * tot / steps).sum(-1).double()


def gate_aware(m, u):
    """A_ctx = E_x[(W_U + Q D_x W_in)^T (W_U + Q D_x W_in)], plus per-prompt readout maps (N x 13 x 64)."""
    WU, Win = m.W_U.weight.double(), m.W_in.weight.double()
    Q = (m.W_U.weight @ m.W_out.weight).double()
    D = (u > 0).double()
    Mx = WU[None] + torch.einsum("cn,xn,nd->xcd", Q, D, Win)
    A = torch.einsum("xcd,xce->de", Mx, Mx) / len(u)
    return A, Mx


@torch.no_grad()
def head_logits(m, h):
    return m.W_U(h + m.W_out(torch.relu(h @ m.W_in.weight.T + m.W_in.bias))).double()


def acc_of(m, h, y, un):
    pr = head_logits(m, h.float()).argmax(-1)
    return float((pr[~un] == y[~un]).double().mean()), float((pr[un] == y[un]).double().mean())


def sectors_gate_aware(m, h, u, y, un):
    A, Mx = gate_aware(m, u)
    mu = h[~un].mean(0)
    hc = h - mu
    lam, V = torch.linalg.eigh(A)
    E = (hc @ V).pow(2).mean(0) * lam.clamp_min(0)
    order = torch.argsort(-E)
    frac = torch.cumsum(E[order], 0) / E.sum().clamp_min(1e-12)
    nb = int((frac < 0.9).sum()) + 1
    Vb, Vd = V[:, order[:nb]], V[:, order[nb:]]
    hd = h - (hc @ Vd) @ Vd.T
    base = head_logits(m, h.float())
    act = (head_logits(m, hd.float()) - base).pow(2).sum(1).mean().sqrt()
    pred = torch.einsum("xd,de,xe->x", hd - h, A, hd - h).mean().sqrt()
    return dict(bright_dims=nb, bright_vectors=Vb, ablate_dark_acc=acc_of(m, hd, y, un),
                ablate_bright_acc=acc_of(m, h - (hc @ Vb) @ Vb.T, y, un),
                dark_actual_rms=float(act), dark_pred_rms=float(pred)), A, Mx


def sectors_interventional(m, h, y, un):
    mu = h[~un].mean(0)
    hc = h - mu
    _, _, Vt = torch.linalg.svd(hc, full_matrices=True)
    base = head_logits(m, h.float())
    eff = []
    for j in range(Vt.shape[0]):
        v = Vt[j]
        hj = h - torch.outer(hc @ v, v)
        eff.append(float((head_logits(m, hj.float()) - base).pow(2).sum(1).mean().sqrt()))
    eff = torch.tensor(eff, dtype=torch.float64)
    order = torch.argsort(-eff)
    frac = torch.cumsum(eff[order], 0) / eff.sum().clamp_min(1e-12)
    nb = int((frac < 0.9).sum()) + 1
    Vd = Vt[order[nb:]].T
    hd = h - (hc @ Vd) @ Vd.T
    act = (head_logits(m, hd.float()) - base).pow(2).sum(1).mean().sqrt()
    pred = eff[order[nb:]].pow(2).sum().sqrt()                       # independent-effects prediction
    return dict(bright_dims=nb, bright_vectors=Vt[order[:nb]].T, ablate_dark_acc=acc_of(m, hd, y, un),
                dark_actual_rms=float(act), dark_pred_rms=float(pred))


def i4_compare(m, h, u, A, Mx, seed=0):
    g = torch.Generator().manual_seed(seed)
    WU = m.W_U.weight.double()
    Jbar_read = WU + (m.W_U.weight @ m.W_out.weight).double() @ torch.diag((u > 0).double().mean(0)) @ m.W_in.weight.double()
    mu = h.mean(0)
    base = head_logits(m, h.float())
    rows = []
    for k in (1, 2, 4, 8, 16, 32):
        for _ in range(20):
            Qr, _ = torch.linalg.qr(torch.randn(m.d, k, generator=g, dtype=torch.float64))
            dh = -((h - mu) @ Qr) @ Qr.T
            actual = (head_logits(m, (h + dh).float()) - base).pow(2).sum(1).mean().sqrt()
            p_avg = (dh @ Jbar_read.T).pow(2).sum(1).mean().sqrt()
            p_exact = torch.einsum("xcd,xd->xc", Mx, dh).pow(2).sum(1).mean().sqrt()
            p_gate = torch.einsum("xd,de,xe->x", dh, A, dh).mean().sqrt()
            rows.append((k, float(actual), float(p_avg), float(p_exact), float(p_gate)))
    t = torch.tensor([r[1:] for r in rows])
    out = {}
    for j, nm in enumerate(("averaged_J", "exact_per_prompt_J", "gate_aware")):
        ratio = t[:, 0] / t[:, j + 1].clamp_min(1e-12)
        rank = lambda x: torch.argsort(torch.argsort(x)).double()
        out[nm] = dict(median_actual_over_pred=float(ratio.median()),
                       spearman=float(torch.corrcoef(torch.stack([rank(t[:, 0]), rank(t[:, j + 1])]))[0, 1]))
    return out


def residual_gate_aware(m, P_, un):
    """u-space analogue: readout map of u is Q D_x; A_u = (Q^T Q) * E[d d^T]. Restore bright vs dark part of rho."""
    L = C.domain()
    U = P_["u"].double()
    oh = RS.onehots(L)
    Uh = RS.fit(RS.design(list(oh.values())), U)
    rho = U - Uh
    Q = P_["q"].double()
    d = (P_["u"] > 0).double()
    A = (Q.T @ Q) * (d.T @ d / len(d))
    lam, V = torch.linalg.eigh(A)
    E = (rho @ V).pow(2).mean(0) * lam.clamp_min(0)
    order = torch.argsort(-E)
    nb = int((torch.cumsum(E[order], 0) / E.sum().clamp_min(1e-12) < 0.9).sum()) + 1
    Vb, Vd = V[:, order[:nb]], V[:, order[nb:]]
    rb, rd = (rho @ Vb) @ Vb.T, (rho @ Vd) @ Vd.T
    y = P_["y"]
    base = P_["logits"].double() - torch.relu(U) @ Q.T
    def acc(uu):
        pr = (base + torch.relu(uu) @ Q.T).argmax(-1)
        return float((pr[~un] == y[~un]).double().mean()), float((pr[un] == y[un]).double().mean())
    return dict(bright_dims_u=nb, rho_energy_frac_in_bright=float(rb.pow(2).sum() / rho.pow(2).sum()),
                acc_alpha0=acc(Uh), acc_plus_bright=acc(Uh + rb), acc_plus_dark=acc(Uh + rd), acc_full=acc(U))


def analyse(m, name, used=None, read=None, mass_pos=True):
    L = C.domain()
    un = C.unseen_mask(L)
    P_ = N.parts(m, L)
    h, u, y = P_["h"].double(), P_["u"], P_["y"]
    out = dict(name=name)
    if read is not None:
        tokens = P_["b"].tokens
        S1 = s_lens_interventional(m, tokens, read.float())
        S1b = s_lens_ig(m, tokens, read.float())
        mp = torch.where(L["order"] == 0, 0, 3)
        def share(S):
            a = S.abs()
            on = a.gather(1, mp[:, None]).squeeze(1) + a.gather(1, (mp + 1)[:, None]).squeeze(1)
            return float(on.sum() / a.sum().clamp_min(1e-12))
        out["I1_interventional_mass_pair_share"] = share(S1)
        out["I1b_integrated_gradients_mass_pair_share"] = share(S1b)
    ga, A, Mx = sectors_gate_aware(m, h, u, y, un)
    it = sectors_interventional(m, h, y, un)
    for nm, sec in (("I2_gate_aware", ga), ("I3_interventional", it)):
        Vb = sec.pop("bright_vectors")
        if used is not None:
            outside = [i for i in range(m.d) if i not in used]
            sec["bright_mass_inside_used"] = float(1 - Vb[outside].pow(2).sum() / Vb.pow(2).sum())
        out[nm] = sec
    out["I4"] = i4_compare(m, h, u, A, Mx)
    out["retest_residual_gate_aware_u"] = residual_gate_aware(m, P_, un)
    return out


def v_checks(r):
    ok = {}
    if "I1_interventional_mass_pair_share" in r:
        ok["V4p_I1"] = r["I1_interventional_mass_pair_share"] >= 0.95
        ok["V4p_I1b"] = r["I1b_integrated_gradients_mass_pair_share"] >= 0.95
    for nm in ("I2_gate_aware", "I3_interventional"):
        s = r[nm]
        ratio_ok = (s["dark_actual_rms"] < 1e-3 and s["dark_pred_rms"] < 1e-3) or \
                   (0.5 <= s["dark_actual_rms"] / max(s["dark_pred_rms"], 1e-12) <= 2)
        ok[f"V5p_{nm}"] = bool(s.get("bright_mass_inside_used", 1.0) >= 0.95 and min(s["ablate_dark_acc"]) >= 0.99 and ratio_ok)
    return ok


def run_validate():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    mC, _ = C.build_C()
    usedC = set(C.DIM["Q"] + C.DIM["A"] + C.DIM["FU"] + C.DIM["MU"] + C.DIM["AU"] + [C.DIM["NULL"]])
    vC = torch.zeros(64); vC[C.DIM["Q"]] = torch.tensor([float(C._lg(x)) for x in range(P)])
    res["designC"] = analyse(mC, "designC", usedC, vC)
    mL, _ = C.build_L()
    usedL = {C.DL[k] for k in ("HM", "HZM", "HA", "HZA", "HS")}
    vL = torch.zeros(64); vL[C.DL["HM"]] = 1
    res["designL"] = analyse(mL, "designL", usedL, vL)
    for k in res:
        res[k]["checks"] = v_checks(res[k])
        print(k, json.dumps(res[k], default=str), flush=True)
    (OUT / "instruments_validation.json").write_text(json.dumps(res, indent=1, default=str))


def learned_qualified(m, name):
    """Only instruments that qualified on Designs C/L (I1, I1b) plus the descriptive I4 comparison are applied
    (doc 57 §I: I2 and I3 failed V5'). Read directions: (a) the leading mass direction of the declared block (first PC of
    its 39 rendering cell means), (b) the W_in rows of the 8 neurons with the largest |margin contribution|."""
    L = C.domain()
    un = C.unseen_mask(L)
    P_ = N.parts(m, L)
    h, u, y = P_["h"].double(), P_["u"], P_["y"]
    tokens = P_["b"].tokens
    cm = torch.stack([h[(L["mu"] == a_) & (L["m"] == v_), :16].mean(0) for a_ in range(3) for v_ in range(P)])
    _, _, Vt = torch.linalg.svd(cm - cm.mean(0), full_matrices=False)
    v_blk = torch.zeros(m.d, dtype=torch.float64); v_blk[:16] = Vt[0]
    mp = torch.where(L["order"] == 0, 0, 3)
    ap = torch.where(L["order"] == 0, 3, 0)
    def shares(S):
        a = S.abs()
        g = lambda pos: (a.gather(1, pos[:, None]).squeeze(1) + a.gather(1, (pos + 1)[:, None]).squeeze(1)).sum()
        tot = a.sum().clamp_min(1e-12)
        return dict(mass_pair=float(g(mp) / tot), acc_pair=float(g(ap) / tot), force_unit=float(a[:, 6].sum() / tot))
    out = dict(name=name)
    out["I1_block_mass_direction"] = shares(s_lens_interventional(m, tokens, v_blk.float()))
    out["I1b_block_mass_direction"] = shares(s_lens_ig(m, tokens, v_blk.float()))
    ystar = N.competitor(P_["logits"], y)
    imp = N.margins(P_, ystar).abs().mean(0)
    out["I1_top8_neurons"] = {}
    for i in torch.argsort(-imp)[:8].tolist():
        out["I1_top8_neurons"][i] = shares(s_lens_interventional(m, tokens, m.W_in.weight[i].detach()))
    A, Mx = gate_aware(m, u)
    out["I4"] = i4_compare(m, h, u, A, Mx)
    return out


def run_learned():
    torch.set_num_threads(4)
    res = {}
    for s in (307, 311, 300, 305, 302, 304, 310):
        n = f"mass_only_s{s}"
        res[n] = r = learned_qualified(R.load(n), n)
        print(n, json.dumps({k: v for k, v in r.items() if k != "name"}, default=str), flush=True)
    (OUT / "instruments_learned.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"validate": run_validate, "learned": run_learned}[sys.argv[1]]()
