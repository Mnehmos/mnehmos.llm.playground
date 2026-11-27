"""Doc 55: bidirectional neuron-level interpretability for the decoder (adapted from neuron-lens: R-/S-lens, sectors,
dark-sector ablation), with exact per-prompt Jacobians. Validation on constructed networks precedes interpretation.

Decoder: h -> u = W_in h + b -> a = ReLU(u) -> logits = W_U (h + W_out a + b_out).
  E1  logits = W_U(h + b_out) + sum_i a_i q_i,  q_i = W_U W_out[:, i]  (exact; a neuron's write is rank one)
  E3  J(x) = dr/dh = I + W_out D(x) W_in (exact); R-lens = W_U mean_x J(x)  (= J-lens: no norms/gates here)
  E4  S-lens K_s = mean_x dh/de(s); exact per prompt via autograd
  E5  u_i = W_in[i].(x0 + layer-1 heads + layer-2 heads at the last position) + b_i (exact split)

    python -m sci.neuron_lens validate | learned | develop | direct
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import construct as C  # noqa: E402
from sci import rescue as R  # noqa: E402
from sci import residual as RS  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase6/atlas"
SC = dict(MU=T.MASS_S, AU=T.ACC_S)


# ------------------------------------------------------------------ exact forward with per-head pieces
def heads(attn, x):
    B, L, d = x.shape
    H, dh = attn.h, attn.dh
    q, k, v = attn.qkv(x).view(B, L, 3, H, dh).unbind(2)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    pat = ((q @ k.transpose(-1, -2)) / math.sqrt(dh)).softmax(-1)
    z = pat @ v                                                     # B,H,L,dh
    Wo = attn.o.weight.view(d, H, dh)
    per = torch.einsum("bhld,ehd->bhle", z, Wo)                     # B,H,L,d (bias separate)
    return per, attn.o.bias, pat


@torch.no_grad()
def parts(m, L):
    b = T.build(L)
    x0 = m.embed(b.tokens, torch.zeros_like(b.tokens))
    p1, b1, pat1 = heads(m.attn1, x0)
    x1 = x0 + p1.sum(1) + b1
    p2, b2, pat2 = heads(m.attn2, x1)
    h = (x1 + p2.sum(1) + b2)[:, -1]
    u = h @ m.W_in.weight.T + m.W_in.bias
    a = torch.relu(u)
    logits = m.W_U(h + m.W_out(a))
    q = m.W_U.weight @ m.W_out.weight                               # 13 x 256
    comp = dict(embed=x0[:, -1], l1=p1[:, :, -1].sum(1) + b1, **{f"l2h{i}": p2[:, i, -1] for i in range(p2.shape[1])},
                l2bias=b2.expand(len(h), -1))
    return dict(b=b, y=b.y, h=h, u=u, a=a, logits=logits, q=q, comp=comp, x1=x1)


def competitor(logits, y):
    return logits.scatter(1, y[:, None], -float("inf")).argmax(-1)


def margins(P_, ystar):
    """per-neuron margin contributions m_i(x) = a_i (q_i[y] - q_i[y*])"""
    q, a, y = P_["q"], P_["a"], P_["y"]
    return a * (q[y] - q[ystar])                                     # N x 256  (q[y]: N x 256)


def fourier_purity(qi):
    """energy share of the best discrete-log frequency in a neuron's write over the 12 nonzero classes."""
    order = [pow(2, l, P) for l in range(12)]
    v = qi[order].double()
    v = v - v.mean()
    F = torch.fft.rfft(v)
    E = F.abs() ** 2
    E[1:6] *= 2
    E = E[1:]
    tot = float(E.sum())
    if tot < 1e-12:
        return 0.0, 0
    k = int(E.argmax()) + 1
    return float(E.max() / tot), k


def r2_1(X, y):
    X1 = torch.cat([X.double(), torch.ones(len(X), 1, dtype=torch.float64)], 1)
    yh = X1 @ torch.linalg.lstsq(X1, y.double(), driver="gelsd").solution
    return 1 - ((y - yh) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-12)


def input_r2(L, u):
    M, A, S = C.canon_scales(L, SC)
    oh = lambda t, n: torch.nn.functional.one_hot(t, n)
    pair = lambda t1, n1, t2, n2: (oh(t1, n1)[:, :, None] * oh(t2, n2)[:, None, :]).reshape(len(t1), -1)
    sets = dict(M=oh(M, P), A=oh(A, P), S=oh(L["fu"], 3),
                written_mass=torch.cat([oh(L["m"], P), oh(L["mu"], 3)], 1), written_acc=torch.cat([oh(L["a"], P), oh(L["au"], 3)], 1),
                mass_pair=pair(L["m"], P, L["mu"], 3), acc_pair=pair(L["a"], P, L["au"], 3), order=oh(L["order"], 2))
    return {k: r2_1(X, u.double()) for k, X in sets.items()}


# ------------------------------------------------------------------ lenses
def jbar(m, u):
    D = (u > 0).float().mean(0)
    return torch.eye(m.d) + m.W_out.weight @ torch.diag(D) @ m.W_in.weight      # mean_x J(x) (D enters linearly)


def s_lens(m, L, batch=2048):
    """Averaged S-lens K_s (8 x 64 x 64) and exact per-prompt reads T[x,s,d] = (dh_d/de(s)) . (W_E[tok_s] - e_bar)."""
    b = T.build(L)
    ebar = m.E_c.weight.mean(0)
    N = len(b.y)
    K = torch.zeros(T.SEQ_LEN, m.d, m.d)
    Tx = torch.zeros(N, T.SEQ_LEN, m.d)
    for i0 in range(0, N, batch):
        tok = b.tokens[i0:i0 + batch]
        x0 = m.embed(tok, torch.zeros_like(tok)).detach().requires_grad_(True)
        x1 = x0 + m.attn1(x0)
        h = (x1 + m.attn2(x1))[:, -1]
        dev = (m.E_c.weight[tok] - ebar).detach()                   # n x 8 x 64
        for d_ in range(m.d):
            g, = torch.autograd.grad(h[:, d_].sum(), x0, retain_graph=True)
            K[:, d_, :] += g.sum(0)
            Tx[i0:i0 + batch, :, d_] = (g * dev).sum(-1)
    return K / N, Tx


# ------------------------------------------------------------------ N3: contribution-weighted residual
def residual_contrib(m, L, P_, un):
    U = P_["u"].double()
    oh = RS.onehots(L)
    Uh = RS.fit(RS.design(list(oh.values())), U)
    rho = U - Uh
    y = P_["y"]
    ystar = competitor(P_["logits"], y)
    q = P_["q"].double()
    gq = q[y] - q[ystar]                                             # N x 256
    a1, a0 = torch.relu(U), torch.relu(Uh)
    dA = a1 - a0                                                     # exact activation change from the residual
    base0 = (P_["logits"].double() - (a1 - a0) @ q.T)                # logits with u = u_hat
    def acc(lg):
        pr = lg.argmax(-1)
        return float((pr[~un] == y[~un]).double().mean()), float((pr[un] == y[un]).double().mean())
    meas = dict(variance=rho.var(0), output_weighted=rho.abs().mean(0) * q.norm(dim=0),
                jacobian_pred=((U > 0).double() * rho * gq).mean(0).abs(), actual_margin=(dA * gq).mean(0).abs())
    # actual single-neuron accuracy effects (restore alone from alpha=0; remove alone from alpha=1)
    full = P_["logits"].double()
    restore_acc = torch.zeros(U.shape[1]); remove_acc = torch.zeros(U.shape[1])
    for i in range(U.shape[1]):
        lg = base0 + dA[:, i:i + 1] * q[:, i][None, :]
        restore_acc[i] = sum(acc(lg))
        lg2 = full - dA[:, i:i + 1] * q[:, i][None, :]
        remove_acc[i] = sum(acc(lg2))
    meas["actual_restore_acc_gain"] = restore_acc - sum(acc(base0))
    meas["actual_remove_acc_loss"] = sum(acc(full)) - remove_acc
    rank = lambda v: torch.argsort(torch.argsort(-v.double())).double()
    names = list(meas)
    spear = {f"{a}~{b}": float(torch.corrcoef(torch.stack([rank(meas[a]), rank(meas[b])]))[0, 1])
             for i, a in enumerate(names) for b in names[i + 1:]}
    def n90(v):
        v = v.double().clamp_min(0); s = torch.sort(v, descending=True).values
        c = torch.cumsum(s, 0) / s.sum().clamp_min(1e-12)
        return int((c < 0.9).sum()) + 1
    topk = {}
    for nm in ("variance", "output_weighted", "jacobian_pred", "actual_margin"):
        order = torch.argsort(-meas[nm].double())
        topk[nm] = {}
        for k in (4, 8, 16, 32, 64):
            idx = order[:k]
            lg = base0 + dA[:, idx] @ q[:, idx].T
            topk[nm][k] = acc(lg)
    return dict(acc_alpha0=acc(base0), acc_full=acc(full), spearman=spear, n_for_90pct={k: n90(v) for k, v in meas.items()},
                topk_restore=topk, rho_var_share=float((rho ** 2).sum() / ((U - U.mean(0)) ** 2).sum()),
                jacobian_vs_actual_margin_rel_err=float(((meas["jacobian_pred"] - meas["actual_margin"]).abs().sum() / meas["actual_margin"].sum()).clamp_max(1e9))), rho, Uh


# ------------------------------------------------------------------ N4: sectors and dark subspaces
@torch.no_grad()
def head_logits(m, h):
    return m.W_U(h + m.W_out(torch.relu(h @ m.W_in.weight.T + m.W_in.bias)))


def sectors(m, L, P_, un, K=None, seed=0):
    h, u, y = P_["h"].double(), P_["u"], P_["y"]
    Jb = jbar(m, u).double()
    WU = m.W_U.weight.double()
    Rm = WU @ Jb                                                     # R-lens readout of h (13 x 64)
    A = Rm.T @ Rm
    def acc_of(hh):
        pr = head_logits(m, hh.float()).argmax(-1)
        return float((pr[~un] == y[~un]).double().mean()), float((pr[un] == y[un]).double().mean())
    mu_f = h[~un].mean(0)
    out = {}
    # readout share per coordinate, native and rotated
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(m.d, m.d, generator=g, dtype=torch.float64))
    foot = None
    if K is not None:
        ebar = m.E_c.weight.mean(0).double()
        dev = m.E_c.weight.double()[P_["b"].tokens] - ebar            # N x 8 x 64
        foot = torch.einsum("sde,nse->nd", K.double(), dev)           # S-lens linear prediction of h
    for basis, B_ in (("native", torch.eye(m.d, dtype=torch.float64)), ("rotated", Q)):
        hb = h @ B_                                                   # coordinates in basis
        Ab = B_.T @ A @ B_
        share = (hb * (hb @ Ab)).sum(0) / (hb * (hb @ Ab)).sum()
        res = dict(readout_share_top7=float(torch.sort(share, descending=True).values[:7].sum()))
        Rset = set(torch.argsort(-share)[:7].tolist())
        if foot is not None:
            fb = foot @ B_
            tok = torch.tensor([float(torch.corrcoef(torch.stack([hb[:, i], fb[:, i]]))[0, 1]) ** 2 if hb[:, i].std() > 1e-9 else 0.0
                                for i in range(m.d)])
            Sset = set(torch.argsort(-tok)[:7].tolist())
        else:
            Sset = set()
        secs = dict(c=Rset & Sset, r=Rset - Sset, s=Sset - Rset)
        secs["d"] = set(range(m.d)) - Rset - Sset
        for nm, S_ in secs.items():
            if not S_:
                res[f"ablate_{nm}"] = None; continue
            idx = torch.tensor(sorted(S_))
            hb2 = hb.clone(); hb2[:, idx] = (mu_f @ B_)[idx]
            res[f"ablate_{nm}"] = dict(n=len(S_), acc=acc_of(hb2 @ B_.T))
        res["keep_only_R"] = None
        idx = torch.tensor(sorted(set(range(m.d)) - Rset))
        hb2 = hb.clone(); hb2[:, idx] = (mu_f @ B_)[idx]
        res["keep_only_R"] = acc_of(hb2 @ B_.T)
        out[basis] = res
    # dark sector of h (R-lens readout singular directions)
    Uv, Sv, Vt = torch.linalg.svd(Rm, full_matrices=True)
    hc = h - mu_f
    en = ((hc @ Vt.T[:, :len(Sv)]) * Sv) ** 2
    frac = torch.cumsum(en.sum(0), 0) / en.sum()
    nb = int((frac < 0.9).sum()) + 1
    Vb, Vd = Vt[:nb].T, Vt[nb:].T
    hd = h - (hc @ Vd) @ Vd.T                                        # dark part replaced by its familiar mean
    hb_ = h - (hc @ Vb) @ Vb.T
    lg_full = head_logits(m, h.float()).double()
    def eff(hh):
        lg = head_logits(m, hh.float()).double()
        actual = (lg - lg_full).abs().mean()
        pred = ((hh - h) @ Rm.T).abs().mean()                       # averaged-lens prediction
        # exact per-prompt Jacobian prediction
        D = (P_["u"] > 0).double()
        dh = (hh - h)
        Jdh = dh + ((dh @ m.W_in.weight.double().T) * D) @ m.W_out.weight.double().T
        pred_exact = (Jdh @ WU.T).abs().mean()
        return dict(acc=acc_of(hh), mean_abs_dlogit_actual=float(actual), lens_predicted=float(pred),
                    exact_jacobian_predicted=float(pred_exact), actual_over_lens=float(actual / max(pred, 1e-12)))
    out["dark_sector_h"] = dict(bright_dims=nb, ablate_dark=eff(hd), ablate_bright=eff(hb_))
    return out


def residual_sectors(m, P_, rho, Uh, un):
    """Is the essential residual rho in a lens-dark subspace of u-space? Lens from u to logits: W_U W_out D_bar."""
    y = P_["y"]
    D = (P_["u"] > 0).double().mean(0)
    Lu = (m.W_U.weight.double() @ m.W_out.weight.double()) * D[None, :]   # 13 x 256
    _, Sv, Vt = torch.linalg.svd(Lu, full_matrices=True)
    rc = rho
    en = ((rc @ Vt.T[:, :len(Sv)]) * Sv) ** 2
    frac = torch.cumsum(en.sum(0), 0) / en.sum()
    nb = int((frac < 0.9).sum()) + 1
    Vb, Vd = Vt[:nb].T, Vt[nb:].T
    rb, rd = (rho @ Vb) @ Vb.T, (rho @ Vd) @ Vd.T
    q = m.W_U.weight.double() @ m.W_out.weight.double()
    base = m.W_U.weight.double() @ (P_["h"].double().T) + (m.W_U.weight.double() @ m.W_out.bias.double())[:, None]
    def acc(uu):
        lg = base.T + torch.relu(uu) @ q.T
        pr = lg.argmax(-1)
        return float((pr[~un] == y[~un]).double().mean()), float((pr[un] == y[un]).double().mean())
    return dict(bright_dims_u=nb, rho_energy_in_dark=float((rd ** 2).sum() / (rho ** 2).sum()),
                acc_uhat=acc(Uh), acc_uhat_plus_bright=acc(Uh + rb), acc_uhat_plus_dark=acc(Uh + rd), acc_full=acc(Uh + rho))


# ------------------------------------------------------------------ N5: native vs rotated basis
def basis_compare(m, P_, un, seed=0):
    a, q, y = P_["a"].double(), P_["q"].double(), P_["y"]
    ystar = competitor(P_["logits"], y)
    base = P_["logits"].double() - a @ q.T
    mu = a[~un].mean(0)
    def acc(lg):
        pr = lg.argmax(-1)
        return float((pr[~un] == y[~un]).double().mean()), float((pr[un] == y[un]).double().mean())
    ac = a - mu
    _, _, Vp = torch.linalg.svd(ac, full_matrices=False)
    g = torch.Generator().manual_seed(seed)
    Qr, _ = torch.linalg.qr(torch.randn(a.shape[1], a.shape[1], generator=g, dtype=torch.float64))
    res = {}
    for name, V in (("native", torch.eye(a.shape[1], dtype=torch.float64)), ("pca", Vp.T), ("random_rotation", Qr)):
        coef = ac @ V                                                 # N x 256
        qv = V.T @ q.T                                                # 256 x 13
        gm = (coef * (qv[:, y].T - qv[:, ystar].T)).abs().mean(0)     # mean |margin contribution| per direction
        order = torch.argsort(-gm)
        tot = gm.sum()
        r = {}
        for k in (1, 2, 4, 8, 16, 32, 64):
            idx = order[:k]
            keep = torch.zeros(a.shape[1], dtype=torch.float64); keep[idx] = 1
            lg_keep = base + (mu + (coef * keep) @ V.T) @ q.T
            lg_rm = base + (mu + (coef * (1 - keep)) @ V.T) @ q.T
            r[k] = dict(share=float(gm[idx].sum() / tot), keep_only=acc(lg_keep), remove_only=acc(lg_rm))
        res[name] = r
    return res


# ------------------------------------------------------------------ N6: programming through output weights
def program_out(m, P_, L, un, imp, minput):
    y = P_["y"]
    q = P_["q"].double()
    a = P_["a"].double()
    direct = P_["logits"].double() - a @ q.T
    sigma = {c: pow(c, 5, P) for c in range(P)}                     # F -> F^5 (5 invertible mod 12)
    sig_inv = {v: c for c, v in sigma.items()}
    perm_inv = torch.tensor([sig_inv[c] for c in range(P)])
    tgt = torch.tensor([sigma[int(t)] for t in y])
    order = torch.argsort(-imp)
    res = {"a_F^5": {}}
    for k in (1, 4, 8, 16, 32, 64, 128, 256):
        qn = q.clone()
        idx = order[:k]
        qn[:, idx] = q[perm_inv][:, idx]                              # q_i'[c] = q_i[sigma^-1(c)]
        pr = (direct + a @ qn.T).argmax(-1)
        res["a_F^5"][k] = (float((pr[~un] == tgt[~un]).double().mean()), float((pr[un] == tgt[un]).double().mean()))
    # (b) operand-selective: M-dominated neurons rewritten so that F(M^-1, A, S) is computed
    M, A, S = C.canon_scales(L, SC)
    Minv = torch.tensor([pow(int(x), -1, P) if x else 0 for x in M])
    tgt_b = T.target(Minv, A, S)
    sel = torch.nonzero(minput).flatten()
    if len(sel) == 0:
        res["b_M_inverse"] = dict(n_neurons=0)
        return res
    fixed = direct + a[:, ~minput] @ q[:, ~minput].T
    Y = 10.0 * torch.nn.functional.one_hot(tgt_b, P).double() - fixed
    Xs = a[:, sel]
    qs = torch.linalg.lstsq(Xs[~un], Y[~un], driver="gelsd").solution    # |sel| x 13, fitted on familiar prompts
    pr = (fixed + Xs @ qs).argmax(-1)
    res["b_M_inverse"] = dict(n_neurons=int(len(sel)), familiar=float((pr[~un] == tgt_b[~un]).double().mean()),
                              unseen=float((pr[un] == tgt_b[un]).double().mean()))
    return res


# ------------------------------------------------------------------ atlas for one network
def atlas(m, name, with_slens=True):
    L = C.domain()
    un = C.unseen_mask(L)
    P_ = parts(m, L)
    y = P_["y"]
    ystar = competitor(P_["logits"], y)
    out = dict(name=name)
    # V0 / E1
    rec = m.W_U(P_["h"] + m.W_out.bias) + P_["a"] @ P_["q"].T
    out["E1_max_abs_err"] = float((rec - P_["logits"]).abs().max())
    mg = margins(P_, ystar)
    # single-neuron ablations (exact by E1): zero and mean
    lg = P_["logits"]
    mu_a = P_["a"][~un].mean(0)
    def accs(lgx):
        pr = lgx.argmax(-1)
        return float((pr[~un] == y[~un]).float().mean()), float((pr[un] == y[un]).float().mean())
    base_acc = accs(lg)
    r2in = input_r2(L, P_["u"])
    neurons = []
    for i in range(P_["u"].shape[1]):
        qi = P_["q"][:, i]
        pur, k = fourier_purity(qi)
        z = lg - P_["a"][:, i:i + 1] * qi[None, :]
        mn = lg - (P_["a"][:, i:i + 1] - mu_a[i]) * qi[None, :]
        za, ma = accs(z), accs(mn)
        conn = {c: float((P_["comp"][c] @ m.W_in.weight[i]).abs().mean()) for c in P_["comp"]}
        neurons.append(dict(i=i, active_frac=float((P_["u"][:, i] > 0).float().mean()), write_norm=float(qi.norm()),
                            write_purity=pur, write_k=k, mean_abs_margin_fam=float(mg[~un, i].abs().mean()),
                            mean_abs_margin_unseen=float(mg[un, i].abs().mean()),
                            zero_ablate_dacc=(base_acc[0] - za[0], base_acc[1] - za[1]),
                            mean_ablate_dacc=(base_acc[0] - ma[0], base_acc[1] - ma[1]),
                            input_r2={kk: float(v[i]) for kk, v in r2in.items()}, connectivity=conn))
    out["neurons"] = neurons
    imp = torch.tensor([n["mean_abs_margin_fam"] + n["mean_abs_margin_unseen"] for n in neurons])
    out["n_neurons_90pct_margin"] = int((torch.cumsum(torch.sort(imp, descending=True).values, 0) / imp.sum() < 0.9).sum()) + 1
    # N2 functional polysemanticity among important neurons
    imp_idx = torch.argsort(-imp)[:64].tolist()
    poly = []
    for i in imp_idx:
        n = neurons[i]
        nvars = sum(1 for kk in ("M", "A", "S") if n["input_r2"][kk] >= 0.2)
        poly.append(dict(i=i, n_input_vars=nvars, purity=n["write_purity"], k=n["write_k"]))
    out["N2_polysemanticity_top64"] = dict(
        frac_multi_input=sum(p["n_input_vars"] >= 2 for p in poly) / len(poly),
        frac_pure_write=sum(p["purity"] >= 0.8 for p in poly) / len(poly),
        frac_multi_input_and_pure=sum(p["n_input_vars"] >= 2 and p["purity"] >= 0.8 for p in poly) / len(poly),
        dominant_k_hist={kk: sum(1 for p in poly if p["k"] == kk) for kk in range(1, 7)}, neurons=poly)
    # N3, N4, N5, N6
    n3, rho, Uh = residual_contrib(m, L, P_, un)
    out["N3_residual"] = n3
    out["N4_residual_sectors"] = residual_sectors(m, P_, rho, Uh, un)
    K = None
    if with_slens:
        K, Tx = s_lens(m, L)
        # exact per-prompt S-reads of the read direction of the 16 most important neurons, by position
        reads = {}
        for i in imp_idx[:16]:
            r = Tx @ m.W_in.weight[i].detach()                         # N x 8
            reads[i] = (r.abs().mean(0) / r.abs().mean(0).sum()).tolist()
        out["S_lens_position_share_top16"] = reads
    out["N4_sectors"] = sectors(m, L, P_, un, K=K)
    out["N5_basis"] = basis_compare(m, P_, un)
    minput = torch.tensor([n["input_r2"]["M"] > max(n["input_r2"]["A"], n["input_r2"]["S"]) and n["active_frac"] > 0
                           for n in neurons])
    out["N6_program"] = program_out(m, P_, L, un, imp, minput)
    out["base_acc"] = base_acc
    return out


# ------------------------------------------------------------------ validation on constructed networks
def validate():
    res = {}
    L = C.domain()
    un = C.unseen_mask(L)
    for nm, (m, _) in (("designC", C.build_C()), ("designL", C.build_L())):
        P_ = parts(m, L)
        r = {}
        rec = m.W_U(P_["h"] + m.W_out.bias) + P_["a"] @ P_["q"].T
        r["V0_E1_max_abs_err"] = float((rec - P_["logits"]).abs().max())
        U = P_["u"].double()
        oh = RS.onehots(L)
        Uh = RS.fit(RS.design(list(oh.values())), U)
        rho = U - Uh
        tot = float(((U - U.mean(0)) ** 2).sum())
        r["V1_token_additive_r2"] = 1 - float((rho ** 2).sum()) / tot
        # V2: pairwise attribution of rho
        import itertools
        pr = {}
        for (k1, _), (k2, _) in itertools.combinations(RS.FACT, 2):
            X = (oh[k1][:, :, None] * oh[k2][:, None, :]).reshape(len(U), -1)
            pr[f"{k1}x{k2}"] = RS.r2_of(RS.fit(RS.design([X]), rho), rho) if float((rho ** 2).sum()) > 1e-9 else None
        both = torch.cat([(oh["m"][:, :, None] * oh["mu"][:, None, :]).reshape(len(U), -1),
                          (oh["a"][:, :, None] * oh["au"][:, None, :]).reshape(len(U), -1)], 1)
        rh = RS.fit(RS.design([both]), rho)
        r["V2_pairs"] = pr
        r["V2_unit_conversion_pairs_r2"] = RS.r2_of(rh, rho) if float((rho ** 2).sum()) > 1e-9 else None
        y = P_["y"]
        q = P_["q"].double()
        base = P_["logits"].double() - torch.relu(U) @ q.T
        def acc(uu):
            p = (base + torch.relu(uu) @ q.T).argmax(-1)
            return float((p[~un] == y[~un]).double().mean()), float((p[un] == y[un]).double().mean())
        r["V2_acc_alpha0"] = acc(Uh)
        r["V2_acc_restore_unit_pairs"] = acc(Uh + rh)
        # V3: active writing units per prompt and write support
        ystar = competitor(P_["logits"], y)
        mg = margins(P_, ystar)
        r["V3_max_units_with_nonzero_margin"] = int((mg.abs() > 1e-6).sum(1).max())
        r["V3_write_support_sizes"] = sorted({int((P_["q"][:, i].abs() > 1e-6).sum()) for i in range(P_["q"].shape[1]) if P_["q"][:, i].abs().sum() > 0})
        # V4: S-lens position attribution of the M-region read (Design C) / mass lift (Design L)
        K, Tx = s_lens(m, L)
        if nm == "designC":
            v = torch.zeros(m.d); v[C.DIM["Q"]] = torch.tensor([float(C._lg(x)) for x in range(P)])
        else:
            v = torch.zeros(m.d); v[C.DL["HM"]] = 1
        rd = (Tx @ v).abs()                                            # N x 8
        mass_pos = torch.where(L["order"] == 0, 0, 3)
        on = rd.gather(1, mass_pos[:, None]).squeeze(1) + rd.gather(1, (mass_pos + 1)[:, None]).squeeze(1)
        r["V4_mass_pair_share"] = float(on.sum() / rd.sum().clamp_min(1e-12))
        # V5: sectors
        s = sectors(m, L, P_, un, K=K)
        used = set(C.DIM["Q"] + C.DIM["A"] + C.DIM["FU"] + C.DIM["MU"] + C.DIM["AU"] + [C.DIM["NULL"]]) if nm == "designC" else \
            set([C.DL[k] for k in ("HM", "HZM", "HA", "HZA", "HS")])
        Jb = jbar(m, P_["u"]).double()
        Rm = m.W_U.weight.double() @ Jb
        _, Sv, Vt = torch.linalg.svd(Rm, full_matrices=True)
        nb = s["dark_sector_h"]["bright_dims"]
        bright = Vt[:nb]
        r["V5_bright_mass_outside_used_dims"] = float(bright[:, [i for i in range(m.d) if i not in used]].pow(2).sum() / bright.pow(2).sum())
        r["V5_sectors"] = s
        res[nm] = r
    return res


def run_validate():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    r = validate()
    (OUT / "validation.json").write_text(json.dumps(r, indent=1, default=str))
    for nm, v in r.items():
        print(nm, {k: (v[k] if not isinstance(v[k], dict) else "...") for k in v if k != "V2_pairs"})
        print("   pairs", {k: (round(x, 3) if x is not None else None) for k, x in v["V2_pairs"].items()})
        print("   sectors", json.dumps(v["V5_sectors"], default=str)[:900])


def run_learned():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    for s in (307, 311, 300, 305, 302, 304, 310):
        n = f"mass_only_s{s}"
        a = atlas(R.load(n), n)
        (OUT / f"{n}.json").write_text(json.dumps(a, indent=1, default=str))
        n3 = a["N3_residual"]
        print(n, "E1", f"{a['E1_max_abs_err']:.1e}", "n90", a["n_neurons_90pct_margin"], "| N2", {k: v for k, v in a["N2_polysemanticity_top64"].items() if k != "neurons"},
              "| N3 spearman", {k: round(v, 2) for k, v in n3["spearman"].items()}, "n90", n3["n_for_90pct"],
              "top16", {k: v[16] for k, v in n3["topk_restore"].items()}, "| N4res", a["N4_residual_sectors"],
              "| dark h", a["N4_sectors"]["dark_sector_h"], "| N5 k8", {k: v[8] for k, v in a["N5_basis"].items()},
              "| N6", a["N6_program"], flush=True)


# ------------------------------------------------------------------ §55.4 developmental tracking
def dev_metrics(m):
    L = C.domain()
    un = C.unseen_mask(L)
    P_ = parts(m, L)
    y = P_["y"]
    ystar = competitor(P_["logits"], y)
    pr = P_["logits"].argmax(-1)
    out = dict(acc=(float((pr[~un] == y[~un]).float().mean()), float((pr[un] == y[un]).float().mean())))
    mg = margins(P_, ystar).abs().mean(0)
    s_ = torch.sort(mg, descending=True).values
    out["n90_margin"] = int((torch.cumsum(s_, 0) / s_.sum().clamp_min(1e-12) < 0.9).sum()) + 1
    top = torch.argsort(-mg)[:32].tolist()
    purs = [fourier_purity(P_["q"][:, i]) for i in top]
    out["top32_median_purity"] = float(torch.tensor([p_ for p_, _ in purs]).median())
    out["top32_k_hist"] = {k: sum(1 for _, kk in purs if kk == k) for k in range(1, 7)}
    h = P_["h"].double()
    Rm = m.W_U.weight.double() @ jbar(m, P_["u"]).double()
    A = Rm.T @ Rm
    sh = (h * (h @ A)).sum(0) / (h * (h @ A)).sum()
    out["block_readout_share"] = float(sh[:16].sum())
    mu_f = h[~un].mean(0)
    _, Sv, Vt = torch.linalg.svd(Rm, full_matrices=True)
    hc = h - mu_f
    en = ((hc @ Vt.T[:, :len(Sv)]) * Sv) ** 2
    nb = int((torch.cumsum(en.sum(0), 0) / en.sum() < 0.9).sum()) + 1
    out["bright_dims_h"] = nb
    Vd = Vt[nb:].T
    hd = h - (hc @ Vd) @ Vd.T
    lg_full = head_logits(m, h.float()).double()
    lgd = head_logits(m, hd.float()).double()
    prd = lgd.argmax(-1)
    out["dark_ablate_acc"] = (float((prd[~un] == y[~un]).double().mean()), float((prd[un] == y[un]).double().mean()))
    out["dark_actual_over_lens"] = float((lgd - lg_full).abs().mean() / ((hd - h) @ Rm.T).abs().mean().clamp_min(1e-12))
    U = P_["u"].double()
    oh = RS.onehots(L)
    Uh = RS.fit(RS.design(list(oh.values())), U)
    rho = U - Uh
    q = P_["q"].double()
    base0 = P_["logits"].double() - (torch.relu(U) - torch.relu(Uh)) @ q.T
    p0 = base0.argmax(-1)
    out["rho_var_share"] = float((rho ** 2).sum() / ((U - U.mean(0)) ** 2).sum())
    out["acc_without_rho"] = (float((p0[~un] == y[~un]).double().mean()), float((p0[un] == y[un]).double().mean()))
    W = m.W_in.weight[top]
    tot = {c: float((P_["comp"][c] @ W.T).abs().mean()) for c in P_["comp"]}
    z = sum(tot.values())
    out["conn_share"] = dict(embed=tot["embed"] / z, layer1=tot["l1"] / z,
                             layer2=sum(v for k, v in tot.items() if k.startswith("l2h")) / z)
    return out


def _dev_job(args):
    net, root, steps = args
    torch.set_num_threads(1)
    from safetensors.torch import load_file
    from nri.model import TinyTyped
    import run_swap as RSW
    m = TinyTyped("mass_only", **RSW.MODEL_KW)
    res = {}
    for st in steps:
        m.load_state_dict(load_file(f"{root}/mass_only_s{net}/ckpt/step_{st:05d}/model.safetensors"))
        res[st] = dev_metrics(m)
    return net, res


def run_develop():
    from concurrent.futures import ProcessPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, "/home/user/swap_window", list(range(600, 2001, 20))) for s in (307, 311, 305, 304, 310)]
    jobs += [(s, "/home/user/swap_recon/zoo", list(range(0, 6001, 100))) for s in (307, 311, 300, 305, 302, 304, 310)]
    res = {"window": {}, "zoo": {}}
    with ProcessPoolExecutor(4) as ex:
        for (net, r), (_, root, _) in zip(ex.map(_dev_job, jobs), jobs):
            res["window" if "window" in root else "zoo"][net] = r
            (OUT / "development.json").write_text(json.dumps(res, indent=1, default=str))
            print("done", net, root, flush=True)


def run_direct_path():
    """Doc 56 §56.5: the residual stream's direct write to the logits vs the MLP's (post-hoc measurement)."""
    L = C.domain()
    res = {}
    for s_ in (307, 311, 300, 305, 302, 304, 310):
        m = R.load(f"mass_only_s{s_}")
        P_ = parts(m, L)
        y = P_["y"]
        q, a, full = P_["q"].double(), P_["a"].double(), P_["logits"].double()
        direct, mlp = full - a @ q.T, a @ q.T
        sigma = torch.tensor([pow(c, 5, P) for c in range(P)])
        inv = torch.empty(P, dtype=torch.long); inv[sigma] = torch.arange(P)
        tgt = sigma[y]
        res[s_] = dict(direct_class_spread=float((direct.max(1).values - direct.min(1).values).mean()),
                       mlp_class_spread=float((mlp.max(1).values - mlp.min(1).values).mean()),
                       corr_direct_mlp=float(torch.corrcoef(torch.stack([direct.flatten(), mlp.flatten()]))[0, 1]),
                       F5_mlp_only=float(((direct + mlp[:, inv]).argmax(-1) == tgt).double().mean()),
                       F5_all_logits=float((full[:, inv].argmax(-1) == tgt).double().mean()))
    (OUT / "direct_path.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"validate": run_validate, "learned": run_learned, "develop": run_develop, "direct": run_direct_path}[sys.argv[1]]()
