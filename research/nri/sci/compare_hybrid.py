"""Doc 53 §53.6-53.7: constructed vs learned computation (K1-K10) and hybrid repair (H1-H4).

Every fitted map uses FAMILIAR prompts only; targets are functions of the input tokens computed from the task
definition (discrete logs of written values and unit scales, canonical codes). No held-out answer is used.

    python -m sci.compare_hybrid    # -> runs/phase6/compare.json, runs/phase6/hybrid.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri.model import TinyTyped  # noqa: E402
from sci import construct as C  # noqa: E402
from sci import rescue as R  # noqa: E402
from sci.construct_qualify import trace  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase6"
LEARNED = (307, 311, 300, 305, 302, 304, 310)
SC = dict(MU=T.MASS_S, AU=T.ACC_S)


def lg(x):
    return C.DLOG.get(int(x), 0)


def lifts(L):
    """Design-L coordinates computed from the tokens: unwrapped logs and zero flags."""
    lm = torch.tensor([lg(v) + C.DLOG[T.MASS_S[u]] if v else 0 for v, u in zip(L["m"], L["mu"])]).double()
    la = torch.tensor([lg(v) + C.DLOG[T.ACC_S[u]] if v else 0 for v, u in zip(L["a"], L["au"])]).double()
    ls = torch.tensor([C.DLOG[C.SINV[f]] for f in L["fu"]]).double()
    return dict(lm=lm, zm=(L["m"] == 0).double(), la=la, za=(L["a"] == 0).double(), ls=ls)


def affine_fit(X, Y, mask):
    X1 = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    W = torch.linalg.lstsq(X1[mask], Y[mask], driver="gelsd").solution
    return lambda Z: torch.cat([Z, torch.ones(len(Z), 1, dtype=Z.dtype)], 1) @ W


def r2(Yh, Y):
    return 1 - float(((Y - Yh) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum().clamp_min(1e-12))


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))


@torch.no_grad()
def learned_h(m, L):
    b = T.build(L)
    return m.resid_pre_mlp(b.tokens, torch.zeros_like(b.tokens)).double(), b.y


@torch.no_grad()
def head(m, h):
    h = h.to(next(m.parameters()).dtype)
    return m.W_U(h + m.W_out(torch.relu(h @ m.W_in.weight.T + m.W_in.bias)))


def acc_split(pred, y, un):
    return dict(familiar=float((pred[~un] == y[~un]).float().mean()), unseen=float((pred[un] == y[un]).float().mean()))


# ------------------------------------------------------------------ constructed decoder for hybrids (Design-L MLP)
def lift_decoder(t, z, tmin=0, tmax=55):
    """Hats on a real-valued t (exact for integer t; tolerant to |error| < 0.5) + zero logit 2*clamp(z, 0, 1).
    Ungated (doc 53 D1 variant for real-valued inputs): with z ~ 1 the zero class (2) beats any hat (<= 1)."""
    js = torch.arange(tmin - 1, tmax + 2, dtype=t.dtype)
    r = torch.relu(t[:, None] - js[None, :])
    hats = r[:, :-2] - 2 * r[:, 1:-1] + r[:, 2:]          # k = tmin..tmax
    logits = torch.zeros(len(t), P, dtype=t.dtype)
    for i, k in enumerate(range(tmin, tmax + 1)):
        logits[:, pow(2, k % 12, P)] += hats[:, i]
    logits[:, 0] += 2 * (torch.relu(z) - torch.relu(z - 1))
    return logits


def H1(h, L, un, y):
    """Learned encoder -> familiar-fitted affine readouts of the lift coordinates -> constructed decoder."""
    lf = lifts(L)
    Y = torch.stack([lf["lm"], lf["la"], lf["ls"], lf["zm"] + lf["za"]], 1)
    f = affine_fit(h, Y, ~un)
    Yh = f(h)
    pred = lift_decoder(Yh[:, 0] + Yh[:, 1] + Yh[:, 2], Yh[:, 3]).argmax(-1)
    return dict(**acc_split(pred, y, un), readout_r2_familiar=r2(Yh[~un], Y[~un]), readout_r2_unseen=r2(Yh[un], Y[un]),
                readout_max_abs_err_unseen=float((Yh[un, :3].sum(1) - Y[un, :3].sum(1)).abs().max()),
                constructed_params=int(4 * 65 + 58 * 2 + 2 + 58 * 3))


@torch.no_grad()
def H2(m, h, L, un, y, regroup=None):
    """Constructed encoder's canonical one-hot(M) -> familiar-fitted affine map into the learned block (dims 0-15);
    the rest of h and the learned decoder are kept."""
    M, _, _ = C.canon_scales(L, SC)
    if regroup is not None:            # control: renderings assigned to random classes (breaks canonical grouping)
        M = regroup[L["mu"] * P + L["m"]]
    X = torch.nn.functional.one_hot(M, P).double()
    f = affine_fit(X, h[:, :16], ~un)
    h2 = h.clone(); h2[:, :16] = f(X)
    pred = head(m, h2).argmax(-1)
    return acc_split(pred, y, un)


@torch.no_grad()
def H4(m, h, L, un, y):
    """Linear correction of the learned network's own block from its own h (no constructed encoder): a familiar-fitted
    affine map h -> canonical block placement. Tests whether the canonical code is linearly available in h."""
    M, _, _ = C.canon_scales(L, SC)
    X = torch.nn.functional.one_hot(M, P).double()
    place = affine_fit(X, h[:, :16], ~un)(X)              # canonical placement for familiar fit targets
    g = affine_fit(h, place, ~un)
    h2 = h.clone(); h2[:, :16] = g(h)
    pred = head(m, h2).argmax(-1)
    return dict(**acc_split(pred, y, un), changed_params_if_folded=int(16 * 65))


def fact_cells(L, Z):
    cells = [(u, v) for u in range(3) for v in range(P)]
    cm = torch.stack([Z[(L["mu"] == u) & (L["m"] == v)].mean(0) for u, v in cells]).double()
    Xv = torch.nn.functional.one_hot(torch.tensor([v for _, v in cells]), P).double()
    Xu = torch.nn.functional.one_hot(torch.tensor([u for u, _ in cells]), 3).double()
    X = torch.cat([Xv, Xu, torch.ones(len(cells), 1, dtype=torch.float64)], 1)
    return r2(X @ torch.linalg.lstsq(X, cm, driver="gelsd").solution, cm)


def compare_one(m, hC, hL, L, un, y, Cm):
    h, _ = learned_h(m, L)
    M, A, S = C.canon_scales(L, SC)
    canon = torch.cat([torch.nn.functional.one_hot(M, P), torch.nn.functional.one_hot(A, P),
                       torch.nn.functional.one_hot(L["fu"], 3)], 1).double()
    written = torch.cat([torch.nn.functional.one_hot(L[k], n) for k, n in (("m", P), ("mu", 3), ("a", P), ("au", 3), ("fu", 3))], 1).double()
    out = {}
    out["K1_cka"] = dict(vs_designC_h=cka(h, hC), vs_designL_h=cka(h, hL), vs_canonical_features=cka(h, canon),
                         vs_written_token_features=cka(h, written))
    # cell means of the block over the 39 mass renderings (mu, m)
    cells = [(u, v) for u in range(3) for v in range(P)]
    def cellmeans(X):
        return torch.stack([X[(L["mu"] == u) & (L["m"] == v)].mean(0) for u, v in cells])
    blk = cellmeans(h[:, :16])
    dist = lambda Z: torch.cdist(Z, Z).flatten()
    rsa = lambda a, b: float(torch.corrcoef(torch.stack([dist(a), dist(b)]))[0, 1])
    out["K1_rsa_block"] = dict(vs_designC_M=rsa(blk, cellmeans(hC[:, C.DIM["Q"]])),
                               vs_designL_lift=rsa(blk, cellmeans(hL[:, [C.DL["HM"], C.DL["HZM"]]])))
    # K3 factorisation of block cell means
    def fact(Z):
        Xv = torch.nn.functional.one_hot(torch.tensor([v for _, v in cells]), P).double()
        Xu = torch.nn.functional.one_hot(torch.tensor([u for u, _ in cells]), 3).double()
        X = torch.cat([Xv, Xu, torch.ones(len(cells), 1, dtype=torch.float64)], 1)
        Zh = X @ torch.linalg.lstsq(X, Z.double(), driver="gelsd").solution
        return r2(Zh, Z.double())
    out["K3_factorisation"] = fact(blk)
    # K2 effective causal dimensionality of the block
    out["K2_causal_dim"] = causal_dim(m, h, L, y)
    # K4 operation-specific units: per-unit R2 of pre-activation by canonical M vs A vs S
    u_ = (h @ m.W_in.weight.double().T + m.W_in.bias.double())
    dom = {}
    for k, X in (("M", torch.nn.functional.one_hot(M, P)), ("A", torch.nn.functional.one_hot(A, P)), ("S", torch.nn.functional.one_hot(L["fu"], 3))):
        X1 = torch.cat([X.double(), torch.ones(len(X), 1, dtype=torch.float64)], 1)
        res = ((u_ - X1 @ torch.linalg.lstsq(X1, u_, driver="gelsd").solution) ** 2).sum(0)
        dom[k] = 1 - res / ((u_ - u_.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    best = torch.stack([dom["M"], dom["A"], dom["S"]]).argmax(0)
    active = (u_ > 0).any(0)
    out["K4_units_dominated_by"] = {k: int(((best == i) & active).sum()) for i, k in enumerate("MAS")}
    out["K4_median_unit_r2"] = {k: float(v[active].detach().median()) for k, v in dom.items()}
    out["K5_nonzero_params"] = int(sum((p != 0).sum() for p in m.parameters()))
    out["K6_active_frac"] = float((u_ > 0).double().mean())
    out["K7_robustness"] = robustness(m, h, y)
    out["K9_alignment"] = alignment(h, hC, hL, un)
    pred = head(m, h).argmax(-1)
    out["K10_argmax_agreement_with_constructed"] = float((pred == y).float().mean())
    out["K10_interchange_equivalence"] = iit_equiv(m, h, Cm, hC, L, y)
    return out


@torch.no_grad()
def causal_dim(m, h, L, y, n=2048, seed=0, dims=tuple(range(16))):
    g = torch.Generator().manual_seed(seed)
    M, A, S = C.canon_scales(L, SC)
    bi = torch.randint(0, len(y), (n,), generator=g)
    si = torch.randint(0, len(y), (n,), generator=g)
    want = T.target(M[si], A[bi], S[bi])
    disc = want != y[bi]
    dims = list(dims)
    blk = h[:, dims]
    cm = torch.stack([blk[(L["mu"] == u) & (L["m"] == v)].mean(0) for u in range(3) for v in range(P)])
    U, Sv, Vt = torch.linalg.svd(cm - cm.mean(0), full_matrices=False)
    res = {}
    for k in range(1, min(len(dims), 16) + 1):
        Pk = Vt[:k].T @ Vt[:k]
        h2 = h[bi].clone()
        h2[:, dims] = blk[bi] + (blk[si] - blk[bi]) @ Pk
        pred = head(m, h2).argmax(-1)
        res[k] = float((pred[disc] == want[disc]).float().mean())
    kmin = next((k for k, v in res.items() if v >= 0.95), None)
    return dict(curve=res, k_for_0_95=kmin)


@torch.no_grad()
def robustness(m, h, y, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {"h_noise": {}, "weight_noise": {}}
    for s in (0.01, 0.05, 0.1, 0.25, 0.5):
        z = torch.randn(h.shape, generator=g, dtype=h.dtype)
        z = z / z.norm(dim=1, keepdim=True) * h.norm(dim=1, keepdim=True) * s
        out["h_noise"][s] = float((head(m, h + z).argmax(-1) == y).float().mean())
    import copy
    L = C.domain()
    b = T.build(L)
    for s in (0.01, 0.05, 0.1):
        e = copy.deepcopy(m)
        for p in e.parameters():
            rms = p.pow(2).mean().sqrt()
            p.add_(torch.randn(p.shape, generator=g) * rms * s)
        out["weight_noise"][s] = float((e(b.tokens, torch.zeros_like(b.tokens)).argmax(-1) == y).float().mean())
    return out


def alignment(h, hC, hL, un):
    D = C.DIM
    res = {}
    for name, Y in (("designC_M", hC[:, D["Q"]]), ("designC_A", hC[:, D["A"]]), ("designC_FU", hC[:, D["FU"]]),
                    ("designL_lifts", hL[:, [C.DL["HM"], C.DL["HZM"], C.DL["HA"], C.DL["HZA"], C.DL["HS"]]])):
        f = affine_fit(h, Y, ~un)
        Yh = f(h)
        res[name] = dict(r2_familiar=r2(Yh[~un], Y[~un]), r2_unseen=r2(Yh[un], Y[un]))
    return res


@torch.no_grad()
def iit_equiv(m, h, Cm, hC, L, y, n=2048, seed=1):
    g = torch.Generator().manual_seed(seed)
    M, A, S = C.canon_scales(L, SC)
    bi = torch.randint(0, len(y), (n,), generator=g)
    si = torch.randint(0, len(y), (n,), generator=g)
    want = T.target(M[si], A[bi], S[bi])
    disc = want != y[bi]
    h2 = h[bi].clone(); h2[:, :16] = h[si, :16]
    pl = head(m, h2).argmax(-1)
    c2 = hC[bi].clone(); c2[:, C.DIM["Q"]] = hC[si][:, C.DIM["Q"]]
    pc = head(Cm, c2.float()).argmax(-1)
    return dict(learned_disc_acc=float((pl[disc] == want[disc]).float().mean()),
                constructed_disc_acc=float((pc[disc] == want[disc]).float().mean()),
                agreement_disc=float((pl[disc] == pc[disc]).float().mean()), n_disc=int(disc.sum()))


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    L = C.domain()
    un = C.unseen_mask(L)
    Cm, _ = C.build_C()
    Lm, _ = C.build_L()
    hC = trace(Cm, L, torch.float32)["h"].double()
    hL = trace(Lm, L, torch.float32)["h"].double()
    y = T.build(L).y
    comp = {"designC": {}, "designL": {}}
    for name, mm, hh in (("designC", Cm, hC), ("designL", Lm, hL)):
        uu = hh @ mm.W_in.weight.double().T + mm.W_in.bias.double()
        comp[name] = dict(kind="constructed", K5_nonzero_params=int(sum((p != 0).sum() for p in mm.parameters())),
                          K6_active_frac=float((uu > 0).double().mean()), K7_robustness=robustness(mm, hh, y),
                          K1_cka=dict(vs_designC_h=cka(hh, hC), vs_designL_h=cka(hh, hL)))
    comp["designC"]["K3_factorisation_M_region"] = fact_cells(L, hC[:, C.DIM["Q"]])
    comp["designL"]["K3_factorisation_lift"] = fact_cells(L, hL[:, [C.DL["HM"], C.DL["HZM"]]])
    comp["designC"]["K2_causal_dim"] = causal_dim(Cm, hC, L, y, dims=C.DIM["Q"])
    comp["designL"]["K2_causal_dim"] = causal_dim(Lm, hL, L, y, dims=[C.DL["HM"], C.DL["HZM"]])
    hyb = {}
    for s in LEARNED:
        n = f"mass_only_s{s}"
        m = R.load(n)
        comp[n] = dict(kind="learned", **compare_one(m, hC, hL, L, un, y, Cm))
        h, _ = learned_h(m, L)
        regroup = torch.randint(0, P, (3 * P,), generator=torch.Generator().manual_seed(5))
        base = acc_split(head(m, h).argmax(-1), y, un)
        hyb[n] = dict(baseline=base, H1_decoder_replacement=H1(h, L, un, y), H2_representation_replacement=H2(m, h, L, un, y),
                      H2_control_random_grouping=H2(m, h, L, un, y, regroup=regroup), H4_linear_self_correction=H4(m, h, L, un, y))
        c = comp[n]
        print(n, "CKA", {k: round(v, 2) for k, v in c["K1_cka"].items()}, "RSA", {k: round(v, 2) for k, v in c["K1_rsa_block"].items()},
              "fact", round(c["K3_factorisation"], 3), "kdim", c["K2_causal_dim"]["k_for_0_95"], "units", c["K4_units_dominated_by"],
              "active", round(c["K6_active_frac"], 2), "| align unseen", {k: round(v["r2_unseen"], 2) for k, v in c["K9_alignment"].items()},
              "| iit-eq", {k: round(v, 2) for k, v in c["K10_interchange_equivalence"].items()}, flush=True)
        hh_ = hyb[n]
        print("   hybrid: base", base, "| H1", {k: round(v, 3) for k, v in hh_["H1_decoder_replacement"].items()}, "| H2", hh_["H2_representation_replacement"],
              "| H2ctrl", hh_["H2_control_random_grouping"], "| H4", hh_["H4_linear_self_correction"], flush=True)
    # bypass controls for H1: random-initialised network and undeclared ('neither') networks
    ctrl = {}
    torch.manual_seed(0)
    rnd = TinyTyped("mass_only", **C.MODEL_KW)
    for name, mm in [("random_init", rnd)] + [(f"neither_s{s}", R.load(f"neither_s{s}")) for s in (304, 307)]:
        h, _ = learned_h(mm, L)
        ctrl[name] = dict(H1=H1(h, L, un, y), baseline=acc_split(head(mm, h).argmax(-1), y, un))
        print("control", name, {k: round(v, 3) for k, v in ctrl[name]["H1"].items()}, ctrl[name]["baseline"], flush=True)
    hyb["bypass_controls"] = ctrl
    (OUT / "compare.json").write_text(json.dumps(comp, indent=1, default=str))
    (OUT / "hybrid.json").write_text(json.dumps(hyb, indent=1))
    print("constructed:", {k: {a: b for a, b in v.items() if a != "kind"} for k, v in comp.items() if k.startswith("design")})


if __name__ == "__main__":
    main()
