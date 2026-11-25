"""Doc 53 §53.5 (R1-R3): the unexplained part of the learned decoders' pre-activations.

u = W_in h + b (256 units) over all 9,126 prompts; u_hat = token-additive fit (one-hot m, mu, a, au, fu, order; full
domain, as in Phase V); rho = u - u_hat.
  R1 graduated removal  u_alpha = u_hat + alpha * rho
  R2 decomposition of rho by candidate families (a)-(g)
  R3 targeted restoration from alpha = 0

    python -m sci.residual      # -> runs/phase6/residual.json
    python -m sci.residual pairs   # -> runs/phase6/residual_pairs.json (exploratory per-pair split)
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import construct as C  # noqa: E402
from sci import rescue as R  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase6"
FACT = (("m", P), ("mu", 3), ("a", P), ("au", 3), ("fu", 3), ("order", 2))


def onehots(L):
    return {k: torch.nn.functional.one_hot(L[k], n).double() for k, n in FACT}


def design(cols):
    X = torch.cat(cols, 1)
    return torch.cat([X, torch.ones(len(X), 1, dtype=torch.float64)], 1)


def fit(X, Y):
    coef = torch.linalg.lstsq(X, Y, driver="gelsd").solution
    return X @ coef


def r2_of(Yhat, Y):
    return 1 - float(((Y - Yhat) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


@torch.no_grad()
def forward_parts(m, L):
    b = T.build(L)
    x0 = m.embed(b.tokens, torch.zeros_like(b.tokens))
    x1 = x0 + m.attn1(x0)
    a2, p2 = m.attn2(x1, return_pattern=True)
    h = (x1 + a2)[:, -1]
    u = h @ m.W_in.weight.T + m.W_in.bias
    return b, h, u, p2


@torch.no_grad()
def readout(m, h, u):
    return m.W_U(h + m.W_out(torch.relu(u)))


def scores(m, h, u, y, un):
    lg = readout(m, h.float(), u.float())
    pred = lg.argmax(-1)
    corr = lg.gather(1, y[:, None]).squeeze(1)
    other = lg.scatter(1, y[:, None], -1e9).max(1).values
    return dict(fam=float((pred[~un] == y[~un]).float().mean()), unseen=float((pred[un] == y[un]).float().mean()),
                margin=float((corr - other).mean()))


def analyse(name):
    m = R.load(name)
    L = C.domain()
    un = C.unseen_mask(L)
    b, h, u, p2 = forward_parts(m, L)
    y = b.y
    U = u.double()
    oh = onehots(L)
    Xadd = design(list(oh.values()))
    Uhat = fit(Xadd, U)
    rho = U - Uhat
    out = dict(kind="learned", r2_additive=r2_of(Uhat, U), rho_var_frac=float((rho ** 2).sum() / ((U - U.mean(0)) ** 2).sum()))
    # ---- R1 graduated removal
    flips_full = (torch.sign(Uhat) != torch.sign(U))
    out["R1"] = {}
    for al in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        ua = Uhat + al * rho
        out["R1"][str(al)] = dict(**scores(m, h, ua, y, un),
                                  relu_state_flips=float(((ua > 0) != (U > 0)).double().mean()))
    out["sign_flip_cells_at_alpha0"] = float(flips_full.double().mean())
    # ---- R2 decomposition of rho
    M, A, S = C.canon_scales(L, dict(MU=T.MASS_S, AU=T.ACC_S))
    Ffull = T.target(M, A, S)
    fams = {}
    fams["a_canonical_operands"] = [torch.nn.functional.one_hot(M, P).double(), torch.nn.functional.one_hot(A, P).double(),
                                    torch.nn.functional.one_hot(L["fu"], 3).double()]
    zm, za = (L["m"] == 0).double()[:, None], (L["a"] == 0).double()[:, None]
    fams["b_zero_handling"] = [zm, za] + [zm * v for k, v in oh.items() if k != "m"] + [za * v for k, v in oh.items() if k != "a"]
    pair = []
    for (k1, _), (k2, _) in itertools.combinations(FACT, 2):
        pair.append((oh[k1][:, :, None] * oh[k2][:, None, :]).reshape(len(U), -1))
    fams["c1_pairwise_tokens"] = pair
    lf = torch.tensor([C.DLOG.get(int(f), 0) for f in Ffull]).double()
    ph = [torch.stack([torch.cos(2 * torch.pi * k * lf / 12), torch.sin(2 * torch.pi * k * lf / 12)], 1) for k in range(1, 7)]
    fams["c2_product_phase"] = ph + [(Ffull == 0).double()[:, None]]
    fams["d_attention_weights_L2"] = [p2[:, :, -1, :].reshape(len(U), -1).double()]
    fams["e_order_x_token"] = [oh["order"][:, 1:2] * v for k, v in oh.items() if k != "order"]
    fit_parts = {}
    out["R2"] = {}
    for k, cols in fams.items():
        rh = fit(design(cols), rho)
        fit_parts[k] = rh
        out["R2"][k] = dict(r2_of_rho=r2_of(rh, rho), n_regressors=int(sum(c.shape[1] for c in cols)))
    allcols = sum(fams.values(), [])
    rh_all = fit(design(allcols), rho)
    out["R2"]["all_families"] = dict(r2_of_rho=r2_of(rh_all, rho))
    fit_parts["all_families"] = rh_all
    out["R2"]["f_boundary_energy_share"] = float((rho[Uhat.abs() < rho.abs()] ** 2).sum() / (rho ** 2).sum())
    out["R2"]["f_boundary_cell_frac"] = float((Uhat.abs() < rho.abs()).double().mean())
    # (g) numerics: float64 forward
    m64 = R.load(name).double()
    _, h64, u64, _ = forward_parts(m64, L)
    out["R2"]["g_float64_vs_float32_max_abs_preact_diff"] = float((u64 - U).abs().max())
    out["R2"]["g_rho_rms"] = float(rho.pow(2).mean().sqrt())
    out["R2"]["g_predictions_identical_f64"] = bool((readout(m64, h64, u64).argmax(-1) == readout(m, h, u).argmax(-1)).all())
    # ---- R3 targeted restoration from alpha = 0
    base0 = out["R1"]["0.0"]
    full = out["R1"]["1.0"]
    def rec(s):
        d_f = full["fam"] - base0["fam"]; d_u = full["unseen"] - base0["unseen"]
        return dict(**s, recovered_fam=(s["fam"] - base0["fam"]) / d_f if abs(d_f) > 1e-9 else None,
                    recovered_unseen=(s["unseen"] - base0["unseen"]) / d_u if abs(d_u) > 1e-9 else None)
    out["R3"] = {}
    flip = flips_full.double()
    out["R3"]["sign_relevant_cells"] = rec(scores(m, h, Uhat + rho * flip, y, un))
    lg0 = readout(m, h.float(), Uhat.float())
    corr0 = lg0.gather(1, y[:, None]).squeeze(1)
    oth0 = lg0.scatter(1, y[:, None], -1e9).max(1).values
    low = ((corr0 - oth0) < 1).double()[:, None]
    out["R3"]["low_margin_prompts"] = rec(scores(m, h, Uhat + rho * low, y, un))
    out["R3"]["low_margin_prompt_frac"] = float(low.mean())
    for k, rh in fit_parts.items():
        out["R3"][f"family:{k}"] = rec(scores(m, h, Uhat + rh, y, un))
    return out


def pairs():
    """Exploratory refinement of R2(c1): which token pair carries rho (per-pair R2 and restoration)."""
    torch.set_num_threads(4)
    L = C.domain()
    un = C.unseen_mask(L)
    out = {}
    for s in (307, 311, 300, 305, 302, 304, 310):
        n = f"mass_only_s{s}"
        m = R.load(n)
        b, h, u, _ = forward_parts(m, L)
        U = u.double()
        oh = onehots(L)
        Uh = fit(design(list(oh.values())), U)
        rho = U - Uh
        r = {}
        for (k1, _), (k2, _) in itertools.combinations(FACT, 2):
            X = (oh[k1][:, :, None] * oh[k2][:, None, :]).reshape(len(U), -1)
            rh = fit(design([X]), rho)
            sc = scores(m, h, Uh + rh, b.y, un)
            r[f"{k1}x{k2}"] = dict(r2=r2_of(rh, rho), fam=sc["fam"], unseen=sc["unseen"])
        out[n] = r
    (OUT / "residual_pairs.json").write_text(json.dumps(out, indent=1))


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for s in (307, 311, 300, 305, 302, 304, 310):
        n = f"mass_only_s{s}"
        res[n] = r = analyse(n)
        print(n, "R2add", round(r["r2_additive"], 3), "| R1:", {a: (round(v["fam"], 2), round(v["unseen"], 2)) for a, v in r["R1"].items()},
              "| R2:", {k: (round(v["r2_of_rho"], 2) if isinstance(v, dict) else round(v, 3) if isinstance(v, float) else v) for k, v in r["R2"].items()},
              "| R3:", {k: ((round(v["fam"], 2), round(v["unseen"], 2)) if isinstance(v, dict) else round(v, 3)) for k, v in r["R3"].items()}, flush=True)
    (OUT / "residual.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    pairs() if sys.argv[1:] == ["pairs"] else main()
