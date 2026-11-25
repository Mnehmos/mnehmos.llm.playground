"""Doc 51 D4: reverse-engineering the learned algorithm over the complete finite domain (9,126 written prompts).

Restricted to prompts with M, A != 0 (F != 0) for the phase analyses; phases are discrete logs base 2 (mod 13):
  phi_M = dlog M, phi_A = dlog A, phi_S = dlog S, phi_F = dlog F = phi_M + phi_A + phi_S (mod 12).
Explanatory models for the decoder's hidden activations z = ReLU(W_in h + b) (and for h, logits):
  cell        one-hot of (written mass value, mass unit, acc value, acc unit, output unit, order)  -- lookup ceiling
  separate_k  cos/sin(k * 2pi * phi_X / 12) for X in {M, A, S}, k = 1..K  (operands encoded separately)
  sum_k       cos/sin(k * 2pi * phi_F / 12), k = 1..K                       (product phase)
  pairs_k     cos/sin of k*(phi_M+phi_A), k*(phi_M+phi_S), k*(phi_A+phi_S)
  product     one-hot of F
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri import dimtask as T  # noqa: E402
from sci import rescue as R  # noqa: E402

P = T.P
DLOG = {}
_x = 1
for _i in range(12):
    DLOG[_x] = _i
    _x = _x * 2 % P


def domain():
    rows = list(itertools.product(range(P), range(3), range(P), range(3), range(3), range(2)))
    t = torch.tensor(rows)
    return dict(m=t[:, 0], mu=t[:, 1], a=t[:, 2], au=t[:, 3], fu=t[:, 4], order=t[:, 5])


def r2(X, Y):
    X = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    coef = torch.linalg.lstsq(X, Y).solution
    res = ((Y - X @ coef) ** 2).sum()
    return 1 - float(res / ((Y - Y.mean(0)) ** 2).sum())


def fourier(phi, K):
    cols = []
    for k in range(1, K + 1):
        th = 2 * math.pi * k * phi.double() / 12
        cols += [torch.cos(th), torch.sin(th)]
    return torch.stack(cols, 1)


@torch.no_grad()
def analyse(m, K=6):
    L = domain()
    b = T.build(L)
    z0 = torch.zeros_like(b.tokens)
    h = m.resid_pre_mlp(b.tokens, z0)
    zh = torch.relu(h @ m.W_in.weight.T + m.W_in.bias)
    logits = m.head_from_resid(h)
    M, A, S = T.canon(L)
    pred = logits.argmax(-1)
    out = dict(domain=len(M), acc_full=float((pred == b.y).float().mean()))
    unseen = ((L["mu"] == 2) & (L["m"] >= 9)) | ((L["au"] == 2) & (L["a"] >= 9))
    out["acc_unseen_full"] = float((pred[unseen] == b.y[unseen]).float().mean())
    nz = (M != 0) & (A != 0)
    pM = torch.tensor([DLOG[int(x)] for x in M[nz]])
    pA = torch.tensor([DLOG[int(x)] for x in A[nz]])
    pS = torch.tensor([DLOG[int(x)] for x in S[nz]])
    pF = (pM + pA + pS) % 12
    cellX = torch.cat([torch.nn.functional.one_hot(L[k][nz], n) for k, n in (("m", P), ("mu", 3), ("a", P), ("au", 3), ("fu", 3), ("order", 2))], 1).double()
    models = {
        "cell_additive": cellX,
        "separate": torch.cat([fourier(pM, K), fourier(pA, K), fourier(pS, K)], 1),
        "sum": fourier(pF, K),
        "pairs": torch.cat([fourier((pM + pA) % 12, K), fourier((pM + pS) % 12, K), fourier((pA + pS) % 12, K)], 1),
        "separate+sum": torch.cat([fourier(pM, K), fourier(pA, K), fourier(pS, K), fourier(pF, K)], 1),
        "product_onehot": torch.nn.functional.one_hot(pF, 12).double(),
    }
    for site, Y in (("h", h[nz].double()), ("hidden", zh[nz].double()), ("logits", logits[nz].double())):
        out[site] = {k: r2(X, Y) for k, X in models.items()}
    # per-frequency share of logits explained by the product phase
    out["logits_sum_by_k"] = {k: r2(fourier(pF, k)[:, -2:], logits[nz].double()) for k in range(1, 7)}
    # per hidden unit: best single explanation
    zz = zh[nz].double()
    active = zz.std(0) > 1e-6
    unit_best = {}
    for name in ("separate", "sum", "pairs"):
        X = torch.cat([models[name], torch.ones(len(zz), 1, dtype=zz.dtype)], 1)
        coef = torch.linalg.lstsq(X, zz).solution
        res = ((zz - X @ coef) ** 2).sum(0)
        tot = ((zz - zz.mean(0)) ** 2).sum(0).clamp_min(1e-12)
        unit_best[name] = (1 - res / tot)[active]
    out["hidden_units_active"] = int(active.sum())
    out["hidden_units_best_model_counts"] = {k: int(sum(1 for i in range(int(active.sum())) if max(unit_best, key=lambda n: float(unit_best[n][i])) == k)) for k in unit_best}
    out["hidden_units_median_r2"] = {k: float(v.median()) for k, v in unit_best.items()}
    return out


def main():
    torch.set_num_threads(4)
    res = {}
    for s in (307, 311, 300, 305, 304, 310):
        n = f"mass_only_s{s}"
        res[n] = analyse(R.load(n))
        r = res[n]
        print(n, "acc", round(r["acc_full"], 3), "unseen", round(r["acc_unseen_full"], 3), "| hidden R2:", {k: round(v, 3) for k, v in r["hidden"].items()},
              "| logits R2:", {k: round(v, 3) for k, v in r["logits"].items()}, "| units best:", r["hidden_units_best_model_counts"], flush=True)
    (R.OUT.parent / "phase5").mkdir(exist_ok=True)
    (R.OUT.parent / "phase5" / "algorithm.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()


@torch.no_grad()
def preact_phases(m, K=6):
    """Pre-activations u = W_in h + b of the decoder's hidden units, explained by Fourier features of
    (i) the five WRITTEN phases separately: dlog(written mass value), dlog(mass unit scale), dlog(written acc value),
        dlog(acc unit scale), dlog(output factor)  -- what a factorised encoder can supply additively;
    (ii) the three CANONICAL phases separately: dlog M, dlog A, dlog S -- requires the encoder to have multiplied
        value and unit already.
    Prompts with any zero written value excluded. Also: per-unit dominant frequency k under model (i)."""
    L = domain()
    keep = (L["m"] != 0) & (L["a"] != 0)
    L = {k: v[keep] for k, v in L.items()}
    b = T.build(L)
    h = m.resid_pre_mlp(b.tokens, torch.zeros_like(b.tokens))
    u = (h @ m.W_in.weight.T + m.W_in.bias).double()
    dl = lambda x: torch.tensor([DLOG[int(t)] for t in x])
    ph_written = [dl(L["m"]), dl(torch.tensor(T.MASS_S)[L["mu"]]), dl(L["a"]), dl(torch.tensor(T.ACC_S)[L["au"]]),
                  dl(torch.tensor([pow(s, -1, P) for s in T.FORCE_S])[L["fu"]])]
    M, A, S = T.canon(L)
    ph_canon = [dl(M), dl(A), dl(S)]
    X_w = torch.cat([fourier(p, K) for p in ph_written], 1)
    X_c = torch.cat([fourier(p, K) for p in ph_canon], 1)
    X_onehot = torch.cat([torch.nn.functional.one_hot(L[k], n) for k, n in (("m", P), ("mu", 3), ("a", P), ("au", 3), ("fu", 3), ("order", 2))], 1).double()
    out = dict(preact_written_phases=r2(X_w, u), preact_canonical_phases=r2(X_c, u), preact_token_additive=r2(X_onehot, u))
    # dominant frequency per unit under written phases
    doms = []
    for k in range(1, K + 1):
        Xk = torch.cat([fourier(p, k)[:, -2:] for p in ph_written], 1)
        X1 = torch.cat([Xk, torch.ones(len(u), 1, dtype=u.dtype)], 1)
        coef = torch.linalg.lstsq(X1, u).solution
        doms.append(1 - ((u - X1 @ coef) ** 2).sum(0) / ((u - u.mean(0)) ** 2).sum(0).clamp_min(1e-12))
    D = torch.stack(doms)                                   # K x units
    best_k = D.argmax(0) + 1
    out["unit_dominant_k_hist"] = {int(k): int((best_k == k).sum()) for k in range(1, K + 1)}
    out["unit_single_k_r2_median"] = float(D.max(0).values.median())
    return out


def main_preact():
    torch.set_num_threads(4)
    res = {}
    for s in (307, 311, 300, 305, 302, 304, 310):
        n = f"mass_only_s{s}"
        res[n] = preact_phases(R.load(n))
        print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[n].items()}, flush=True)
    (R.OUT.parent / "phase5" / "preact_phases.json").write_text(json.dumps(res, indent=1))


@torch.no_grad()
def program_validation(m, k):
    """Explicit program: u_j = c_j + sum_t [p_tj cos(k theta_t) + q_tj sin(k theta_t)], theta_t = 2 pi dlog(t)/12 over the
    five written phases; z = ReLU(u); logits = W_U(h + W_out z) with the network's own readout and residual.
    The coefficients are fitted by least squares on FAMILIAR prompts only (no unseen rendering in the fit) and
    validated over the complete domain (nonzero written values). Reports agreement with the network's predictions
    and accuracy on familiar / unseen renderings. A frequency-scrambled control uses k' with gcd(k', 12) > 1."""
    L = domain()
    keep = (L["m"] != 0) & (L["a"] != 0)
    L = {kk: v[keep] for kk, v in L.items()}
    b = T.build(L)
    h = m.resid_pre_mlp(b.tokens, torch.zeros_like(b.tokens))
    u = (h @ m.W_in.weight.T + m.W_in.bias).double()
    dl = lambda x: torch.tensor([DLOG[int(t)] for t in x])
    ph = [dl(L["m"]), dl(torch.tensor(T.MASS_S)[L["mu"]]), dl(L["a"]), dl(torch.tensor(T.ACC_S)[L["au"]]),
          dl(torch.tensor([pow(s, -1, P) for s in T.FORCE_S])[L["fu"]])]
    unseen = ((L["mu"] == 2) & (L["m"] >= 9)) | ((L["au"] == 2) & (L["a"] >= 9))
    net_pred = m.head_from_resid(h).argmax(-1)
    out = {}
    for name, kk in (("program", k), ("control_k2", 2), ("control_k6", 6)):
        X = torch.cat([fourier(p, kk)[:, -2:] for p in ph] + [torch.ones(len(u), 1, dtype=torch.double)], 1)
        coef = torch.linalg.lstsq(X[~unseen], u[~unseen]).solution
        uh = (X @ coef).float()
        logits = m.W_U(h + m.W_out(torch.relu(uh)))
        pred = logits.argmax(-1)
        out[name] = dict(k=kk, agreement_with_network=float((pred == net_pred).float().mean()),
                         acc_familiar=float((pred[~unseen] == b.y[~unseen]).float().mean()),
                         acc_unseen=float((pred[unseen] == b.y[unseen]).float().mean()),
                         preact_r2_familiar=1 - float(((u[~unseen] - uh.double()[~unseen]) ** 2).sum() / ((u[~unseen] - u[~unseen].mean(0)) ** 2).sum()),
                         preact_r2_unseen=1 - float(((u[unseen] - uh.double()[unseen]) ** 2).sum() / ((u[unseen] - u[unseen].mean(0)) ** 2).sum()))
    out["network"] = dict(acc_familiar=float((net_pred[~unseen] == b.y[~unseen]).float().mean()), acc_unseen=float((net_pred[unseen] == b.y[unseen]).float().mean()))
    return out


def main_program():
    torch.set_num_threads(4)
    res = {}
    for s, k in ((307, 1), (311, 5), (300, 1), (305, 1), (304, 1), (310, 5), (302, 2)):
        n = f"mass_only_s{s}"
        res[n] = program_validation(R.load(n), k)
        print(n, json.dumps({a: {c: (round(d, 3) if isinstance(d, float) else d) for c, d in v.items()} for a, v in res[n].items()}), flush=True)
    (R.OUT.parent / "phase5" / "program_validation.json").write_text(json.dumps(res, indent=1))


@torch.no_grad()
def additive_program(m, include_order=True):
    """Program B: u_j = c_j + sum over tokens of a learned per-token contribution (all harmonics; = one-hot parts of
    written mass value, mass unit, acc value, acc unit, output unit[, order]), fitted on FAMILIAR prompts only.
    z = ReLU(u), logits = W_U(h_hat + W_out z) where h_hat is the same additive token model of the residual h.
    If the network's computation is 'additive token encoding + ReLU readout', this program reproduces the network on
    unseen renderings (new sums of familiar parts). Includes zero values (all prompts)."""
    L = domain()
    b = T.build(L)
    h = m.resid_pre_mlp(b.tokens, torch.zeros_like(b.tokens))
    u = (h @ m.W_in.weight.T + m.W_in.bias).double()
    facs = (("m", P), ("mu", 3), ("a", P), ("au", 3), ("fu", 3)) + ((("order", 2),) if include_order else ())
    X = torch.cat([torch.nn.functional.one_hot(L[kk], n) for kk, n in facs] + [torch.ones(len(u), 1, dtype=torch.long)], 1).double()
    unseen = ((L["mu"] == 2) & (L["m"] >= 9)) | ((L["au"] == 2) & (L["a"] >= 9))
    cu = torch.linalg.lstsq(X[~unseen], u[~unseen]).solution
    ch = torch.linalg.lstsq(X[~unseen], h.double()[~unseen]).solution
    uh, hh = (X @ cu).float(), (X @ ch).float()
    net = m.head_from_resid(h).argmax(-1)
    res = {}
    for name, (U_, H_) in {"additive_preact_real_skip": (uh, h), "additive_preact_and_skip": (uh, hh),
                           "real_preact_additive_skip": (u.float(), hh)}.items():
        pred = m.W_U(H_ + m.W_out(torch.relu(U_))).argmax(-1)
        res[name] = dict(agreement=float((pred == net).float().mean()),
                         acc_familiar=float((pred[~unseen] == b.y[~unseen]).float().mean()),
                         acc_unseen=float((pred[unseen] == b.y[unseen]).float().mean()))
    res["preact_r2_familiar"] = 1 - float(((u[~unseen] - uh.double()[~unseen]) ** 2).sum() / ((u[~unseen] - u[~unseen].mean(0)) ** 2).sum())
    res["preact_r2_unseen"] = 1 - float(((u[unseen] - uh.double()[unseen]) ** 2).sum() / ((u[unseen] - u[unseen].mean(0)) ** 2).sum())
    res["network"] = dict(acc_familiar=float((net[~unseen] == b.y[~unseen]).float().mean()), acc_unseen=float((net[unseen] == b.y[unseen]).float().mean()))
    return res


def main_additive():
    torch.set_num_threads(4)
    res = {}
    for s in (307, 311, 300, 305, 302, 304, 310):
        n = f"mass_only_s{s}"
        res[n] = additive_program(R.load(n))
        print(n, json.dumps({a: ({c: round(d, 3) for c, d in v.items()} if isinstance(v, dict) else round(v, 3)) for a, v in res[n].items()}), flush=True)
    (R.OUT.parent / "phase5" / "additive_program.json").write_text(json.dumps(res, indent=1))


@torch.no_grad()
def pair_program(m):
    """Program C: additive token parts + pairwise cell terms for one quantity, fitted on familiar prompts.
    Variants: +acc cell (a, au) | +mass cell (m, mu) | +both. Scored on prompts whose ACCELERATION rendering is
    familiar (so acc cells are fittable), separately for familiar and unseen MASS renderings."""
    L = domain()
    b = T.build(L)
    h = m.resid_pre_mlp(b.tokens, torch.zeros_like(b.tokens))
    u = (h @ m.W_in.weight.T + m.W_in.bias).double()
    oh = lambda kk, n: torch.nn.functional.one_hot(L[kk], n)
    base = [oh("m", P), oh("mu", 3), oh("a", P), oh("au", 3), oh("fu", 3), oh("order", 2), torch.ones(len(u), 1, dtype=torch.long)]
    acc_cell = torch.nn.functional.one_hot(L["au"] * P + L["a"], 3 * P)
    mass_cell = torch.nn.functional.one_hot(L["mu"] * P + L["m"], 3 * P)
    um = (L["mu"] == 2) & (L["m"] >= 9)
    ua = (L["au"] == 2) & (L["a"] >= 9)
    fam = ~(um | ua)
    net = m.head_from_resid(h).argmax(-1)
    res = {}
    for name, extra in (("additive", []), ("+acc_cell", [acc_cell]), ("+mass_cell", [mass_cell]), ("+both", [acc_cell, mass_cell])):
        X = torch.cat(base + extra, 1).double()
        cu = torch.linalg.lstsq(X[fam], u[fam]).solution
        uh = (X @ cu).float()
        pred = m.W_U(h + m.W_out(torch.relu(uh))).argmax(-1)
        sel_f = fam
        sel_u = um & ~ua
        res[name] = dict(agree_familiar=float((pred[sel_f] == net[sel_f]).float().mean()),
                         agree_unseen_mass=float((pred[sel_u] == net[sel_u]).float().mean()),
                         acc_unseen_mass=float((pred[sel_u] == b.y[sel_u]).float().mean()))
    res["network_acc_unseen_mass"] = float((net[um & ~ua] == b.y[um & ~ua]).float().mean())
    return res


def main_pair():
    torch.set_num_threads(4)
    res = {}
    for s in (307, 311, 300, 305, 302, 304, 310):
        n = f"mass_only_s{s}"
        res[n] = pair_program(R.load(n))
        print(n, json.dumps({a: ({c: round(d, 3) for c, d in v.items()} if isinstance(v, dict) else round(v, 3)) for a, v in res[n].items()}), flush=True)
    (R.OUT.parent / "phase5" / "pair_program.json").write_text(json.dumps(res, indent=1))
