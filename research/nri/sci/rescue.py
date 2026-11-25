"""Doc 48 E1-E3: single-pass interventions on the declared block (no training). All fitted quantities use
calibration data (trained renderings, fixed seeds disjoint from the test panels). Oracle diagnostics are labelled.

Decomposition of the block (dims 0-15 of the last-token residual):  h = g + a_v + b_u + I(u,v) + e(x)
  per-example interaction estimate  I_hat(x) = h - g - a_v - b_u - e_hat(x),  e_hat from h[16:] by ridge regression.

    python -m sci.rescue        # writes runs/phase4/rescue.json, compat.json, inject.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_dimensional as D  # noqa: E402
import run_swap as RS  # noqa: E402
from nri import dimtask as T  # noqa: E402
from sci import window_analysis as W  # noqa: E402

P = T.P
OUT = ROOT / "runs" / "phase4"
CAL, TEST_ID, TEST_OOD, SWAP_B, SWAP_S = 770_001, 770_002, 770_003, 770_004, 770_005
N = 8192


def resid(m, L):
    _, _, h, _, _, _ = W.states(m, L)
    return h


class Decomp:
    """Calibration-fitted additive model of a network's block."""

    def __init__(self, m, n=32768):
        L = W.lat(n, CAL)
        h = resid(m, L)
        blk = h[:, :16].double()
        cell = L["mu"] * P + L["m"]
        ids = torch.unique(cell)
        means = torch.stack([blk[cell == c].mean(0) for c in ids])
        u, v = ids // P, ids % P
        X = torch.cat([torch.nn.functional.one_hot(v, P), torch.nn.functional.one_hot(u, 3)], 1).double()
        coef = torch.linalg.lstsq(X, means).solution
        self.a, self.b = coef[:P], coef[P:]
        self.I = {int(c): means[i] - X[i] @ coef for i, c in enumerate(ids)}
        cm = torch.stack([self.I[int(c)] + X[i] @ coef for i, c in enumerate(ids)])
        e = blk - cm[torch.searchsorted(ids, cell)]
        R = torch.cat([h[:, 16:].double(), torch.ones(n, 1, dtype=torch.double)], 1)
        self.E = torch.linalg.solve(R.T @ R + 1e-2 * torch.eye(R.shape[1], dtype=R.dtype), R.T @ e)
        Im = torch.stack(list(self.I.values()))
        s = torch.linalg.svdvals(Im)
        self.int_rank = int((s > 0.01 * s[0]).sum())
        self.int_rel = float(Im.norm(dim=1).mean() / torch.cat([self.a, self.b]).norm(dim=1).mean())
        # causal subspace S: span of canonical-M class means of the block (trained renderings)
        M, _, _ = T.canon(L)
        mu = torch.stack([blk[M == k].mean(0) for k in range(P)])
        _, _, Vh = torch.linalg.svd(mu - mu.mean(0), full_matrices=False)
        self.S = Vh[:12].T
        self.Mmean = mu
        self.mu_blk = blk.mean(0)

    def additive(self, L):
        return self.a[L["m"]] + self.b[L["mu"]]

    def ehat(self, h):
        R = torch.cat([h[:, 16:].double(), torch.ones(len(h), 1, dtype=torch.double)], 1)
        return R @ self.E

    def ihat(self, h, L):
        return h[:, :16].double() - self.additive(L) - self.ehat(h)


def with_block(h, blk):
    h2 = h.clone()
    h2[:, :16] = blk.float()
    return h2


@torch.no_grad()
def score(m, transform, transform_src=None):
    """transform(h, L) -> h'. Returns ID acc, OOD-M acc, discriminating unseen-source interchange (PR #34 estimand)."""
    out = {}
    for name, L in (("id", W.lat(N, TEST_ID)), ("oodM", W.ood_mass(N, TEST_OOD))):
        b = T.build(L)
        out[name] = float((m.head_from_resid(transform(resid(m, L), L)).argmax(-1) == b.y).float().mean())
    base, src = W.lat(4096, SWAP_B), W.ood_mass(4096, SWAP_S)
    hb = transform(resid(m, base), base)
    hs = (transform_src or transform)(resid(m, src), src)
    hp = hb.clone()
    hp[:, :16] = hs[:, :16]
    pred = m.head_from_resid(hp).argmax(-1)
    M, A, S = T.canon(base)
    Ms, As, _ = T.canon(src)
    intended, unchanged, wrong = T.target(Ms, A, S), T.target(M, A, S), T.target(M, As, S)
    el = (intended != unchanged) & (intended != wrong) & (unchanged != wrong)
    out["disc_swap"] = float((pred[el] == intended[el]).float().mean())
    out["disc_n"] = int(el.sum())
    return out


def rescue_suite(m, dec, seed=0):
    g = torch.Generator().manual_seed(seed)
    res = {}
    ident = lambda h, L: h
    res["R0_none"] = score(m, ident)
    res["R1_remove"] = score(m, lambda h, L: with_block(h, h[:, :16].double() - dec.ihat(h, L)))
    for a in (0.75, 0.5, 0.25):
        res[f"R2_keep{a}"] = score(m, lambda h, L, a=a: with_block(h, h[:, :16].double() - (1 - a) * dec.ihat(h, L)))
    PS = dec.S @ dec.S.T
    res["R3_inS"] = score(m, lambda h, L: with_block(h, h[:, :16].double() - dec.ihat(h, L) @ PS))
    res["R3_outS"] = score(m, lambda h, L: with_block(h, h[:, :16].double() - dec.ihat(h, L) @ (torch.eye(16, dtype=PS.dtype) - PS)))
    res["R4_additive"] = score(m, lambda h, L: with_block(h, dec.additive(L) + dec.ehat(h)))
    # oracle diagnostic: block <- class mean of trained renderings with the same canonical M (+ context estimate)
    res["D1_oracle_canonical"] = score(m, lambda h, L: with_block(h, dec.Mmean[T.canon(L)[0]] + dec.ehat(h)))
    # controls
    r = max(dec.int_rank, 1)
    Q, _ = torch.linalg.qr(torch.randn(16, r, generator=g, dtype=torch.double))
    res["C_random_rank_removal"] = score(m, lambda h, L: with_block(h, h[:, :16].double() - ((h[:, :16].double() - dec.mu_blk) @ Q) @ Q.T))

    def mag(h, L):
        ih = dec.ihat(h, L)
        z = torch.randn(ih.shape, generator=g, dtype=torch.double)
        z = z / z.norm(dim=1, keepdim=True) * ih.norm(dim=1, keepdim=True)
        return with_block(h, h[:, :16].double() + z)
    res["C_magnitude_matched"] = score(m, mag)
    res["meta"] = dict(int_rank=dec.int_rank, int_rel_norm=dec.int_rel)
    return res


@torch.no_grad()
def compat(models, n_cal=16384):
    """Representation x decoder: map R's full residual (or block) into D's residual space by an affine map fitted on
    calibration inputs; run D's head on OOD-M and ID panels."""
    Lc = W.lat(n_cal, CAL + 1)
    Lid, Lood = W.lat(N, TEST_ID), W.ood_mass(N, TEST_OOD)
    H = {k: (resid(m, Lc), resid(m, Lid), resid(m, Lood)) for k, m in models.items()}
    out = {}
    for r, (hc_r, hid_r, hood_r) in H.items():
        for d_, (hc_d, hid_d, hood_d) in H.items():
            X = torch.cat([hc_r, torch.ones(len(hc_r), 1)], 1).double()
            A = torch.linalg.solve(X.T @ X + 1e-3 * torch.eye(65, dtype=X.dtype), X.T @ hc_d.double())
            mp = lambda h: (torch.cat([h, torch.ones(len(h), 1)], 1).double() @ A).float()
            md = models[d_]
            full_ood = float((md.head_from_resid(mp(hood_r)).argmax(-1) == T.build(Lood).y).float().mean())
            full_id = float((md.head_from_resid(mp(hid_r)).argmax(-1) == T.build(Lid).y).float().mean())
            hb = hood_d.clone()
            hb[:, :16] = mp(hood_r)[:, :16]
            blk_ood = float((md.head_from_resid(hb).argmax(-1) == T.build(Lood).y).float().mean())
            out[f"{r}->{d_}"] = dict(full_ood=full_ood, full_id=full_id, block_ood=blk_ood)
    return out


@torch.no_grad()
def inject(m, dec, seed=0):
    """E3: add a synthetic double-centred interaction I*(u,v) to the block of a generaliser."""
    out = {}
    g = torch.Generator().manual_seed(seed)
    scale = float(torch.cat([dec.a, dec.b]).norm(dim=1).mean())
    PS = dec.S @ dec.S.T
    cells = [(u, v) for u in range(3) for v in range(P)]
    for r in (1, 2, 4, 8):
        Z = torch.randn(3, P, r, generator=g, dtype=torch.double)
        Z = Z - Z.mean(0, keepdim=True) - Z.mean(1, keepdim=True) + Z.mean((0, 1), keepdim=True)   # double-centred
        B = torch.linalg.qr(torch.randn(16, 16, generator=g, dtype=torch.double))[0][:, :r]
        for orient in ("inS", "outS"):
            Br = (PS @ B) if orient == "inS" else ((torch.eye(16, dtype=PS.dtype) - PS) @ B)
            Ist = Z @ Br.T                                                    # 3 x 13 x 16
            Ist = Ist / Ist.norm(dim=-1).mean()
            for spec in ("all", "unit2", "unseen"):
                for k in (0.25, 0.5, 1.0, 2.0):
                    def tf(h, L, Ist=Ist, k=k, spec=spec):
                        add = Ist[L["mu"], L["m"]] * k * scale
                        if spec == "unit2":
                            add = add * (L["mu"] == 2)[:, None]
                        elif spec == "unseen":
                            add = add * ((L["mu"] == 2) & (L["m"] >= 9))[:, None]
                        return with_block(h, h[:, :16].double() + add)
                    s = score(m, tf)
                    out[f"r{r}_{orient}_{spec}_k{k}"] = dict(rank=r, orient=orient, spec=spec, kappa=k, **s)
    return out


def load(name):
    m, _ = D.load(ROOT / "runs/zoo_swap" / name)
    return m


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    nets = [f"mass_only_s{s}" for s in (304, 310, 302, 300, 305, 307, 311)]
    models = {n: load(n) for n in nets}
    decs = {n: Decomp(models[n]) for n in nets}
    rescue = {n: rescue_suite(models[n], decs[n]) for n in nets}
    (OUT / "rescue.json").write_text(json.dumps(rescue, indent=1))
    for n, r in rescue.items():
        print(n, r["meta"], {k: (round(v["oodM"], 3), round(v["id"], 3), round(v["disc_swap"], 3)) for k, v in r.items() if k != "meta"}, flush=True)
    cm = compat({n.split("_s")[1]: models[n] for n in nets})
    (OUT / "compat.json").write_text(json.dumps(cm, indent=1))
    inj = {n: inject(models[n], decs[n]) for n in ("mass_only_s307", "mass_only_s311")}
    (OUT / "inject.json").write_text(json.dumps(inj, indent=1))


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ R6: geometric completion of missing cells
def completion_suite(m, dec, ks=(2, 3, 4, 6)):
    """Non-oracle repair of the encoder's missing (unit 2, value 9-12) cells: in the top-k principal plane of the
    trained cell means, fit a map z(0,v) -> z(2,v) on trained values v = 0..8 (calibration only), extrapolate to
    v = 9..12, and write the completed cell mean (+ the context estimate) into the block of unseen renderings. Trained
    renderings are left untouched. Two map families: affine (least squares) and orthogonal (Procrustes + shift)."""
    L = W.lat(32768, CAL)
    h = resid(m, L)
    blk = h[:, :16].double()
    cm = {}
    for u in range(3):
        for v in range(P):
            s = (L["mu"] == u) & (L["m"] == v)
            if s.sum() > 3:
                cm[(u, v)] = blk[s].mean(0)
    C = torch.stack(list(cm.values()))
    mu = C.mean(0)
    _, _, Vh = torch.linalg.svd(C - mu, full_matrices=False)
    out = {}
    for k in ks:
        Bk = Vh[:k].T
        z = {c: (x - mu) @ Bk for c, x in cm.items()}
        tr = [v for v in range(9) if (0, v) in z and (2, v) in z]
        Z0 = torch.stack([z[(0, v)] for v in tr])
        Z2 = torch.stack([z[(2, v)] for v in tr])
        X = torch.cat([Z0, torch.ones(len(tr), 1, dtype=Z0.dtype)], 1)
        A = torch.linalg.lstsq(X, Z2).solution
        m0, m2 = Z0.mean(0), Z2.mean(0)
        U, _, Vt = torch.linalg.svd((Z0 - m0).T @ (Z2 - m2))
        R = U @ Vt
        for fam in ("affine", "orthogonal"):
            def comp(v):
                z0 = z[(0, v)]
                zz = (torch.cat([z0, torch.ones(1, dtype=z0.dtype)]) @ A) if fam == "affine" else ((z0 - m0) @ R + m2)
                return mu + zz @ Bk.T + ((cm[(0, v)] - mu) - ((cm[(0, v)] - mu) @ Bk) @ Bk.T) * 0
            filled = {v: comp(v) for v in range(9, 13)}

            def tf(hh, LL, filled=filled):
                out_b = hh[:, :16].double().clone()
                sel = (LL["mu"] == 2) & (LL["m"] >= 9)
                if sel.any():
                    idx = sel.nonzero().flatten()
                    eh = dec.ehat(hh)
                    for i in idx.tolist():
                        out_b[i] = filled[int(LL["m"][i])] + eh[i]
                return with_block(hh, out_b)
            res = score(m, tf)
            # fidelity: does the completed cell match the oracle canonical class mean for that cell?
            Mo = {v: (v * T.MASS_S[2]) % P for v in range(9, 13)}
            err = float(torch.stack([(filled[v] - dec.Mmean[Mo[v]]).norm() / dec.Mmean[Mo[v]].norm() for v in filled]).mean())
            out[f"R6_{fam}_k{k}"] = dict(**res, rel_err_to_canonical_mean=err)
    return out


def run_completion():
    torch.set_num_threads(2)
    nets = [f"mass_only_s{s}" for s in (304, 310, 302, 300, 305, 307, 311)]
    res = {}
    for n in nets:
        m = load(n)
        res[n] = completion_suite(m, Decomp(m))
        print(n, {k: (round(v["oodM"], 3), round(v["id"], 3), round(v["disc_swap"], 3), round(v["rel_err_to_canonical_mean"], 2)) for k, v in res[n].items()}, flush=True)
    (OUT / "completion.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ R7: class completion (equivalence classes + bijectivity)
def class_completion(m, dec, k_geo=4, seed=0):
    """Non-oracle completion using the network's own structure:
    1. cluster the 39 trained cell means of the block into 13 groups (k-means, 50 restarts; no labels used);
    2. a unit acts as an invertible rescaling, so unit 2's 4 unseen cells must belong to the 4 clusters its 9 trained
       cells do not use (an inductive prior about units, not the answer);
    3. assign the 4 unseen cells to those 4 clusters by matching geometric predictions (R6 orthogonal, top-k plane)
       to cluster centroids (Hungarian); control: random assignment.
    Labels are used only to report clustering purity."""
    import itertools
    L = W.lat(32768, CAL)
    h = resid(m, L)
    blk = h[:, :16].double()
    cells = [(u, v) for u in range(3) for v in range(P) if not (u == 2 and v >= 9)]
    cm = torch.stack([blk[(L["mu"] == u) & (L["m"] == v)].mean(0) for u, v in cells])
    g = torch.Generator().manual_seed(seed)
    best = None
    for _ in range(50):
        cent = cm[torch.randperm(len(cm), generator=g)[:P]].clone()
        for _ in range(100):
            lab = torch.cdist(cm, cent).argmin(1)
            new = torch.stack([cm[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(P)])
            if torch.allclose(new, cent):
                break
            cent = new
        inertia = float(((cm - cent[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab.clone(), cent.clone())
    _, lab, cent = best
    Mtrue = torch.tensor([v * T.MASS_S[u] % P for u, v in cells])
    purity = sum(int(torch.bincount(Mtrue[lab == j], minlength=P).max()) for j in range(P) if (lab == j).any()) / len(cells)
    used = {int(lab[i]) for i, (u, v) in enumerate(cells) if u == 2}
    free = [j for j in range(P) if j not in used]
    # geometric prediction (orthogonal map, top-k plane) for the unseen cells
    mu = cm.mean(0)
    _, _, Vh = torch.linalg.svd(cm - mu, full_matrices=False)
    Bk = Vh[:k_geo].T
    z = lambda x: (x - mu) @ Bk
    idx = {c: i for i, c in enumerate(cells)}
    Z0 = torch.stack([z(cm[idx[(0, v)]]) for v in range(9)])
    Z2 = torch.stack([z(cm[idx[(2, v)]]) for v in range(9)])
    m0, m2 = Z0.mean(0), Z2.mean(0)
    U, _, Vt = torch.linalg.svd((Z0 - m0).T @ (Z2 - m2))
    R = U @ Vt
    pred = {v: mu + ((z(cm[idx[(0, v)]]) - m0) @ R + m2) @ Bk.T for v in range(9, 13)}
    out = dict(purity=purity, n_free=len(free))
    if len(free) != 4:
        out["note"] = "cluster structure did not leave exactly 4 free classes; completion not defined"
        return out
    best_perm = min(itertools.permutations(free), key=lambda p: sum(float((pred[v] - cent[j]).norm()) for v, j in zip(range(9, 13), p)))
    rand_perm = list(free)
    rand_perm = [rand_perm[i] for i in torch.randperm(4, generator=g).tolist()]

    def make_tf(perm):
        fill = {v: cent[j] for v, j in zip(range(9, 13), perm)}

        def tf(hh, LL):
            out_b = hh[:, :16].double().clone()
            sel = ((LL["mu"] == 2) & (LL["m"] >= 9)).nonzero().flatten().tolist()
            eh = dec.ehat(hh)
            for i in sel:
                out_b[i] = fill[int(LL["m"][i])] + eh[i]
            return with_block(hh, out_b)
        return tf
    # correctness of the assignment (diagnostic only): does the matched cluster contain cells with the true M?
    truth = {v: (v * T.MASS_S[2]) % P for v in range(9, 13)}
    cl_M = {j: int(torch.bincount(Mtrue[lab == j], minlength=P).argmax()) for j in free}
    out["assignment_correct"] = sum(cl_M[j] == truth[v] for v, j in zip(range(9, 13), best_perm))
    out["R7_geometric"] = score(m, make_tf(best_perm))
    out["R7_random_assignment_control"] = score(m, make_tf(rand_perm))
    out["R7_best_of_24_oracle"] = max((score(m, make_tf(p)) for p in itertools.permutations(free)), key=lambda r: r["oodM"])
    return out


def run_class_completion():
    torch.set_num_threads(2)
    res = {}
    for s in (304, 310, 302, 300, 305, 307, 311):
        n = f"mass_only_s{s}"
        m = load(n)
        res[n] = class_completion(m, Decomp(m))
        r = res[n]
        print(n, {k: (round(v["oodM"], 3), round(v["id"], 3), round(v["disc_swap"], 3)) if isinstance(v, dict) else v for k, v in r.items()}, flush=True)
    (OUT / "class_completion.json").write_text(json.dumps(res, indent=1))


def labelled_class_completion(m, dec, ks=(2, 3, 4, 6, 8)):
    """R7L: as R7, but the classes are the canonical-M classes of TRAINED renderings (training labels; the
    interchange objective itself is trained on them). Unit 2's trained cells cover 9 classes; bijectivity leaves the
    4 free classes. The 4 unseen cells are assigned to them by geometric matching of the R6-orthogonal prediction to
    the class means (Hungarian over 24 permutations). The answer for an unseen cell is never used, except to report
    whether the geometric assignment was correct (diagnostic)."""
    import itertools
    L = W.lat(32768, CAL)
    h = resid(m, L)
    blk = h[:, :16].double()
    cm = {(u, v): blk[(L["mu"] == u) & (L["m"] == v)].mean(0) for u in range(3) for v in range(P) if not (u == 2 and v >= 9)}
    used = {(v * T.MASS_S[2]) % P for v in range(9)}
    free = [k for k in range(P) if k not in used]
    C = torch.stack(list(cm.values()))
    mu = C.mean(0)
    _, _, Vh = torch.linalg.svd(C - mu, full_matrices=False)
    truth = {v: (v * T.MASS_S[2]) % P for v in range(9, 13)}
    out = {"free_classes": free}
    for k in ks:
        Bk = Vh[:k].T
        z = lambda x: (x - mu) @ Bk
        Z0 = torch.stack([z(cm[(0, v)]) for v in range(9)])
        Z2 = torch.stack([z(cm[(2, v)]) for v in range(9)])
        m0, m2 = Z0.mean(0), Z2.mean(0)
        U, _, Vt = torch.linalg.svd((Z0 - m0).T @ (Z2 - m2))
        R = U @ Vt
        pred = {v: mu + ((z(cm[(0, v)]) - m0) @ R + m2) @ Bk.T for v in range(9, 13)}
        perm = min(itertools.permutations(free), key=lambda p: sum(float((pred[v] - dec.Mmean[j]).norm()) for v, j in zip(range(9, 13), p)))
        fill = {v: dec.Mmean[j] for v, j in zip(range(9, 13), perm)}

        def tf(hh, LL, fill=fill):
            out_b = hh[:, :16].double().clone()
            eh = dec.ehat(hh)
            for i in ((LL["mu"] == 2) & (LL["m"] >= 9)).nonzero().flatten().tolist():
                out_b[i] = fill[int(LL["m"][i])] + eh[i]
            return with_block(hh, out_b)
        out[f"k{k}"] = dict(**score(m, tf), assignment_correct=sum(j == truth[v] for v, j in zip(range(9, 13), perm)))
    return out


def run_labelled_completion():
    torch.set_num_threads(2)
    res = {}
    for s in (304, 310, 302, 300, 305, 307, 311):
        n = f"mass_only_s{s}"
        m = load(n)
        res[n] = labelled_class_completion(m, Decomp(m))
        print(n, {k: (round(v["oodM"], 3), round(v["id"], 3), round(v["disc_swap"], 3), v["assignment_correct"]) for k, v in res[n].items() if k.startswith("k")}, flush=True)
    (OUT / "labelled_completion.json").write_text(json.dumps(res, indent=1))


def angle_completion(m, dec):
    """R7A: assignment by angle on the circle of the canonical class means (training labels only). Plane = top-2 PCs
    of the 13 class means; Delta_2 = circular mean of angle(c(2,v)) - angle(c(0,v)) over trained v=1..8; the unseen
    cell (2,v) is assigned to the free class whose angle is nearest to angle(c(0,v)) + Delta_2 (Hungarian)."""
    import itertools
    import math
    L = W.lat(32768, CAL)
    blk = resid(m, L)[:, :16].double()
    cm = {(u, v): blk[(L["mu"] == u) & (L["m"] == v)].mean(0) for u in range(3) for v in range(P) if not (u == 2 and v >= 9)}
    mu = dec.Mmean[1:].mean(0)
    _, _, Vh = torch.linalg.svd(dec.Mmean[1:] - mu, full_matrices=False)
    B2 = Vh[:2].T
    ang = lambda x: math.atan2(float((x - mu) @ B2[:, 1]), float((x - mu) @ B2[:, 0]))
    d = [ang(cm[(2, v)]) - ang(cm[(0, v)]) for v in range(1, 9)]
    delta = math.atan2(sum(math.sin(x) for x in d), sum(math.cos(x) for x in d))
    spread = 1 - math.hypot(sum(math.sin(x) for x in d), sum(math.cos(x) for x in d)) / len(d)
    used = {(v * T.MASS_S[2]) % P for v in range(9)}
    free = [k for k in range(P) if k not in used]
    phi = {k: ang(dec.Mmean[k]) for k in free}
    tgt = {v: ang(cm[(0, v)]) + delta for v in range(9, 13)}
    cd = lambda a, b: abs(math.atan2(math.sin(a - b), math.cos(a - b)))
    perm = min(itertools.permutations(free), key=lambda p: sum(cd(tgt[v], phi[j]) for v, j in zip(range(9, 13), p)))
    fill = {v: dec.Mmean[j] for v, j in zip(range(9, 13), perm)}

    def tf(hh, LL):
        out_b = hh[:, :16].double().clone()
        eh = dec.ehat(hh)
        for i in ((LL["mu"] == 2) & (LL["m"] >= 9)).nonzero().flatten().tolist():
            out_b[i] = fill[int(LL["m"][i])] + eh[i]
        return with_block(hh, out_b)
    truth = {v: (v * T.MASS_S[2]) % P for v in range(9, 13)}
    return dict(**score(m, tf), assignment_correct=sum(j == truth[v] for v, j in zip(range(9, 13), perm)),
                rotation_spread=spread)


def run_angle():
    torch.set_num_threads(2)
    res = {}
    for s in (304, 310, 302, 300, 305, 307, 311):
        n = f"mass_only_s{s}"
        m = load(n)
        res[n] = angle_completion(m, Decomp(m))
        print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[n].items()}, flush=True)
    (OUT / "angle_completion.json").write_text(json.dumps(res, indent=1))
