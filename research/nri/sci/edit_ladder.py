"""Generalisation x editability ladder for the role-swap networks (owner INIT §5). Retrospective, on copies.

Rungs, for the declared mass block (h[:, :16], last token before the MLP):
  L1 localisation       designed block (known); IIA (ID) and IIA from never-trained sources (validated)
  L2 replacement        copy a REAL donor activation carrying the edited mass into the block (doc 15 'copy')
  L3 transformation     map the block activation with a fitted map, no real donor:
                          global      one affine map fitted on class means of canonical M (all renderings pooled)
                          per_unit    one affine map per mass unit (fitted on that unit's trained renderings)
                          held_class  global map fitted with the target classes held out (geometry linearity)
  L4 permanent          the global map folded into W_in only (doc 15 style: the residual skip path is not edited)
  L5 generalisation     L3/L4 success on never-trained renderings (mass unit 2, value 9-12)
  L6 preservation       accuracy on fixed points of the edit (inversion: M in {0,1,12}), ID and never-trained
Edit: inversion mod 13 (primary, as doc 15) and doubling. Success = intended answer on eligible examples
(intended answer differs from the unedited answer).

    python -m sci.edit_ladder        # writes runs/swap_recon/edit_ladder.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_dimensional as D  # noqa: E402
import run_swap as RS  # noqa: E402
from nri import dimtask as T  # noqa: E402
from nri.model import TinyTyped  # noqa: E402
from sci import window_analysis as W  # noqa: E402

P = T.P
EDITS = {"inv": lambda q: torch.tensor([pow(int(x), -1, P) if int(x) else 0 for x in q]), "double": lambda q: 2 * q % P}
N = 4096


def block_means(m, L, key="M"):
    _, _, h, _, _, _ = W.states(m, L)
    M, _, _ = T.canon(L)
    return h, M


def fit_affine(Zs, Zt):
    A = torch.cat([Zs, torch.ones(len(Zs), 1)], 1).double()
    return torch.linalg.lstsq(A, Zt.double()).solution


def apply_affine(X, coef):
    return (torch.cat([X, torch.ones(len(X), 1)], 1).double() @ coef).float()


@torch.no_grad()
def score(m, h, Lb, newM, keep):
    S = T.canon(Lb)[2]
    _, A, _ = T.canon(Lb)
    want = T.target(newM, A, S)
    pred = m.head_from_resid(h).argmax(-1)
    return float((pred[keep] == want[keep]).float().mean()) if keep.any() else None


@torch.no_grad()
def ladder(m):
    """All maps act only on the coordinates of the block inside the span S of the canonical class means (top-12 of the
    centred means); the component of each activation outside S (context: acceleration, output unit, order, route) is
    kept. Fitting a full affine map to 13 means instead (minimum-norm least squares) erases that component and is not
    used (doc 46 deviation note)."""
    m.eval()
    out = {}
    fit = W.lat(16384, 880_001)
    hf, Mf = block_means(m, fit)
    blk = hf[:, :16]
    mu = torch.stack([blk[Mf == k].mean(0) for k in range(P)])
    gm = mu.mean(0)
    _, Sv, Vh = torch.linalg.svd(mu - gm, full_matrices=False)
    S = Vh[:12].T                                                       # 16 x 12
    Z = (mu - gm) @ S
    Zu = {}
    for u in range(3):
        sel = fit["mu"] == u
        Zu[u] = {k: ((blk[sel & (Mf == k)].mean(0) - gm) @ S) for k in range(P) if (sel & (Mf == k)).sum() > 3}
    out["block_class_mean_spectrum"] = [round(float(x), 3) for x in Sv]
    test_id = W.lat(N, 880_002)
    test_ood = W.ood_mass(N, 880_003)
    fk = lambda f, k: int(f(torch.tensor([k]))[0])

    def edit_block(h, coef_by_row):
        z = (h[:, :16] - gm) @ S
        zn = torch.stack([apply_affine(z[i:i + 1], c)[0] if c is not None else z[i] for i, c in enumerate(coef_by_row)])
        h2 = h.clone()
        h2[:, :16] = h[:, :16] + (zn - z) @ S.T
        return h2

    for ename, f in EDITS.items():
        g = fit_affine(Z, torch.stack([Z[fk(f, k)] for k in range(P)]))
        cu = {}
        for u in range(3):
            ks = [k for k in Zu[u] if fk(f, k) in Zu[u]]
            cu[u] = fit_affine(torch.stack([Zu[u][k] for k in ks]), torch.stack([Zu[u][fk(f, k)] for k in ks])) if len(ks) >= 3 else None
        for tname, Lt in (("id", test_id), ("ood", test_ood)):
            h, M = block_means(m, Lt)
            newM = f(M)
            keep = newM != M
            _, A, Sx = T.canon(Lt)
            base = T.target(M, A, Sx)
            want = T.target(newM, A, Sx)
            hg = edit_block(h, [g] * len(h))
            pg = m.head_from_resid(hg).argmax(-1)
            out[f"{ename}_global_{tname}"] = float((pg[keep] == want[keep]).float().mean())
            if ename == "inv":
                out[f"inv_global_fixedpoint_{tname}"] = float((pg[~keep] == base[~keep]).float().mean())
            hu = edit_block(h, [cu[int(u)] for u in Lt["mu"]])
            pu = m.head_from_resid(hu).argmax(-1)
            out[f"{ename}_perunit_{tname}"] = float((pu[keep] == want[keep]).float().mean())
            rng = torch.Generator().manual_seed(880_010)
            hd, _ = block_means(m, W.rerender_mass(Lt, newM, rng))
            h4 = h.clone()
            h4[:, :16] = hd[:, :16]
            out[f"{ename}_copy_{tname}"] = float((m.head_from_resid(h4).argmax(-1)[keep] == want[keep]).float().mean())
            # copy restricted to the subspace S (replace only the in-span coordinates by the donor's)
            h5 = h.clone()
            h5[:, :16] = h[:, :16] + (((hd[:, :16] - gm) @ S) - ((h[:, :16] - gm) @ S)) @ S.T
            out[f"{ename}_copy_subspace_{tname}"] = float((m.head_from_resid(h5).argmax(-1)[keep] == want[keep]).float().mean())
            # permanent: the global in-span map folded into W_in only (residual skip path not edited)
            Aff = torch.eye(64)
            Aff[:16, :16] = torch.eye(16) + S @ (g[:12].T.float() - torch.eye(12)) @ S.T
            tvec = torch.zeros(64)
            tvec[:16] = (S @ (g[12].float() + (gm @ S) @ (torch.eye(12) - g[:12].float()).T * 0) ) if False else S @ (g[12].float() - ((gm @ S) @ (torch.eye(12) - g[:12].float())))
            Wn, bn = m.W_in.weight, m.W_in.bias
            hh = h
            pf = m.W_U(hh + m.W_out(torch.relu(hh @ (Wn @ Aff).T + bn + Wn @ tvec))).argmax(-1)
            out[f"{ename}_fold_win_{tname}"] = float((pf[keep] == want[keep]).float().mean())
        hits = tot = 0
        h, M = block_means(m, test_id)
        _, A, Sx = T.canon(test_id)
        for fold in ([1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]):
            fitk = [k for k in range(P) if k not in fold]
            c = fit_affine(Z[fitk], torch.stack([Z[fk(f, k)] for k in fitk]))
            newM = f(M)
            sel = torch.tensor([int(x) in fold for x in M]) & (newM != M)
            p_ = m.head_from_resid(edit_block(h, [c] * len(h))).argmax(-1)
            want = T.target(newM, A, Sx)
            hits += int((p_[sel] == want[sel]).sum())
            tot += int(sel.sum())
        out[f"{ename}_heldclass_id"] = hits / tot if tot else None
    return out


def load(run_dir, step=None):
    m = TinyTyped("mass_only", **RS.MODEL_KW)
    if step is None:
        m.load_state_dict(torch.load(Path(run_dir) / "model.pt", weights_only=True))
    else:
        m.load_state_dict(load_file(str(Path(run_dir) / "ckpt" / f"step_{step:05d}" / "model.safetensors")))
    return m


def main():
    torch.set_num_threads(3)
    res = {"endpoints": {}, "trajectory": {}}
    for s in range(300, 312):
        n = f"mass_only_s{s}"
        res["endpoints"][n] = ladder(load(ROOT / "runs/zoo_swap" / n))
    for s in (307, 311, 305, 304, 310):
        n = f"mass_only_s{s}"
        res["trajectory"][n] = {st: ladder(load(f"/home/user/swap_recon/zoo/{n}", st)) for st in (800, 1000, 1200, 1500, 2000, 4000, 6000)}
    (ROOT / "runs/swap_recon/edit_ladder.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main()
    for n, v in r["endpoints"].items():
        print(n, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()})
