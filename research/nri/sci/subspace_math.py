"""Mathematical organisation of the canonical subspace in the concept factorial (owner INIT §4-5). Retrospective,
exploratory; reads verified weight copies only. Site: residual after block 1 at the source-number token.

Distinguishes three levels:
  present      canonical classes are separable (nearest-class-mean accuracy on held-out bases)
  readable     a linear readout fitted on trained bases decodes q on held-out bases
  controlling  replacing / transforming the activation inside the subspace changes the output as predicted
and adds necessity: mean-ablating the subspace (vs a random subspace of equal rank) during ordinary prediction.

Per network and step:
  spectrum       singular values of the centred 13 canonical-class means (trained bases); effective rank (PR)
  topk_switch    I1-style canonical switch (trained base | implied held-out) patching only the top-k directions
  ablate_*       accuracy on trained / development-held-out questions after mean-ablating the subspace / random
  edit_*         a single linear map inside the subspace fitted on trained class means to send class q to f(q)
                 (double, invert, add-one), applied to development-held-out bases: rate of the answer implied by f(q)
Across networks (step 6,000):
  rsa            Spearman correlation between seeds of the 13x13 class-mean distance matrices
  transplant     donor activation from network A mapped into network B by an orthogonal Procrustes map fitted on the
                 class means (trained), patched into B's subspace: canonical switch in B (vs B's own I4, vs a
                 random orthogonal map)

    python -m sci.subspace_math     # writes runs/instruments/subspace_math.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import causal_instruments as CI  # noqa: E402
from sci import concept as C  # noqa: E402
from sci.mechanism_sswd import spearman  # noqa: E402
from sem import data as D  # noqa: E402
from sem import lang as L  # noqa: E402

P = C.P
STRAT = "base_trained|can_implied_heldout"
EDITS = {"double": lambda q: 2 * q % P, "invert": lambda q: pow(q, -1, P) if q else 0, "add1": lambda q: (q + 1) % P}
OUT = CI.OUT_DIR / "subspace_math.json"


@torch.no_grad()
def x1_src(ad, combos):
    enc = ad.encode(combos)
    return enc, enc["states"]["x1"][torch.arange(len(combos)), enc["pos"]["src"]]


@torch.no_grad()
def class_means(ad, split):
    combos = list(split["train"])
    _, x = x1_src(ad, combos)
    q = torch.tensor([C.canonical(r[0], v, r[1]) for r, v in combos])
    mu = torch.stack([x[q == k].mean(0) for k in range(P)])
    return mu, x, q


@torch.no_grad()
def readout_from(ad, enc, x_src, site="x1"):
    xs = {k: v.clone() for k, v in enc["states"].items()}
    xs[site][torch.arange(len(x_src)), enc["pos"]["src"]] = x_src
    return ad.readout(xs, enc, site)


@torch.no_grad()
def network_measures(ad, split, cp):
    mu, xtr, qtr = class_means(ad, split)
    gm = mu.mean(0)
    U, S, Vh = torch.linalg.svd(mu - gm, full_matrices=False)
    B = Vh[:12].T
    pr = float(S.sum() ** 2 / (S ** 2).sum())
    out = dict(spectrum=[round(float(s), 4) for s in S[:12]], eff_rank=pr)
    # top-k causal dimension
    out["topk_switch"] = {}
    for k in (1, 2, 4, 6, 8, 12):
        r = CI.strata(ad, cp, 300, site="x1", basis=B[:, :k], mode="sub")
        out["topk_switch"][k] = r[STRAT]["canonical_switch"]
    # present / readable on development-held-out bases
    dev = list(split["dev"])
    enc_d, xd = x1_src(ad, dev)
    qd = torch.tensor([C.canonical(r[0], v, r[1]) for r, v in dev])
    out["present_nearest_mean"] = float((torch.cdist(xd, mu).argmin(1) == qd).float().mean())
    X = torch.cat([xtr, torch.ones(len(xtr), 1)], 1).double()
    W = torch.linalg.solve(X.T @ X + 1e-2 * torch.eye(X.shape[1], dtype=X.dtype), X.T @ torch.nn.functional.one_hot(qtr, P).double())
    out["readable_linear"] = float(((torch.cat([xd, torch.ones(len(xd), 1)], 1).double() @ W).argmax(1) == qd).float().mean())
    # necessity: mean-ablate the subspace vs a random subspace of equal rank
    ans = lambda combos: torch.tensor([C.answer(r[0], v, r[1], r[2]) for r, v in combos])
    tr = list(split["train"])
    enc_t, xt = x1_src(ad, tr)
    for tag, Bx in (("canonical", B), ("random", CI.random_basis(B.shape[0], B.shape[1], 77))):
        for name, enc, x, combos in (("trained", enc_t, xt, tr), ("heldout", enc_d, xd, dev)):
            xa = x - ((x - gm) @ Bx) @ Bx.T
            out[f"ablate_{tag}_{name}"] = float((readout_from(ad, enc, xa) == ans(combos)).float().mean())
    out["acc_trained"] = float((readout_from(ad, enc_t, xt) == ans(tr)).float().mean())
    out["acc_heldout"] = float((readout_from(ad, enc_d, xd) == ans(dev)).float().mean())
    # linear edits inside the subspace. A single affine map in subspace coordinates is fitted on the class means of
    # the TRAINING-FOLD classes only and evaluated on development-held-out bases whose canonical class was held out of
    # the fit (4 folds over the 12 non-zero classes; class 0 always in the fit). An exact fit on all 13 means is
    # trivially possible and therefore uninformative; generalisation across classes tests whether the edit is linear
    # in this geometry.
    Z = (mu - gm) @ B
    zd = (xd - gm) @ B
    folds = [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
    for ename, f in EDITS.items():
        hits, tot, hits_in, tot_in = 0, 0, 0, 0
        for fold in folds:
            fit = [k for k in range(P) if k not in fold]
            Za = torch.cat([Z[fit], torch.ones(len(fit), 1)], 1)
            A = torch.linalg.lstsq(Za, torch.stack([Z[f(k)] for k in fit])).solution
            zn = torch.cat([zd, torch.ones(len(zd), 1)], 1) @ A
            pred = readout_from(ad, enc_d, xd + (zn - zd) @ B.T)
            for i, ((r, v), q) in enumerate(zip(dev, qd)):
                want = f(int(q)) * L.INV[L.SCALES[r[0]][r[2]]] % P
                if want == C.answer(r[0], v, r[1], r[2]):
                    continue
                ok = int(pred[i]) == want
                if int(q) in fold:
                    hits, tot = hits + ok, tot + 1
                elif int(q) in fit and fold == folds[0]:
                    hits_in, tot_in = hits_in + ok, tot_in + 1
        out[f"edit_{ename}_heldclass"] = hits / tot if tot else None
        out[f"edit_{ename}_fitclass"] = hits_in / tot_in if tot_in else None
    return out, mu, B


def procrustes(A, B):
    """Orthogonal R minimising ||A R - B|| (rows centred)."""
    U, _, Vh = torch.linalg.svd(A.T @ B)
    return U @ Vh


@torch.no_grad()
def transplant(adA, adB, split, cp, muA, muB, BB, n=300, seed=0, random_R=False):
    """Donor from network A, mapped into B's space by Procrustes on class means; patched into B's subspace."""
    gA, gB = muA.mean(0), muB.mean(0)
    R = procrustes(muA - gA, muB - gB)
    if random_R:
        R = CI.random_basis(R.shape[0], R.shape[1], 99)
    rng = random.Random(seed)
    pairs = [p for p in cp if p["base_pool"] == "trained" and p["can_implied"] == "heldout"]
    pairs = rng.sample(pairs, min(n, len(pairs)))
    eb = adB.encode([p["base"] for p in pairs])
    edA = adA.encode([p["donor"] for p in pairs])
    a = torch.arange(len(pairs))
    xb = eb["states"]["x1"][a, eb["pos"]["src"]]
    xdA = edA["states"]["x1"][a, edA["pos"]["src"]]
    z = gB + (xdA - gA) @ R
    xp = xb + ((z - xb) @ BB) @ BB.T
    pred = readout_from(adB, eb, xp)
    base_pred = adB.readout(eb["states"], eb, "x1")
    tgt = torch.tensor([p["a_can"] for p in pairs])
    nb = base_pred != tgt
    return float((pred[nb] == tgt[nb]).float().mean())


def main():
    torch.set_num_threads(4)
    split = C.make_split()
    cp = CI.cross_pairs(split)
    res = dict(networks={}, rsa={}, transplant={})
    keep = {}
    for cond in ("wd1.0_C", "wd0.1_C", "wd1.0_D", "wd1.0_B"):
        for s in range(2001, 2013):
            run = f"{cond}_s{s}"
            res["networks"][run] = {}
            for step in (1000, 3000, 6000):
                ad = CI.load_student(run, step)
                m, mu, B = network_measures(ad, split, cp)
                res["networks"][run][step] = m
                if step == 6000:
                    keep[run] = (ad, mu, B)
            print(run, {k: res["networks"][run][6000][k] for k in ("eff_rank", "present_nearest_mean", "readable_linear",
                                                                  "ablate_canonical_heldout", "edit_double_heldclass", "edit_invert_heldclass", "edit_add1_heldclass")}, flush=True)
    # RSA across seeds within condition (13x13 class-mean distances)
    for cond in ("wd1.0_C", "wd0.1_C", "wd1.0_D", "wd1.0_B"):
        mats = [torch.cdist(keep[f"{cond}_s{s}"][1], keep[f"{cond}_s{s}"][1]) for s in range(2001, 2013)]
        iu = torch.triu_indices(P, P, 1)
        vecs = [m_[iu[0], iu[1]].tolist() for m_ in mats]
        rs = [spearman(vecs[i], vecs[j]) for i in range(12) for j in range(i + 1, 12)]
        res["rsa"][cond] = dict(mean=sum(rs) / len(rs), min=min(rs), max=max(rs))
    # cross-network transplant within C1.0 (s -> s+1) and controls
    for s in range(2001, 2013):
        A_, B_ = f"wd1.0_C_s{s}", f"wd1.0_C_s{2001 + (s - 2000) % 12}"
        adA, muA, _ = keep[A_]
        adB, muB, BB = keep[B_]
        res["transplant"][f"{A_}->{B_}"] = dict(procrustes=transplant(adA, adB, split, cp, muA, muB, BB),
                                                random_rotation=transplant(adA, adB, split, cp, muA, muB, BB, random_R=True),
                                                own_I4=res["networks"][B_][6000]["topk_switch"][12])
    OUT.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main()
    print("RSA", json.dumps(r["rsa"], indent=0))
    for k, v in r["transplant"].items():
        print(k, v)
