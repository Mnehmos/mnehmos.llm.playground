"""Doc 51 D1-D3: the decoder (head) isolated from the encoder.

Parts are fitted on FAMILIAR renderings only (calibration seed). For a network's last-token residual h:
  block parts   a_v (mass value), b_u (mass unit)  -- additive model of the declared block's cell means
  twin distance for each unseen cell (unit 2, value 9-12): distance from the synthetic block a_v + b_2 to the block of
                the trained rendering of the same canonical mass (the "twin"); near < 0.3, far > 2 (relative; doc 49)
Synthetic decoder input for an example x: h_syn = h(x) with its block replaced by a_v + b_u + e_hat(x), i.e. the mass
part is constructed from familiar-fitted parts, the context (acceleration, output unit, order) is the network's own.
Classes (doc 51): algorithmic / interpolative / lookup / mixed from synthetic accuracy on trained, near, far cells.

    from sci.decoder import profile
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri import dimtask as T  # noqa: E402
from sci import rescue as R  # noqa: E402
from sci import window_analysis as W  # noqa: E402

P = T.P
UNSEEN = [(2, v) for v in range(9, 13)]


def cells_test(n_per=256, seed=990_500):
    """One latent batch per (unit, value) mass cell, with random context."""
    out = {}
    for u in range(3):
        for v in range(P):
            L = W.lat(n_per, seed + u * 100 + v)
            L["mu"] = torch.full((n_per,), u)
            L["m"] = torch.full((n_per,), v)
            out[(u, v)] = L
    return out


@torch.no_grad()
def profile(m, n_per=256, noise=(0.1, 0.25, 0.5), seed=0):
    dec = R.Decomp(m)
    a, b = dec.a.float(), dec.b.float()
    cells = cells_test(n_per)
    g = torch.Generator().manual_seed(seed)
    res = {"cells": {}}
    twin = {}
    for (u, v), L in cells.items():
        h = R.resid(m, L)
        y = T.build(L).y
        syn = h.clone()
        blk = a[v] + b[u] + dec.ehat(h).float()
        syn[:, :16] = blk
        acc_real = float((m.head_from_resid(h).argmax(-1) == y).float().mean())
        acc_syn = float((m.head_from_resid(syn).argmax(-1) == y).float().mean())
        M = (v * T.MASS_S[u]) % P
        # canonical placement (class mean of trained renderings with the same M)
        can = h.clone()
        can[:, :16] = dec.Mmean[M].float() + dec.ehat(h).float()
        acc_can = float((m.head_from_resid(can).argmax(-1) == y).float().mean())
        pert = {}
        for s in noise:
            z = torch.randn(blk.shape, generator=g)
            z = z / z.norm(dim=1, keepdim=True) * blk.norm(dim=1, keepdim=True) * s
            hp = syn.clone()
            hp[:, :16] = blk + z
            pert[s] = float((m.head_from_resid(hp).argmax(-1) == y).float().mean())
        res["cells"][f"{u},{v}"] = dict(real=acc_real, syn=acc_syn, canonical=acc_can, perturbed=pert)
        if (u, v) in UNSEEN:
            # twin: trained rendering of the same canonical mass in unit 0 (always trained for every M)
            vt = (M * pow(T.MASS_S[0], -1, P)) % P
            d = float((a[v] + b[u] - a[vt] - b[0]).norm() / (a[vt] + b[0]).norm())
            twin[(u, v)] = d
    tr = [res["cells"][f"{u},{v}"]["syn"] for u in range(3) for v in range(P) if (u, v) not in UNSEEN]
    near = [res["cells"][f"{u},{v}"]["syn"] for (u, v) in UNSEEN if twin[(u, v)] < 0.3]
    far = [res["cells"][f"{u},{v}"]["syn"] for (u, v) in UNSEEN if twin[(u, v)] > 2]
    mid = [res["cells"][f"{u},{v}"]["syn"] for (u, v) in UNSEEN if 0.3 <= twin[(u, v)] <= 2]
    mean = lambda x: sum(x) / len(x) if x else None
    res["summary"] = dict(syn_trained=mean(tr), syn_near=mean(near), syn_far=mean(far), syn_mid=mean(mid),
                          syn_unseen_all=mean([res["cells"][f"{u},{v}"]["syn"] for (u, v) in UNSEEN]),
                          real_unseen=mean([res["cells"][f"{u},{v}"]["real"] for (u, v) in UNSEEN]),
                          canonical_unseen=mean([res["cells"][f"{u},{v}"]["canonical"] for (u, v) in UNSEEN]),
                          canonical_trained=mean([res["cells"][f"{u},{v}"]["canonical"] for u in range(3) for v in range(P) if (u, v) not in UNSEEN]),
                          perturbed_unseen={s: mean([res["cells"][f"{u},{v}"]["perturbed"][s] for (u, v) in UNSEEN]) for s in noise},
                          n_near=len(near), n_far=len(far), twin_dist={f"{u},{v}": d for (u, v), d in twin.items()})
    s = res["summary"]
    s["class"] = classify(s)
    return res


def classify(s):
    tr, near, far = s["syn_trained"], s["syn_near"], s["syn_far"]
    if tr is None or tr < 0.95:
        return "mixed (synthetic trained < 0.95)"
    unseen = s["syn_unseen_all"]
    if far is None and near is None:
        return "undetermined (no near/far cells)"
    if (far is None or far >= 0.95) and (near is None or near >= 0.95) and unseen >= 0.95:
        return "algorithmic"
    if (near is None or near >= 0.9) and (far is not None and far < 0.5):
        return "interpolative"
    if (near is None or near < 0.5) and (far is None or far < 0.5):
        return "lookup"
    return "mixed"


@torch.no_grad()
def profile_exact(m, n_per=256, interaction_frac=0.0):
    """Diagnostic variant (doc 51 deviation 1): synthetic block = h_block - cellmean_batch(u,v) + (a_v + b_u)
    + interaction_frac * I_cal(u,v)  [I_cal = calibration interaction, only for trained cells].
    The per-example context deviation is kept exactly; for unseen cells the batch cell mean uses the encoder's own
    activations on those inputs (allowed for characterising the decoder; never for a qualifying repair)."""
    dec = R.Decomp(m)
    a, b = dec.a.float(), dec.b.float()
    cells = cells_test(n_per)
    out = {}
    twin = {}
    for (u, v), L in cells.items():
        h = R.resid(m, L)
        y = T.build(L).y
        cm = h[:, :16].mean(0)
        add = a[v] + b[u]
        I = dec.I.get(u * P + v)
        syn = h.clone()
        syn[:, :16] = h[:, :16] - cm + add + (interaction_frac * I.float() if (I is not None) else 0)
        out[(u, v)] = float((m.head_from_resid(syn).argmax(-1) == y).float().mean())
        if (u, v) in UNSEEN:
            M = (v * T.MASS_S[u]) % P
            vt = (M * pow(T.MASS_S[0], -1, P)) % P
            twin[(u, v)] = float((a[v] + b[u] - a[vt] - b[0]).norm() / (a[vt] + b[0]).norm())
    tr = [x for c, x in out.items() if c not in UNSEEN]
    near = [out[c] for c in UNSEEN if twin[c] < 0.3]
    far = [out[c] for c in UNSEEN if twin[c] > 2]
    mean = lambda x: sum(x) / len(x) if x else None
    s = dict(syn_trained=mean(tr), syn_near=mean(near), syn_far=mean(far), syn_unseen_all=mean([out[c] for c in UNSEEN]),
             per_unseen={f"{u},{v}": out[(u, v)] for (u, v) in UNSEEN}, twin={f"{u},{v}": twin[(u, v)] for (u, v) in UNSEEN})
    s["class"] = classify(s)
    return s
