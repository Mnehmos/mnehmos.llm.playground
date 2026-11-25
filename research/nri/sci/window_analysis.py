"""Doc 45 analyses: measurements per saved state of the role-swap networks (any checkpoint directory). Read-only.

Measures (mass is the declared variable M; site h = last token before the MLP; block = h[:, :16]):
  beh        acc_id, oodM (accuracy when only the mass is a never-trained (unit 2, value 9-12) rendering), oodA
  iia_oodsrc block interchange with a never-trained mass source (validated causal measure, doc 41R)
  alias      unit invariance: distance between activations of two renderings of the SAME canonical mass, divided by
             the distance for a DIFFERENT canonical mass, everything else fixed. r_id: both renderings trained;
             r_ood: one rendering is the never-trained (unit 2, value 9-12). Measured at the block, at the rest of h,
             and at the layer-1 residual of the mass value token and mass unit token. 1 = no invariance, 0 = full.
  attn       layer-1 attention from the mass value token to its unit token (binding), and layer-2 attention from the
             last token to the mass value / unit tokens (mean over heads and examples)
  rsi_M      route-specificity of the block (1 - R^2 of (unit, value) class means on canonical M)
  circle     multiplicative structure: best fraction of variance of the class means of canonical M (block) explained by
             a single discrete-log Fourier frequency (cos/sin 2*pi*k*dlog(M)/12); also for the value-token embeddings
  emb        mass-unit embedding geometry: cosine of unit-2 row with unit-0/1 rows; norms of value rows 9-12 vs 1-8

    python -m sci.window_analysis <ckpt_root> <out.json> [--every N]
"""
from __future__ import annotations

import json
import math
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

N = 1024
P = T.P
INV8 = pow(T.MASS_S[2], -1, P)
DLOG = {}
_x = 1
for _i in range(12):
    DLOG[_x] = _i
    _x = _x * 2 % P


def gen(seed):
    return torch.Generator().manual_seed(seed)


def lat(n, seed, split="train"):
    return T.sample_latents(n, gen(seed), split)


def ood_mass(n, seed):
    g = gen(seed)
    L = T.sample_latents(n, g)
    L["mu"] = torch.full((n,), T.OOD_UNIT)
    L["m"] = torch.randint(9, 13, (n,), generator=g)
    return L


def rerender_mass(L, M_target, rng, ood=False):
    """Same everything, mass re-rendered so that canonical mass = M_target. ood=True uses unit 2 with value 9-12."""
    out = {k: v.clone() for k, v in L.items()}
    for i in range(len(M_target)):
        M = int(M_target[i])
        if ood:
            out["mu"][i] = T.OOD_UNIT
            out["m"][i] = M * INV8 % P
            continue
        opts = [(u, M * pow(s, -1, P) % P) for u, s in enumerate(T.MASS_S)
                if u != int(L["mu"][i]) and not (u == T.OOD_UNIT and M * pow(s, -1, P) % P >= 9)]
        u, v = opts[int(torch.randint(0, len(opts), (1,), generator=rng))]
        out["mu"][i], out["m"][i] = u, v
    return out


@torch.no_grad()
def states(m, L):
    b = T.build(L)
    z = torch.zeros_like(b.tokens)
    x0 = m.embed(b.tokens, z)
    a1, p1 = m.attn1(x0, return_pattern=True)
    x1 = x0 + a1
    a2, p2 = m.attn2(x1, return_pattern=True)
    h = (x1 + a2)[:, -1]
    mpos = torch.where(L["order"] == 0, 0, 3)
    return b, x1, h, p1, p2, mpos


def dist(a, b):
    return float((a - b).norm(dim=-1).mean())


@torch.no_grad()
def measure(m):
    m.eval()
    r = {}
    r["acc_id"] = D.acc(m, n=N, seed=990_001)
    for var, key in (("m", "oodM"), ("a", "oodA")):
        g = gen(990_010 if var == "m" else 990_011)
        L = T.sample_latents(N, g)
        uk = "mu" if var == "m" else "au"
        L[uk] = torch.full((N,), T.OOD_UNIT)
        L[var] = torch.randint(9, 13, (N,), generator=g)
        b = T.build(L)
        r[key] = float((m(b.tokens, torch.zeros_like(b.tokens)).argmax(-1) == b.y).float().mean())
    mk = D.mask(*RS.BLOCK)
    base, src = lat(N, 990_020), ood_mass(N, 990_021)
    _, _, hb, _, _, _ = states(m, base)
    _, _, hs, _, _, _ = states(m, src)
    r["iia_oodsrc"] = float((m.head_from_resid(hb * (1 - mk) + hs * mk).argmax(-1) == D.counterfactual(base, src, "M")).float().mean())
    # alias invariance
    rng = gen(990_030)
    L = lat(N, 990_031)
    M, _, _ = T.canon(L)
    Lid = rerender_mass(L, M, rng)
    Mdiff = (M + 1 + torch.randint(0, P - 1, (N,), generator=rng)) % P
    Ldf = rerender_mass(L, Mdiff, rng)
    sel = torch.tensor([int(x) in (2, 5, 7, 10) for x in M])                     # canonical values with an OOD rendering
    Lood = rerender_mass(L, M, rng, ood=True)
    S = {k: states(m, x) for k, x in (("b", L), ("id", Lid), ("df", Ldf), ("ood", Lood))}
    ar = torch.arange(N)
    sites = {"block": lambda s: s[2][:, :16], "rest": lambda s: s[2][:, 16:],
             "x1val": lambda s: s[1][ar, s[5]], "x1unit": lambda s: s[1][ar, s[5] + 1]}
    for name, f in sites.items():
        fb, fi, fd, fo = (f(S[k]) for k in ("b", "id", "df", "ood"))
        den = dist(fb, fd)
        r[f"alias_id_{name}"] = dist(fb, fi) / den if den else None
        r[f"alias_ood_{name}"] = dist(fb[sel], fo[sel]) / dist(fb[sel], fd[sel]) if sel.any() else None
    pb = m(T.build(L).tokens, torch.zeros_like(T.build(L).tokens)).argmax(-1)
    po = m(T.build(Lood).tokens, torch.zeros_like(T.build(Lood).tokens)).argmax(-1)
    r["alias_ood_pred_agree"] = float((pb[sel] == po[sel]).float().mean())
    # attention
    b, x1, h, p1, p2, mpos = S["b"]
    r["attn1_val_to_unit"] = float(p1[ar, :, mpos, mpos + 1].mean())
    r["attn1_unit_to_val"] = float(p1[ar, :, mpos + 1, mpos].mean())
    r["attn2_last_to_val"] = float(p2[ar, :, -1, mpos].mean())
    r["attn2_last_to_unit"] = float(p2[ar, :, -1, mpos + 1].mean())
    # route specificity and circle structure of the block
    big = lat(8192, 990_040)
    _, _, hbig, _, _, _ = states(m, big)
    Mb, _, _ = T.canon(big)
    blk = hbig[:, :16].double()
    keys = sorted(set(zip(big["mu"].tolist(), big["m"].tolist())))
    means, q = [], []
    for u, v in keys:
        s = (big["mu"] == u) & (big["m"] == v)
        if s.sum() >= 5:
            means.append(blk[s].mean(0))
            q.append(int(Mb[s][0]))
    Mm = torch.stack(means)
    Mc = Mm - Mm.mean(0)
    Q = torch.nn.functional.one_hot(torch.tensor(q), P).double()
    beta = torch.linalg.lstsq(Q, Mc).solution
    r["rsi_M"] = 1 - float(((Mc - Q @ beta) ** 2).sum() / (Mc ** 2).sum())
    cm = torch.stack([blk[Mb == k].mean(0) for k in range(1, P)])               # canonical classes 1..12
    r["circle_block"] = circle_score(cm, list(range(1, P)))
    E = m.E_c.weight.detach().double()
    r["circle_valemb"] = circle_score(E[1:P], list(range(1, P)))
    mu = [E[T.MU[i]] for i in range(3)]
    cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
    r["emb_cos_u2_u0"], r["emb_cos_u2_u1"], r["emb_cos_u0_u1"] = cos(mu[2], mu[0]), cos(mu[2], mu[1]), cos(mu[0], mu[1])
    r["emb_norm_v9_12"] = float(E[9:13].norm(dim=1).mean())
    r["emb_norm_v1_8"] = float(E[1:9].norm(dim=1).mean())
    return r


def circle_score(X, vals):
    """Max over discrete-log frequencies k=1..6 of the fraction of centred variance explained by cos/sin(2 pi k dlog/12)."""
    X = X - X.mean(0)
    tot = float((X ** 2).sum())
    if tot == 0:
        return None
    th = torch.tensor([2 * math.pi * DLOG[v] / 12 for v in vals], dtype=X.dtype)
    best = 0.0
    for k in range(1, 7):
        B = torch.stack([torch.cos(k * th), torch.sin(k * th)], 1) if k < 6 else torch.cos(k * th)[:, None]
        coef = torch.linalg.lstsq(B, X).solution
        best = max(best, 1 - float(((X - B @ coef) ** 2).sum()) / tot)
    return best


def run_dir(run_dir, every=1):
    run_dir = Path(run_dir)
    m = TinyTyped("mass_only", **RS.MODEL_KW)
    out = {}
    cks = sorted((run_dir / "ckpt").iterdir())
    for i, d in enumerate(cks):
        if i % every:
            continue
        step = int(d.name.split("_")[1])
        m.load_state_dict(load_file(str(d / "model.safetensors")))
        out[step] = measure(m)
    return out


def _job(a):
    torch.set_num_threads(1)
    return a[0], run_dir(a[1], a[2])


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--runs", nargs="*", default=["mass_only_s307", "mass_only_s311", "mass_only_s304", "mass_only_s310", "mass_only_s305"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    jobs = [(r, str(Path(a.root) / r), a.every) for r in a.runs]
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(_job, jobs))
    Path(a.out).write_text(json.dumps(res))
