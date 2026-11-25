"""P4 (doc 46 §46.5), retrospective on copies: edit the declared block on its composable parts.

The block's (unit, value) class means are fitted with an additive model c(u, v) = g + a_v + b_u (trained renderings
only). An edit f on the canonical mass is implemented per part: value v -> v_f and unit u -> u_f with
v_f * s_{u_f} = f(v * s_u) (mod 13), and the block is shifted by (a_{v_f} - a_v) + (b_{u_f} - b_u).
  inversion: v -> v^-1, u -> unit with scale s_u^-1  (mass scales 1,5,8: 5 and 8 are mutual inverses)
  doubling:  v -> 2v,   u unchanged
Compared with the class-mean edit (doc 46 §46.3). Success = intended answer on eligible examples; preservation =
fixed points of the edit; tested on trained and never-trained (unit 2, value 9-12) renderings.

    python -m sci.part_edit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_dimensional as D  # noqa: E402
from nri import dimtask as T  # noqa: E402
from sci import window_analysis as W  # noqa: E402

P = T.P
UINV = {u: T.MASS_S.index(pow(s, -1, P)) for u, s in enumerate(T.MASS_S)}
VINV = lambda v: pow(v, -1, P) if v else 0


@torch.no_grad()
def additive(m, n=32768):
    L = W.lat(n, 7)
    _, _, h, _, _, _ = W.states(m, L)
    X = h[:, :16].double()
    keys = sorted(set(zip(L["mu"].tolist(), L["m"].tolist())))
    C = torch.stack([X[(L["mu"] == u) & (L["m"] == v)].mean(0) for u, v in keys])
    Dm = torch.zeros(len(keys), 16, dtype=torch.double)
    for i, (u, v) in enumerate(keys):
        Dm[i, v] = 1
        Dm[i, 13 + u] = 1
    co = torch.linalg.lstsq(Dm, C).solution
    return co[:13].float(), co[13:].float()                   # a_v (13 x 16), b_u (3 x 16)  (up to a shared offset)


@torch.no_grad()
def run(m):
    a, b = additive(m)
    out = {}
    for ename, fv, fu in (("inv", VINV, lambda u: UINV[u]), ("double", lambda v: 2 * v % P, lambda u: u)):
        for tname, L in (("id", W.lat(4096, 880_002)), ("ood", W.ood_mass(4096, 880_003))):
            _, _, h, _, _, _ = W.states(m, L)
            v = L["m"]
            u = L["mu"]
            v2 = torch.tensor([fv(int(x)) for x in v])
            u2 = torch.tensor([fu(int(x)) for x in u])
            h2 = h.clone()
            h2[:, :16] = h[:, :16] + (a[v2] - a[v]) + (b[u2] - b[u])
            M, A, S = T.canon(L)
            Mn = v2 * torch.tensor(T.MASS_S)[u2] % P
            want, base = T.target(Mn, A, S), T.target(M, A, S)
            pred = m.head_from_resid(h2).argmax(-1)
            keep = want != base
            out[f"{ename}_part_{tname}"] = float((pred[keep] == want[keep]).float().mean())
            out[f"{ename}_part_preserve_{tname}"] = float((pred[~keep] == base[~keep]).float().mean()) if (~keep).any() else None
            # is the edited target rendering itself trained? (u2, v2) with u2 == 2 and v2 >= 9 would be never-trained
            out[f"{ename}_target_untrained_frac_{tname}"] = float(((u2 == 2) & (v2 >= 9)).float().mean())
    return out


if __name__ == "__main__":
    torch.set_num_threads(3)
    res = {}
    for g in ("mass_only", "neither"):
        for s in range(300, 312):
            m, cfg = D.load(ROOT / f"runs/zoo_swap/{g}_s{s}")
            if cfg["acc_id"] < 0.95:
                continue
            res[f"{g}_s{s}"] = run(m)
            print(f"{g}_s{s}", {k: round(x, 2) for k, x in res[f'{g}_s{s}'].items() if x is not None and 'frac' not in k}, flush=True)
    (ROOT / "runs/swap_recon/part_edit.json").write_text(json.dumps(res, indent=1))
