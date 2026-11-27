"""Phase V drivers (doc 52 §52.8), as run; logic unchanged, reorganised into functions.

    python -m sci.phase5_drivers d1          # runs/phase5/decoder_profiles_registered.json (registered construction)
    python -m sci.phase5_drivers d1_exact    # runs/phase5/decoder_profiles_exact.json (deviation-1 variant)
    python -m sci.phase5_drivers d2 [WINDOW] [ZOO]   # runs/phase5/decoder_birth.json
    python -m sci.phase5_drivers d3 [ZOO]    # runs/phase5/tolerance.json
    python -m sci.phase5_drivers d5          # runs/phase5/transplant.json
D4: python -c "import sci.algorithm as A; A.main(); A.main_preact(); A.main_program(); A.main_additive(); A.main_pair()"
D7: python -m sci.operator_edit
WINDOW defaults to /home/user/swap_window (10-step replay), ZOO to /home/user/swap_recon/zoo (extracted archives).
"""
from __future__ import annotations

import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file  # noqa: E402

import run_swap as RS  # noqa: E402
from nri.model import TinyTyped  # noqa: E402
from sci import decoder as DC  # noqa: E402
from sci import rescue as R  # noqa: E402

OUT = ROOT / "runs/phase5"
WINDOW = "/home/user/swap_window"
ZOO = "/home/user/swap_recon/zoo"


def _ckpt(root, s, st):
    return load_file(f"{root}/mass_only_s{s}/ckpt/step_{st:05d}/model.safetensors")


# ---------------------------------------------------------------- D1
def d1():
    torch.set_num_threads(4)
    res = {}
    for g in ("mass_only", "neither"):
        for s in range(300, 312):
            n = f"{g}_s{s}"
            res[n] = DC.profile(R.load(n))
    json.dump(res, open(OUT / "decoder_profiles_registered.json", "w"))


def d1_exact():
    torch.set_num_threads(4)
    res = {}
    for s in range(300, 312):
        n = f"mass_only_s{s}"
        m = R.load(n)
        res[n] = dict(additive=DC.profile_exact(m, interaction_frac=0.0),
                      additive_plus_interaction=DC.profile_exact(m, interaction_frac=1.0))
    json.dump(res, open(OUT / "decoder_profiles_exact.json", "w"))


# ---------------------------------------------------------------- D2
def _birth_job(args):
    s, window, zoo = args
    torch.set_num_threads(1)
    m = TinyTyped("mass_only", **RS.MODEL_KW)
    out = {}
    for root, steps in ((window, range(600, 2001, 20)), (zoo, (0, 300, 500, 3000, 6000))):
        for st in steps:
            m.load_state_dict(_ckpt(root, s, st))
            p = DC.profile_exact(m, n_per=128)
            out[st] = dict(syn_trained=p["syn_trained"], syn_unseen=p["syn_unseen_all"], syn_far=p["syn_far"],
                           syn_near=p["syn_near"], cls=p["class"])
    return s, out


def d2(window=WINDOW, zoo=ZOO):
    with ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(_birth_job, [(s, window, zoo) for s in (307, 311, 305, 304, 310)]))
    json.dump(res, open(OUT / "decoder_birth.json", "w"))


# ---------------------------------------------------------------- D3
def tol(m, k=0.5, r=4, seed=0):
    """Run-time interaction injection inside (inS) / outside (outS) the causal subspace, plus isotropic control."""
    dec = R.Decomp(m)
    g = torch.Generator().manual_seed(seed)
    scale = float(torch.cat([dec.a, dec.b]).norm(dim=1).mean())
    PS = dec.S @ dec.S.T
    out = {}
    base = R.score(m, lambda h, L: h)
    for orient in ("inS", "outS"):
        Z = torch.randn(3, 13, r, generator=g, dtype=torch.double)
        Z = Z - Z.mean(0, keepdim=True) - Z.mean(1, keepdim=True) + Z.mean((0, 1), keepdim=True)
        B = torch.linalg.qr(torch.randn(16, 16, generator=g, dtype=torch.double))[0][:, :r]
        Br = (PS @ B) if orient == "inS" else ((torch.eye(16, dtype=PS.dtype) - PS) @ B)
        I = Z @ Br.T
        I = I / I.norm(dim=-1).mean()
        s = R.score(m, lambda h, L: R.with_block(h, h[:, :16].double() + I[L["mu"], L["m"]] * k * scale))
        out[orient] = dict(oodM=s["oodM"], id=s["id"])

    def iso(h, L):
        z = torch.randn(len(h), 16, generator=g, dtype=torch.double)
        z = z / z.norm(dim=1, keepdim=True) * k * scale
        return R.with_block(h, h[:, :16].double() + z)
    s = R.score(m, iso)
    out["isotropic"] = dict(oodM=s["oodM"], id=s["id"])
    out["clean"] = dict(oodM=base["oodM"], id=base["id"])
    return out


def _tol_job(args):
    s, zoo = args
    torch.set_num_threads(1)
    m = TinyTyped("mass_only", **RS.MODEL_KW)
    res = {}
    for st in (1200, 1400, 1600, 2000, 3000, 4000, 6000):
        m.load_state_dict(_ckpt(zoo, s, st))
        res[st] = tol(m)
    return s, res


def d3(zoo=ZOO):
    with ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(_tol_job, [(s, zoo) for s in (307, 311, 300, 305)]))
    json.dump(res, open(OUT / "tolerance.json", "w"))


# ---------------------------------------------------------------- D5
def d5():
    """Variants (a) direct head and (d) layer-wise; (b)/(e) are doc 49 §49.2 (`sci.rescue` compat)."""
    torch.set_num_threads(4)
    donors = {s: R.load(f"mass_only_s{s}") for s in (307, 311)}
    res = {}
    for r in (300, 305, 302, 304, 310):
        rec = R.load(f"mass_only_s{r}")
        for d, dm in donors.items():
            for name, parts in (("direct_head", ("W_in", "W_out", "W_U")), ("W_in_only", ("W_in",)),
                                ("W_out+W_U", ("W_out", "W_U"))):
                e = copy.deepcopy(rec)
                for p in parts:
                    getattr(e, p).load_state_dict(getattr(dm, p).state_dict())
                res[f"{d}->{r}:{name}"] = R.score(e, lambda h, L: h)
    json.dump(res, open(OUT / "transplant.json", "w"), indent=1)


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"d1": d1, "d1_exact": d1_exact, "d2": d2, "d3": d3, "d5": d5}[cmd](*rest)
