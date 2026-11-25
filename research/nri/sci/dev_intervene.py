"""Doc 48 E4: developmental interventions (P1-P3) as training continuations from bitwise-exact checkpoints.

The training arithmetic is run_swap.train_one (aaa67c6). Interventions modify only the loss, the weight decay, the
interchange weight, or (freeze/projection) the update of attention-2 query/key rows. Every (network, resume step)
also has a matched no-intervention control, whose state at the next 100-step multiple(s) is compared bitwise with
the reconstruction checkpoints.

Interaction penalty L_int = 1 - R^2_add of the batch's block (unit, value) cell means under the additive model
(value one-hot + unit one-hot), count-weighted, differentiable. lambda > 0 penalises interaction; lambda < 0 rewards it.

    python -m sci.dev_intervene <out_dir> [--only name ...] [--workers 4]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_dimensional as D  # noqa: E402
import run_swap as RS  # noqa: E402
from nri import dimtask as T  # noqa: E402
from nri.model import TinyTyped  # noqa: E402
from nri.train import set_determinism  # noqa: E402

RECON = Path("/home/user/swap_recon/zoo")
WINDOW = Path("/home/user/swap_window")
P = T.P


def l_int(hblock, lat):
    """1 - R^2 of the additive (value + unit) model on the batch's block cell means (count-weighted)."""
    cell = lat["mu"] * P + lat["m"]
    ids, inv, cnt = torch.unique(cell, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(ids), hblock.shape[1]).index_add_(0, inv, hblock)
    means = sums / cnt[:, None]
    w = cnt.float() / cnt.sum()
    u, v = ids // P, ids % P
    X = torch.cat([F.one_hot(v, P), F.one_hot(u, 3)], 1).float()
    sw = w.sqrt()[:, None]
    coef = torch.linalg.lstsq(X * sw, means * sw).solution
    resid = means - X @ coef
    mu = (w[:, None] * means).sum(0)
    tot = (w[:, None] * (means - mu) ** 2).sum()
    return (w[:, None] * resid ** 2).sum() / tot


def ckpt_path(group, seed, step):
    for root in (WINDOW, RECON):
        d = root / f"{group}_s{seed}" / "ckpt" / f"step_{step:05d}"
        if (d / "resume.pt").exists():
            return d
    raise FileNotFoundError(f"{group}_s{seed} step {step}")


@torch.no_grad()
def measure(m):
    from sci import window_analysis as W
    from sci.factorization import factor_scores
    r = W.measure(m)
    out = {k: r[k] for k in ("acc_id", "oodM", "oodA", "iia_oodsrc", "rsi_M")}
    out["r2add"] = factor_scores(m, "block", n=16384)["r2_additive"]
    return out


def run(spec, out_dir):
    """spec: dict(name, seed, resume, steps, lam_int, int_steps, wd, iit_mult, freeze_qk_steps, project_steps)."""
    out = Path(out_dir) / spec["name"]
    if (out / "result.json").exists():
        return spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    set_determinism()
    group = "mass_only"
    seed, start = spec["seed"], spec["resume"]
    torch.manual_seed(seed)
    m = TinyTyped(group, **RS.MODEL_KW)
    opt = torch.optim.AdamW(m.parameters(), lr=RS.LR, weight_decay=RS.WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=RS.LR, total_steps=RS.STEPS, pct_start=0.1)
    g_base, g_src = torch.Generator(), torch.Generator()
    src = ckpt_path(group, seed, start)
    m.load_state_dict(load_file(str(src / "model.safetensors")))
    st = torch.load(src / "resume.pt", weights_only=False)
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    g_base.set_state(st["g_base"])
    g_src.set_state(st["g_src"])
    torch.set_rng_state(st["torch_rng"])
    random.setstate(st["py_rng"])
    if spec.get("wd") is not None:
        for gp in opt.param_groups:
            gp["weight_decay"] = spec["wd"]
    mk = D.mask(*RS.BLOCK)
    qkw = m.attn2.qkv.weight
    qkb = m.attn2.qkv.bias
    d = RS.MODEL_KW.get("d", 64) if "d" in RS.MODEL_KW else 64
    end = min(RS.STEPS, start + spec["steps"])
    log = {"spec": spec, "trajectory": {}}
    log["trajectory"][start] = measure(m)
    for step in range(start + 1, end + 1):
        lat_b = T.sample_latents(RS.BATCH, g_base)
        bb = T.build(lat_b)
        _, hb = D.encode(m, bb.tokens)
        loss = F.cross_entropy(m.head_from_resid(hb), bb.y)
        lat_s = T.sample_latents(RS.BATCH, g_src)
        _, hs = D.encode(m, T.build(lat_s).tokens)
        iit = F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk), D.counterfactual(lat_b, lat_s, "M"))
        loss = loss + spec.get("iit_mult", 1.0) * RS.lam_at(step, "warmup") * iit
        active = step <= start + spec.get("int_steps", spec["steps"])
        lam = spec.get("lam_int", 0.0)
        li = None
        if lam and active:
            li = l_int(hb[:, :16], lat_b)
            loss = loss + lam * li
        opt.zero_grad()
        proj = spec.get("project_steps", 0) and step <= start + spec["project_steps"]
        if proj:
            li_p = l_int(hb[:, :16], lat_b)
            gI = torch.autograd.grad(li_p, [qkw, qkb], retain_graph=True)
        loss.backward()
        if proj:
            for p_, gi in ((qkw, gI[0]), (qkb, gI[1])):
                rows = slice(0, 2 * 64)
                g = p_.grad[rows]
                gi = gi[rows]
                dot = (g * gi).sum()
                if dot < 0:                                   # descending on loss would increase L_int
                    p_.grad[rows] = g - dot / (gi * gi).sum().clamp_min(1e-12) * gi
        frz = spec.get("freeze_qk_steps", 0) and step <= start + spec["freeze_qk_steps"]
        if frz:
            saved = (qkw.data[: 2 * 64].clone(), qkb.data[: 2 * 64].clone())
        opt.step()
        if frz:
            qkw.data[: 2 * 64] = saved[0]
            qkb.data[: 2 * 64] = saved[1]
        sched.step()
        if step % 100 == 0 or step == end:
            r = measure(m)
            if li is not None:
                r["l_int_batch"] = float(li.detach())
            log["trajectory"][step] = r
            if spec.get("control"):
                ref = RECON / f"{group}_s{seed}" / "ckpt" / f"step_{step:05d}" / "model.safetensors"
                if step % 100 == 0 and ref.exists():
                    a = load_file(str(ref))
                    r["bitwise_vs_reconstruction"] = all(torch.equal(a[k], m.state_dict()[k]) for k in a)
    save_file({k: v.detach().clone().contiguous() for k, v in m.state_dict().items()}, str(out / "final.safetensors"))
    log["final"] = log["trajectory"][end]
    (out / "result.json").write_text(json.dumps(log, indent=1))
    return spec["name"]


def specs():
    S = []
    add = lambda **k: S.append(dict(steps=2000, **k))
    for s in (304, 310, 302):
        add(name=f"ctl_{s}_2000", seed=s, resume=2000, control=True)
        add(name=f"P1a_{s}_2000", seed=s, resume=2000, lam_int=1.0)
        add(name=f"P1b_{s}_2000", seed=s, resume=2000, lam_int=0.1)
        add(name=f"P3a_{s}_2000", seed=s, resume=2000, wd=1.0)
    for s, r in ((304, 700), (304, 850), (310, 1350), (310, 1450)):
        add(name=f"ctl_{s}_{r}", seed=s, resume=r, control=True)
        add(name=f"P1early_{s}_{r}", seed=s, resume=r, lam_int=1.0, int_steps=500)
    for r in (700, 850, 1000, 1500, 4000):
        if r not in (700, 850):
            add(name=f"ctl_304_{r}", seed=304, resume=r, control=True)
        add(name=f"P1path_304_{r}", seed=304, resume=r, lam_int=1.0)
    for s in (307, 311):
        add(name=f"ctl_{s}_850", seed=s, resume=850, control=True)
        add(name=f"P2a_{s}_850", seed=s, resume=850, lam_int=-1.0)
        add(name=f"P2b_{s}_850", seed=s, resume=850, iit_mult=3.0)
        add(name=f"ctl_{s}_2000", seed=s, resume=2000, control=True)
        add(name=f"P2c_{s}_2000", seed=s, resume=2000, lam_int=-1.0)
    add(name="F1_304_700", seed=304, resume=700, freeze_qk_steps=500)
    add(name="G1_304_700", seed=304, resume=700, project_steps=500)
    return S


def _job(a):
    try:
        return run(*a)
    except Exception as e:  # report, do not hide
        return f"{a[0]['name']} FAILED: {e!r}"


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sp = [s for s in specs() if not a.only or s["name"] in a.only]
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_job, [(s, a.out) for s in sp]):
            print("[done]", r, flush=True)
