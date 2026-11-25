"""Doc 45: exact replay of the critical window (steps 600-2,000) at 10-step resolution, with per-step separation
of the ordinary-task gradient and the interchange (IIT) gradient. Resumes from verified step-600 checkpoints of the
reconstruction; the training arithmetic is the original run_swap.train_one loop (aaa67c6).

    python -m sci.window_replay <recon_zoo> <out_dir> 307 311 304 310 305
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

START, END, EVERY = 600, 2000, 10


def replay(seed, recon_zoo, out_dir, group="mass_only", schedule="warmup"):
    out = Path(out_dir) / f"{group}_s{seed}"
    if (out / "DONE").exists():
        return str(out)
    out.mkdir(parents=True, exist_ok=True)
    set_determinism()
    torch.manual_seed(seed)
    m = TinyTyped(group, **RS.MODEL_KW)
    opt = torch.optim.AdamW(m.parameters(), lr=RS.LR, weight_decay=RS.WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=RS.LR, total_steps=RS.STEPS, pct_start=0.1)
    g_base, g_src = torch.Generator(), torch.Generator()
    src = Path(recon_zoo) / f"{group}_s{seed}" / "ckpt" / f"step_{START:05d}"
    m.load_state_dict(load_file(str(src / "model.safetensors")))
    st = torch.load(src / "resume.pt", weights_only=False)
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    g_base.set_state(st["g_base"])
    g_src.set_state(st["g_src"])
    torch.set_rng_state(st["torch_rng"])
    random.setstate(st["py_rng"])
    var = RS.VAR_OF[group]
    mk = D.mask(*RS.BLOCK)
    names = [n for n, _ in m.named_parameters()]
    params = [p for _, p in m.named_parameters()]

    def save(step):
        d = out / "ckpt" / f"step_{step:05d}"
        d.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().clone().contiguous() for k, v in m.state_dict().items()}, str(d / "model.safetensors"))
        torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), g_base=g_base.get_state(), g_src=g_src.get_state(),
                        torch_rng=torch.get_rng_state(), py_rng=random.getstate(), step=step), d / "resume.pt")

    save(START)
    with open(out / "steps.jsonl", "w") as log:
        for step in range(START + 1, END + 1):
            lat_b = T.sample_latents(RS.BATCH, g_base)
            bb = T.build(lat_b)
            _, hb = D.encode(m, bb.tokens)
            task = F.cross_entropy(m.head_from_resid(hb), bb.y)
            lat_s = T.sample_latents(RS.BATCH, g_src)
            _, hs = D.encode(m, T.build(lat_s).tokens)
            iit = F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk), D.counterfactual(lat_b, lat_s, var))
            lam = RS.lam_at(step, schedule)
            loss = task + lam * iit
            # read-only gradient decomposition on the retained graph (never written into .grad)
            gt = torch.autograd.grad(task, params, retain_graph=True, allow_unused=True)
            gi = torch.autograd.grad(iit, params, retain_graph=True, allow_unused=True)
            opt.zero_grad()
            loss.backward()
            lr = opt.param_groups[0]["lr"]
            before = [p.detach().clone() for p in params]
            opt.step()
            sched.step()
            rec = dict(step=step, lr=lr, lam=lam, task=task.item(), iit=iit.item(), t={})
            for n, p, a, b, p0 in zip(names, params, gt, gi, before):
                if p.grad is None:
                    continue
                a = torch.zeros_like(p) if a is None else a
                b = torch.zeros_like(p) if b is None else b
                na, nb = float(a.norm()), float(b.norm())
                cos = float((a * b).sum() / (na * nb)) if na > 0 and nb > 0 else None
                rec["t"][n] = [round(na, 6), round(lam * nb, 6), None if cos is None else round(cos, 5),
                               round(float(p.grad.norm()), 6), round(float((p.detach() - p0).norm()), 8)]
            log.write(json.dumps(rec) + "\n")
            if step % EVERY == 0:
                save(step)
    (out / "DONE").write_text("ok")
    return str(out)


def qualify(seed, recon_zoo, out_dir, group="mass_only"):
    res = {}
    for step in (1000, 2000):
        a = load_file(str(Path(recon_zoo) / f"{group}_s{seed}" / "ckpt" / f"step_{step:05d}" / "model.safetensors"))
        b = load_file(str(Path(out_dir) / f"{group}_s{seed}" / "ckpt" / f"step_{step:05d}" / "model.safetensors"))
        res[step] = all(torch.equal(a[k], b[k]) for k in a)
    return res


def _job(a):
    return replay(*a)


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    zoo, out = sys.argv[1], sys.argv[2]
    seeds = [int(s) for s in sys.argv[3:]]
    with ProcessPoolExecutor(min(4, len(seeds))) as ex:
        list(ex.map(_job, [(s, zoo, out) for s in seeds]))
    q = {s: qualify(s, zoo, out) for s in seeds}
    (Path(out) / "qualification.json").write_text(json.dumps(q, indent=1))
    print(json.dumps(q))
