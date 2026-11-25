"""Phase III: instrumented reconstruction of the historical role-swap networks (doc 41 §41.6).

The training arithmetic is a line-for-line copy of `run_swap.train_one` at commit aaa67c6 (the training files are
unchanged between aaa67c6 and HEAD). The instrumentation only READS state:
  * every 100 steps (and step 0): a complete resumable checkpoint
      model.safetensors              model weights
      resume.pt                      AdamW state, OneCycle state, both data-generator states, torch + python RNG
      updates.safetensors            exact AdamW decomposition summed over the interval: decay part (-lr*wd*theta_before)
                                     and adaptive part (delta - decay part), per parameter
      COMPLETE                       written last
  * every step: loss (task, IIT), grad norm per parameter group, lr -> interval.jsonl
  * the original 250-step log line, byte-for-byte as the original writes it -> log.jsonl
No audit runs inside training (the analysis code reseeds the global RNG); audits run afterwards from checkpoints.

    python -m sci.swap_recon train  <zoo> group:seed [group:seed ...] [--workers N]
    python -m sci.swap_recon resume <zoo> group:seed --from 3000 --out <zoo2>
    python -m sci.swap_recon compare <runA> <runB>          # bitwise state_dict + log comparison
"""
from __future__ import annotations

import json
import random
import sys
import time
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

CKPT_EVERY = 100


def _ckpt(run, step, m, opt, sched, g_base, g_src, dec, ada):
    d = run / "ckpt" / f"step_{step:05d}"
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().clone().contiguous() for k, v in m.state_dict().items()}, str(d / "model.safetensors"))
    torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), g_base=g_base.get_state(), g_src=g_src.get_state(),
                    torch_rng=torch.get_rng_state(), py_rng=random.getstate(), step=step), d / "resume.pt")
    if dec is not None:
        upd = {f"decay.{k}": v.contiguous() for k, v in dec.items()}
        upd.update({f"adaptive.{k}": v.contiguous() for k, v in ada.items()})
        save_file(upd, str(d / "updates.safetensors"))
    (d / "COMPLETE").write_text(str(step))


def train(group, seed, zoo, schedule="warmup", resume_from=None, src_zoo=None):
    """Exact copy of run_swap.train_one's arithmetic, plus read-only instrumentation."""
    run = Path(zoo) / f"{group}_s{seed}"
    if (run / "model.pt").exists():
        return str(run)
    run.mkdir(parents=True, exist_ok=True)
    set_determinism()
    torch.manual_seed(seed)                                   # matched initialization
    m = TinyTyped(group, **RS.MODEL_KW)
    opt = torch.optim.AdamW(m.parameters(), lr=RS.LR, weight_decay=RS.WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=RS.LR, total_steps=RS.STEPS, pct_start=0.1)
    g_base = torch.Generator().manual_seed(10_000 + seed)
    g_src = torch.Generator().manual_seed(20_000 + seed)
    var = RS.VAR_OF.get(group)
    mk = D.mask(*RS.BLOCK)
    names = [n for n, _ in m.named_parameters()]
    start = 1
    if resume_from is not None:
        src = Path(src_zoo) / f"{group}_s{seed}" / "ckpt" / f"step_{resume_from:05d}"
        m.load_state_dict(load_file(str(src / "model.safetensors")))
        st = torch.load(src / "resume.pt", weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        g_base.set_state(st["g_base"])
        g_src.set_state(st["g_src"])
        torch.set_rng_state(st["torch_rng"])
        random.setstate(st["py_rng"])
        start = resume_from + 1
    cfg = dict(group=group, arm=group, condition=group, seed=seed, schedule=schedule, steps=RS.STEPS,
               batch=RS.BATCH, lr=RS.LR, wd=RS.WD, lambda_iit=RS.LAMBDA if var else 0.0, declared=var,
               block=RS.BLOCK, warmup_steps=RS.WARMUP_STEPS if schedule == "warmup" else 0,
               n_params=m.n_params(), model_kw=RS.MODEL_KW, torch=torch.__version__,
               encoder_forwards_per_step=2 if var else 1, reconstruction=True, resumed_from=resume_from)
    if resume_from is None:
        _ckpt(run, 0, m, opt, sched, g_base, g_src, None, None)
    dec = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    ada = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    t0 = time.time()
    mode = "a" if resume_from is not None else "w"
    with open(run / "log.jsonl", mode) as log, open(run / "interval.jsonl", mode) as ilog:
        for step in range(start, RS.STEPS + 1):
            lat_b = T.sample_latents(RS.BATCH, g_base)
            bb = T.build(lat_b)
            _, hb = D.encode(m, bb.tokens)
            task = F.cross_entropy(m.head_from_resid(hb), bb.y)
            loss = task
            iit = None
            if var:
                lat_s = T.sample_latents(RS.BATCH, g_src)
                _, hs = D.encode(m, T.build(lat_s).tokens)
                iit = F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk),
                                      D.counterfactual(lat_b, lat_s, var))
                loss = loss + RS.lam_at(step, schedule) * iit
            opt.zero_grad()
            loss.backward()
            # ---- read-only instrumentation (no RNG, no in-place change to params or grads) ----
            lr = opt.param_groups[0]["lr"]
            gn = {n: float(p.grad.norm()) for n, p in m.named_parameters() if p.grad is not None}
            before = {n: p.detach().clone() for n, p in m.named_parameters()}
            opt.step()
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if before[n] is not None and p.grad is not None:
                        d_dec = -lr * RS.WD * before[n]
                        dec[n] += d_dec
                        ada[n] += (p.detach() - before[n]) - d_dec
            sched.step()
            ilog.write(json.dumps(dict(step=step, lr=lr, loss=loss.item(), task=task.item(),
                                       iit=None if iit is None else iit.item(), grad_norm=gn)) + "\n")
            if step % 250 == 0 or step == RS.STEPS:
                log.write(json.dumps(dict(step=step, loss=loss.item(), acc=D.acc(m, n=4096))) + "\n")
                log.flush()
            if step % CKPT_EVERY == 0:
                ilog.flush()
                _ckpt(run, step, m, opt, sched, g_base, g_src, dec, ada)
                dec = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
                ada = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    cfg.update(acc_id=D.acc(m), acc_ood=D.acc(m, "ood"), train_seconds=round(time.time() - t0, 1))
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.save(m.state_dict(), run / "model.pt")
    print(f"[done] {run.name} id={cfg['acc_id']:.4f} ood={cfg['acc_ood']:.4f} ({cfg['train_seconds']}s)", flush=True)
    return str(run)


def compare(run_a, run_b):
    """Bitwise state_dict equality, log.jsonl byte equality, max abs diff."""
    A = torch.load(Path(run_a) / "model.pt", weights_only=True)
    B = torch.load(Path(run_b) / "model.pt", weights_only=True)
    keys_equal = list(A) == list(B)
    bitwise = keys_equal and all(torch.equal(A[k], B[k]) for k in A)
    maxdiff = max(float((A[k].float() - B[k].float()).abs().max()) for k in A) if keys_equal else None
    la, lb = (Path(run_a) / "log.jsonl").read_bytes(), (Path(run_b) / "log.jsonl").read_bytes()
    return dict(bitwise_state_dict=bitwise, max_abs_diff=maxdiff, log_identical=la == lb,
                log_lines=(la.count(b"\n"), lb.count(b"\n")))


def _job(a):
    return train(*a)


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "resume", "compare"))
    ap.add_argument("args", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--from", dest="frm", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.mode == "compare":
        print(json.dumps(compare(*a.args), indent=1))
    elif a.mode == "resume":
        g, s = a.args[1].split(":")
        train(g, int(s), a.out, resume_from=a.frm, src_zoo=a.args[0])
    else:
        zoo, specs = a.args[0], [x.split(":") for x in a.args[1:]]
        jobs = [(g, int(s), zoo) for g, s in specs]
        with ProcessPoolExecutor(min(a.workers, len(jobs))) as ex:
            list(ex.map(_job, jobs))
