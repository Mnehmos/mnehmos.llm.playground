"""Independent replay of run_swap.train_one, splitting only its three seed inputs.

Files outside Git contain recoverable states. Training pauses at full audit boundaries until a
review receipt names the card's SHA-256. Training streams never enter the evaluator.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save

import observe as O
from observe import D, R, T
from nri.model import TinyTyped


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def append(path, data):
    with path.open("a") as f:
        f.write(json.dumps(data, allow_nan=False) + "\n")


def norm(t):
    return float(t.detach().norm())


def cosine(x, y):
    den = x.norm() * y.norm()
    return float(x.flatten() @ y.flatten() / den) if den > 0 else None


def seed_state(i, b, s):
    R.set_determinism()
    random.seed(i)
    torch.manual_seed(i)
    m = TinyTyped("mass_only", **R.MODEL_KW)
    opt = torch.optim.AdamW(m.parameters(), lr=R.LR, weight_decay=R.WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=R.LR, total_steps=R.STEPS, pct_start=0.1
    )
    gb = torch.Generator().manual_seed(10000 + b)
    gs = torch.Generator().manual_seed(20000 + s)
    return m, opt, sched, gb, gs


def task_losses(m, lb, ls, mk):
    bb = T.build(lb)
    _, hb = D.encode(m, bb.tokens)
    task = F.cross_entropy(m.head_from_resid(hb), bb.y)
    _, hs = D.encode(m, T.build(ls).tokens)
    iit = F.cross_entropy(
        m.head_from_resid(hb * (1 - mk) + hs * mk), D.counterfactual(lb, ls, "M")
    )
    return task, iit


def grad_panel(m):
    lb = O.sample(64, 72101)
    ls = O.sample(64, 72102)
    left, right = task_losses(m, lb, ls, D.mask(*R.BLOCK))
    weights = list(m.parameters())
    a = torch.autograd.grad(left, weights, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(right, weights, allow_unused=True)
    out = {}
    for (n, _), g, h in zip(m.named_parameters(), a, b):
        out[n] = (
            None
            if g is None or h is None
            else {
                "ordinary_l2": norm(g),
                "interchange_l2": norm(h),
                "cos": cosine(g, h),
            }
        )
    return out


def state_dump(fp, step, seeds, m, opt, sched, gb, gs):
    temp = fp.with_suffix(".pt.tmp")
    torch.save(
        dict(
            step=step,
            seeds=seeds,
            version=torch.__version__,
            model=m.state_dict(),
            opt=opt.state_dict(),
            sched=sched.state_dict(),
            base=gb.get_state(),
            source=gs.get_state(),
            cpu=torch.get_rng_state(),
            python=random.getstate(),
            config={
                "schedule": "warmup",
                "params": m.n_params(),
                "batch": R.BATCH,
                "steps": R.STEPS,
                "lr": R.LR,
                "wd": R.WD,
                "lambda_iit": R.LAMBDA,
                "warmup": R.WARMUP_STEPS,
            },
        ),
        temp,
    )
    os.replace(temp, fp)


def await_review(run, step, card, wait):
    receipt = run / "reviews" / f"{step:04}.json"
    expected = sha(card)
    while not receipt.is_file():
        if not wait:
            return False
        time.sleep(1)
    r = json.loads(receipt.read_text())
    if (
        r.get("card") != expected
        or r.get("step") != step
        or not r.get("assessment")
        or not r.get("prediction")
    ):
        raise ValueError(f"incomplete or stale review at {run.name} step {step}")
    return True


def run_one(root, seeds, wait, upto):
    i, b, s = seeds
    run = root / f"i{i}_b{b}_s{s}"
    run.mkdir(parents=True, exist_ok=True)
    for directory in ("states", "cards", "reviews"):
        (run / directory).mkdir(exist_ok=True)
    # Prevent simultaneous writers. An operating-system lock releases on process exit.
    import fcntl

    lock = (run / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    m, opt, sched, gb, gs = seed_state(i, b, s)
    mk = D.mask(*R.BLOCK)
    initial = {n: p.detach().clone() for n, p in m.named_parameters()}
    prev = initial
    start = 0
    manifest = run / "manifest.jsonl"
    entries = (
        [json.loads(x) for x in manifest.read_text().splitlines()]
        if manifest.exists()
        else []
    )
    if entries:
        latest = entries[-1]
        start = latest["step"]
        fp = run / "states" / f"{start:04}.pt"
        if sha(fp) != latest["file_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        st = torch.load(fp, weights_only=False, map_location="cpu")
        assert tuple(st["seeds"]) == seeds
        m.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        gb.set_state(st["base"])
        gs.set_state(st["source"])
        torch.set_rng_state(st["cpu"])
        random.setstate(st["python"])
        pstep = (
            max(0, ((start - 1) // 100) * 100)
            if start % 100 == 0
            else start // 100 * 100
        )
        old = torch.load(
            run / "states" / f"{pstep:04}.pt", weights_only=False, map_location="cpu"
        )["model"]
        prev = {n: old[n] for n, _ in m.named_parameters()}
        # Remove an uncommitted telemetry suffix after an unexpected interruption. Published states are immutable.
        for name in ("steps.jsonl", "log.jsonl"):
            f = run / name
            if f.is_file():
                raw = f.read_bytes().splitlines(keepends=True)
                kept = []
                for ln in raw:
                    try:
                        item = json.loads(ln)
                    except ValueError:
                        break
                    if item["step"] <= start:
                        kept.append(ln)
                if len(raw) != len(kept):
                    f.write_bytes(b"".join(kept))

    def snapshot(step, already=False):
        nonlocal prev
        fp = run / "states" / f"{step:04}.pt"
        card = run / "cards" / f"{step:04}.json"
        full = step % 100 == 0
        if not already:
            state_dump(fp, step, seeds, m, opt, sched, gb, gs)
            model_sha = hashlib.sha256(
                save({k: v.detach().contiguous() for k, v in m.state_dict().items()})
            ).hexdigest()
            entry = {
                "step": step,
                "file_sha256": sha(fp),
                "model_safetensors_sha256": model_sha,
                "size": fp.stat().st_size,
            }
            append(manifest, entry)
        if not card.is_file():
            with torch.random.fork_rng(devices=[]):
                r = O.audit(m, step, full=full)
                r["seeds"] = seeds
                if full:
                    r["parameters"] = {
                        n: dict(
                            change=norm(p - prev[n]),
                            from_init=norm(p - initial[n]),
                            weight=norm(p),
                            step_cos=cosine(p.detach(), prev[n]),
                            max_abs_change=float((p.detach() - prev[n]).abs().max()),
                        )
                        for n, p in m.named_parameters()
                    }
                    r["gradients"] = grad_panel(m)
                if step == 6000:
                    r["endpoint"] = O.endpoint(m)
                r["state_hash"] = sha(fp)
                record(card, r)
        if full:
            prev = {n: p.detach().clone() for n, p in m.named_parameters()}
            return await_review(run, step, card, wait)
        return True

    if not snapshot(start, already=bool(entries)) or start >= upto:
        return
    for step in range(start + 1, min(upto, R.STEPS) + 1):
        lb = T.sample_latents(R.BATCH, gb)
        ls = T.sample_latents(R.BATCH, gs)
        task, iit = task_losses(m, lb, ls, mk)
        loss = task + R.lam_at(step, "warmup") * iit
        opt.zero_grad()
        loss.backward()
        lr = opt.param_groups[0]["lr"]
        grads = {n: norm(p.grad) for n, p in m.named_parameters() if p.grad is not None}
        old = {
            n: p.detach().clone() for n, p in m.named_parameters() if p.grad is not None
        }
        opt.step()
        sched.step()
        delta = {
            n: norm(p.detach() - old[n]) for n, p in m.named_parameters() if n in old
        }

        # Hash latent fields in fixed order, keeping separate provenance for the ordinary and donor streams.
        def batch_sha(L):
            return hashlib.sha256(
                torch.stack(list(L.values())).numpy().tobytes()
            ).hexdigest()[:24]

        append(
            run / "steps.jsonl",
            dict(
                step=step,
                loss=loss.item(),
                task=task.item(),
                iit=iit.item(),
                lr=lr,
                grad=grads,
                update=delta,
                base=batch_sha(lb),
                source=batch_sha(ls),
            ),
        )
        if step % 250 == 0:
            append(
                run / "log.jsonl",
                dict(step=step, loss=loss.item(), acc=D.acc(m, n=4096)),
            )
        if step % 100 == 0 or (600 <= step <= 2000 and step % 10 == 0):
            if not snapshot(step):
                return
    lock.close()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("root", type=Path)
    a.add_argument("seeds", nargs=3, type=int)
    a.add_argument("--wait", action="store_true")
    a.add_argument("--upto", type=int, default=6000)
    args = a.parse_args()
    run_one(args.root, tuple(args.seeds), args.wait, args.upto)
