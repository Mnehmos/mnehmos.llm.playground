"""Deterministic training of one (condition, seed) model.

    python -m nri.train --condition typed --seed 0 --out runs/zoo

Writes <out>/<condition>_s<seed>/{model.pt, config.json, log.jsonl}.
Existing checkpoints are never overwritten (pass --force to retrain).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import tasks
from .model import TinyTyped

EVAL_SEED = 999_983


def set_determinism(threads: int = 1):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def evaluate(model, condition, n=8192, seed=EVAL_SEED):
    g = torch.Generator().manual_seed(seed)
    b = tasks.make_batch(n, condition, g)
    with torch.no_grad():
        pred = model(b.tokens, b.types).argmax(-1)
    acc = (pred == b.y).float()
    per_op = {tasks.OPS[o]: acc[b.op == o].mean().item() for o in range(len(tasks.OPS))}
    return acc.mean().item(), per_op


def train(condition: str, seed: int, out: Path, steps: int = 6000, batch: int = 512,
          lr: float = 2e-3, wd: float = 0.01, force: bool = False, **model_kw):
    run = out / f"{condition}_s{seed}"
    if (run / "model.pt").exists() and not force:
        print(f"[skip] {run} exists")
        return run
    run.mkdir(parents=True, exist_ok=True)
    set_determinism()
    torch.manual_seed(seed)
    model = TinyTyped(condition, **model_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(10_000 + seed)
    cfg = dict(condition=condition, seed=seed, steps=steps, batch=batch, lr=lr, wd=wd,
               n_params=model.n_params(), model_kw=model_kw, torch=torch.__version__,
               P=tasks.P, ops=tasks.OPS, seq_len=tasks.SEQ_LEN)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    t0 = time.time()
    with open(run / "log.jsonl", "w") as log:
        for step in range(1, steps + 1):
            b = tasks.make_batch(batch, condition, g)
            loss = F.cross_entropy(model(b.tokens, b.types), b.y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if (step <= 1000 and step % 50 == 0) or step % 250 == 0 or step == steps:
                acc, per_op = evaluate(model, condition, n=4096)
                rec = dict(step=step, loss=loss.item(), eval_acc=acc, per_op=per_op,
                           t=round(time.time() - t0, 1))
                log.write(json.dumps(rec) + "\n")
                log.flush()
    acc, per_op = evaluate(model, condition)
    cfg.update(final_eval_acc=acc, final_per_op=per_op, train_seconds=round(time.time() - t0, 1))
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.save(model.state_dict(), run / "model.pt")
    print(f"[done] {run.name} acc={acc:.4f} ({cfg['train_seconds']}s)")
    return run


def load(run: Path) -> TinyTyped:
    cfg = json.loads((run / "config.json").read_text())
    m = TinyTyped(cfg["condition"], **cfg.get("model_kw", {}))
    m.load_state_dict(torch.load(run / "model.pt", weights_only=True))
    m.eval()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=tasks.CONDITIONS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/zoo"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    train(a.condition, a.seed, a.out, steps=a.steps, force=a.force)
