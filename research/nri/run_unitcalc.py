"""Unit-calculus study (design docs/20; registration docs/21). Training + pilot.

Arms:
  natural      ordinary training on all four modes (primary)
  declared     + interchange training declaring canonical mass M (dims 0-15) and acceleration A
               (dims 16-31) in mul mode (positive control for localization)
  meaningless  labels carry no meaning (negative control)
  untrained    random initialization (no training)

    python run_unitcalc.py pilot --seeds 900 901 902 903 904 --budgets 6000 12000
    python run_unitcalc.py train --seeds ... --steps <locked>
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from nri import unitcalc as U
from nri.model import TinyTyped
from nri.train import set_determinism

ROOT = Path(__file__).parent
MODEL_KW = dict(n_types=1, aux_classes=0, vocab=U.VOCAB, seq_len=U.SEQ_LEN, n_out=U.P)
BATCH, LR, WD = 512, 2e-3, 0.01
LAMBDA, WARMUP_FRAC = 3.0, 1 / 3
BLOCKS = {"M": (0, 16), "A": (16, 32)}
ARMS = ("natural", "declared", "meaningless", "untrained")
CONV_ACC = 0.95


def new_model(arm, size=None):
    return TinyTyped(arm, **MODEL_KW, **(size or {}))


def encode(m, tok):
    zero = torch.zeros_like(tok)
    x1 = m.resid_layer1(tok, zero)
    return (x1 + m.attn2(x1))[:, -1]


@torch.no_grad()
def accuracy(m, meaningless=False, n=8192, seed=999_331):
    """Accuracy per mode on in-distribution (train-split) and held-out label combinations."""
    out = {}
    for split in ("train", "held"):
        g = torch.Generator().manual_seed(seed + (split == "held"))
        lat = U.sample(n, g, split)
        b = U.build(lat, meaningless)
        ok = (m.head_from_resid(encode(m, b.tokens)).argmax(-1) == b.y)
        out[split] = {mode: ok[lat["mode"] == i].float().mean().item() for i, mode in enumerate(U.MODES)}
        out[split]["all"] = ok.float().mean().item()
    return out


def iit_targets(base, src, var):
    """mul mode: answer when canonical `var` (M or A) comes from src, all else from base."""
    qb, qs = U.quantities(base), U.quantities(src)
    M = qs["M_mul"] if var == "M" else qb["M_mul"]
    A = qs["A_mul"] if var == "A" else qb["A_mul"]
    s_out = U.scale(U.FORCE_LOG)[base["uo"]]
    return M * A * U.INV[s_out] % U.P


def train_one(args):
    arm, seed, steps, zoo, *rest = args
    hp = {"size": {}, "wd": WD, "lr": LR, "modes": U.MODES, "sched": "onecycle", **(rest[0] if rest else {})}
    run = Path(zoo) / f"{arm}_s{seed}"
    if (run / "model.pt").exists():
        return str(run)
    run.mkdir(parents=True, exist_ok=True)
    set_determinism()
    torch.manual_seed(seed)
    m = new_model(arm, hp["size"])
    meaningless = arm == "meaningless"
    cfg = dict(arm=arm, seed=seed, steps=steps, batch=BATCH, lr=hp["lr"], wd=hp["wd"], model_kw=MODEL_KW,
               size=hp["size"], modes=list(hp["modes"]), sched=hp["sched"],
               lambda_iit=LAMBDA if arm == "declared" else 0.0, blocks=BLOCKS if arm == "declared" else None,
               n_params=m.n_params(), torch=torch.__version__)
    t0 = time.time()
    if arm != "untrained":
        opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        if hp["sched"] == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=steps, pct_start=0.1)
        else:  # constant after a 5% linear warm-up
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: min(1.0, (s_ + 1) / (0.05 * steps)))
        g = torch.Generator().manual_seed(10_000 + seed)
        g_iit = torch.Generator().manual_seed(20_000 + seed)
        with open(run / "log.jsonl", "w") as log:
            for step in range(1, steps + 1):
                b = U.build(U.sample(BATCH, g, modes=tuple(hp["modes"])), meaningless)
                loss = F.cross_entropy(m.head_from_resid(encode(m, b.tokens)), b.y)
                if arm == "declared":
                    lam = LAMBDA * min(1.0, step / (WARMUP_FRAC * steps))
                    for var in ("M", "A"):
                        base = U.sample(BATCH // 2, g_iit, modes=("mul",))
                        src = U.sample(BATCH // 2, g_iit, modes=("mul",))
                        hb = encode(m, U.build(base).tokens)
                        hs = encode(m, U.build(src).tokens)
                        mk = torch.zeros(m.d)
                        mk[BLOCKS[var][0]:BLOCKS[var][1]] = 1
                        loss = loss + lam * F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk),
                                                            iit_targets(base, src, var))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                if step % 500 == 0 or step == steps:
                    a = accuracy(m, meaningless, n=2048)
                    log.write(json.dumps(dict(step=step, loss=loss.item(), acc=a["train"]["all"],
                                              per_mode=a["train"], held=a["held"]["all"])) + "\n")
                    log.flush()
    cfg["acc"] = accuracy(m, meaningless)
    cfg["train_seconds"] = round(time.time() - t0, 1)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.save(m.state_dict(), run / "model.pt")
    print(f"[done] {run.name} steps={steps} acc={cfg['acc']['train']['all']:.3f} "
          f"held={cfg['acc']['held']['all']:.3f} ({cfg['train_seconds']}s)", flush=True)
    return str(run)


def load(run):
    cfg = json.loads((Path(run) / "config.json").read_text())
    m = new_model(cfg["arm"], cfg.get("size"))
    m.load_state_dict(torch.load(Path(run) / "model.pt", weights_only=True))
    return m.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pilot", "train"))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[6000, 12000])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "pilot":
        jobs = [(arm, s, b, str(ROOT / "runs" / "zoo_unitcalc_pilot" / f"b{b}"))
                for b in a.budgets for arm in ("natural", "declared") for s in a.seeds]
    else:
        assert a.steps, "pass the budget locked in docs/21"
        jobs = [(arm, s, a.steps, str(ROOT / "runs" / "zoo_unitcalc")) for s in a.seeds for arm in a.arms]
    with ProcessPoolExecutor(a.workers) as ex:
        runs = list(ex.map(train_one, jobs))
    if a.mode == "pilot":
        rows = []
        for r in runs:
            m, cfg = load(r)
            rows.append(dict(run=str(Path(r).relative_to(ROOT)), arm=cfg["arm"], steps=cfg["steps"],
                             acc=cfg["acc"]))
        out = ROOT / "results" / "unitcalc"
        out.mkdir(parents=True, exist_ok=True)
        (out / "pilot.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
