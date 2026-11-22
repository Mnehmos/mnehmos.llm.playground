"""Variance at the site explained by an additive (op + a + b) model vs a full cell-mean model.
Low additive R^2 = variables are mixed (interaction features), so no linear 'register' exists.
    python additivity.py   -> results/additivity.json
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nri import analysis as A
from nri.train import load, set_determinism

ROOT = Path(__file__).parent


def r2(model, cond, n=16384):
    lat = A.latents(n, A.FIT_SEED)
    _, h = A.collect(model, lat, cond)
    tot = ((h - h.mean(0)) ** 2).sum()
    X = torch.cat([F.one_hot(lat["op"], 3), F.one_hot(lat["a"], 11), F.one_hot(lat["b"], 11)], 1).float()
    add = 1 - ((h - X @ torch.linalg.lstsq(X, h).solution) ** 2).sum() / tot
    cell = lat["op"] * 121 + lat["a"] * 11 + lat["b"]
    mu = torch.zeros(363, h.shape[1]).index_add_(0, cell, h) / torch.bincount(cell, minlength=363).clamp(min=1)[:, None]
    full = 1 - ((h - mu[cell]) ** 2).sum() / tot
    return add.item(), full.item()


if __name__ == "__main__":
    set_determinism(4)
    out = {}
    for zoo in sorted((ROOT / "runs").glob("zoo_*")):
        for run in sorted(zoo.iterdir()):
            cond = json.loads((run / "config.json").read_text())["condition"]
            a, f = r2(load(run), cond)
            out[f"{zoo.name}/{run.name}"] = dict(condition=cond, additive_r2=a, cell_r2=f)
    (ROOT / "results" / "additivity.json").write_text(json.dumps(out, indent=2))
    for reg in sorted({k.split("/")[0] for k in out}):
        for c in ("none", "typed", "shuffled", "canonical", "typed_fact"):
            v = [x["additive_r2"] for k, x in out.items() if k.startswith(reg) and x["condition"] == c]
            print(reg, c, f"additive R2 mean {sum(v) / len(v):.3f}")
