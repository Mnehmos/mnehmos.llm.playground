"""Programs x hosts: rerun each interpreter-written edit program on networks it never saw.

Separates "this interpreter wrote a better program" from "this network is easier to edit",
and tests program transfer across independently trained hosts and across conditions.
The goal is always the true one: '+' (token 11) behaves as '-' (token 12).

    python pilot/cross_eval.py --blind <dir with net_*/> --zoo runs/zoo_wd0.01 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nri import tasks  # noqa: E402
from nri.nri_exec import Host, load_program  # noqa: E402
from nri.train import load, set_determinism  # noqa: E402


def true_goal_eval(host, cond, dW, db, model):
    lat = tasks.sample_latents(8192, torch.Generator().manual_seed(999))
    b = tasks.build(lat, cond)
    h = host.site("resid_pre_mlp", b.tokens, b.types)
    pred = host.head(h, dW, db)
    in_scope = lat["op"] == 0
    b2 = tasks.build(tasks.with_latent(lat, op=torch.where(in_scope, 1, lat["op"])), cond)
    return dict(success=(pred[in_scope] == b2.y[in_scope]).float().mean().item(),
                preserve=(pred[~in_scope] == b.y[~in_scope]).float().mean().item(),
                rank=int(torch.linalg.matrix_rank(dW)),
                rel_norm=(dW.norm() / model.W_in.weight.norm()).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", type=Path, required=True)
    ap.add_argument("--zoo", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--conditions", nargs="+", default=["none", "typed"])
    ap.add_argument("--hosts", type=Path, nargs="*", default=None,
                    help="explicit host run dirs (overrides --zoo/--conditions/--seeds)")
    ap.add_argument("--input-condition", default=None,
                    help="task condition used to render host inputs (default: host condition)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "pilot" / "cross_eval.json")
    ap.add_argument("--allow-programs", action="store_true",
                    help="execute interpreter-written edit_program.py files (review first, or run isolated)")
    a = ap.parse_args()
    set_determinism(4)
    rows = []
    for pkg in sorted(p for p in a.blind.glob("net_*") if p.is_dir()):
        key = json.loads((a.blind / "answer_keys" / f"{pkg.name}.json").read_text())
        doc = json.loads((pkg / "nri.json").read_text())
        args = next((c["program"].get("args", {}) for c in doc["claims"] if c["kind"] == "edit"), {})
        # interpreter-written code: requires --allow-programs (review first, or run isolated)
        program = load_program(pkg / "edit_program.py", "program", allow_programs=a.allow_programs)
        host_runs = a.hosts or [a.zoo / f"{c}_s{s}" for c in a.conditions for s in a.seeds]
        for run in host_runs:
                cond = a.input_condition or json.loads((run / "config.json").read_text())["condition"]
                model = load(run)
                host = Host(model, (lambda c: lambda n, g: (lambda b: (b.tokens, b.types))(
                    tasks.make_batch(n, c, g)))(cond), tasks.P)
                t = time.time()
                try:
                    dW, db = program(host, doc, **args)
                    r = true_goal_eval(host, cond, dW, db, model)
                except Exception as e:  # noqa: BLE001
                    r = dict(error=f"{type(e).__name__}: {e}")
                r.update(program=pkg.name, program_written_on=key["condition"],
                         program_mode=key.get("mode", "white"), host=f"{run.parent.name}/{run.name}",
                         host_condition=json.loads((run / "config.json").read_text())["condition"],
                         own_host=Path(key["run"]).resolve() == run.resolve(),
                         seconds=round(time.time() - t, 1))
                rows.append(r)
                print(json.dumps(r))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
