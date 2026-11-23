"""Score a blinded interpreter's NRI document against the answer key.

    python pilot/score_pilot.py --pkg <out>/net_A --keys <out>/answer_keys

Scores (docs/03_nri_spec.md §3.5):
  verified-claim rate   executor verdicts on the interpreter's own claims
  variable recall       for each ground-truth variable, does some interpreter variable agree with
                        it on >= 99% of inputs up to a relabeling (bijection)?
  edit score            the interpreter's edit program, scored against the TRUE goal
                        (in-scope = token tx present; target = output with tx replaced by ty)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nri import tasks  # noqa: E402
from nri.nri_exec import Host, _vars, execute, load_program  # noqa: E402
from nri.reference_nri import reference_doc  # noqa: E402
from nri.train import load  # noqa: E402


def agree_up_to_relabel(x: torch.Tensor, y: torch.Tensor) -> float:
    """Best agreement between x and y under a mapping x-value -> y-value (majority map)."""
    pairs = Counter(zip(x.tolist(), y.tolist()))
    best = {}
    for (a, b), c in pairs.items():
        if c > best.get(a, (None, 0))[1]:
            best[a] = (b, c)
    mapped = sum(c for _, c in best.values())
    injective = len({b for b, _ in best.values()}) == len(best)
    return mapped / len(x) if injective else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, required=True)
    ap.add_argument("--keys", type=Path, required=True)
    ap.add_argument("--allow-programs", action="store_true",
                    help="execute the interpreter's edit_program.py (review it first, or run isolated)")
    a = ap.parse_args()
    key = json.loads((a.keys / f"{a.pkg.name}.json").read_text())
    run, cond = Path(key["run"]), key.get("input_condition", key["condition"])
    model = load(run)
    sampler = lambda n, g: (lambda b: (b.tokens, b.types))(tasks.make_batch(n, cond, g))  # noqa
    host = Host(model, sampler, tasks.P)
    doc = json.loads((a.pkg / "nri.json").read_text())
    for c in doc.get("claims", []):  # resolve program paths relative to the package
        if c.get("kind") == "edit":
            p = Path(c["program"]["path"])
            c["program"]["path"] = str(p if p.is_absolute() else a.pkg / p)
    # The interpreter's edit program is code: run only after review or inside an isolated env.
    ex = execute(doc, host, seed=123, allow_programs=a.allow_programs)  # fresh seed

    # variable recall
    g = torch.Generator().manual_seed(77)
    tok, _ = sampler(4000, g)
    ref = reference_doc("ref", "0" * 64, cond, "unused")
    Vref = _vars(ref, tok)
    try:
        Vint = _vars(doc, tok)
    except Exception as e:  # noqa: BLE001
        Vint = {}
        print("binding error:", e)
    recall = {}
    for rid, rv in Vref.items():
        best = max((agree_up_to_relabel(iv, rv) for iv in Vint.values()), default=0.0)
        recall[rid] = best
    # edit against the TRUE goal
    tx, ty = key["edit_goal"]["token_from"], key["edit_goal"]["token_to"]
    edit_res = None
    prog = a.pkg / "edit_program.py"
    if prog.exists():
        program = load_program(prog, "program", allow_programs=a.allow_programs)
        args = next((c["program"].get("args", {}) for c in doc["claims"] if c["kind"] == "edit"), {})
        dW, db = program(host, doc, **args)
        g = torch.Generator().manual_seed(999)
        lat = tasks.sample_latents(8192, g)
        b = tasks.build(lat, cond)
        h = host.site("resid_pre_mlp", b.tokens, b.types)
        pred = host.head(h, dW, db)
        in_scope = (b.tokens == tx).any(1)
        b2 = tasks.build(tasks.with_latent(lat, op=torch.where(lat["op"] == tx - tasks.OP_TOKENS[0],
                                                                ty - tasks.OP_TOKENS[0], lat["op"])), cond)
        edit_res = dict(success=(pred[in_scope] == b2.y[in_scope]).float().mean().item(),
                        preserve=(pred[~in_scope] == b.y[~in_scope]).float().mean().item(),
                        rank=int(torch.linalg.matrix_rank(dW)),
                        rel_norm=(dW.norm() / model.W_in.weight.norm()).item())
    score = dict(blind_id=a.pkg.name, condition=cond, executor_summary=ex["summary"],
                 claim_status={c["id"]: (c["kind"], c["status"], c.get("measured")) for c in ex["claims"]},
                 variable_recall=recall, edit_vs_true_goal=edit_res)
    (a.pkg / "score.json").write_text(json.dumps(score, indent=2))
    (a.pkg / "nri.executed.json").write_text(json.dumps(ex, indent=2))
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    main()
