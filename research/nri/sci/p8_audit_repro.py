"""Doc 59 A1: reproduce Phase VII hybrid runs with the unchanged sci/hybrid.py, writing to runs/phase8/audit/ (never
touching runs/phase7), then compare full metric histories with the stored Phase VII JSONs.

    python -m sci.p8_audit_repro run      # B, K2, K5 full s0-2; K2, K5 q25 s0  (3 workers, resumable: skips done)
    python -m sci.p8_audit_repro compare
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import hybrid as H  # noqa: E402

AUD = ROOT / "runs/phase8/audit"
ORIG = ROOT / "runs/phase7/hybrid"
JOBS = [(c, "full", s) for c in ("B", "K2", "K5") for s in (0, 1, 2)] + [("K2", "q25", 0), ("K5", "q25", 0)]


def _job(j):
    H.OUT = AUD                      # redirect output; train_one is otherwise unchanged
    return H.train_one(j)


def run():
    todo = [j for j in JOBS if not (AUD / f"{j[0]}_{j[1]}_s{j[2]}.json").exists()]
    with ProcessPoolExecutor(3) as ex:
        for r in ex.map(_job, todo):
            print(r[:4], flush=True)


def compare():
    out = {}
    for c, g, s in JOBS:
        f = f"{c}_{g}_s{s}.json"
        a, b = json.loads((ORIG / f).read_text()), json.loads((AUD / f).read_text())
        ha, hb = a["history"], b["history"]
        maxdiff = max(abs(x[k] - y[k]) for x, y in zip(ha, hb) for k in x if isinstance(x[k], float))
        out[f] = dict(bitwise_identical_history=ha == hb, max_abs_diff=maxdiff, crit_orig=a["steps_to_OP2_criterion"],
                      crit_repro=b["steps_to_OP2_criterion"])
        print(f, out[f])
    (AUD / "comparison.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1]]()
