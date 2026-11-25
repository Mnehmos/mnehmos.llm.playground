"""Drive each new cohort to one boundary and wait for the scientific review before resuming."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
import time

from train import run_one

PAIRS = ((307, 304), (311, 310))


def configs():
    return [v for pair in PAIRS for v in product(pair, repeat=3)]


def work(arg):
    return run_one(*arg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("qualification", type=Path)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    q = json.loads(args.qualification.read_text())
    assert set(q) == {"307", "311", "305", "304", "310"} and all(
        x["result"] in ("EXACT", "FUNCTIONAL") for x in q.values()
    )
    jobs = [v for v in configs() if len(set(v)) > 1]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for step in range(0, 6001, 100):
            list(pool.map(work, [(args.root, ss, False, step) for ss in jobs]))
            runs = [args.root / f"i{i}_b{b}_s{s}" for i, b, s in jobs]
            files = [r / "cards" / f"{step:04}.json" for r in runs]
            if not all(f.is_file() for f in files):
                raise RuntimeError(f"missing measurement at {step}")
            print(f"{step}: all {len(runs)} new cards ready", flush=True)
            while not all((r / "reviews" / f"{step:04}.json").is_file() for r in runs):
                time.sleep(1)
