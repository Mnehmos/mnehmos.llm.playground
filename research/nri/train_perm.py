"""Train the NEW networks for the permanent-edit study (docs/18) with the training procedure
frozen in docs/14 (run_swap.train_one, warm-up schedule). Fresh seeds 400-411, never used
before. Training never sees any editor or evaluation code.

    python train_perm.py
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import run_swap as S

ZOO = Path(__file__).parent / "runs" / "zoo_perm"
SEEDS = list(range(400, 412))
GROUPS = ["mass_only", "acc_only"]

if __name__ == "__main__":
    jobs = [(g, s, "warmup", str(ZOO)) for s in SEEDS for g in GROUPS]
    with ProcessPoolExecutor(4) as ex:
        list(ex.map(S.train_one, jobs))
