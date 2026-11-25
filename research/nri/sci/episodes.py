"""Instability-episode counter over interval logs (exploratory instrument, built at cohort 1,100).

An *episode block* is a 100-step block after step 300 (all populations fit their trained set by then) in which the
batch task loss exceeds 0.2 at least once AND at least 3 steps have a gradient norm above 3x the block median.
Counts per network and per condition up to a given step; first episode step per network.

    python -m sci.episodes <upto_step>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import factorial as Fx  # noqa: E402

LOSS_T, START = 0.2, 300


def blocks(rows, upto):
    out = []
    for s in range(START + Fx.BLOCK, upto + 1, Fx.BLOCK):
        b = [r for r in rows if s - Fx.BLOCK < r["step"] <= s]
        if len(b) < Fx.BLOCK:
            break
        med = sorted(r["grad_norm"] for r in b)[len(b) // 2]
        hot = sum(r["grad_norm"] > 3 * med for r in b)
        out.append(dict(step=s, episode=max(r["task"] for r in b) > LOSS_T and hot >= 3,
                        max_task=max(r["task"] for r in b), hot=hot))
    return out


def count(upto, live=Fx.LIVE):
    res = {}
    for p, w, s in Fx.all_runs():
        f = live / Fx.run_name(p, w, s) / "interval.jsonl"
        rows = [json.loads(l) for l in f.read_text().splitlines() if l] if f.exists() else []
        bl = blocks(rows, upto)
        ep = [b["step"] for b in bl if b["episode"]]
        res[Fx.run_name(p, w, s)] = dict(pop=p, wd=w, seed=s, n_blocks=len(bl), episodes=len(ep), first=ep[0] if ep else None, steps=ep)
    return res


def summary(res):
    out = {}
    for w in Fx.WDS:
        for p in Fx.POPS:
            rs = [r for r in res.values() if r["pop"] == p and r["wd"] == w]
            out[f"wd{w}_{p}"] = dict(networks_with_episode=sum(r["episodes"] > 0 for r in rs), total_episode_blocks=sum(r["episodes"] for r in rs),
                                     blocks=sum(r["n_blocks"] for r in rs))
    return out


if __name__ == "__main__":
    r = count(int(sys.argv[1]))
    for k, v in summary(r).items():
        print(k, v)
