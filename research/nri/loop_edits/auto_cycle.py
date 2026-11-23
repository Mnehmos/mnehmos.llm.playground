"""The agent's codified per-checkpoint procedure from step 1,700 on (journal §1,600):
advance -> packet -> OBS pre-check of the step-1,500 editability method at scales 0.3 / 0.4 ->
submit ONE candidate to the gate only if its OBS pre-check clears every gate criterion with margin.
Usage: python loop_edits/auto_cycle.py N   (advance N checkpoints)"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import run_loop as L  # noqa: E402
from common import rewire_edit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UP = ["attn1.qkv.weight", "attn1.qkv.bias", "attn1.o.weight", "attn1.o.bias", "attn2.qkv.weight",
      "attn2.qkv.bias", "attn2.o.weight", "attn2.o.bias"]
POST = ["W_in.weight", "W_in.bias", "W_out.weight", "W_out.bias"]
KW = dict(kl=30, iit=1, translate=1.0, l2=3, steps=400, lr=1e-3, noise_kl=10, route_keep=5)
MARGIN = dict(gain=0.02, d_ood_dev=-0.005, keep=0.995, d_iia=-0.005, d_acc=-0.002, d_wrong=0.015,
              d_route_probe=0.0, d_unit_probe=0.0)
TEMPLATE = '''"""Step {step}: auto-cycle submission (journal: codified procedure from step 1,700).
Packet: {packet}
Method: the step-1,500 editability edit (rewire_edit: random-permutation class-mean translation,
KL x30, block interchange, noise-KL x10, frozen route probe x5, L2 leash), scaled {scale}.
OBS pre-check: {pre}
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import rewire_edit  # noqa: E402

TARGET = "edit_inv"
RATIONALE = __doc__
UP = {UP}
POST = {POST}


def propose(model, ctx):
    P = dict(model.named_parameters())
    up = {{k: torch.zeros_like(P[k]) for k in UP}}
    d = rewire_edit(model, ctx, up, post=POST, also_upstream=True, **{KW})
    return {{k: v * {scale} for k, v in d.items() if v.abs().sum() > 0}}
'''


def ok(r):
    return (r["gain"] >= MARGIN["gain"] and r["d_ood_dev"] >= MARGIN["d_ood_dev"] and r["keep"] >= MARGIN["keep"]
            and r["d_iia"] >= MARGIN["d_iia"] and r["d_acc"] >= MARGIN["d_acc"] and r["d_wrong"] <= MARGIN["d_wrong"]
            and r["d_route_probe"] >= MARGIN["d_route_probe"] and r["d_unit_probe"] >= MARGIN["d_unit_probe"])


def one():
    out = subprocess.run([sys.executable, "run_loop.py", "advance"], cwd=ROOT, capture_output=True, text=True).stdout
    keep = [l for l in out.splitlines() if any(t in l for t in ("===", "acc OBS", "acc OOD", "block IIA", "pooled", "interaction share",
                                                                 "unit-transfer", "identical"))]
    print("\n".join(keep), flush=True)
    st, s, meta = L.load_current_model()
    m = st["model"]
    P = dict(m.named_parameters())
    d = rewire_edit(m, L.ctx(), {k: torch.zeros_like(P[k]) for k in UP}, post=POST, also_upstream=True, **KW)
    d = {k: v for k, v in d.items() if v.abs().sum() > 0}
    best = None
    log = {"step": s, "prechecks": {}}
    for sc in (0.4, 0.3):
        r = L.precheck(m, {k: v * sc for k, v in d.items()}, "edit_inv")
        log["prechecks"][sc] = {k: round(v, 4) for k, v in r.items()}
        print(f"  pre x{sc}:", {k: round(v, 3) for k, v in r.items()}, flush=True)
        if best is None and ok(r):
            best = (sc, r)
    if best:
        sc, r = best
        pk = "; ".join(l.strip() for l in keep[1:6])
        path = ROOT / "loop_edits" / f"step_{s:04d}_auto.py"
        path.write_text(TEMPLATE.format(step=s, packet=pk, scale=sc, pre={k: round(v, 3) for k, v in r.items()},
                                        UP=UP, POST=POST, KW=KW))
        g = subprocess.run([sys.executable, "run_loop.py", "gate", str(path), "--install"], cwd=ROOT,
                           capture_output=True, text=True).stdout
        print(g, flush=True)
        log["submitted"] = str(path.name)
        log["gate"] = g.splitlines()[0] if g else "ERROR"
    else:
        print("  no submission (no scale cleared the margins)", flush=True)
    with open(ROOT / "results" / "llm_loop" / "auto_cycle_log.jsonl", "a") as fh:
        fh.write(json.dumps(log) + "\n")


if __name__ == "__main__":
    for _ in range(int(sys.argv[1])):
        one()
