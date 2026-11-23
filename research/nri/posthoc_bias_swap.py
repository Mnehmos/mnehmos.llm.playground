"""Supervisor's read-only test (step 2,700 review; stored snapshots of the CONTROL only; no training):
do the attention biases carry the post-convergence held-out (OOD_DEV) oscillation? Swap bias sets
between a high-OOD (1,800) and a low-OOD (2,000) checkpoint and measure OOD_DEV and OBS accuracy."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_swap as S  # noqa: E402
from nri import atlas as AT  # noqa: E402
from nri.model import TinyTyped  # noqa: E402

ZOO = Path(__file__).parent / "runs" / "zoo_lineage" / "A_control"


def load(s):
    return torch.load(ZOO / f"step_{s:04d}" / "model.pt", weights_only=True)


def score(sd):
    m = TinyTyped("mass_only", **S.MODEL_KW)
    m.load_state_dict(sd)
    d = AT.partition()
    return dict(ood_dev=round(AT.accuracy(m, d["OOD_DEV"]), 4), obs=round(AT.accuracy(m, d["OBS"]), 4))


SETS = {"attn1 biases": ["attn1.qkv.bias", "attn1.o.bias"], "attn2 biases": ["attn2.qkv.bias", "attn2.o.bias"],
        "all attention biases": ["attn1.qkv.bias", "attn1.o.bias", "attn2.qkv.bias", "attn2.o.bias"],
        "attn1 weights": ["attn1.qkv.weight", "attn1.o.weight"], "attn2 weights": ["attn2.qkv.weight", "attn2.o.weight"],
        "MLP + unembed": ["W_in.weight", "W_in.bias", "W_out.weight", "W_out.bias", "W_U.weight"]}
if __name__ == "__main__":
    hi, lo = load(1800), load(2000)
    out = {"1800 (high)": score(hi), "2000 (low)": score(lo)}
    for name, keys in SETS.items():
        a = dict(hi)
        a.update({k: lo[k] for k in keys})
        b = dict(lo)
        b.update({k: hi[k] for k in keys})
        out[f"1800 with 2000's {name}"] = score(a)
        out[f"2000 with 1800's {name}"] = score(b)
    print(json.dumps(out, indent=1))
    (Path(__file__).parent / "results" / "lineage").mkdir(parents=True, exist_ok=True)
    (Path(__file__).parent / "results" / "lineage" / "posthoc_bias_swap.json").write_text(json.dumps(out, indent=1))
