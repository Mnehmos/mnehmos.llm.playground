"""Doc 59 A2/A4: parameter accounting and analytic training-FLOP model for TinyTyped-shaped networks.

FLOPs: multiply-accumulate counted as 2 FLOPs. Dense counting: frozen / zeroed weights still multiply.
Forward per prompt (seq L = 8, d, heads H, dh = d/H, MLP width m at the last position only, out = 13):
  per attention layer: qkv 2*L*d*3d + scores 2*H*L*L*dh + mix 2*H*L*L*dh + o 2*L*d*d
  MLP (last position): 2*d*m + 2*m*d ; readout 2*d*13
Training step ~ 3 x forward (backward ~ 2 x forward) per example, plus auxiliary passes.

    python -m sci.p8_cost accounting      # -> runs/phase8/accounting.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs/phase8"


def forward_flops(d, heads, m, L=8, out=13):
    dh = d // heads
    attn = 2 * L * d * 3 * d + 4 * heads * L * L * dh + 2 * L * d * d
    return 2 * attn + 2 * d * m + 2 * m * d + 2 * d * out


def resid_flops(d, heads, L=8):
    dh = d // heads
    return 2 * (2 * L * d * 3 * d + 4 * heads * L * L * dh + 2 * L * d * d)


def step_flops(d, heads, m, batch=256, interchange_frac=0.0):
    """training FLOPs per optimiser step. Interchange loss (K5/D): for the OP1 fraction of the batch, two extra residual
    forwards (base, source) + one head forward, all with backward (x3)."""
    f = 3 * batch * forward_flops(d, heads, m)
    if interchange_frac:
        n = batch * interchange_frac
        f += 3 * n * (2 * resid_flops(d, heads) + 2 * d * m * 2 + 2 * d * 13)
    return f


def eval_flops(d, heads, m, n_prompts=18252):
    return n_prompts * forward_flops(d, heads, m)


def accounting():
    from sci import hybrid as H
    res = {}
    for cond in ("B", "Bprime", "C", "K2", "K3", "K0", "D", "K5"):
        m = H.make_model("Bprime" if cond == "Bprime" else "full", 0)
        total = sum(p.numel() for p in m.parameters())
        cons = zeroed = frozen_init = 0
        if cond not in ("B", "Bprime"):
            mask = H.install(m)
            km = H.k2_mask(m)
            if cond in ("C", "K2", "K3", "K5"):
                cons = int(sum(v.sum() for v in mask.values()))
            if cond in ("K2", "K5"):
                zeroed = int(sum((km[n] & ~mask[n]).sum() for n in km))
            if cond == "K3":
                frozen_init = int(sum((km[n] & ~mask[n]).sum() for n in km))
        eff = total - cons - zeroed - frozen_init
        d, heads, mlp = (120, 8, 480) if cond == "Bprime" else (128, 8, 512)
        res[cond] = dict(total=total, constructed_frozen=cons, structurally_zeroed=zeroed, frozen_at_init=frozen_init,
                         effective_trainable=eff, phase7_reported_trainable=total - cons if cons else total,
                         forward_flops_per_prompt=forward_flops(d, heads, mlp),
                         train_flops_per_step=step_flops(d, heads, mlp, interchange_frac=0.5 if cond in ("D", "K5") else 0.0))
        print(cond, res[cond], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "accounting.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    accounting()
