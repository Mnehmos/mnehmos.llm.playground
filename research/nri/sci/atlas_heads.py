"""Concept Formation Atlas, head level (doc 48 §9): for each attention head in layers 1 and 2 of the role-swap
networks, (i) its share of the declared block's variance at the last token (layer 2 writes the block directly),
(ii) the effect of zero-ablating the head on familiar accuracy and unseen-mass accuracy.
    python -m sci.atlas_heads
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri import dimtask as T  # noqa: E402
from sci import rescue as R  # noqa: E402
from sci import window_analysis as W  # noqa: E402


@torch.no_grad()
def forward(m, L, ablate=None):
    """Manual forward with optional zero-ablation of one head: ablate=(layer, head)."""
    b = T.build(L)
    x = m.embed(b.tokens, torch.zeros_like(b.tokens))
    for layer, attn in ((1, m.attn1), (2, m.attn2)):
        B, Ln, d = x.shape
        q, k, v = attn.qkv(x).view(B, Ln, 3, attn.h, attn.dh).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        pat = ((q @ k.transpose(-1, -2)) / attn.dh ** 0.5).softmax(-1)
        o = pat @ v                                                     # B, H, L, dh
        if ablate and ablate[0] == layer:
            o[:, ablate[1]] = 0
        heads_out = torch.stack([o[:, hh] @ attn.o.weight[:, hh * attn.dh:(hh + 1) * attn.dh].T for hh in range(attn.h)], 1)
        x = x + heads_out.sum(1) + attn.o.bias
        if layer == 2:
            contrib = heads_out[:, :, -1, :16]                         # B, H, 16
    h = x[:, -1]
    return m.head_from_resid(h), b.y, contrib


def atlas(m):
    out = {}
    Lid, Lood = W.lat(4096, 330_001), W.ood_mass(4096, 330_002)
    _, _, contrib = forward(m, Lid)
    var = contrib.var(0).sum(-1)
    out["block_variance_share_layer2"] = (var / var.sum()).tolist()
    for layer in (1, 2):
        for hh in range(m.attn1.h):
            lid, yid, _ = forward(m, Lid, (layer, hh))
            lod, yod, _ = forward(m, Lood, (layer, hh))
            out[f"ablate_L{layer}H{hh}"] = dict(id=float((lid.argmax(-1) == yid).float().mean()),
                                                oodM=float((lod.argmax(-1) == yod).float().mean()))
    lid, yid, _ = forward(m, Lid)
    lod, yod, _ = forward(m, Lood)
    out["none"] = dict(id=float((lid.argmax(-1) == yid).float().mean()), oodM=float((lod.argmax(-1) == yod).float().mean()))
    return out


def main():
    torch.set_num_threads(2)
    res = {}
    for s in (307, 311, 300, 305, 304, 310):
        n = f"mass_only_s{s}"
        res[n] = atlas(R.load(n))
        r = res[n]
        print(n, "L2 block share", [round(x, 2) for x in r["block_variance_share_layer2"]], {k: (round(v["id"], 2), round(v["oodM"], 2)) for k, v in r.items() if k.startswith("ablate") or k == "none"}, flush=True)
    (R.OUT / "atlas_heads.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
