"""Doc 48 §5: where does correctly represented information stop being used? For networks with factorised blocks
(307, 311: generalise; 300, 305: partial) the only computation after the block is the head MLP
(h -> h + W_out relu(W_in h + b) -> W_U). Measures per unseen cell (mass unit 2, value 9-12):
  acc            accuracy on that cell
  route_jacc     Jaccard overlap of the head's active hidden units (relu > 0, averaged over contexts) with those of
                 TRAINED renderings carrying the same canonical mass in the same context (canonical twins)
  logit_cos      cosine between the head's logit vectors for the unseen rendering and its trained twin
  block_err      distance of the unseen rendering's block from the additive prediction (relative)
    python -m sci.decoder_trace
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri import dimtask as T  # noqa: E402
from sci import rescue as RS_  # noqa: E402
from sci import window_analysis as W  # noqa: E402

P = T.P


@torch.no_grad()
def trace(m, dec, n=4096):
    out = {}
    for v in range(9, 13):
        L = W.lat(n, 440_000 + v)
        L["mu"] = torch.full((n,), 2)
        L["m"] = torch.full((n,), v)
        M = (v * T.MASS_S[2]) % P
        # canonical twin: same context, mass written in a trained rendering of the same M
        u_tw = 0 if (M * pow(T.MASS_S[0], -1, P)) % P <= 12 else 1
        Lt = {k: x.clone() for k, x in L.items()}
        Lt["mu"] = torch.full((n,), u_tw)
        Lt["m"] = torch.full((n,), (M * pow(T.MASS_S[u_tw], -1, P)) % P)
        h, ht = RS_.resid(m, L), RS_.resid(m, Lt)
        y = T.build(L).y
        z, zt = torch.relu(h @ m.W_in.weight.T + m.W_in.bias), torch.relu(ht @ m.W_in.weight.T + m.W_in.bias)
        a, at = (z > 0).float(), (zt > 0).float()
        jacc = float(((a * at).sum(1) / ((a + at).clamp(max=1).sum(1)).clamp(min=1)).mean())
        lg, lgt = m.head_from_resid(h), m.head_from_resid(ht)
        cos = float(torch.nn.functional.cosine_similarity(lg, lgt, dim=1).mean())
        blk_pred = (dec.additive(L) + dec.ehat(h)).float()
        err = float(((h[:, :16] - blk_pred).norm(dim=1) / blk_pred.norm(dim=1)).mean())
        out[f"v{v}"] = dict(acc=float((lg.argmax(-1) == y).float().mean()), twin_acc=float((lgt.argmax(-1) == y).float().mean()),
                            route_jacc=jacc, logit_cos=cos, block_err=err,
                            block_dist_to_twin=float((h[:, :16] - ht[:, :16]).norm(dim=1).mean() / ht[:, :16].norm(dim=1).mean()))
    return out


def main():
    torch.set_num_threads(3)
    res = {}
    for s in (307, 311, 300, 305, 304, 310):
        n = f"mass_only_s{s}"
        m = RS_.load(n)
        res[n] = trace(m, RS_.Decomp(m))
        print(n, {k: {a: round(b, 2) for a, b in v.items()} for k, v in res[n].items()}, flush=True)
    (RS_.OUT / "decoder_trace.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
