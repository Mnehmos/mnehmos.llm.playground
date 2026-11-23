"""Agent's bespoke inspection (not an edit): per layer-2 head, where it attends from the '=' position,
how much it writes into the mass block, and whether ITS contribution to the block's mass code is the
same for every mass unit (per-unit CKA of that head's class means) -- i.e. which head carries the
unit-1 route split. Also the layer-1 residual at the value token: per-unit CKA of mass class means.
Usage: python loop_edits/inspect_heads.py <model.pt> [...]"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_swap as S  # noqa: E402
from nri import atlas as AT  # noqa: E402
from nri.model import TinyTyped  # noqa: E402


def pair_cka(Z, fo):
    out = {}
    for u1, u2 in ((0, 1), (0, 2), (1, 2)):
        cs = [k for k in range(AT.P) if ((fo["M"] == k) & (fo["mu"] == u1)).any() and ((fo["M"] == k) & (fo["mu"] == u2)).any()]
        A = torch.stack([Z[(fo["M"] == k) & (fo["mu"] == u1)].mean(0) for k in cs])
        B = torch.stack([Z[(fo["M"] == k) & (fo["mu"] == u2)].mean(0) for k in cs])
        out[f"{u1}{u2}"] = round(AT.linear_cka(A, B), 2)
    return out


@torch.no_grad()
def inspect(m):
    d = AT.partition()
    obs = d["OBS"]
    c = AT.capture(m, obs)
    fo = AT.factors(obs)
    rp = AT.role_pos(obs)
    idx = torch.arange(len(obs["m"]))
    Wo = m.attn2.o.weight
    rows = []
    for hd in range(4):
        pat = c["pat2"][:, hd, -1]                       # (n, 8) attention from '='
        att = {r: round(pat[idx, rp[r]].mean().item(), 2) for r in ("mval", "munit", "aval", "aunit")}
        att["fu"] = round(pat[:, 6].mean().item(), 2)
        contrib = c["pre2"][:, 16 * hd:16 * (hd + 1)] @ Wo[:, 16 * hd:16 * (hd + 1)].T   # (n, 64)
        blk = contrib[:, AT.BLK]
        dec, _ = AT.interaction_decomp(blk, fo["M"], fo["route"])
        mu = AT.class_means(blk, fo["M"], AT.P)
        rows.append(dict(head=hd, attends=att, block_write_rms=round(blk.pow(2).sum(1).mean().sqrt().item(), 3),
                         M_between_var=round(((mu - mu.mean(0)) ** 2).sum(1).mean().item(), 4),
                         interaction=round(dec["interaction_share"], 2), unit_cka=pair_cka(blk, fo)))
    x1v = c["x1"][idx, rp["mval"]]
    return rows, dict(L1_value_token_unit_cka=pair_cka(x1v, fo),
                      x1_last_block_rms=round(c["x1"][:, -1][:, AT.BLK].pow(2).sum(1).mean().sqrt().item(), 3))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        m = TinyTyped("mass_only", **S.MODEL_KW)
        m.load_state_dict(torch.load(p, weights_only=True))
        rows, extra = inspect(m)
        print("==", p, extra)
        for r in rows:
            print("  ", r)
