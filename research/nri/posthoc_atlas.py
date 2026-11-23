"""POST HOC (not registered) analyses for the Issue #22 atlas pilot (docs/25 §25.8).

Written after training, to explain the registered results; nothing here feeds the policy or the
registered outcome. Uses OBS only (FINAL is untouched by these analyses).

  P1  Where does the block's mass x route interaction come from? Split the block of h into its two
      additive sources, x1[last] (layer 1 at the '=' position) and the attn2 output (the edited
      tensor writes only the second), and decompose each over (M, route) cells.
  P2  How far could attn2.o reach? Fraction of the block interaction that the registered ridge fit
      can express from the attn2 head outputs (the alpha = 1 solution's achieved reduction).
  P3  Per-unit geometry of mass at the block: are the three unit-specific mass codes the same code
      shifted (route main effect) or differently arranged (interaction)? Pairwise CKA per unit pair.

    python posthoc_atlas.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

import run_atlas as RA
from nri import atlas as AT

OUT = RA.OUT


@torch.no_grad()
def p1_p2(m):
    d = AT.partition()
    obs = d["OBS"]
    c = AT.capture(m, obs)
    fo = AT.factors(obs)
    x1b = c["x1"][:, -1][:, AT.BLK]
    a2b = c["a2"][:, AT.BLK]
    out = {}
    for name, Z in (("block", c["h"][:, AT.BLK]), ("x1_last_part", x1b), ("attn2_part", a2b)):
        dec, tab = AT.interaction_decomp(Z, fo["M"], fo["route"])
        out[name] = dict(interaction_ss=dec["interaction_share"] * dec["within_ss"], within_ss=dec["within_ss"],
                         interaction_share=dec["interaction_share"], route_main_share=dec["route_main_share"])
    # share of the block's interaction table carried by each part (inner products of the tables)
    _, Tb = AT.interaction_decomp(c["h"][:, AT.BLK], fo["M"], fo["route"])
    _, Tx = AT.interaction_decomp(x1b, fo["M"], fo["route"])
    _, Ta = AT.interaction_decomp(a2b, fo["M"], fo["route"])
    tot = (Tb ** 2).sum().item()
    out["interaction_from_x1_last"] = ((Tx * Tb).sum() / tot).item()
    out["interaction_from_attn2"] = ((Ta * Tb).sum() / tot).item()
    # P2: the full alpha = 1 registered direction, applied, measured on OBS
    import copy
    mm = copy.deepcopy(m)
    dW, db = AT.interaction_direction(mm)
    AT.apply_delta(mm, dW, db)
    c2 = AT.capture(mm, obs)
    dec2, _ = AT.interaction_decomp(c2["h"][:, AT.BLK], fo["M"], fo["route"])
    out["alpha1_rel_norm"] = torch.cat([dW.flatten(), db]).norm().item() / AT.block_row_norm(m)
    out["alpha1_interaction_share_after"] = dec2["interaction_share"]
    out["alpha1_acc_after"] = AT.accuracy(mm, obs, c2)
    out["acc_before"] = AT.accuracy(m, obs, c)
    return out


@torch.no_grad()
def p3(m):
    d = AT.partition()
    obs = d["OBS"]
    c = AT.capture(m, obs)
    fo = AT.factors(obs)
    Z = c["h"][:, AT.BLK]
    res = {}
    for u1, u2 in ((0, 1), (0, 2), (1, 2)):
        cs = [k for k in range(AT.P) if ((fo["M"] == k) & (fo["mu"] == u1)).any() and ((fo["M"] == k) & (fo["mu"] == u2)).any()]
        A = torch.stack([Z[(fo["M"] == k) & (fo["mu"] == u1)].mean(0) for k in cs])
        B = torch.stack([Z[(fo["M"] == k) & (fo["mu"] == u2)].mean(0) for k in cs])
        res[f"units {u1}-{u2}"] = round(AT.linear_cka(A, B), 3)
    return res


def main():
    rows = {}
    for name in RA.TRAJ:
        rows[name] = []
        for s in (600, 800, 900, 1000, 1100, 1200, 1500, 2000, 3000, 6000):
            m = RA.load_model(name, s)
            r = dict(step=s, **p1_p2(m), per_unit_cka_block=p3(m))
            rows[name].append(r)
            print(name, s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                            if k in ("interaction_from_x1_last", "interaction_from_attn2", "alpha1_interaction_share_after",
                                     "alpha1_acc_after", "alpha1_rel_norm")}, r["per_unit_cka_block"], flush=True)
    (OUT / "posthoc.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
