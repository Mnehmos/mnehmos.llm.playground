"""Secondary, fixed-location causal controls on immutable checkpoints. No model fitting or training."""

import argparse
import json
from pathlib import Path
import torch
from observe import T, D, R, sample, forced, hidden, correct, interval_change
from nri.model import TinyTyped


@torch.no_grad()
def measure(m, n=1024):
    b = sample(n, 81001)
    s = forced(n, 81002, "M")
    hb = hidden(m, b)
    hs = hidden(m, s)
    next_m = D.counterfactual(b, s, "M")
    base = T.build(b).y
    wrong = D.counterfactual(b, s, "A")
    ok = (next_m != base) & (next_m != wrong) & (wrong != base)
    # Same mass in a different written unit, while donor acceleration and output context vary.
    alias = {k: v.clone() for k, v in s.items()}
    alias["mu"] = (b["mu"] + 1) % 3
    alias["m"] = T.canon(b)[0] * torch.tensor([1, 8, 5])[alias["mu"]] % 13
    assert T.canon(alias)[0].equal(T.canon(b)[0])
    ha = hidden(m, alias)
    read = lambda h: m.head_from_resid(h).argmax(-1)
    q, _ = torch.linalg.qr(
        torch.randn(64, 16, generator=torch.Generator().manual_seed(81003))
    )
    rand = read(hb + ((hs - hb) @ q) @ q.T)
    intact = read(hb)
    eq = read(interval_change(hb, ha))
    planted = read(interval_change(hb, hs))
    good = ok & (intact != next_m)
    return {
        "n": n,
        "eligible": int(ok.sum()),
        "base_hit": correct(intact, base),
        "base_matches_target": correct(intact[ok], next_m[ok]),
        "known_block": correct(planted[ok], next_m[ok]),
        "random_16d": correct(rand[ok], next_m[ok]),
        "switch_eligible": int(good.sum()),
        "switch_intended": correct(planted[good], next_m[good]),
        "same_mass_prediction_preserved": correct(eq, intact),
        "same_mass_oracle_preserved": correct(eq, base),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    R.set_determinism()
    out = {}
    for run in sorted(args.root.glob("i*")):
        m = TinyTyped("mass_only", **R.MODEL_KW)
        rows = {}
        for step in (0, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 6000):
            checkpoint = torch.load(
                run / "states" / f"{step:04}.pt", map_location="cpu", weights_only=False
            )
            m.load_state_dict(checkpoint["model"])
            rows[step] = measure(m)
        out[run.name] = rows
    args.out.write_text(json.dumps(out, indent=2) + "\n")
