"""Instrument validity: do our localization metrics detect localization when it exists,
and reject it when it does not?

Two synthetic hosts share the real head (W_in, W_out, W_U, trained here), but their site
activation h is constructed, not learned:

  disentangled  h = A_op[op] + A_a[a] + A_b[b] in disjoint coordinate blocks
  entangled     h = A[(op, a, b)]  -- an independent random code per triple
                (every variable perfectly decodable, none causally separable)

Expected: class-mean IIA and DAS IIA ~1 for `disentangled`; ~chance for `entangled`;
probes ~1 for both. Writes results/validity.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nri import analysis as A
from nri import tasks
from nri.model import TinyTyped
from nri.train import set_determinism

ROOT = Path(__file__).parent


class Synthetic(TinyTyped):
    def __init__(self, mode, d=64):
        super().__init__("none", d=d)
        g = torch.Generator().manual_seed(1)
        self.mode = mode
        if mode == "disentangled":
            E = torch.zeros(3 + 2 * tasks.P, d)
            E[0:3, 0:4] = torch.randn(3, 4, generator=g)
            E[3:3 + tasks.P, 4:20] = torch.randn(tasks.P, 16, generator=g)
            E[3 + tasks.P:, 20:36] = torch.randn(tasks.P, 16, generator=g)
        else:
            E = torch.randn(3 * tasks.P * tasks.P, d, generator=g)
        self.register_buffer("code", E * 2.0)

    def resid_pre_mlp(self, tokens, types):
        t = tokens[:, :-1]
        is_op = (t >= tasks.OP_TOKENS[0]) & (t <= tasks.OP_TOKENS[-1])
        j = is_op.float().argmax(1)
        i = torch.arange(len(t))
        op = t[i, j] - tasks.OP_TOKENS[0]
        a, b = t[i, j - 1], t[i, j + 1]
        if self.mode == "disentangled":
            return self.code[op] + self.code[3 + a] + self.code[3 + tasks.P + b]
        return self.code[(op * tasks.P + a) * tasks.P + b]


def fit_head(m, steps=3000):
    torch.manual_seed(0)
    opt = torch.optim.Adam([*m.W_in.parameters(), *m.W_out.parameters(), *m.W_U.parameters()], lr=3e-3)
    g = torch.Generator().manual_seed(5)
    for _ in range(steps):
        b = tasks.make_batch(512, "none", g)
        loss = F.cross_entropy(m(b.tokens, b.types), b.y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    b = tasks.make_batch(8192, "none", torch.Generator().manual_seed(6))
    with torch.no_grad():
        return (m(b.tokens, b.types).argmax(-1) == b.y).float().mean().item()


def main():
    set_determinism(4)
    out = {}
    for mode in ("disentangled", "entangled"):
        m = Synthetic(mode)
        acc = fit_head(m)
        m.eval()
        r = {"task_acc": acc}
        for v in ("op", "a", "b"):
            c = A.iia_curve(m, "none", v)
            d, _ = A.das_curve(m, "none", v, ks=(2, 4, 16))
            r[v] = {"probe": A.probe_acc(m, "none", v), "classmean_iia": c["curve"][-1],
                    "das": d["curve"]}
        sw = A.all_swaps(m, "none")
        r["edit_success"] = sum(x["success"] for x in sw.values()) / len(sw)
        r["edit_preserve"] = sum(x["preserve"] for x in sw.values()) / len(sw)
        out[mode] = r
        print(mode, json.dumps(r))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "validity.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
