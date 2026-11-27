"""Doc 59 P1/P3: is programmed computation guaranteed correct regardless of the trainable parameters?

P  = structural read-only regions + write masks + deterministic output-ownership gate.
S4 = the same structure WITHOUT the output gate (= K2/K5 architecture; K5 adds only a loss, which cannot constrain
     arbitrary parameter values).
P1: 20 random draws of all trainable parameters (2 seeds x 10 scale factors, 1..1e4) and adversarial optimisation
    (500 Adam steps maximising OP1 cross-entropy). P3: no gradient reaches masked entries; masked writes are exactly 0.

    python -m sci.p8_p1      # -> runs/phase8/p1_guarantee.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import hybrid as H  # noqa: E402
from sci import p8  # noqa: E402

OUT = ROOT / "runs/phase8"


@torch.no_grad()
def op1_status(m, L, tok, y, un):
    e = p8.evaluate(m, L, tok, y, un)
    return dict(OP1_endtoend=e["OP1_familiar"], OP1_endtoend_unseen=e["OP1_unseen"], OP1_subcircuit_exact=e["OP1_subcircuit_exact"])


def adversarial(m, L, tok, y, steps=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.nonzero(L["op"] == 0).flatten()
    opt = torch.optim.Adam(m.base.parameters(), lr=1e-2)
    for _ in range(steps):
        b = idx[torch.randint(0, len(idx), (256,), generator=g)]
        loss = -F.cross_entropy(m(tok[b]), y[b])
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def audit(m, L, tok, y):
    """P3: gradients at masked entries are exactly zero; effective weights at write-blocked entries are exactly zero."""
    m.zero_grad()
    b = torch.arange(0, len(y), 37)
    F.cross_entropy(m(tok[b]), y[b]).backward()
    bad_grad = 0
    e = m.eff()
    bad_write = 0
    for n, p in m.base.named_parameters():
        k = n.replace(".", "__")
        Fm, Zm = getattr(m, f"F__{k}"), getattr(m, f"Z__{k}")
        blocked = (Fm if Fm.numel() else torch.zeros_like(p, dtype=torch.bool)) | (Zm if Zm.numel() else torch.zeros_like(p, dtype=torch.bool))
        if p.grad is not None:
            bad_grad += int((p.grad[blocked] != 0).sum())
        if Zm.numel():
            bad_write += int((e[n][Zm] != 0).sum())
    return dict(nonzero_grads_at_blocked_entries=bad_grad, nonzero_effective_writes_at_blocked=bad_write)


def main():
    torch.set_num_threads(1)
    L = H.domain2()
    tok, y = H.build2(L)
    un = H.unseen2(L)
    res = {}
    for cond in ("P", "S4"):
        r = {"random": [], "adversarial": []}
        for seed in (0, 1):
            for sc in torch.logspace(0, 4, 10).tolist():
                m, _ = p8.build(cond, 100 + seed)
                with torch.no_grad():
                    for p in m.base.parameters():
                        p.mul_(sc)
                r["random"].append(dict(seed=seed, scale=sc, **op1_status(m, L, tok, y, un)))
        for seed in (0, 1):
            m, _ = p8.build(cond, 200 + seed)
            adversarial(m, L, tok, y, seed=seed)
            r["adversarial"].append(dict(seed=seed, **op1_status(m, L, tok, y, un)))
        m, _ = p8.build(cond, 300)
        r["audit"] = audit(m, L, tok, y)
        r["min_endtoend_random"] = min(x["OP1_endtoend"] for x in r["random"])
        r["min_endtoend_adversarial"] = min(x["OP1_endtoend"] for x in r["adversarial"])
        r["all_subcircuit_exact"] = all(x["OP1_subcircuit_exact"] == 1.0 for x in r["random"] + r["adversarial"])
        res[cond] = r
        print(cond, {k: v for k, v in r.items() if k not in ("random", "adversarial")}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p1_guarantee.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
