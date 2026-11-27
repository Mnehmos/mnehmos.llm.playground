"""Compare trace runs against the baseline trace: data digest, losses, final model/optimizer state."""
import json, sys, torch
from pathlib import Path
base = Path(sys.argv[1]); out = {}
B = json.loads((base / "metrics.json").read_text()); SB = torch.load(base / "state.pt", weights_only=False)
for c in sys.argv[2:]:
    c = Path(c); M = json.loads((c / "metrics.json").read_text()); SC = torch.load(c / "state.pt", weights_only=False)
    pd = max(float((SB["model"][k] - SC["model"][k]).abs().max()) for k in SB["model"])
    pex = all(torch.equal(SB["model"][k], SC["model"][k]) for k in SB["model"])
    ost = [(a, b) for a, b in zip(SB["opt"]["state"].values(), SC["opt"]["state"].values())]
    oex = all(torch.equal(x[k], y[k]) for x, y in ost for k in x)
    om = max(float((x[k].float() - y[k].float()).abs().max()) for x, y in ost for k in x)
    lm = max(abs(x - y) for x, y in zip(B["losses"], M["losses"]))
    out[c.name] = dict(data_digest_equal=B["batch_digest"] == M["batch_digest"], loss_max_abs=lm, loss_bitwise=lm == 0,
                       param_max_abs=pd, param_bitwise=pex, adam_state_bitwise=oex, adam_state_max_abs=om,
                       sched_equal=SB["sched"] == SC["sched"],
                       tier="exact" if (lm == 0 and pex and oex) else "numeric")
    print(c.name, out[c.name], flush=True)
print(json.dumps(out, indent=1))
