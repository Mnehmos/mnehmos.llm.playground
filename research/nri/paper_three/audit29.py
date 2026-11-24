"""Recompute claims directly from an extracted, trusted Experiment 29 archive."""
import importlib.util
import json
from pathlib import Path
import sys

import torch

archive = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(archive))
spec = importlib.util.spec_from_file_location("exp29", archive / "experiment.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
torch.set_num_threads(1)
v, u, d = e.CV.T
q = v * e.SCALES[u] % 13
truth = torch.tensor([next(k for k in range(13) if k * e.SCALES[dst] % 13 == c)
                      for dst, c in zip(d, q)])
assert torch.equal(truth, e.Y)
results = []
with torch.no_grad():
    for seed in e.SEEDS:
        for arm in e.ARMS:
            for step in range(0, 601, 100):
                path = archive / "runs" / f"{seed}-{arm}" / f"{step:04}.pt"
                ck = torch.load(path, weights_only=True)
                card = json.loads(path.with_suffix(".json").read_text())
                receipt = json.loads((archive / f"review-{step:04}.json").read_text())
                assert e.h(path) == card["sha256"] == receipt["hashes"][path.parent.name]
                model = e.Student(seed, arm)
                model.load_state_dict(ck["model"])
                logits = model.convert(torch.arange(len(truth)))
                full = logits.argmax(1)
                raw = model.residual(e.CV_TOK, torch.zeros_like(e.CV_TOK))[:,:13]
                base = model.encoder[u,v]
                project = lambda x, r: torch.einsum("ni,nij->nj", x, model.decoder[d[r]])
                i,j,want,_ = e.SWAPS
                isolated = project(base, torch.arange(len(truth))).argmax(1)
                swap = model.logits(base[j],d[i],torch.nn.functional.pad(raw[i],(0,13)),"convert").argmax(1)
                isolated_swap = project(base[j],i).argmax(1)
                result = dict(seed=seed, arm=arm, step=step,
                              practice=e.rate(full[e.cv_ids],truth[e.cv_ids]),
                              authority=e.rate(swap,want), integrity=e.rate(isolated,truth),
                              isolated_authority=e.rate(isolated_swap,want))
                assert abs(result["practice"]-card["conversion"]["practiced"])<1e-8
                assert abs(result["authority"]-card["intervention"]["intended"])<1e-8
                if arm != "A" and step == 0:
                    assert all(result[k]==1 for k in ("authority","integrity","isolated_authority"))
                if arm in "CD":
                    assert torch.equal(full,truth) and result["authority"]==1
                if arm == "B" and step == 600:
                    model.residual.W_U.weight[:13].zero_()
                    assert torch.equal(model.convert(torch.arange(len(truth))).argmax(1),truth)
                    muted_force = torch.cat([model.forward_force(x) for x in torch.arange(len(e.FR)).split(256)])
                    model.load_state_dict(ck["model"])
                    full_force = torch.cat([model.forward_force(x) for x in torch.arange(len(e.FR)).split(256)])
                    result["force_logits_equal_after_zeroing"] = bool(torch.equal(muted_force,full_force))
                    fresh = e.Student(seed, arm)
                    model.encoder.copy_(fresh.encoder)
                    model.decoder.copy_(fresh.decoder)
                    result["restored_swap"] = e.rate(model.logits(model.encoder[u[j],v[j]],d[i],torch.nn.functional.pad(raw[i],(0,13)),"convert").argmax(1),want)
                results.append(result)
print(json.dumps(results,indent=2))
