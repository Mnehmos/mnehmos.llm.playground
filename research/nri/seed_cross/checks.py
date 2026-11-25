"""Known-answer tests and an independent restart test, without changing a live run."""

import argparse
import json
from pathlib import Path
from itertools import product
import shutil
import tempfile
import torch
from analyse import cube, onset
from observe import controls, enumerate_domain
from train import R, run_one, seed_state


def test_inputs():
    lat, masks = enumerate_domain()
    result = controls()
    assert all(torch.equal(v, v.clone()) for v in lat.values())
    for name, coordinates in [
        ("initial", (307, 304, 304)),
        ("base", (304, 307, 304)),
        ("source", (304, 304, 307)),
    ]:
        ref = seed_state(304, 304, 304)
        other = seed_state(*coordinates)
        m0 = all(
            torch.equal(a, b)
            for a, b in zip(ref[0].parameters(), other[0].parameters())
        )
        bo = torch.equal(ref[3].get_state(), other[3].get_state())
        so = torch.equal(ref[4].get_state(), other[4].get_state())
        assert (m0, bo, so) == (name != "initial", name != "base", name != "source")
    vals = {k: (1 if k[1] == 307 else 0) for k in product((307, 304), repeat=3)}
    effects = cube(vals, 307, 304)["contrasts"]
    assert effects["base"] == 1 and sum(abs(i) for i in effects.values()) == 1
    inter = {k: float((k[0] == 307) == (k[2] == 307)) for k in vals}
    ef = cube(inter, 307, 304)["contrasts"]
    assert ef["initial*donor"] == 1 and sum(abs(i) for i in ef.values()) == 1
    assert onset({0: 1, 100: 0, 200: 1, 300: 1, 400: 1}, bool) == {
        "first": 200,
        "third": 400,
        "previous_observation": 100,
    }
    return result | {
        "orthogonal_seed_streams": True,
        "planted_main_and_interaction_contrasts": True,
        "domain_count": len(lat["m"]),
    }


def resume(root):
    src = root / "i307_b307_s307"
    step = 3000
    end = 3100
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        dst = tmp / src.name
        (dst / "states").mkdir(parents=True)
        (dst / "cards").mkdir()
        (dst / "reviews").mkdir()
        for k in (2900, 3000):
            shutil.copy2(src / "states" / f"{k:04}.pt", dst / "states" / f"{k:04}.pt")
        for kind in ("cards", "reviews"):
            shutil.copy2(src / kind / "3000.json", dst / kind / "3000.json")
        entries = [
            json.loads(s) for s in (src / "manifest.jsonl").read_text().splitlines()
        ]
        start = next(x for x in entries if x["step"] == step)
        (dst / "manifest.jsonl").write_text(json.dumps(start) + "\n")
        run_one(tmp, (307, 307, 307), False, end)
        old = torch.load(src / "states" / f"{end}.pt", weights_only=False)
        new = torch.load(dst / "states" / f"{end}.pt", weights_only=False)

        def eq(a, b):
            if isinstance(a, torch.Tensor):
                return torch.equal(a, b)
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)):
                return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
            return a == b

        items = ["model", "opt", "sched", "base", "source", "cpu", "python"]
        res = {key: eq(old[key], new[key]) for key in items}
        assert all(res.values())
        return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    R.set_determinism()
    r = test_inputs()
    if a.resume:
        r["step_3000_to_3100"] = resume(a.resume)
    print(json.dumps(r))
    if a.output:
        a.output.write_text(json.dumps(r, indent=2) + "\n")
