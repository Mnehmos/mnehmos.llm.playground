"""Prespecified finite-cube contrasts and developmental thresholds; seed triples are experimental units."""

import argparse
import csv
from itertools import product, combinations
import json
from pathlib import Path
from statistics import mean

PAIRS = ((307, 304), (311, 310))
NAMES = ("initial", "base", "donor")


def read(p):
    return json.loads(p.read_text())


def rate(v, key="hit", n="n"):
    return v[key] / v[n]


def cube(ys, hi, lo):
    keys = list(product((hi, lo), repeat=3))
    assert set(ys) == set(keys)
    effects = {}
    edges = []
    for dim in range(3):
        for k in keys:
            if k[dim] != lo:
                continue
            other = list(k)
            other[dim] = hi
            other = tuple(other)
            edges.append(
                dict(
                    factor=NAMES[dim],
                    from_cell=k,
                    to_cell=other,
                    difference=ys[other] - ys[k],
                )
            )
    for order in range(1, 4):
        for dims in combinations(range(3), order):
            positive = [k for k in keys if sum(k[d] == lo for d in dims) % 2 == 0]
            negative = [k for k in keys if k not in positive]
            effects["*".join(NAMES[d] for d in dims)] = mean(
                ys[k] for k in positive
            ) - mean(ys[k] for k in negative)
    return {"contrasts": effects, "simple_effects": edges}


def onset(history, select):
    ts = sorted(history)
    n = 0
    for i, t in enumerate(ts):
        n = n + 1 if select(history[t]) else 0
        if n == 3:
            return {
                "first": ts[i - 2],
                "third": t,
                "previous_observation": ts[i - 3] if i >= 3 else None,
            }
    return None


def full_manifest(root, configs):
    import hashlib

    def stream_id(a):
        return hashlib.sha256(("\n".join(a) + "\n").encode()).hexdigest()

    streams = {k: {} for k in ("base", "source")}
    init = {}
    checks = {}
    for i, b, s in configs:
        path = root / f"i{i}_b{b}_s{s}"
        name = path.name
        stepdata = [
            json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()
        ]
        assert [x["step"] for x in stepdata] == list(range(1, 6001))
        mf = [json.loads(x) for x in (path / "manifest.jsonl").read_text().splitlines()]
        expected = [
            t for t in range(6001) if t % 100 == 0 or (600 <= t <= 2000 and t % 10 == 0)
        ]
        assert [x["step"] for x in mf] == expected
        raw = mf[0]["model_safetensors_sha256"]
        assert init.setdefault(i, raw) == raw
        for label, seed in [("base", b), ("source", s)]:
            digest = stream_id(x[label] for x in stepdata)
            assert streams[label].setdefault(seed, digest) == digest
        for t in expected:
            d = path / "cards" / f"{t:04}.json"
            card = read(d)
            assert (
                card["step"] == t
                and card["state_hash"] == mf[expected.index(t)]["file_sha256"]
            )
            if t % 100 == 0:
                assert (
                    read(path / "reviews" / d.name)["card"]
                    == hashlib.sha256(d.read_bytes()).hexdigest()
                )
        checks[name] = {
            "states": len(mf),
            "reviews": 61,
            "base_stream": streams["base"][b],
            "donor_stream": streams["source"][s],
            "init": raw,
        }
    return checks


def main(root, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    results = {}
    trajectory = []
    for hi, lo in PAIRS:
        Y = {}
        hist = {}
        for k in product((hi, lo), repeat=3):
            i, b, s = k
            run = root / f"i{i}_b{b}_s{s}"
            e = read(run / "cards/6000.json")["endpoint"]
            u = e["universe"]
            Y[k] = rate(u["unfamiliar"])
            cells = {
                int(p.stem): read(p) for p in sorted((run / "cards").glob("*.json"))
            }
            hist[k] = {}
            for t, d in cells.items():
                z = d["swap"]
                sr = {name: rate(v) for name, v in d["panels"].items()}
                sr.update(
                    familiar_swap=rate(z["familiar"], "intended", "eligible"),
                    unfamiliar_swap=rate(z["unfamiliar"], "intended", "eligible"),
                    unfamiliar_excess=(
                        z["unfamiliar"]["intended"]
                        - z["unfamiliar"]["unpatched_target"]
                    )
                    / z["unfamiliar"]["eligible"],
                )
                hist[k][t] = sr
                tr = dict(pair=f"{hi}-{lo}", initial=i, base=b, source=s, step=t, **sr)
                if "permanent_inversion" in d:
                    tr["permanent_specificity"] = d["permanent_inversion"][
                        "specificity"
                    ]
                    tr["alias_both_correct"] = rate(
                        z["unfamiliar"], "alias_both_correct", "eligible"
                    )
                    tr["alias_agreement"] = rate(
                        z["unfamiliar"], "alias_prediction_agreement", "eligible"
                    )
                    tr["complement_target"] = rate(
                        z["unfamiliar"], "complement_target", "eligible"
                    )
                    tr["mlp_only_target"] = rate(
                        z["unfamiliar"], "mlp_only_target", "eligible"
                    )
                    tr["residual_only_target"] = rate(
                        z["unfamiliar"], "residual_only_target", "eligible"
                    )
                trajectory.append(tr)
            orig = e["frozen"]
            rep = cells[6000]["permanent_inversion"]
            row = dict(
                pair=f"{hi}-{lo}",
                initial=i,
                base=b,
                source=s,
                historical=(i == b == s),
                **{key: rate(v) for key, v in u.items()},
                historical_draw=rate(orig["unfamiliar"]),
                practiced_correct=u["familiar"]["hit"],
                unfamiliar_correct=u["unfamiliar"]["hit"],
                swap_familiar_raw=rate(e["swap"]["familiar"]),
                swap_unfamiliar_raw=rate(e["swap"]["unfamiliar"]),
                swap_unfamiliar_eligible=rate(
                    e["swap"]["unfamiliar"], "intended", "eligible"
                ),
                canonical_alias_both=rate(
                    e["swap"]["unfamiliar"], "alias_both_correct", "eligible"
                ),
                permanent_intended=rep["intended"],
                permanent_wrong=rep["wrong"],
                permanent_unchanged=rep["unchanged"],
                permanent_other=rep["other"],
                permanent_specificity=rep["specificity"],
                escape=next(
                    (t for t, r in hist[k].items() if r["familiar"] >= 0.5), None
                ),
                settle=next(
                    (t for t, r in hist[k].items() if r["familiar"] >= 0.95), None
                ),
            )
            rows.append(row)
        res = cube(Y, hi, lo)
        high = [k for k in Y if Y[k] >= 0.95]
        low = [k for k in Y if Y[k] <= 0.50]
        dev = {}
        if high and low:
            for metric in ("unfamiliar", "unfamiliar_swap", "familiar"):
                g = {
                    t: mean(hist[k][t][metric] for k in high)
                    - mean(hist[k][t][metric] for k in low)
                    for t in next(iter(hist.values()))
                }
                dev[metric] = onset(g, lambda x: x >= 0.20)
        res.update(high=high, low=low, first_sustained_gap=dev)
        results[f"{hi}-{lo}"] = res
    for name, table in [("cells", rows), ("trajectories", trajectory)]:
        with (out / (name + ".csv")).open("w") as f:
            w = csv.DictWriter(
                f, fieldnames=list(dict.fromkeys(k for row in table for k in row))
            )
            w.writeheader()
            w.writerows(table)
    (out / "factorial.json").write_text(json.dumps(results, indent=2) + "\n")
    flags = full_manifest(
        root, [(r["initial"], r["base"], r["source"]) for r in rows] + [(305, 305, 305)]
    )
    (out / "stream_verification.json").write_text(json.dumps(flags, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    main(args.root, args.out)
