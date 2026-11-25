"""Display each blocked network's evidence and attach an actual supervisor assessment."""

import argparse
import hashlib
import json
from pathlib import Path


def observed(run, step):
    cp = run / "cards" / f"{step:04}.json"
    d = json.loads(cp.read_text())
    obs = {}
    for k in ("familiar", "unfamiliar", "unfamiliar_mass", "unfamiliar_acc"):
        v = d["panels"][k]
        obs[k] = round(v["hit"] / v["n"], 3)
    for k, t in [("familiar", "F"), ("unfamiliar", "U")]:
        a = d["swap"][k]
        n = a["eligible"]
        for label in (
            "intended",
            "unpatched_target",
            "alias_both_correct",
            "complement_target",
            "mlp_only_target",
        ):
            obs[t + "_" + label] = round(a[label] / n, 3)
    obs["edit"] = round(d["permanent_inversion"]["specificity"], 3)
    g = {k: v for k, v in d["gradients"].items() if v}
    top = max(g, key=lambda k: g[k]["ordinary_l2"])
    w = d["parameters"]
    change = max(w, key=lambda k: w[k]["change"])
    obs["largest_update"] = [change, round(w[change]["change"], 3)]
    obs["largest_gradient"] = [
        top,
        round(g[top]["ordinary_l2"], 3),
        round(g[top]["cos"] or 0, 3),
    ]
    if step:
        prev = run / "reviews" / f"{step - 100:04}.json"
        if prev.is_file():
            pr = json.loads(prev.read_text())
            k = pr["prediction_metric"]
            value = obs[k]
            obs["previous_prediction"] = {
                "metric": k,
                "value": value,
                "range": pr["prediction_range"],
                "in_range": pr["prediction_range"][0]
                <= value
                <= pr["prediction_range"][1],
            }
    return cp, obs


def pending(root):
    items = []
    for p in sorted(root.glob("i*/cards/*.json")):
        step = int(p.stem)
        run = p.parent.parent
        if step % 100 == 0 and not (run / "reviews" / p.name).exists():
            items.append((run, step))
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("mode", choices=["show", "sign"])
    ap.add_argument("--step", type=int)
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--assessment")
    ap.add_argument("--metric", default="unfamiliar")
    ap.add_argument("--range", nargs=2, type=float)
    ap.add_argument("--prediction")
    args = ap.parse_args()
    jobs = pending(args.root)
    if args.step is not None:
        jobs = [(r, s) for r, s in jobs if s == args.step]
    if args.runs:
        jobs = [(r, s) for r, s in jobs if r.name in args.runs]
    for run, step in jobs:
        cp, o = observed(run, step)
        if args.mode == "show":
            print(
                run.name,
                step,
                "ID/OOD/m/a",
                *[
                    o[k]
                    for k in (
                        "familiar",
                        "unfamiliar",
                        "unfamiliar_mass",
                        "unfamiliar_acc",
                    )
                ],
                "swap F/U/base/alias/comp/mlp",
                *[
                    o[k]
                    for k in (
                        "F_intended",
                        "U_intended",
                        "U_unpatched_target",
                        "U_alias_both_correct",
                        "U_complement_target",
                        "U_mlp_only_target",
                    )
                ],
                "edit",
                o["edit"],
                "delta",
                o["largest_update"],
                "grad",
                o["largest_gradient"],
                "prior",
                o.get("previous_prediction", {}).get("in_range"),
            )
        else:
            assert (
                args.assessment
                and args.prediction
                and args.range
                and len(args.range) == 2
                and args.metric in o
            )
            receipt = {
                "step": step,
                "run": run.name,
                "card": hashlib.sha256(cp.read_bytes()).hexdigest(),
                "evidence": o,
                "assessment": args.assessment,
                "prediction": args.prediction,
                "prediction_metric": args.metric,
                "prediction_range": args.range,
                "decision": "continue frozen trajectory"
                if step < 6000
                else "endpoint reviewed; training complete",
            }
            out = run / "reviews" / cp.name
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(receipt, sort_keys=True) + "\n")
            tmp.replace(out)
