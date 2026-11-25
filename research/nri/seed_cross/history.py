"""Bind a retrospective scientific review to an exactly reproduced historical tensor state."""

import argparse
import gzip
import json
import subprocess
from pathlib import Path
import time

from cohort import observed
import hashlib

REFERENCE = "672e94f42616741ae6a65af2648f05b83a5bd651"
SEEDS = (307, 311, 305, 304, 310)
STUDY = Path(__file__).resolve().parent


def historical_file(seed, path):
    name = f"research/nri/runs/swap_recon/mass_only_s{seed}/{path}"
    return subprocess.check_output(
        ["git", "-C", str(STUDY), "show", f"{REFERENCE}:{name}"]
    )


def read_sources():
    data = {}
    for seed in SEEDS:
        audits = [
            json.loads(x)
            for x in gzip.decompress(
                historical_file(seed, "audits.jsonl.gz")
            ).splitlines()
        ]
        interval = [
            json.loads(x)
            for x in gzip.decompress(
                historical_file(seed, "interval.jsonl.gz")
            ).splitlines()
        ]
        hashes = json.loads(historical_file(seed, "manifest.json"))["files"]
        data[seed] = {}
        for row in audits:
            step = row["step"]
            block = interval[max(step - 100, 0) : step]
            grads = {}
            if block:
                for k in block[0]["grad_norm"]:
                    grads[k] = sum(x["grad_norm"][k] for x in block) / len(block)
            data[seed][step] = {
                "id": row["acc_id"],
                "ood": row["acc_ood"],
                "familiar_swap": row["iia_M"],
                "unknown_swap": row["iia_M_oodsrc"],
                "same_quantity_variance": row["rsi_M"],
                "unknown_mass": sum(
                    v for k, v in row["ood_combo_acc"].items() if k.startswith("M")
                )
                / 4,
                "highest_grad": max(grads, key=grads.get) if grads else None,
                "grad_l2": max(grads.values()) if grads else None,
                "model_sha": hashes[f"ckpt/step_{step:05}/model.safetensors"],
            }
    return data


def show(d):
    for t in range(0, 6001, 100):
        cells = [d[s][t] for s in SEEDS]
        print(
            "%4d" % t,
            " ".join(
                "/".join(
                    f"{100 * c[k]:.0f}"
                    for k in ("id", "ood", "unknown_swap", "same_quantity_variance")
                )
                for c in cells
            ),
            "grad",
            ",".join(
                f"{c['grad_l2']:.2g}" if c["grad_l2"] is not None else "-"
                for c in cells
            ),
            ",".join((c["highest_grad"] or "-").replace(".weight", "") for c in cells),
        )


def attach(root, d, review_path):
    # A prior assessment is reusable only for byte-identical network tensors. Local probes remain separately identified.
    reviews = {
        int(x["step"]): x for x in map(json.loads, review_path.read_text().splitlines())
    }
    for s in SEEDS:
        run = root / f"i{s}_b{s}_s{s}"
        mf = run / "manifest.jsonl"
        if not mf.exists():
            continue
        items = [json.loads(x) for x in mf.read_text().splitlines()]
        for item in items:
            t = item["step"]
            card = run / "cards" / f"{t:04}.json"
            receipt = run / "reviews" / card.name
            if t % 100 or t not in reviews or receipt.exists() or not card.exists():
                continue
            hist = d[s][t]
            assess = reviews[t]
            if item["model_safetensors_sha256"] != hist["model_sha"]:
                raise RuntimeError(
                    f"historical review cannot transfer, tensor mismatch: {s}/{t}"
                )
            _, obs = observed(run, t)
            result = dict(
                step=t,
                run=run.name,
                card=hashlib.sha256(card.read_bytes()).hexdigest(),
                evidence=obs,
                assessment=assess["assessment"],
                prediction=assess["prediction"],
                prediction_metric=assess["metric"],
                prediction_range=[
                    max(0, hist["ood"] - 0.15),
                    min(1, hist["ood"] + 0.15),
                ],
                historical_evidence=hist,
                provenance="LLM retrospective reading of archived measured state; model tensor SHA matched before release",
                decision="continue frozen trajectory"
                if t < 6000
                else "endpoint reviewed; training complete",
            )
            tmp = receipt.with_suffix(".revtmp")
            tmp.write_text(json.dumps(result, sort_keys=True) + "\n")
            tmp.replace(receipt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["show", "bind"])
    ap.add_argument("--root", type=Path)
    ap.add_argument("--reviews", type=Path)
    args = ap.parse_args()
    data = read_sources()
    if args.mode == "show":
        show(data)
    else:
        while True:
            attach(args.root, data, args.reviews)
            if all(
                (args.root / f"i{s}_b{s}_s{s}" / "reviews/6000.json").is_file()
                for s in SEEDS
            ):
                break
            time.sleep(1)
