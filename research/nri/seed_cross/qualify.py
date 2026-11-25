"""Verify reconstructed endpoints, exact archive trajectories, logs, predictions, and stream identity."""

import argparse
import gzip
import json
from pathlib import Path

import torch
from observe import NRI, R, D, endpoint
from history import historical_file, SEEDS
from train import sha


def validate_one(root, seed):
    run = root / f"i{seed}_b{seed}_s{seed}"
    orig = NRI / "runs/zoo_swap" / f"mass_only_s{seed}"
    entries = [json.loads(x) for x in (run / "manifest.jsonl").read_text().splitlines()]
    assert [e["step"] for e in entries] == [
        t for t in range(6001) if t % 100 == 0 or (600 <= t <= 2000 and t % 10 == 0)
    ]
    archive = json.loads(historical_file(seed, "manifest.json"))["files"]
    full = 0
    measured = 0
    for entry in entries:
        t = entry["step"]
        fp = run / "states" / f"{t:04}.pt"
        assert sha(fp) == entry["file_sha256"]
        if t % 100 == 0:
            measured += 1
            full += (
                entry["model_safetensors_sha256"]
                == archive[f"ckpt/step_{t:05}/model.safetensors"]
            )
            receipt = json.loads((run / "reviews" / f"{t:04}.json").read_text())
            assert receipt["card"] == sha(run / "cards" / f"{t:04}.json")
    gold = torch.load(orig / "model.pt", weights_only=True, map_location="cpu")
    new = torch.load(fp, weights_only=False, map_location="cpu")["model"]
    maxdiff = {k: float((new[k] - v).abs().max()) for k, v in gold.items()}
    same = all(torch.equal(new[k], v) for k, v in gold.items())
    logged = (run / "log.jsonl").read_bytes() == (orig / "log.jsonl").read_bytes()
    source = [
        json.loads(x)
        for x in gzip.decompress(
            historical_file(seed, "interval.jsonl.gz")
        ).splitlines()
    ]
    first = None
    matches = 0
    with (run / "steps.jsonl").open() as f:
        for a, b in zip(source, f, strict=True):
            b = json.loads(b)
            equal = (
                all(a[k] == b[k] for k in ("step", "loss", "task", "iit", "lr"))
                and a["grad_norm"] == b["grad"]
            )
            if not equal and first is None:
                first = a["step"]
            matches += int(equal)
    model, cfg = D.load(orig)
    model.load_state_dict(new)
    rec = endpoint(model)
    hist = json.loads(
        (Path(__file__).parent / "results/archived_endpoints.json").read_text()
    )["archive"][str(seed)]
    panel = {
        k: rec["frozen"][k]["pred_sha256"] == hist["frozen"][k]["pred_sha256"]
        for k in ("familiar", "unfamiliar")
    }
    ratings = {
        k: round(rec["frozen"][k]["hit"] / 8192, 4)
        == round(hist["frozen"][k]["hit"] / 8192, 4)
        for k in panel
    }
    audit = {
        "seed": seed,
        "result": "EXACT"
        if same and logged
        else "FUNCTIONAL"
        if all(panel.values()) and all(ratings.values())
        else "FAIL",
        "model_tensor_count": len(gold),
        "maxdiff_by_tensor": maxdiff,
        "identical_historical_log_lines": logged,
        "n_log_lines": len((orig / "log.jsonl").read_text().splitlines()),
        "exact_100_step_hashes": full,
        "expected_hashes": measured,
        "archived_gradient_and_loss_steps_exact": matches,
        "first_differing_telemetry_step": first,
        "original_panel_predictions_identical": panel,
        "exhaustive_predictions_identical": rec["pred_sha256"] == hist["pred_sha256"],
        "frozen_scores": rec["frozen"],
        "source_model_file_sha256": sha(orig / "model.pt"),
        "our_final_checkpoint_sha256": sha(fp),
    }
    return audit


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("root", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("seeds", nargs="*", type=int, default=list(SEEDS))
    opt = a.parse_args()
    R.set_determinism()
    res = {s: validate_one(opt.root, s) for s in opt.seeds}
    opt.out.write_text(json.dumps(res, indent=2) + "\n")
    print({k: v["result"] for k, v in res.items()})
    if any(x["result"] == "FAIL" for x in res.values()):
        raise SystemExit(1)
