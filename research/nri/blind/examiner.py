"""Owner-side custody for the blind pilot. Run from research/nri with python -m blind.examiner."""
import argparse
from contextlib import redirect_stdout
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import secrets
import shutil
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import torch
from safetensors import safe_open
from safetensors.torch import load, save

import run_dimensional as D
import run_swap as S
from nri import dimtask as T
from . import design
from .participant import new_model


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PACKET_FILES = {"participant.py": ROOT / "blind/participant.py",
                "START_HERE.md": ROOT / "blind/brief.md",
                **{f"nri/{name}.py": ROOT / f"nri/{name}.py"
                   for name in ("__init__", "model", "tasks", "dimtask")}}
CASES = ("first", "second")
ROLES = ("M", "A")
WIDTH = 16


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return sha256(data).hexdigest()


def outside_repo(path):
    path = Path(path).resolve()
    if path.is_relative_to(REPO):
        raise ValueError("use a private directory outside the checkout")
    return path


def zip_bytes(members):
    buf = BytesIO()
    with ZipFile(buf, "w") as z:
        for name, data in sorted(members.items()):
            z.writestr(ZipInfo(name), data, compress_type=ZIP_DEFLATED)
    return buf.getvalue()


def record_once(path, data):
    """Install a complete receipt atomically. Identical resubmissions are safe."""
    with tempfile.NamedTemporaryFile(dir=path.parent) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            os.link(tmp.name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"a different {path.name} has already been sealed") from None
    return digest(data)


@torch.no_grad()
def issue(vault, source=None):
    vault = outside_repo(vault)
    if vault.exists():
        raise ValueError("choose a new vault directory")
    vault.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".blind-", dir=vault.parent))
    try:
        if source is None:
            # Keep training provenance, including stdout, on the custodian's side.
            group = secrets.choice(list(S.VAR_OF))
            seed = (1 << 59) + secrets.randbits(59)
            with torch.enable_grad(), (stage / "training.log").open("w") as log, redirect_stdout(log):
                source = Path(S.train_one((group, seed, "warmup", stage / "source")))
        source = outside_repo(source)
        model, cfg = D.load(source)
        if (cfg["group"] not in S.VAR_OF or cfg["declared"] != S.VAR_OF[cfg["group"]] or
                cfg["block"] != list(S.BLOCK) or cfg["steps"] != S.STEPS or cfg["schedule"] != "warmup"):
            raise ValueError("expected a fresh single-role doc 14 warm-up run")
        S.set_determinism()
        rng = secrets.SystemRandom()
        fit, test = design.choose_split(rng)
        cal_bytes = save(design.calibration(fit))
        donor_seed = secrets.randbits(62)
        tokens = T.build(design.universe()).tokens
        reference = design.logits(model, tokens)
        source_sha = digest((source / "model.pt").read_bytes())
        receipts = {}
        sites = []
        for case in CASES:
            order = rng.sample(range(model.d), model.d)
            while {i for i, old in enumerate(order) if old < WIDTH} in sites:
                order = rng.sample(range(model.d), model.d)
            sites.append({i for i, old in enumerate(order) if old < WIDTH})
            state = design.permute_residual(model.state_dict(), order)
            public_model = new_model()
            public_model.load_state_dict(state)
            check = design.logits(public_model, tokens)
            torch.testing.assert_close(check, reference, atol=2e-4, rtol=1e-5)
            if not torch.equal(check.argmax(1), reference.argmax(1)):
                raise ValueError("permutation changed the prediction on an input")
            weights = save(state)
            key = dict(version=1, salt=secrets.token_hex(32), training_config=cfg,
                       source_sha256=source_sha, residual_order=order, fit_pairs=fit, test_pairs=test,
                       donor_seed=donor_seed, checkpoint_sha256=digest(weights),
                       calibration_sha256=digest(cal_bytes))
            manifest = {k: key[k] for k in ("version", "checkpoint_sha256", "calibration_sha256")}
            manifest["examiner_commitment"] = digest(canonical(key))
            files = {name: path.read_bytes() for name, path in PACKET_FILES.items()}
            files.update({"manifest.json": canonical(manifest), "checkpoint.safetensors": weights,
                          "calibration.safetensors": cal_bytes})
            packet = zip_bytes(files)
            dest = stage / case
            dest.mkdir()
            (dest / "participant.zip").write_bytes(packet)
            (dest / "key.json").write_bytes(canonical(key))
            receipts[case] = {"packet_sha256": digest(packet),
                              "examiner_commitment": manifest["examiner_commitment"]}
        (stage / "receipt.json").write_bytes(canonical(receipts))
        stage.rename(vault)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return receipts


def open_case(case):
    case = outside_repo(case)
    key = json.loads((case / "key.json").read_bytes())
    packet = (case / "participant.zip").read_bytes()
    receipt = json.loads((case.parent / "receipt.json").read_bytes())[case.name]
    commitment = digest(canonical(key))
    if receipt != {"packet_sha256": digest(packet), "examiner_commitment": commitment}:
        raise ValueError("case does not match the receipt")
    with ZipFile(BytesIO(packet)) as z:
        info = json.loads(z.read("manifest.json"))
        weights, calibration = z.read("checkpoint.safetensors"), z.read("calibration.safetensors")
    if (info["examiner_commitment"] != commitment or
            digest(weights) != info["checkpoint_sha256"] or digest(weights) != key["checkpoint_sha256"] or
            digest(calibration) != info["calibration_sha256"] or digest(calibration) != key["calibration_sha256"]):
        raise ValueError("case contents do not match the sealed key")
    return key, load(weights), load(calibration)


def validate_claim(data, checkpoint):
    if (not isinstance(data, dict) or set(data) != {"checkpoint_sha256", "declared_guess", *ROLES} or
            data["checkpoint_sha256"] != checkpoint or data["declared_guess"] not in ROLES):
        raise ValueError("invalid checkpoint or role in claim")
    for role in ROLES:
        entry = data[role]
        if not isinstance(entry, dict) or set(entry) != {"coordinates", "predicted_swap_intended_rate"}:
            raise ValueError("expected coordinates and predicted_swap_intended_rate")
        coords, rate = entry["coordinates"], entry["predicted_swap_intended_rate"]
        if (not isinstance(coords, list) or len(coords) != WIDTH or
                any(type(i) is not int or i not in range(64) for i in coords) or len(set(coords)) != WIDTH):
            raise ValueError("each role needs 16 distinct residual coordinates in 0..63")
        if type(rate) not in (float, int) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("swap rate must be a finite probability")


def lock(case, claim):
    case = outside_repo(case)
    key, _, _ = open_case(case)
    data = json.loads(Path(claim).read_bytes())
    validate_claim(data, key["checkpoint_sha256"])
    return record_once(case / "locked_claim.json", canonical(data))


def checked_patch(path, state, role, checkpoint):
    # Only inert, bounded tensors can cross from the participant into the trusted evaluator.
    if path.stat().st_size > sum(t.numel() * t.element_size() for t in state.values()) + 65536:
        raise ValueError("patch exceeds the model size allowance")
    params = set(dict(new_model().named_parameters()))
    result = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        if f.metadata() != {"checkpoint_sha256": checkpoint, "target": role} or not set(f.keys()) <= params:
            raise ValueError("wrong case, target, or parameter names")
        for name in f.keys():
            t, expected = f.get_tensor(name), state[name]
            if t.dtype != expected.dtype or t.shape != expected.shape or not t.isfinite().all():
                raise ValueError(f"invalid tensor for {name}")
            result[name] = t
    return result


def submit(case, mass, acceleration):
    case = outside_repo(case)
    key, state, _ = open_case(case)
    if not (case / "locked_claim.json").is_file():
        raise ValueError("lock the prediction before accepting edits")
    claim = (case / "locked_claim.json").read_bytes()
    validate_claim(json.loads(claim), key["checkpoint_sha256"])
    files = {"claim.sha256": digest(claim).encode()}
    with tempfile.TemporaryDirectory(dir=case) as tmp:
        for role, path in zip(ROLES, (mass, acceleration)):
            # Snapshot each untrusted file once, then validate and seal those exact bytes.
            bound = sum(t.numel() * t.element_size() for t in state.values()) + 65536
            with Path(path).open("rb") as inp:
                raw = inp.read(bound + 1)
            if len(raw) > bound:
                raise ValueError("patch exceeds the model size allowance")
            snapshot = Path(tmp) / f"{role}.safetensors"
            snapshot.write_bytes(raw)
            checked_patch(snapshot, state, role, key["checkpoint_sha256"])
            files[snapshot.name] = raw
    return record_once(case / "sealed_edits.zip", zip_bytes(files))


@torch.no_grad()
def grade_case(case):
    key, state, cal = open_case(case)
    claim_bytes = (case / "locked_claim.json").read_bytes()
    claim = json.loads(claim_bytes)
    validate_claim(claim, key["checkpoint_sha256"])
    sealed = (case / "sealed_edits.zip").read_bytes()
    model = new_model()
    model.load_state_dict(state)
    lat = design.split_rows(key["test_pairs"])
    tokens = T.build(lat).tokens
    h = design.resid(model, tokens)
    original = model.head_from_resid(h).argmax(1)
    cal_h = design.resid(model, cal["tokens"])
    pairs = design.swap_pairs(lat, h, key["donor_seed"])
    role = key["training_config"]["declared"]
    known = [i for i, old in enumerate(key["residual_order"]) if old in range(*S.BLOCK)]
    chance_site = random.Random(key["donor_seed"]).sample(range(model.d), WIDTH)
    report = dict(checkpoint_sha256=key["checkpoint_sha256"], claim_sha256=digest(claim_bytes),
                  sealed_edits_sha256=digest(sealed), n_questions=len(tokens), n_donor_pairs=len(pairs[0]),
                  base_accuracy=(original == T.build(lat).y).float().mean().item(),
                  declared=role, correct_role_guess=claim["declared_guess"] == role, roles={})
    with ZipFile(BytesIO(sealed)) as z, tempfile.TemporaryDirectory() as tmp:
        if z.read("claim.sha256").decode() != digest(claim_bytes):
            raise ValueError("prediction differs from the one sealed with the edits")
        for target in ROLES:
            coords = claim[target]["coordinates"]
            predicted = claim[target]["predicted_swap_intended_rate"]
            observed = design.swap_rate(model, pairs, coords, target)
            ranked = design.baseline_coordinates(cal_h, cal[target], WIDTH)
            path = Path(tmp) / f"{target}.safetensors"
            path.write_bytes(z.read(path.name))
            patch = checked_patch(path, state, target, key["checkpoint_sha256"])
            edited = new_model()
            edited.load_state_dict({**state, **patch})
            effect = design.logits(edited, tokens).argmax(1)
            scores = dict(predicted_swap=predicted, observed_swap=observed,
                          absolute_prediction_error=abs(observed - predicted),
                          changed_parameters=sum(int((t != state[name]).sum()) for name, t in patch.items()),
                          changed_tensors=sorted(name for name, t in patch.items() if not torch.equal(t, state[name])),
                          random_site_swap=design.swap_rate(model, pairs, chance_site, target),
                          python_site_swap=design.swap_rate(model, pairs, ranked, target),
                          unchanged_model=design.edit_scores(original, original, lat, target),
                          submitted_weights=design.edit_scores(effect, original, lat, target),
                          python_affine=design.edit_scores(design.python_baseline(
                              model, cal_h, cal[target], ranked, h), original, lat, target))
            if target == role:
                scores.update(overlap_with_designed_site=len(set(coords) & set(known)),
                              designed_site_swap=design.swap_rate(model, pairs, known, target),
                              designed_site_affine=design.edit_scores(design.python_baseline(
                                  model, cal_h, cal[target], known, h), original, lat, target))
            report["roles"][target] = scores
    return report


def grade(vault):
    vault = outside_repo(vault)
    if any(not (vault / side / "sealed_edits.zip").is_file() for side in CASES):
        raise ValueError("both isolated cases must be submitted before any scores are released")
    S.set_determinism()
    binding = {side: digest((vault / side / "sealed_edits.zip").read_bytes()) for side in CASES}
    report_path = vault / "report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_bytes())
        if all(existing[s]["sealed_edits_sha256"] == binding[s] for s in CASES):
            return existing
        raise ValueError("existing report belongs to a different submission")
    result = {side: grade_case(vault / side) for side in CASES}
    mapped = {}
    for side in CASES:
        key, _, _ = open_case(vault / side)
        claim = json.loads((vault / side / "locked_claim.json").read_bytes())
        mapped[side] = {r: {key["residual_order"][i] for i in claim[r]["coordinates"]} for r in ROLES}
    result["twins"] = {"n_independent_networks": 1,
                       "coordinate_overlap_out_of_16": {
                           r: len(mapped["first"][r] & mapped["second"][r]) for r in ROLES}}
    record_once(report_path, canonical(result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    issue_cmd = commands.add_parser("issue")
    issue_cmd.add_argument("--vault", required=True, type=Path)
    issue_cmd.add_argument("--source", type=Path, help="optional trusted fresh run outside this checkout")
    for name in ("lock", "submit", "grade"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--vault" if name == "grade" else "--case", required=True, type=Path)
        if name == "lock":
            cmd.add_argument("--claim", required=True, type=Path)
        if name == "submit":
            cmd.add_argument("--mass", required=True, type=Path)
            cmd.add_argument("--acceleration", required=True, type=Path)
    args = p.parse_args()
    if args.command == "issue":
        print(json.dumps(issue(args.vault, args.source), indent=2))
    if args.command == "lock":
        print(lock(args.case, args.claim))
    if args.command == "submit":
        print(submit(args.case, args.mass, args.acceleration))
    if args.command == "grade":
        print(json.dumps(grade(args.vault), indent=2))


if __name__ == "__main__":
    main()
