"""Custody and causal-scoring regressions using a disposable model, never the live keys."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest
import torch

pytest.importorskip("safetensors")
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blind import design, examiner as E
from blind.participant import load_case, new_model, save_edit
import run_swap as S


@pytest.fixture(scope="module")
def issued(tmp_path_factory):
    root = tmp_path_factory.mktemp("custody")
    source = root / "source"
    source.mkdir()
    S.set_determinism()
    with torch.random.fork_rng():
        torch.manual_seed(8136)
        model = new_model()
    torch.save(model.state_dict(), source / "model.pt")
    cfg = dict(group="mass_only", arm="mass_only", declared="M", seed=8136, block=list(S.BLOCK),
               steps=S.STEPS, schedule="warmup", model_kw=S.MODEL_KW)
    (source / "config.json").write_text(json.dumps(cfg))
    vault = root / "vault"
    E.issue(vault, source)
    return vault, model


def public_case(case, directory):
    with ZipFile(case / "participant.zip") as z:
        assert set(z.namelist()) == set(E.PACKET_FILES) | {
            "manifest.json", "checkpoint.safetensors", "calibration.safetensors"}
        z.extractall(directory)
    return load_case(directory)


def make_claim(info):
    sites = {"coordinates": list(range(16)), "predicted_swap_intended_rate": 0.5}
    return dict(checkpoint_sha256=info["checkpoint_sha256"], declared_guess="M",
                M=sites.copy(), A=sites.copy())


def test_blind_release(issued, tmp_path):
    vault, original = issued
    orders, keys = [], []
    for name in E.CASES:
        case = vault / name
        key, _, _ = E.open_case(case)
        orders.append(key["residual_order"])
        keys.append(key)
        folder = tmp_path / name
        model, rows, info = public_case(case, folder)
        assert set(info) == {"version", "checkpoint_sha256", "calibration_sha256", "examiner_commitment"}
        assert set(rows) == {"tokens", "force", "M", "A", "S"}
        assert len(set(map(tuple, key["fit_pairs"])) & set(map(tuple, key["test_pairs"]))) == 0
        assert set(map(tuple, torch.stack((rows["M"], rows["A"]), 1).tolist())) == set(map(tuple, key["fit_pairs"]))
        x = rows["tokens"][:37]
        with torch.no_grad():
            expected = design.resid(original, x)[:, key["residual_order"]]
            torch.testing.assert_close(design.resid(model, x), expected, atol=2e-5, rtol=1e-5)
        filename = folder / "calibration.safetensors"
        filename.write_bytes(filename.read_bytes() + b"altered")
        with pytest.raises(ValueError, match="digest"):
            load_case(folder)
    assert orders[0] != orders[1]
    assert keys[0]["source_sha256"] == keys[1]["source_sha256"]
    assert keys[0]["test_pairs"] == keys[1]["test_pairs"]
    assert keys[0]["donor_seed"] == keys[1]["donor_seed"]
    assert {i for i, n in enumerate(orders[0]) if n < 16} != {i for i, n in enumerate(orders[1]) if n < 16}


def test_seals_and_scores(issued, tmp_path):
    vault, _ = issued
    with pytest.raises(ValueError, match="both isolated"):
        E.grade(vault)
    for case in E.CASES:
        directory = vault / case
        work = tmp_path / case
        model, _, info = public_case(directory, work)
        claim = make_claim(info)
        claim_path = work / "claim.json"
        claim_path.write_text(json.dumps(claim))
        no_op, patch = work / "M.safetensors", work / "A.safetensors"
        save_edit(no_op, model, model, info, "M")
        changed = deepcopy(model)
        with torch.no_grad():
            changed.W_U.weight.zero_()
        save_edit(patch, model, changed, info, "A")
        with pytest.raises(ValueError, match="lock"):
            E.submit(directory, no_op, patch)
        lock_hash = E.lock(directory, claim_path)
        assert E.lock(directory, claim_path) == lock_hash
        claim["declared_guess"] = "A"
        claim_path.write_text(json.dumps(claim))
        with pytest.raises(ValueError, match="already been sealed"):
            E.lock(directory, claim_path)
        with pytest.raises(ValueError, match="wrong case"):
            E.submit(directory, patch, no_op)
        receipt = E.submit(directory, no_op, patch)
        assert E.submit(directory, no_op, patch) == receipt
        assert receipt == sha256((directory / "sealed_edits.zip").read_bytes()).hexdigest()
        save_edit(patch, model, model, info, "A")
        with pytest.raises(ValueError, match="already been sealed"):
            E.submit(directory, no_op, patch)
        if case == "first":
            with pytest.raises(ValueError, match="both isolated"):
                E.grade(vault)
    report = E.grade(vault)
    assert E.grade(vault) == report
    assert report["twins"]["n_independent_networks"] == 1
    assert 0 <= report["twins"]["coordinate_overlap_out_of_16"]["M"] <= 16
    for name in E.CASES:
        case = report[name]
        roles = case["roles"]
        assert case["correct_role_guess"] and case["n_donor_pairs"] > 0
        assert roles["M"]["submitted_weights"] == roles["M"]["unchanged_model"]
        assert roles["M"]["changed_parameters"] == 0
        assert roles["A"]["changed_tensors"] == ["W_U.weight"]
        assert roles["M"]["submitted_weights"]["agreement_where_math_unchanged"] == 1.0
        assert roles["A"]["submitted_weights"]["other"] == pytest.approx(1.0)
        assert "designed_site_swap" in roles["M"] and "designed_site_swap" not in roles["A"]


def test_patch_and_claim_validation(tmp_path):
    model = new_model()
    state = model.state_dict()
    info = {"checkpoint_sha256": "a" * 64}
    claim = make_claim(info)
    claim["A"]["coordinates"] = [0] * 16
    with pytest.raises(ValueError, match="distinct"):
        E.validate_claim(claim, info["checkpoint_sha256"])
    path = tmp_path / "invalid.safetensors"
    for invalid in (torch.ones(1), torch.full_like(state["W_in.bias"], float("inf")), state["W_in.bias"].double()):
        save_file({"W_in.bias": invalid}, path, metadata={**info, "target": "M"})
        with pytest.raises(ValueError, match="invalid tensor"):
            E.checked_patch(path, state, "M", info["checkpoint_sha256"])
    save_file({"mask_c": state["mask_c"]}, path, metadata={**info, "target": "M"})
    with pytest.raises(ValueError, match="parameter names"):
        E.checked_patch(path, state, "M", info["checkpoint_sha256"])
    save_edit(path, model, model, info, "M")
    with pytest.raises(ValueError, match="wrong case"):
        E.checked_patch(path, state, "M", "b" * 64)


def test_tampered_key_and_reissue(issued, tmp_path):
    import shutil
    vault, _ = issued
    with pytest.raises(ValueError, match="new vault"):
        E.issue(vault)
    other = tmp_path / "copy"
    shutil.copytree(vault, other)
    path = other / "first/key.json"
    key = json.loads(path.read_bytes())
    key["donor_seed"] += 1
    path.write_bytes(E.canonical(key))
    with pytest.raises(ValueError, match="receipt"):
        E.open_case(other / "first")
