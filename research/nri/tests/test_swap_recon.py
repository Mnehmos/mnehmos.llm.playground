"""Mechanics of the Phase III reconstruction trainer on a shortened schedule (300 steps).
Full-length qualification (exact replay of the historical networks) is in doc 41R, not here."""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_swap as RS  # noqa: E402
from sci import swap_recon as SR  # noqa: E402


def test_resume_and_decomposition(tmp_path, monkeypatch):
    monkeypatch.setattr(RS, "STEPS", 300)
    monkeypatch.setattr(RS, "WARMUP_STEPS", 100)
    full = SR.train("mass_only", 7, tmp_path / "a")
    res = SR.train("mass_only", 7, tmp_path / "b", resume_from=100, src_zoo=tmp_path / "a")
    A = torch.load(Path(full) / "model.pt", weights_only=True)
    B = torch.load(Path(res) / "model.pt", weights_only=True)
    assert all(torch.equal(A[k], B[k]) for k in A)                  # resume is bitwise exact
    # log lines after the resume point are identical
    la = (Path(full) / "log.jsonl").read_text().splitlines()
    lb = (Path(res) / "log.jsonl").read_text().splitlines()
    assert la[-len(lb):] == lb
    # decay + adaptive parts sum exactly (to float tolerance) to the checkpoint-to-checkpoint change
    c = Path(full) / "ckpt"
    w0, w1 = load_file(str(c / "step_00100/model.safetensors")), load_file(str(c / "step_00200/model.safetensors"))
    u = load_file(str(c / "step_00200/updates.safetensors"))
    for k in w0:
        if f"decay.{k}" in u:
            assert torch.allclose(u[f"decay.{k}"] + u[f"adaptive.{k}"], w1[k] - w0[k], atol=1e-5)
