"""Checkpoint-correction pilot (Issue #20): trainer fidelity and edit scope.  python -m pytest -q tests/"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_correction as C  # noqa: E402
import run_perm as RP  # noqa: E402
import run_swap as S  # noqa: E402


def _weights(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def test_wrapper_reproduces_run_swap_train_one(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "STEPS", 40)
    run = S.train_one(("mass_only", 7, "warmup", str(tmp_path / "ref")))
    ref = torch.load(Path(run) / "model.pt", weights_only=True)
    st = C.new_state(7)
    C.train(st, 40)
    got = st["model"].state_dict()
    assert all(torch.equal(ref[k], got[k]) for k in ref)


def test_split_resume_is_bitwise_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "STEPS", 40)
    a = C.new_state(8)
    ha = C.train(a, 40)
    b = C.new_state(8)
    hb1 = C.train(b, 20)
    C.save_state(b, tmp_path / "s.pt")
    torch.manual_seed(12345)                     # disturb global RNG: must not matter
    c = C.load_state(tmp_path / "s.pt")
    hb2 = C.train(c, 40)
    assert ha == hb1 + hb2                       # identical batches
    wa, wc = _weights(a["model"]), _weights(c["model"])
    assert all(torch.equal(wa[k], wc[k]) for k in wa)


def test_fold_changes_only_block_columns_of_W_in_and_b_in():
    st = C.new_state(9)
    m = st["model"]
    cal = RP.pool_lats(2048, 1, RP.reserved_set())
    before = _weights(m)
    Wi, bi = C.canon_fold(m, cal, 1.0)
    C.apply_fold(m, Wi, bi)
    after = _weights(m)
    for k in before:
        if k == "W_in.weight":
            d = after[k] - before[k]
            assert torch.equal(d[:, 16:], torch.zeros_like(d[:, 16:]))
            assert d[:, :16].abs().sum() > 0
        elif k != "W_in.bias":
            assert torch.equal(before[k], after[k]), k
