"""Issue #22 two-trajectory atlas / continuous upstream-edit loop.  python -m pytest -q tests/test_atlas.py

Checks: reproducible initialization; identical batch streams; instrumented training is bitwise the
original trainer; checkpoint/resume exactness; the edit touches only attn2.o rows 0-15 and bias and
respects its norm budget; a rejected/skipped decision leaves model AND optimizer state bitwise
unchanged; a no-op supervised trajectory equals the control; the data partition is disjoint.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_atlas as RA  # noqa: E402
import run_correction as C  # noqa: E402
import run_swap as S  # noqa: E402
from nri import atlas as AT  # noqa: E402


def _w(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _opt_same(o1, o2):
    s1, s2 = o1.state_dict(), o2.state_dict()
    if s1["param_groups"] != s2["param_groups"]:
        return False
    for k in s1["state"]:
        for kk, v in s1["state"][k].items():
            if not torch.equal(v, s2["state"][k][kk]):
                return False
    return True


def test_reproducible_initialization():
    a, b = C.new_state(RA.SEED), C.new_state(RA.SEED)
    assert _same(_w(a["model"]), _w(b["model"]))
    assert 69_000 < a["model"].n_params() < 71_000


def test_identical_batch_streams_between_trajectories(monkeypatch):
    monkeypatch.setattr(S, "STEPS", 30)
    a, b = C.new_state(11), C.new_state(11)
    ha = C.train(a, 30)
    # perturb b's weights: the batch stream must not depend on the model
    with torch.no_grad():
        b["model"].attn2.o.weight.add_(0.1)
    hb = C.train(b, 30)
    assert ha == hb


def test_instrumented_trajectory_is_bitwise_original_trainer(tmp_path, monkeypatch):
    """Observation, shadow policy, gradients and checkpointing never change training."""
    monkeypatch.setattr(S, "STEPS", 40)
    run = S.train_one(("mass_only", 7, "warmup", str(tmp_path / "ref")))
    ref = torch.load(Path(run) / "model.pt", weights_only=True)
    RA.run_trajectory("control", zoo=tmp_path / "zoo", ckpts=(0, 20, 40), seed=7, comparators=False)
    got = torch.load(tmp_path / "zoo" / "control" / "step_0040" / "model.pt", weights_only=True)
    assert _same(ref, got)


def test_noop_supervised_equals_control(tmp_path, monkeypatch):
    """Before any representation exists every proposal is skipped, so the supervised trajectory must
    be bitwise the control (same hashes, same batches)."""
    monkeypatch.setattr(S, "STEPS", 40)
    for n in RA.TRAJ:
        RA.run_trajectory(n, zoo=tmp_path, ckpts=(0, 20, 40), seed=7, comparators=False)
    ca = [json.loads(l) for l in open(tmp_path / "control" / "cards.jsonl")]
    cb = [json.loads(l) for l in open(tmp_path / "supervised" / "cards.jsonl")]
    assert [c["state_hash"] for c in ca] == [c["state_hash"] for c in cb]
    assert [c["batch_digest"] for c in ca] == [c["batch_digest"] for c in cb]
    assert all(not c["decision"].get("applied") for c in cb)


def test_checkpoint_restore_is_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "STEPS", 40)
    a = C.new_state(8)
    C.train(a, 40)
    b = C.new_state(8)
    C.train(b, 20)
    C.save_state(b, tmp_path / "s.pt")
    torch.manual_seed(999)
    c = C.load_state(tmp_path / "s.pt")
    assert _same(_w(b["model"]), _w(c["model"])) and _opt_same(b["opt"], c["opt"])
    C.train(c, 40)
    assert _same(_w(a["model"]), _w(c["model"])) and _opt_same(a["opt"], c["opt"])


@pytest.fixture(scope="module")
def formed():
    """A network with a formed mass representation (an existing doc-23 checkpoint, read only)."""
    st = C.load_state(ROOT / "runs" / "zoo_correction" / "s601" / "trunk" / "step_1500" / "train_state.pt")
    return st


def test_edit_scope_and_budget(formed):
    m = formed["model"]
    before = _w(m)
    prop = AT.make_proposal(m, AT.RHO)
    rel = torch.cat([prop.dW.flatten(), prop.db]).norm().item() / AT.block_row_norm(m)
    assert rel <= AT.RHO + 1e-6
    AT.apply_delta(m, prop.dW, prop.db)
    after = _w(m)
    for k in before:
        d = after[k] - before[k]
        if k == "attn2.o.weight":
            assert torch.equal(d[16:], torch.zeros_like(d[16:])) and d[:16].abs().sum() > 0
        elif k == "attn2.o.bias":
            assert torch.equal(d[16:], torch.zeros_like(d[16:]))
        else:
            assert torch.equal(d, torch.zeros_like(d)), k
    AT.apply_delta(m, prop.dW, prop.db, -1.0)
    assert all(torch.allclose(before[k], v, atol=1e-6) for k, v in _w(m).items())


def test_decision_never_mutates_training_state(formed):
    """decide() works on a copy: model and optimizer are bitwise unchanged whatever the outcome
    (an accepted delta is only installed by the trainer)."""
    m, opt = formed["model"], formed["opt"]
    w0 = _w(m)
    o0 = {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in opt.state_dict()["state"].items()}
    rec, delta = AT.decide(m, comparators=True)
    assert rec["status"] in ("accepted", "rejected", "skipped")
    assert _same(w0, _w(m))
    o1 = opt.state_dict()["state"]
    assert all(torch.equal(o0[k][kk], o1[k][kk]) for k in o0 for kk in o0[k])
    if delta is not None:
        assert delta[0].shape == (16, 64) and delta[1].shape == (16,)


def test_budget_exhaustion_skips(formed):
    rec, delta = AT.decide(formed["model"], cum_used=AT.RHO_CUM, cum_cap=AT.RHO_CUM, comparators=False)
    assert delta is None and rec["status"] == "skipped"


def test_partition_disjoint_and_complete():
    d = AT.partition()
    sets = {k: set(zip(*[d[k][c].tolist() for c in ("m", "mu", "a", "au", "fu", "order")]))
            for k in ("FIT", "VAL", "OBS", "FINAL", "OOD_DEV", "OOD_FINAL")}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), (a, b)
    assert len(sets["FIT"] | sets["VAL"] | sets["OBS"] | sets["FINAL"]) == 7350
    assert all(v in (11, 12) for q in sets["OOD_FINAL"] for v in AT._ood_vals(q))
    assert all(v in (9, 10) for q in sets["OOD_DEV"] for v in AT._ood_vals(q))
