"""Spec v1.0 / docs/28 pre-experiment checks: identical initialization and data streams; deterministic
restoration; no-op supervision reproduces the control; rejected interventions cannot change the training
state; the supervisor cannot read the sealed split; the executor refuses out-of-scope proposals."""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_correction as C  # noqa: E402
import run_lineage as RL  # noqa: E402
from nri import interventions as IV  # noqa: E402

BASE = dict(rationale="t", mechanism="t", predicted="t", falsification="t", collateral_risk="t")


@pytest.fixture(scope="module")
def formed():
    return C.load_state(ROOT / "runs" / "zoo_correction" / "s601" / "trunk" / "step_1500" / "train_state.pt")


def test_identical_init_and_streams():
    a, b = C.new_state(RL.SEED), C.new_state(RL.SEED)
    assert IV.state_fingerprint(a) == IV.state_fingerprint(b)
    assert C.train(a, 5) == C.train(b, 5)


def test_noop_supervision_reproduces_control(tmp_path, monkeypatch):
    monkeypatch.setattr(RL, "ZOO", tmp_path)
    monkeypatch.setattr(RL, "SEED", 9)
    RL.init()
    rev = {k: "t" for k in RL.REVIEW_FIELDS}
    rev["investigation_level"] = 1
    (tmp_path / "r.json").write_text(json.dumps(rev))
    RL.advance()
    RL.review(tmp_path / "r.json")
    RL.advance()
    a = [json.loads(l) for l in open(tmp_path / "A_control" / "records.jsonl")]
    b = [json.loads(l) for l in open(tmp_path / "B_supervised" / "records.jsonl")]
    assert [r["state_fingerprint"] for r in a] == [r["state_fingerprint"] for r in b]
    with pytest.raises(SystemExit):           # no review for step 200 yet
        RL.advance()


def test_clone_and_lookahead_do_not_mutate(formed):
    fp = IV.state_fingerprint(formed)
    st2 = IV.clone_state(formed)
    assert IV.state_fingerprint(st2) == fp
    IV.lookahead(formed, {}, n=3)
    assert IV.state_fingerprint(formed) == fp


def test_rejected_validation_leaves_state_unchanged(formed):
    fp = IV.state_fingerprint(formed)
    g = torch.Generator().manual_seed(0)
    delta = {"W_in.weight": 0.02 * torch.randn(formed["model"].W_in.weight.shape, generator=g)}
    v = IV.validate(formed, delta, "edit_inv", 0.0, run_lookahead=False)
    assert not v["passed"]
    assert IV.state_fingerprint(formed) == fp


def test_executor_scope(formed):
    m = formed["model"]
    p = dict(BASE, primitive="restricted_gradient_update", scope=["attn2.qkv.weight[head:2]"], target_claim="iia",
             objective={"kl": 1, "iit": 1}, steps=5, lr=1e-3, max_rel_model=0.01)
    delta, _ = IV.execute(p, m)
    d = delta["attn2.qkv.weight"]
    rows = [q * 64 + 32 + i for q in range(3) for i in range(16)]
    mask = torch.zeros(d.shape[0], dtype=torch.bool)
    mask[rows] = True
    assert d[~mask].abs().sum() == 0 and d[mask].abs().sum() > 0
    for bad in (dict(p, scope=["aux.weight"]), dict(p, steps=10_000), dict(p, primitive="exec_python"),
                dict(p, objective={"anything": 1}), dict(p, max_rel_model=0.5)):
        with pytest.raises(IV.ProposalError):
            IV.execute(bad, m)


def test_low_rank_rank_and_scope(formed):
    m = formed["model"]
    p = dict(BASE, primitive="low_rank_adjust", scope=["attn2.qkv.weight[head:0]"], target_claim="interaction",
             target="head_interaction_removal", rank=2, max_rel_model=0.02)
    delta, _ = IV.execute(p, m)
    d = delta["attn2.qkv.weight"]
    assert torch.linalg.matrix_rank(d).item() <= 2
    assert d[:128].abs().sum() == 0 and d[144:].abs().sum() == 0


def test_sealed_guard():
    from nri import atlas as AT
    AT.partition()
    orig = dict(AT._PART)
    try:
        RL.guard_sealed()
        with pytest.raises(PermissionError):
            AT.partition()["FINAL"]
        with pytest.raises(PermissionError):
            AT.partition()["OOD_FINAL"]
        AT.partition()["VAL"]
    finally:
        AT._PART = orig
