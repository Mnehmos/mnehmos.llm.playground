"""LLM-in-the-loop gate (docs/26): never mutates the training state; a random edit cannot pass."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_correction as C  # noqa: E402
import run_loop as L  # noqa: E402


def test_gate_does_not_mutate_and_random_edit_fails():
    st = C.load_state(ROOT / "runs" / "zoo_correction" / "s601" / "trunk" / "step_1500" / "train_state.pt")
    m, opt = st["model"], st["opt"]
    w0 = {k: v.clone() for k, v in m.state_dict().items()}
    o0 = {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in opt.state_dict()["state"].items()}
    g = torch.Generator().manual_seed(0)
    delta = {"W_in.weight": 0.05 * torch.randn(m.W_in.weight.shape, generator=g) * m.W_in.weight.abs().mean()}
    v = L.run_gate(m, delta, "edit_inv", 0.0)
    assert not v["passed"]
    assert all(torch.equal(w0[k], x) for k, x in m.state_dict().items())
    o1 = opt.state_dict()["state"]
    assert all(torch.equal(o0[k][kk], o1[k][kk]) for k in o0 for kk in o0[k])


def test_budget_check():
    st = C.load_state(ROOT / "runs" / "zoo_correction" / "s601" / "trunk" / "step_1500" / "train_state.pt")
    m = st["model"]
    delta = {"attn2.o.weight": m.attn2.o.weight.detach().clone()}   # 100% change of one tensor
    v = L.run_gate(m, delta, "edit_inv", 0.0)
    assert not v["S7_budget"] and not v["passed"]
