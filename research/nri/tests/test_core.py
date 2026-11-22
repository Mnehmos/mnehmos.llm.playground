"""Deterministic unit tests:  python -m pytest -q tests/"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri import analysis, tasks  # noqa: E402
from nri.model import TinyTyped  # noqa: E402
from nri.nri_exec import Host, _fn, _vars, execute  # noqa: E402
from nri.reference_nri import reference_doc  # noqa: E402
from nri.train import train, set_determinism  # noqa: E402


def g(seed=0):
    return torch.Generator().manual_seed(seed)


def test_roles_are_contextual_and_correct():
    lat = tasks.sample_latents(2000, g())
    b = tasks.build(lat, "typed")
    for i in range(200):
        t, ty = b.tokens[i].tolist(), b.types[i].tolist()
        j = next(k for k, x in enumerate(t) if x in tasks.OP_TOKENS)
        assert ty[j] == tasks.T_OP and ty[j - 1] == tasks.T_LEFT and ty[j + 1] == tasks.T_RIGHT
        assert t[j - 1] == b.a[i] and t[j + 1] == b.b[i]
        assert sum(x == tasks.T_DISTRACTOR for x in ty[:-1]) == 3
    # the same number token appears in several roles across the batch -> contextual types
    roles_of_7 = {ty for t_row, ty_row in zip(b.tokens.tolist(), b.types.tolist())
                  for t, ty in zip(t_row, ty_row) if t == 7}
    assert {tasks.T_LEFT, tasks.T_RIGHT, tasks.T_DISTRACTOR} <= roles_of_7


def test_conditions_share_latents_and_labels():
    lat = tasks.sample_latents(500, g(1))
    ys = [tasks.build(lat, c).y for c in tasks.CONDITIONS]
    assert all(torch.equal(ys[0], y) for y in ys)
    sh = tasks.build(lat, "shuffled").types[:, :-1]
    ty = tasks.build(lat, "typed").types[:, :-1]
    assert torch.equal(sh.sort(1).values, ty.sort(1).values)  # same multiset per row
    assert not torch.equal(sh, ty)
    can = tasks.build(lat, "canonical").tokens
    assert torch.equal(can[:, 0], lat["a"]) and torch.equal(can[:, 2], lat["b"])
    assert (tasks.build(lat, "none").types == 0).all()


def test_param_counts_match_across_conditions():
    n = {c: TinyTyped(c).n_params() for c in tasks.CONDITIONS}
    assert len(set(n.values())) == 1, n


def test_reference_bindings_recover_latents():
    for c in tasks.CONDITIONS:
        lat = tasks.sample_latents(300, g(2))
        b = tasks.build(lat, c)
        doc = reference_doc("x", "0" * 64, c, "unused")
        V = _vars(doc, b.tokens)
        assert torch.equal(V["V_op"], lat["op"]) and torch.equal(V["V_a"], lat["a"])
        assert torch.equal(V["V_b"], lat["b"])
        out = _fn(doc["mechanisms"][0]["output_fn"])
        assert all(out({"V_op": int(o), "V_a": int(x), "V_b": int(y)}) == int(t)
                   for o, x, y, t in zip(lat["op"], lat["a"], lat["b"], b.y))


def test_rank_one_fold_equals_activation_intervention():
    torch.manual_seed(0)
    m = TinyTyped("typed").eval()
    dW, db, spec = analysis.op_swap_program(m, "typed", 0, 1, n_fit=2048)
    assert torch.linalg.matrix_rank(dW) == 1
    h = torch.randn(64, m.d)
    gh = h @ spec["w"] + spec["c"]
    z_fold = h @ (m.W_in.weight + dW).T + (m.W_in.bias + db)
    z_act = (h + gh[:, None] * spec["delta"]) @ m.W_in.weight.T + m.W_in.bias
    assert torch.allclose(z_fold, z_act, atol=1e-4)
    # g is exactly 1 on mu_x and 0 on the other class means
    gm = spec["mu"] @ spec["w"] + spec["c"]
    assert torch.allclose(gm, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)


def test_training_is_bitwise_deterministic(tmp_path):
    set_determinism()
    r1 = train("typed", 0, tmp_path / "a", steps=40)
    r2 = train("typed", 0, tmp_path / "b", steps=40)
    s1 = torch.load(r1 / "model.pt", weights_only=True)
    s2 = torch.load(r2 / "model.pt", weights_only=True)
    assert all(torch.equal(s1[k], s2[k]) for k in s1)


def test_executor_runs_and_refutes_on_untrained_model():
    torch.manual_seed(0)
    m = TinyTyped("typed").eval()
    host = Host(m, lambda n, gg: (lambda b: (b.tokens, b.types))(tasks.make_batch(n, "typed", gg)),
                tasks.P)
    doc = reference_doc("untrained", "0" * 64, "typed",
                        str(ROOT / "programs" / "op_swap.py"))
    ex = execute(doc, host)
    assert ex["summary"]["errors"] == 0, json.dumps(ex["claims"], indent=1)[:2000]
    st = {c["id"]: c["status"] for c in ex["claims"]}
    assert st["C_behavior"] == "refuted"  # an untrained net does not implement the task
