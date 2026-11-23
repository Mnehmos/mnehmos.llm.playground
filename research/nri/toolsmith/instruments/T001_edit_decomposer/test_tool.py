import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("t001", Path(__file__).parent / "tool.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def test_controls_discriminate_hypotheses():
    res = t.controls(None)
    assert all(c["pass"] for c in res["cases"]), res


def test_edit_functions():
    import torch
    y = torch.arange(13)
    inv = t.edit_fn("inv")(y)
    assert all((int(a) * int(b)) % 13 == 1 for a, b in zip(y[1:], inv[1:])) and int(inv[0]) == 0
    assert t.edit_fn("add1")(y).tolist() == [(i + 1) % 13 for i in range(13)]
