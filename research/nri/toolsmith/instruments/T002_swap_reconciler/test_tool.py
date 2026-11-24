import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("t002", Path(__file__).parent / "tool.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def test_stratify_counts():
    st = T.stratify(torch.tensor([True, False, True, True]), [dict(k=0), dict(k=0), dict(k=1), dict(k=1)])
    assert st["k"] == {"0": [0.5, 2], "1": [1.0, 2]}


def test_swap_success_identity_head():
    head = lambda h: torch.nn.functional.one_hot(h[:, 0].long() % 13, 13).float()
    hb = torch.zeros(4, 8)
    hs = torch.arange(4).float().unsqueeze(1).repeat(1, 8)
    assert T.swap_success(head, hb, hs, torch.arange(4), block=(0, 1)).all()
    assert not T.swap_success(head, hb, hs, torch.arange(4), block=(1, 2))[1:].any()
