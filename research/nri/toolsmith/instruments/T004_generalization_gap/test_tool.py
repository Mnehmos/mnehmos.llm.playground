import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("t004", Path(__file__).parent / "tool.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def test_histogram_top4():
    h = T.answer_histogram(torch.tensor([0, 0, 1, 2, 3, 4, 0]), [str(i) for i in range(5)])
    assert list(h)[0] == "0" and h["0"] == 3 and len(h) == 4
