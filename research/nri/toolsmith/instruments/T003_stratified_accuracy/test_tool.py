import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("t003", Path(__file__).parent / "tool.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def test_stratify():
    st = T.stratify([1, 0, 1], [dict(k="a"), dict(k="a"), dict(k="b")])
    assert st["k"] == {"a": [0.5, 2], "b": [1.0, 1]}
