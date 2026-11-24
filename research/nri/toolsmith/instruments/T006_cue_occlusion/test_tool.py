import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("t006", Path(__file__).parent / "tool.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def test_frame_words_after_last_period():
    fw = T.frame_words([["a", "b", ".", "is", "x", "?"]], {"a": 0, "is": 1, "x": 2, "?": 3}, {"?"})
    assert set(fw) == {"is", "x"} and fw["is"] == [(0, 3)]
