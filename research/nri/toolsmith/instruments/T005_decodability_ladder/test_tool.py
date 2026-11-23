import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("t005", Path(__file__).parent / "tool.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def test_labels_canonical():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from sem import lang as L
    ex = dict(ast=dict(task="convert", dim="mass", v=3, u=1, t=2))
    lab = T.labels(L, [ex])
    assert int(lab["canon"][0]) == 3 * L.MASS_S[1] % 13 and int(lab["answer"][0]) == int(L.answer(ex["ast"]))
