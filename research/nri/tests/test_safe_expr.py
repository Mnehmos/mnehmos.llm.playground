"""The NRI expression evaluator: equivalence with Python on every stored document's
expressions, and rejection of code-execution attempts.   python -m pytest -q tests/"""
import ast
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri.safe_expr import UnsafeExpression, compile_expr  # noqa: E402

FIELDS = ("binding", "output_fn", "scope", "edit_fn")


def stored_expressions():
    """Every expression string in the repository's NRI documents (results/**/*.json)."""
    out = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in FIELDS and isinstance(v, str):
                    out.add(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for f in (ROOT / "results").rglob("*.json"):
        try:
            walk(json.loads(f.read_text()))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    from nri.reference_nri import reference_doc
    for c in ("none", "typed", "canonical", "shuffled", "typed_fact"):
        walk(reference_doc("x", "0" * 64, c, "unused"))
    return sorted(out)


def _reference(src):
    """Python's own evaluation, used ONLY on the repository's stored, reviewed expressions."""
    return eval(src, {"__builtins__": {"len": len, "min": min, "max": max, "abs": abs,  # noqa: S307
                                       "range": range, "int": int, "list": list, "sum": sum,
                                       "any": any, "all": all, "enumerate": enumerate}}, {})


def _arity(src):
    return len(ast.parse(src, mode="eval").body.args.args)


def _inputs(src, rng):
    arg = ast.parse(src, mode="eval").body.args.args[0].arg
    names = sorted({n.value for n in ast.walk(ast.parse(src, mode="eval"))
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)})
    for _ in range(400):
        if names:   # function over a dict of variable values
            yield {k: rng.randrange(0, 400) for k in names}
        else:       # function over a token list (the tasks use length-6..8 sequences of ints < 26)
            yield [rng.randrange(0, 26) for _ in range(rng.choice((6, 7, 8)))]


def _run(f, x):
    try:
        return ("ok", f(x))
    except Exception as e:  # noqa: BLE001 -- both evaluators must fail the same way
        return ("err", type(e).__name__)


def test_stored_expressions_found():
    assert len(stored_expressions()) >= 40


@pytest.mark.parametrize("src", stored_expressions())
def test_same_answers_as_python_on_stored_expressions(src):
    assert _arity(src) == 1
    ours, ref = compile_expr(src), _reference(src)
    rng = random.Random(hash(src) & 0xFFFF)
    for x in _inputs(src, rng):
        assert _run(ours, x) == _run(ref, x), (src, x)


@pytest.mark.parametrize("src", [
    "().__class__.__mro__[1].__subclasses__()",
    "lambda t: t.__class__",
    "__import__('os').system('true')",
    "lambda t: open('/etc/passwd')",
    "lambda t: eval('1')",
    "lambda t: exec('1')",
    "lambda t: getattr(t, 'x')",
    "lambda t: 2 ** 10000",
    "lambda t: t[0:2]",
    "lambda t: [x async for x in t]",
    "lambda t: {1: 2}",
    "lambda t: (y := 3)",
    "lambda t: f'{t}'",
    "lambda *a: 1",
    "lambda t, k=1: k",
    "lambda t: sum(x for x in t)",
    "lambda t: 1.5",
])
def test_rejects_unsafe_constructs(src):
    with pytest.raises(UnsafeExpression):
        f = compile_expr(src)
        f([1, 2, 3])


def test_resource_bounds():
    with pytest.raises(UnsafeExpression):
        compile_expr("lambda t: range(10 ** 9)")([])
    with pytest.raises(UnsafeExpression):
        compile_expr("lambda t: [0 for i in range(9999) for j in range(9999) if 0]")([])
    with pytest.raises(UnsafeExpression):
        compile_expr("lambda t: 7 * 10 ** 70 * 10 ** 70")([])


def test_budget_resets_between_calls():
    # ~505k steps per call: 10 calls exceed one budget (2M) only if it were never reset
    f = compile_expr("lambda t: len([i for i in range(5000) for j in range(100) if 0])")
    assert all(f([]) == 0 for _ in range(10))


def test_calls_cannot_reach_python_objects():
    with pytest.raises(UnsafeExpression):
        compile_expr("lambda t: t(1)")(print)   # a non-whitelisted callable passed in as data


def stored_documents():
    docs = []
    for f in sorted((ROOT / "results").rglob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and isinstance(d.get("variables"), list) and d["variables"]:
            docs.append((str(f.relative_to(ROOT)), d))
    return docs


@pytest.mark.parametrize("name,doc", stored_documents(), ids=lambda x: x if isinstance(x, str) else "")
def test_stored_documents_give_same_answers(name, doc):
    """Bindings, mechanism outputs, scopes and edit functions of every stored NRI document,
    evaluated on real task inputs: the safe evaluator must agree exactly with Python's."""
    import torch
    from nri import tasks
    exprs = [("binding", v["binding"]) for v in doc["variables"]]
    for m in doc.get("mechanisms", []):
        exprs.append(("fn", m["output_fn"]))
    for c in doc.get("claims", []):
        for k in ("scope", "edit_fn"):
            if isinstance(c.get(k), str):
                exprs.append(("fn", c[k]))
    for cond in tasks.CONDITIONS:
        b = tasks.make_batch(256, cond, torch.Generator().manual_seed(5))
        rows = b.tokens.tolist()
        vals_ref, vals_ours = {}, {}
        for v in doc["variables"]:
            vals_ref[v["id"]] = [_run(_reference(v["binding"]), t) for t in rows]
            vals_ours[v["id"]] = [_run(compile_expr(v["binding"]), t) for t in rows]
            assert vals_ours[v["id"]] == vals_ref[v["id"]], (name, cond, v["id"])
        ids = list(vals_ref)
        dicts = [{k: vals_ref[k][i][1] for k in ids} for i in range(len(rows))
                 if all(vals_ref[k][i][0] == "ok" for k in ids)]
        for kind, src in exprs:
            if kind == "fn":
                assert [_run(compile_expr(src), d) for d in dicts] == [_run(_reference(src), d) for d in dicts], (name, cond, src)
