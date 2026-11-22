"""Safe evaluator for NRI expression strings (bindings, output_fn, scope, edit_fn).

NRI documents come from interpreters (human or LLM) and must be treated as untrusted input.
Python's eval() is not a sandbox, even with restricted builtins: attribute access such as
().__class__.__mro__ reaches arbitrary objects. This module therefore never calls
eval/exec. It parses the string with ast and interprets a small whitelisted subset of Python
expressions, the subset the research actually uses:

  lambda (including nested and immediately-invoked lambdas)
  int / bool / str / None constants, names bound by an enclosing lambda or comprehension
  + - * // %, unary - + not, comparisons (== != < <= > >= in, not in), and / or,
  x if c else y, indexing t[i] / v['key'] (no slices), tuple and list displays,
  list comprehensions with if-filters
  calls to: lambdas defined in the expression, and range len min max abs sum any all int
            list enumerate

Everything else (attribute access, **, slicing, star-args, keyword args, other builtins,
assignment expressions, generators, dict/set displays, f-strings) is rejected at parse
time with UnsafeExpression. Resource bounds: range() spans, list lengths, comprehension steps per call and nesting depth
are capped, and integer results beyond MAX_INT_BITS are rejected.
"""
from __future__ import annotations

import ast
import operator as op

MAX_RANGE = 10_000
MAX_LIST = 10_000
MAX_INT_BITS = 256
MAX_DEPTH = 200
MAX_STEPS = 2_000_000   # comprehension iterations per top-level call
_BUDGET = "\0budget"  # env key that no expression can name


class UnsafeExpression(ValueError):
    """The expression uses a construct outside the whitelisted subset."""


def _rng(*a):
    if not all(isinstance(x, int) for x in a):
        raise TypeError("range() takes ints")
    r = range(*a)
    if len(r) > MAX_RANGE:
        raise UnsafeExpression(f"range() longer than {MAX_RANGE}")
    return r


def _int(x):
    if not isinstance(x, (int, bool)):
        raise TypeError("int() only accepts ints/bools here")
    return int(x)


BUILTINS = {"range": _rng, "len": len, "min": min, "max": max, "abs": abs, "sum": sum,
            "any": any, "all": all, "int": _int, "list": list, "enumerate": enumerate}

BINOPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.FloorDiv: op.floordiv,
          ast.Mod: op.mod}
UNARY = {ast.USub: op.neg, ast.UAdd: op.pos, ast.Not: op.not_}
CMPS = {ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt,
        ast.GtE: op.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}


class _Closure:
    """A lambda from the expression. Only callable with positional args."""

    __slots__ = ("params", "body", "env")

    def __init__(self, params, body, env):
        self.params, self.body, self.env = params, body, env

    def __call__(self, *args):
        if len(args) != len(self.params):
            raise TypeError(f"lambda takes {len(self.params)} args, got {len(args)}")
        return _eval(self.body, {**self.env, **dict(zip(self.params, args))}, 0)


def _check_int(x):
    if isinstance(x, int) and not isinstance(x, bool) and x.bit_length() > MAX_INT_BITS:
        raise UnsafeExpression("integer too large")
    return x


def _validate(tree):
    """Reject any node type outside the whitelist before anything is evaluated."""
    allowed = (ast.Expression, ast.Lambda, ast.arguments, ast.arg, ast.Constant, ast.Name,
               ast.Load, ast.Store, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
               ast.Subscript, ast.Tuple, ast.List, ast.ListComp, ast.comprehension, ast.Call,
               ast.And, ast.Or, *BINOPS, *UNARY, *CMPS)
    for n in ast.walk(tree):
        if not isinstance(n, allowed):
            raise UnsafeExpression(f"construct not allowed: {type(n).__name__}")
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, bool, str, type(None))):
            raise UnsafeExpression(f"constant type not allowed: {type(n.value).__name__}")
        if isinstance(n, ast.Name) and n.id.startswith("__"):
            raise UnsafeExpression(f"name not allowed: {n.id}")
        if isinstance(n, ast.arguments):
            if (n.vararg or n.kwarg or n.kwonlyargs or n.defaults or n.kw_defaults
                    or n.posonlyargs):
                raise UnsafeExpression("only plain positional lambda parameters are allowed")
        if isinstance(n, ast.Call) and n.keywords:
            raise UnsafeExpression("keyword arguments are not allowed")
        if isinstance(n, ast.Call) and any(isinstance(a, ast.Starred) for a in n.args):
            raise UnsafeExpression("star-arguments are not allowed")
        if isinstance(n, ast.comprehension) and n.is_async:
            raise UnsafeExpression("async comprehensions are not allowed")
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
            raise UnsafeExpression("slices are not allowed")
    bound = set(BUILTINS)
    for n in ast.walk(tree):
        if isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound:
            raise UnsafeExpression(f"unknown name: {n.id}")


def _eval(n, env, depth):
    if depth > MAX_DEPTH:
        raise UnsafeExpression("expression nested too deeply")
    d = depth + 1
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.Name):
        if n.id in env:
            return env[n.id]
        if n.id in BUILTINS:
            return BUILTINS[n.id]
        raise NameError(f"name '{n.id}' is not defined")
    if isinstance(n, ast.Lambda):
        return _Closure([a.arg for a in n.args.args], n.body, env)
    if isinstance(n, ast.BinOp):
        return _check_int(BINOPS[type(n.op)](_eval(n.left, env, d), _eval(n.right, env, d)))
    if isinstance(n, ast.UnaryOp):
        return UNARY[type(n.op)](_eval(n.operand, env, d))
    if isinstance(n, ast.BoolOp):
        if isinstance(n.op, ast.And):
            v = True
            for x in n.values:
                v = _eval(x, env, d)
                if not v:
                    return v
            return v
        v = False
        for x in n.values:
            v = _eval(x, env, d)
            if v:
                return v
        return v
    if isinstance(n, ast.Compare):
        left = _eval(n.left, env, d)
        for o, c in zip(n.ops, n.comparators):
            right = _eval(c, env, d)
            if not CMPS[type(o)](left, right):
                return False
            left = right
        return True
    if isinstance(n, ast.IfExp):
        return _eval(n.body, env, d) if _eval(n.test, env, d) else _eval(n.orelse, env, d)
    if isinstance(n, ast.Subscript):
        return _eval(n.value, env, d)[_eval(n.slice, env, d)]
    if isinstance(n, ast.Tuple):
        return tuple(_eval(x, env, d) for x in n.elts)
    if isinstance(n, ast.List):
        return [_eval(x, env, d) for x in n.elts]
    if isinstance(n, ast.ListComp):
        out = []
        _comp(n, n.generators, env, out, d)
        return out
    if isinstance(n, ast.Call):
        f = _eval(n.func, env, d)
        if not (isinstance(f, _Closure) or any(f is b for b in BUILTINS.values())):
            raise UnsafeExpression("only lambdas and whitelisted builtins can be called")
        args = [_eval(a, env, d) for a in n.args]
        r = f(*args)
        if isinstance(r, list) and len(r) > MAX_LIST:
            raise UnsafeExpression("list too long")
        return _check_int(r)
    raise UnsafeExpression(f"construct not allowed: {type(n).__name__}")


def _target_names(t):
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, ast.Tuple) and all(isinstance(e, ast.Name) for e in t.elts):
        return tuple(e.id for e in t.elts)
    raise UnsafeExpression("comprehension targets must be names or tuples of names")


def _bind(names, value, env):
    if isinstance(names, str):
        return {**env, names: value}
    vals = tuple(value)
    if len(vals) != len(names):
        raise ValueError("cannot unpack")
    return {**env, **dict(zip(names, vals))}


def _comp(node, gens, env, out, d):
    if not gens:
        out.append(_eval(node.elt, env, d))
        if len(out) > MAX_LIST:
            raise UnsafeExpression("list too long")
        return
    g, rest = gens[0], gens[1:]
    names = _target_names(g.target)
    budget = env[_BUDGET]
    for item in _eval(g.iter, env, d):
        budget[0] -= 1
        if budget[0] < 0:
            raise UnsafeExpression(f"more than {MAX_STEPS} comprehension steps")
        e2 = _bind(names, item, env)
        if all(_eval(c, e2, d) for c in g.ifs):
            _comp(node, rest, e2, out, d)


def compile_expr(src: str):
    """Parse and validate `src`; return a callable equivalent to the Python expression."""
    if not isinstance(src, str) or len(src) > 20_000:
        raise UnsafeExpression("expression must be a string of at most 20,000 characters")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpression(f"syntax error: {e.msg}") from None
    _validate(tree)
    budget = [MAX_STEPS]
    value = _eval(tree.body, {_BUDGET: budget}, 0)
    if not isinstance(value, _Closure):
        return value

    def call(*args):
        budget[0] = MAX_STEPS  # fresh budget for every top-level call
        return value(*args)

    return call
