"""Neural Representation Interface (NRI) v0.1 -- executor.

An NRI document is JSON (see ../schema/nri-0.1.schema.json and docs/03_nri_spec.md).
Its key property: every claim is *executable*. Variables are bound to inputs by code,
mechanisms state an output function over variables, and each claim carries a
predicate that this module checks against the actual network. Nothing in an NRI
document is accepted because it sounds right.

Claim kinds implemented (v0.1):
  behavior      model(x) == output_fn(vars(x)) on >= threshold of samples
  linear_rep    interchange intervention on the rank-r class-mean subspace of V at a
                site moves the output to output_fn(vars with V from source)  (IIA)
  necessity     mean-ablating that subspace drops behavior accuracy to <= threshold
  edit          a weight program returns (dW_in, db_in); measures success on in-scope
                inputs (vs edit_fn) and preservation out of scope

Binding/function strings are Python lambda expressions written by the interpreter (human or
LLM). They are UNTRUSTED. They are never passed to eval(): nri/safe_expr.py parses them and
interprets a small whitelisted subset of Python (no attribute access, no imports, bounded
loops). See safe_expr.py for the exact subset.

Edit claims reference a weight program: an arbitrary Python file. That IS code execution, so
it is refused unless the caller passes allow_programs=True, which it should do only for
programs it has reviewed, or inside an isolated environment (separate container or VM with no
credentials or network). A refused program makes the claim an error, never evidence.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from nri.safe_expr import UnsafeExpression, compile_expr  # noqa: F401 (re-exported)


class ProgramNotAllowed(PermissionError):
    """An edit program was referenced but the caller did not opt in to running code."""


def _fn(src: str):
    """Compile an NRI expression string safely (no eval)."""
    return compile_expr(src)


def load_program(path, entry="program", *, allow_programs=False):
    """Import a weight-program file and return its entry function. This executes the file, so
    it requires an explicit opt-in (see module docstring)."""
    if not allow_programs:
        raise ProgramNotAllowed(f"refusing to execute program {path}: pass allow_programs=True "
                                "only for reviewed programs or inside an isolated environment")
    spec = importlib.util.spec_from_file_location("nri_prog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry)


class Host:
    """Adapter: what the executor needs from a model + its input distribution."""

    def __init__(self, model, sampler, n_classes_out: int):
        self.model, self.sampler, self.n_out = model, sampler, n_classes_out

    @torch.no_grad()
    def site(self, name, tokens, types):
        if name != "resid_pre_mlp":
            raise ValueError(f"unknown site {name}")
        return self.model.resid_pre_mlp(tokens, types)

    @torch.no_grad()
    def head(self, h, dW=None, db=None):
        m = self.model
        W = m.W_in.weight if dW is None else m.W_in.weight + dW
        b = m.W_in.bias if db is None else m.W_in.bias + db
        return m.head_from_resid(h, W, b).argmax(-1)


def _vars(doc, tokens):
    tl = tokens.tolist()
    out = {}
    for v in doc["variables"]:
        f = _fn(v["binding"])
        out[v["id"]] = torch.tensor([f(t) for t in tl])
    return out


def _outputs(fn_src, vars_):
    f = _fn(fn_src)
    n = next(iter(vars_.values())).shape[0]
    keys = list(vars_)
    return torch.tensor([f({k: int(vars_[k][i]) for k in keys}) for i in range(n)])


def _subspace(h, labels):
    vals = labels.unique()
    mu = torch.stack([h[labels == c].mean(0) for c in vals])
    U, S, Vh = torch.linalg.svd(mu - mu.mean(0), full_matrices=False)
    r = int((S > 1e-6 * S[0]).sum())
    return Vh[:r].T


def run_claim(doc, claim, host: Host, n=4096, seed=0, allow_programs=False):
    g = torch.Generator().manual_seed(seed)
    mech = {m["id"]: m for m in doc.get("mechanisms", [])}
    kind = claim["kind"]
    tok, typ = host.sampler(n, g)
    V = _vars(doc, tok)
    if kind == "behavior":
        y = _outputs(mech[claim["mechanism"]]["output_fn"], V)
        h = host.site("resid_pre_mlp", tok, typ)
        val = (host.head(h) == y).float().mean().item()
    elif kind in ("linear_rep", "necessity"):
        site, var = claim["site"], claim["variable"]
        h = host.site(site, tok, typ)
        B = _subspace(h, V[var])
        if claim.get("rank"):
            B = B[:, : claim["rank"]]
        if kind == "linear_rep":
            tok2, typ2 = host.sampler(n, g)
            V2 = _vars(doc, tok2)
            h2 = host.site(site, tok2, typ2)
            Vcf = dict(V)
            Vcf[var] = V2[var]
            y_cf = _outputs(mech[claim["mechanism"]]["output_fn"], Vcf)
            hp = h + (h2 - h) @ B @ B.T
            val = (host.head(hp) == y_cf).float().mean().item()
        else:
            y = _outputs(mech[claim["mechanism"]]["output_fn"], V)
            hp = h - (h - h.mean(0)) @ B @ B.T
            val = (host.head(hp) == y).float().mean().item()
    elif kind == "edit":
        prog = claim["program"]
        entry = load_program(prog["path"], prog["entry"], allow_programs=allow_programs)
        dW, db = entry(host, doc, **prog.get("args", {}))
        h = host.site("resid_pre_mlp", tok, typ)
        pred = host.head(h, dW, db)
        in_scope = torch.tensor([bool(_fn(claim["scope"])({k: int(V[k][i]) for k in V}))
                                 for i in range(n)])
        y_new = _outputs(claim["edit_fn"], V)
        y_old = _outputs(mech[claim["mechanism"]]["output_fn"], V)
        succ = (pred[in_scope] == y_new[in_scope]).float().mean().item()
        pres = (pred[~in_scope] == y_old[~in_scope]).float().mean().item()
        val = {"success": succ, "preserve": pres,
               "rel_norm": (dW.norm() / host.model.W_in.weight.norm()).item(),
               "rank": int(torch.linalg.matrix_rank(dW).item())}
    else:
        raise ValueError(kind)
    return val


def _check(pred: dict, val):
    if isinstance(val, dict):
        return all(_cmp(val[k], op_thr) for k, op_thr in pred.items())
    return _cmp(val, pred)


def _cmp(x, op_thr):
    op, thr = op_thr
    return x >= thr if op == ">=" else x <= thr


def execute(doc: dict, host: Host, seed=0, allow_programs=False) -> dict:
    """Run every claim; return the doc with status/measured filled in (copy). Edit claims run
    their weight program only with allow_programs=True (see module docstring)."""
    doc = json.loads(json.dumps(doc))
    for c in doc["claims"]:
        try:
            val = run_claim(doc, c, host, seed=seed, allow_programs=allow_programs)
            c["measured"] = val
            c["status"] = "verified" if _check(c["predicate"], val) else "refuted"
        except Exception as e:  # a claim that cannot run is not evidence
            c["measured"] = None
            c["status"] = f"error: {type(e).__name__}: {e}"
    n = len(doc["claims"])
    doc["summary"] = {"claims": n,
                      "verified": sum(c["status"] == "verified" for c in doc["claims"]),
                      "refuted": sum(c["status"] == "refuted" for c in doc["claims"]),
                      "errors": sum(str(c["status"]).startswith("error") for c in doc["claims"])}
    return doc


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
