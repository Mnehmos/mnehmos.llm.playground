"""Toolsmith runner: static check, sandboxed execution, validation and provenance registry (Issue #25 §1).

A tool lives in toolsmith/instruments/<tool_id>/ with
  tool.py        def controls(ctx) -> {"cases": [{"name", "expected", "observed", "pass"}...]}
                 def run(ctx) -> JSON-serialisable result
  manifest.json  tool_id, purpose, hypothesis, input_schema, output_schema, read_scope, created_at_checkpoint,
                 kind (observational | causal-readonly), author
  test_tool.py   optional pytest tests of pure helper functions

    python -m toolsmith.runner validate <tool_id>               # static check + controls in the sandbox
    python -m toolsmith.runner run <tool_id> <ckpt...> [--args JSON] [--retrospective]
Every validation and run is appended to results/sem/tool_registry.jsonl with source hash, git commit,
inputs, input-checkpoint hashes before/after (nonmutation), runtime and result reference.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "toolsmith" / "instruments"
REGISTRY = ROOT / "results" / "sem" / "tool_registry.jsonl"
EVIDENCE = ROOT / "results" / "sem" / "evidence"
ALLOWED_IMPORTS = {"math", "json", "collections", "itertools", "statistics", "random", "functools", "dataclasses",
                   "torch", "torch.nn.functional", "numpy"}
FORBIDDEN_NAMES = {"exec", "eval", "compile", "__import__", "open", "globals", "locals", "vars", "getattr", "setattr",
                   "delattr", "breakpoint", "input", "memoryview"}
LIMITS = dict(cpu_s=900, address_space=6 * 1024 ** 3, nproc=256)
REQUIRED_MANIFEST = ("tool_id", "purpose", "hypothesis", "input_schema", "output_schema", "read_scope", "kind",
                     "created_at_checkpoint", "author")


class ToolRejected(Exception):
    pass


def static_check(src: str):
    """Reject tool code outside the policy (before it is ever executed)."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    raise ToolRejected(f"import {a.name} not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_IMPORTS:
                raise ToolRejected(f"from {node.module} import ... not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ToolRejected(f"name {node.id} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ToolRejected(f"private/dunder attribute {node.attr} not allowed")
        elif isinstance(node, ast.Attribute) and ("lock" in node.attr.lower() or "sealed" in node.attr.lower()):
            raise ToolRejected(f"attribute {node.attr} not allowed (sealed-area guard)")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and ("sealed" in node.value.lower() or "SEALED" in node.value):
            raise ToolRejected("string constants mentioning the sealed area are not allowed")
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if not {"controls", "run"} <= names:
        raise ToolRejected("a tool must define controls(ctx) and run(ctx)")


def _sha_tree(p: Path):
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(p)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["cpu_s"], LIMITS["cpu_s"]))
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS["address_space"], LIMITS["address_space"]))
    os.setsid()


def execute(tool_id, checkpoints: dict, mode="run", args=None, vocab_path=None):
    """Run a tool in the sandbox on COPIES of the given checkpoints {name: dir}. Returns (result, record)."""
    tdir = TOOLS / tool_id
    src = (tdir / "tool.py").read_text()
    static_check(src)
    man = json.loads((tdir / "manifest.json").read_text())
    missing = [k for k in REQUIRED_MANIFEST if k not in man]
    if missing:
        raise ToolRejected(f"manifest missing {missing}")
    before = {n: _sha_tree(Path(d)) for n, d in checkpoints.items()}
    with tempfile.TemporaryDirectory(prefix="toolsmith_") as tmp:
        tmp = Path(tmp)
        ck = tmp / "ckpt"
        out = tmp / "out"
        ck.mkdir()
        out.mkdir()
        for n, d in checkpoints.items():
            (ck / n).mkdir()
            shutil.copy(Path(d) / "model.safetensors", ck / n / "model.safetensors")
        shutil.copy(vocab_path or (ROOT / "runs" / "zoo_sem" / "eval" / "vocab.json"), ck / "vocab.json")
        af = tmp / "args.json"
        af.write_text(json.dumps(args or {}))
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "HTTP_PROXY": "http://0.0.0.0:9", "HTTPS_PROXY": "http://0.0.0.0:9",
               "NO_PROXY": "", "OMP_NUM_THREADS": "2", "LANG": "C.UTF-8"}
        t0 = time.time()
        p = subprocess.run([sys.executable, "-I", str(ROOT / "toolsmith" / "sandbox_child.py"), str(ROOT), str(tdir), str(ck), str(out),
                            mode, str(af)], cwd=tmp, env=env, capture_output=True, text=True, preexec_fn=_limits,
                           timeout=LIMITS["cpu_s"] + 60)
        wall = round(time.time() - t0, 1)
        res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else None
        err = p.stderr[-3000:] if p.returncode != 0 else ""
    after = {n: _sha_tree(Path(d)) for n, d in checkpoints.items()}
    rec = dict(tool_id=tool_id, mode=mode, source_sha256=hashlib.sha256(src.encode()).hexdigest(), git_commit=_git_commit(),
               manifest=man, checkpoints={n: str(d) for n, d in checkpoints.items()}, args=args or {},
               nonmutation=dict(before=before, after=after, ok=before == after), returncode=p.returncode, wall_s=wall,
               error=err, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return res, rec


def register(rec, result, retrospective=False, evidence_name=None):
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    if evidence_name:
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        (EVIDENCE / evidence_name).write_text(json.dumps(dict(record=rec, result=result), indent=1, default=str))
    rec = dict(rec, retrospective=retrospective, evidence=evidence_name)
    with open(REGISTRY, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def validate(tool_id, checkpoints):
    res, rec = execute(tool_id, checkpoints, mode="controls")
    ok = rec["returncode"] == 0 and res is not None and all(c["pass"] for c in res["cases"]) and rec["nonmutation"]["ok"]
    rec["validated"] = ok
    register(rec, res, evidence_name=f"{tool_id}_controls.json")
    return ok, res, rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("validate", "run"))
    ap.add_argument("tool_id")
    ap.add_argument("ckpts", nargs="+", help="name=dir pairs")
    ap.add_argument("--args", default="{}")
    ap.add_argument("--retrospective", action="store_true")
    ap.add_argument("--evidence", default=None)
    a = ap.parse_args()
    cks = dict(c.split("=", 1) for c in a.ckpts)
    if a.mode == "validate":
        ok, res, rec = validate(a.tool_id, cks)
        print(json.dumps(dict(validated=ok, cases=res and res["cases"], error=rec["error"]), indent=1))
        return
    reg = [json.loads(l) for l in open(REGISTRY)] if REGISTRY.exists() else []
    src_sha = hashlib.sha256((TOOLS / a.tool_id / "tool.py").read_text().encode()).hexdigest()
    if not any(r["tool_id"] == a.tool_id and r.get("validated") and r["source_sha256"] == src_sha for r in reg):
        raise SystemExit("tool has no passing validation for its current source; run validate first")
    res, rec = execute(a.tool_id, cks, args=json.loads(a.args))
    register(rec, res, retrospective=a.retrospective, evidence_name=a.evidence)
    print(json.dumps(dict(result=res, nonmutation=rec["nonmutation"]["ok"], error=rec["error"]), indent=1, default=str)[:8000])


if __name__ == "__main__":
    main()
