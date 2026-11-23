"""Sandbox child process for supervisor-authored instruments (Issue #25 §1; docs/30 §30.4).

Started by toolsmith.runner with `python -I` (isolated: no user site, no PYTHON* env, no cwd on the
path), a minimal environment (no secrets, proxies blanked), resource limits and a temporary working
directory. Before the tool's code is imported:
  * an audit hook blocks sockets, subprocesses/exec, ctypes, pickle loading, file WRITES outside the
    output directory and file READS of the sealed area or of anything outside the allowlisted roots;
  * the sealed lock of sem.data is ON;
  * the tool source has already passed the runner's static check (import allowlist, no exec/eval/
    open/__import__/dunder access).
The tool receives a Ctx with model copies, the guarded data API and an output writer.

    python -I sandbox_child.py <repo_nri_root> <tool_dir> <ckpt_root_copy> <out_dir> <mode> [args.json]
    mode: controls | run
"""
import json
import os
import sys

NRI, TOOL_DIR, CKPT, OUT, MODE = sys.argv[1:6]
ARGS = json.loads(open(sys.argv[6]).read()) if len(sys.argv) > 6 else {}
NRI, TOOL_DIR, CKPT, OUT = (os.path.realpath(p) for p in (NRI, TOOL_DIR, CKPT, OUT))
READ_ROOTS = [NRI, TOOL_DIR, CKPT, OUT, os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix), "/usr/lib", "/usr/local/lib",
              "/proc/self", "/dev/null", "/dev/urandom", "/etc/ld.so.cache", "/sys/devices/system/cpu", "/tmp"]
DENY_READ = [os.path.join(NRI, "results", "sem", "final_sealed.json"), os.path.join(NRI, "runs", "zoo_sem", "sealed")]
BLOCKED = ("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "ctypes.",
           "pickle.find_class", "os.kill", "shutil.rmtree", "os.remove", "os.unlink", "os.rename", "webbrowser")
_ready = [False]


def _hook(event, args):
    if not _ready[0]:
        return
    if event.startswith(BLOCKED):
        raise PermissionError(f"sandbox: {event} is not permitted")
    if event == "open":
        path, mode = args[0], args[1] if len(args) > 1 else "r"
        if isinstance(path, int):
            return
        p = os.path.realpath(os.fsdecode(path))
        writing = isinstance(mode, str) and any(c in mode for c in "wax+")
        if isinstance(args[2] if len(args) > 2 else 0, int) and (args[2] if len(args) > 2 else 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            writing = True
        if writing and not p.startswith(OUT + os.sep) and not p.startswith("/tmp/"):
            raise PermissionError(f"sandbox: write outside the output directory: {p}")
        if any(p.startswith(d) for d in DENY_READ):
            raise PermissionError(f"sandbox: sealed area is not readable: {p}")
        if not any(p == r or p.startswith(r.rstrip("/") + "/") for r in READ_ROOTS):
            raise PermissionError(f"sandbox: read outside the allowlisted roots: {p}")


sys.addaudithook(_hook)
sys.path.insert(0, NRI)

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from sem import data as D  # noqa: E402
from sem import lang as L  # noqa: E402
from sem import model as SM  # noqa: E402
from sem import observe as O  # noqa: E402

D.SEALED_LOCK[0] = True
# sealed phrasing families, contexts and evaluation specs do not exist inside the sandbox
L.FAMILIES[:] = [f for f in L.FAMILIES if f.split != "sealed"]
for _k in [k for k, f in L.FAMILY.items() if f.split == "sealed"]:
    del L.FAMILY[_k]
L.CONTEXTS[:] = [c for c in L.CONTEXTS if c["split"] != "sealed"]
for _k in [k for k, c in L.CONTEXT.items() if c["split"] == "sealed"]:
    del L.CONTEXT[_k]
for _k in [k for k in D.EVAL_SPECS if not k.startswith("dev/")]:
    del D.EVAL_SPECS[_k]
torch.set_num_threads(2)


class Ctx:
    """What a tool may use. Models are loaded from COPIES of checkpoints; nothing here writes to the
    live training state."""

    def __init__(self):
        self.vocab = D.Vocab.from_json(json.loads(open(os.path.join(CKPT, "vocab.json")).read()))
        self.data, self.lang, self.observe, self.torch = D, L, O, torch
        self.args = ARGS
        self.checkpoints = sorted(d for d in os.listdir(CKPT) if os.path.isdir(os.path.join(CKPT, d)))

    def load_model(self, name):
        m = SM.SemStudent(len(self.vocab.itos))
        m.load_state_dict(load_file(os.path.join(CKPT, name, "model.safetensors")))
        m.eval()
        return m

    def dev_sets(self):
        """The frozen development sets (runs/zoo_sem/eval/dev); sealed sets do not exist on disk."""
        from pathlib import Path
        return O.dev_sets(Path(NRI) / "runs" / "zoo_sem" / "eval")

    def encode(self, examples):
        return D.encode_batch(examples, self.vocab)

    def out(self, name, obj):
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(obj, f, indent=1, default=str)


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tool", os.path.join(TOOL_DIR, "tool.py"))
    mod = importlib.util.module_from_spec(spec)
    _ready[0] = True                                       # from here on the audit hook is enforced
    spec.loader.exec_module(mod)
    ctx = Ctx()
    res = mod.controls(ctx) if MODE == "controls" else mod.run(ctx)
    ctx.out("result.json", res)


if __name__ == "__main__":
    main()
