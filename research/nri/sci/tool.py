"""Run a supervisor-authored JSON evidence instrument in an isolated process."""
import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time

from sci.loop import verify, sha, write


def seccomp(path):
    lib = ctypes.CDLL("libseccomp.so.2")
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7fff0000)
    if not ctx:
        raise RuntimeError("could not create syscall filter")
    try:
        # Native network creation and alternate kernel submission paths are denied.
        for name in (b"socket", b"socketpair", b"io_uring_setup", b"ptrace", b"bpf"):
            syscall = lib.seccomp_syscall_resolve_name(name)
            if syscall < 0 or lib.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, syscall, 0):
                raise RuntimeError(f"could not block {name}")
        with path.open("wb") as f:
            if lib.seccomp_export_bpf(ctx, f.fileno()):
                raise RuntimeError("could not export syscall filter")
    finally:
        lib.seccomp_release(ctx)


def execute(checkpoint, directory):
    card = verify(checkpoint)
    manifest_hash = sha(checkpoint/"manifest.json")
    config = json.loads((checkpoint.parent/"run.json").read_text())
    if config["arm"] != "toolsmith":
        raise ValueError("dynamic instrument permission is limited to the toolsmith arm")
    manifest = json.loads((directory/"manifest.json").read_text())
    if not all(manifest.get(k) for k in ("name","hypothesis","author","positive_control","negative_control")):
        raise ValueError("missing instrument manifest fields")
    tool_source = (directory/"tool.py").read_bytes()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch)
        (staged/"instrument.py").write_bytes(tool_source)
        write(staged/"evidence.json", card)
        for name in ("model.safetensors","traces.safetensors"):
            (staged/name).write_bytes((checkpoint/name).read_bytes())
        seccomp(staged/"network.bpf")
        launcher = """import importlib.util, json
spec = importlib.util.spec_from_file_location('instrument','/input/instrument.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open('/input/evidence.json') as f:
    evidence = json.load(f)
print(json.dumps(dict(controls=mod.controls(), result=mod.run(evidence)), allow_nan=False))
"""
        (staged/"launch.py").write_text(launcher)
        def limits():
            resource.setrlimit(resource.RLIMIT_CPU,(15,15))
            resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
            resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024**2,2*1024**2))
        with (staged/"network.bpf").open("rb") as filt, (staged/"stdout").open("w+") as output:
            subprocess.run(["bwrap","--unshare-user","--unshare-pid","--die-with-parent",
                            "--clearenv","--ro-bind","/usr","/usr","--symlink","usr/lib","/lib",
                            "--symlink","usr/lib64","/lib64","--tmpfs","/tmp","--dir","/input",
                            *[arg for name in ("instrument.py","launch.py","evidence.json","model.safetensors",
                                               "traces.safetensors") for arg in
                              ("--ro-bind",str(staged/name),f"/input/{name}")],
                            "--seccomp",str(filt.fileno()),"--chdir","/tmp","--",
                            "/usr/bin/python3","-I","-B","/input/launch.py"],
                           pass_fds=(filt.fileno(),),preexec_fn=limits,timeout=22,check=True,
                           stdout=output,stdin=subprocess.DEVNULL)
            output.seek(0)
            outcome = json.load(output)
    if outcome["controls"] != {"positive": True,"negative": True}:
        raise ValueError("instrument controls failed")
    verify(checkpoint)
    if sha(checkpoint/"manifest.json") != manifest_hash:
        raise ValueError("instrument changed checkpoint")
    dest = checkpoint/"tools"
    dest.mkdir(exist_ok=True)
    digest = hashlib.sha256(tool_source).hexdigest()
    path = dest/f"{digest}.json"
    write(path, dict(manifest=manifest, checkpoint_sha256=manifest_hash,
                     source=tool_source.decode(), source_sha256=digest, outcome=outcome,
                     runtime_s=time.perf_counter()-started,
                     isolation="read-only mounts, user/PID namespaces, seccomp, resource limits"))
    print(path)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("instrument",type=Path)
    args=parser.parse_args()
    execute(args.checkpoint.resolve(),args.instrument.resolve())
