"""Birth of a Concept — registered 4 x 2 factorial (doc 37): populations A/B/C/D x weight decay {0.1, 1.0} x 12
matched seeds = 96 networks, 6,000 steps each.

* Training is never intervened on. Every 100 steps each network saves a COMPLETE checkpoint (weights, auxiliary
  head, AdamW state, torch + python RNG) plus the exact decomposition of the 100 AdamW updates into the decoupled
  weight-decay part (-lr * wd * theta_before) and the adaptive-gradient part, summed per parameter
  (updates.safetensors) and summarised per parameter group (updates.json); every step appends to interval.jsonl.
* Networks advance round-robin in 100-step blocks inside long-lived workers, so each 100-step cohort completes
  together and can be reviewed while training continues; any worker can be restarted and resumes bit-exactly.
* Audits (sci/concept_audit.py) run in separate processes on saved checkpoints only (never on live state).
* The reserved test combinations are evaluated once, at step 6,000, by `final`.

Live checkpoints live OUTSIDE Git under $NRI_FACTORIAL (default /home/user/nri_factorial); Git keeps manifests,
audit summaries, journals and code.

    python -m sci.factorial train <worker_index> <n_workers>
    python -m sci.factorial audit [--loop]
    python -m sci.factorial final
    python -m sci.factorial archive <run>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import concept as C  # noqa: E402
from sem import data as D  # noqa: E402
from sem import model as SM  # noqa: E402

LIVE = Path(os.environ.get("NRI_FACTORIAL", "/home/user/nri_factorial"))
ARCHIVE = Path(os.environ.get("NRI_ARCHIVE", "/home/user/nri_archives"))
REPO_RUNS = ROOT / "runs" / "concept_factorial"
POPS = ("A", "B", "C", "D")
WDS = (0.1, 1.0)
SEEDS = tuple(range(2001, 2013))                 # 12 registered seeds; pilot used 101-103
STEPS, BLOCK = C.STEPS, C.AUDIT_EVERY


def run_name(pop, wd, seed):
    return f"wd{wd}_{pop}_s{seed}"


def all_runs():
    return [(p, w, s) for w in WDS for p in POPS for s in SEEDS]


class Net:
    """One network's complete training state, resumable from its latest checkpoint."""

    def __init__(self, pop, wd, seed, vocab, live=LIVE):
        self.pop, self.wd, self.seed, self.vocab = pop, wd, seed, vocab
        self.name = run_name(pop, wd, seed)
        self.dir = live / self.name
        self.ck = self.dir / "ckpt"
        self.spec = C.population_spec(pop, C.make_split())
        torch.manual_seed(seed)                                  # matched initialisation across all 8 conditions
        self.model = SM.SemStudent(len(vocab.itos))
        self.aux = nn.Linear(self.model.d, C.P) if pop in ("C", "D") else None
        self.names = [n for n, _ in self.model.named_parameters()] + ([f"auxhead.{n}" for n, _ in self.aux.named_parameters()] if self.aux else [])
        self.params = list(self.model.parameters()) + (list(self.aux.parameters()) if self.aux else [])
        self.opt = torch.optim.AdamW(self.params, lr=C.LR, weight_decay=wd)
        self.rng = random.Random(seed)
        self.step = 0
        if self.ck.exists() and any(self.ck.glob("step_*/state.json")):
            last = max(d for d in self.ck.glob("step_*") if (d / "state.json").exists())
            self.step = C.load_checkpoint(last, self.model, self.aux, self.opt, self.rng)
            self._truncate_interval()
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            split = C.make_split()
            meta = dict(population=pop, wd=wd, seed=seed, steps=STEPS, lr=C.LR, warmup=C.WARMUP, batch=C.BATCH,
                        aux_weight=C.AUX_W if self.aux else 0, n_train_combos=len(self.spec["combos"]), frames=list(self.spec["frames"]),
                        split_sha=hashlib.sha256(json.dumps(split).encode()).hexdigest()[:16], n_params_inference=self.model.n_params(False),
                        checkpoint_every=BLOCK, init_sha=self._weights_sha())
            (self.dir / "meta.json").write_text(json.dumps(meta, indent=1))
            self._save(self._zero_acc())

    def _weights_sha(self):
        h = hashlib.sha256()
        for k, v in self.model.state_dict().items():
            h.update(k.encode()); h.update(v.contiguous().numpy().tobytes())
        return h.hexdigest()[:16]

    def _truncate_interval(self):
        f = self.dir / "interval.jsonl"
        if f.exists():
            keep = [l for l in f.read_text().splitlines() if l and json.loads(l)["step"] <= self.step]
            f.write_text("".join(l + "\n" for l in keep))

    def _zero_acc(self):
        return dict(adam=[torch.zeros_like(p) for p in self.params], wd=[torch.zeros_like(p) for p in self.params],
                    path_adam={}, path_wd={})

    def _save(self, acc):
        d = self.ck / f"step_{self.step:05d}"
        C.save_checkpoint(d, self.model, self.aux, self.opt, self.rng, self.step)
        tens, groups = {}, {}
        for n, a, w in zip(self.names, acc["adam"], acc["wd"]):
            tens[n + "::adam"], tens[n + "::wd"] = a.contiguous(), w.contiguous()
            g = n.split(".")[0]
            e = groups.setdefault(g, dict(a2=0.0, w2=0.0, aw=0.0))
            e["a2"] += a.pow(2).sum().item(); e["w2"] += w.pow(2).sum().item(); e["aw"] += (a * w).sum().item()
        save_file(tens, d / "updates.safetensors")
        summ = {g: dict(adam_net=round(math.sqrt(e["a2"]), 6), wd_net=round(math.sqrt(e["w2"]), 6),
                        cos_adam_wd=round(e["aw"] / max(math.sqrt(e["a2"] * e["w2"]), 1e-12), 4) if e["a2"] and e["w2"] else None,
                        adam_path=round(acc["path_adam"].get(g, 0.0), 6), wd_path=round(acc["path_wd"].get(g, 0.0), 6))
                for g, e in groups.items()}
        (d / "updates.json").write_text(json.dumps(summ))
        (d / "COMPLETE").write_text("")                               # written last: the checkpoint is whole

    def train_block(self, n=BLOCK):
        acc = self._zero_acc()
        f = open(self.dir / "interval.jsonl", "a")
        for _ in range(n):
            self.step += 1
            lr = C.LR * min(1.0, self.step / C.WARMUP)
            for g in self.opt.param_groups:
                g["lr"] = lr
            exs, aux_t = C.batch(self.spec, C.BATCH, self.rng, self.pop)
            b = D.encode_batch(exs, self.vocab)
            self.model.train()
            h, x1 = self.model.trunk(b["tok"])
            items = F.cross_entropy(self.model.head(h), b["ans"], reduction="none")
            loss = items.mean()
            parts = dict(task=loss.item())
            if self.aux is not None:
                la = F.cross_entropy(self.aux(x1[torch.arange(len(exs)), C.src_pos(b)]), aux_t)
                parts["aux"] = la.item()
                loss = loss + C.AUX_W * la
            self.opt.zero_grad()
            loss.backward()
            gn = math.sqrt(sum(p.grad.pow(2).sum().item() for p in self.params if p.grad is not None))
            before = [p.detach().clone() for p in self.params]
            stepped = [p.grad is not None for p in self.params]                # AdamW skips grad-less parameters
            self.opt.step()
            for i, (n, p, b0) in enumerate(zip(self.names, self.params, before)):
                if not stepped[i]:
                    continue
                wdd = -lr * self.wd * b0                                    # decoupled weight decay (exact)
                ad = p.detach() - b0 - wdd                                  # adaptive-gradient part
                acc["wd"][i] += wdd
                acc["adam"][i] += ad
                g = n.split(".")[0]
                acc["path_adam"][g] = acc["path_adam"].get(g, 0.0) + ad.norm().item()
                acc["path_wd"][g] = acc["path_wd"].get(g, 0.0) + wdd.norm().item()
            f.write(json.dumps(dict(step=self.step, lr=lr, grad_norm=round(gn, 5),
                                    batch_sha=hashlib.sha256(b["tok"].numpy().tobytes()).hexdigest()[:12],
                                    loss_max_item=round(items.max().item(), 4), **{k: round(v, 5) for k, v in parts.items()})) + "\n")
        f.close()
        self.model.eval()
        self._save(acc)


def train_worker(idx, n_workers, live=LIVE, runs=None, steps=STEPS, threads=1):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    vocab = D.Vocab.from_json(json.loads((ROOT / "runs/zoo_sem/eval/vocab.json").read_text()))
    mine = [r for i, r in enumerate(runs or all_runs()) if i % n_workers == idx]
    nets = [Net(p, w, s, vocab, live) for p, w, s in mine]
    while any(n.step < steps for n in nets):
        target = min(n.step for n in nets) + BLOCK
        for n in nets:
            if n.step < target and n.step < steps:
                n.train_block()
        print(f"worker {idx}: cohort {target} complete for {len(nets)} nets", flush=True)
    return nets


# ------------------------------------------------------------------ audits (separate processes, checkpoints only)
def audit_pending(live=LIVE, max_n=None, threads=1):
    from sci import concept_audit as A
    torch.set_num_threads(threads)
    vocab = D.Vocab.from_json(json.loads((ROOT / "runs/zoo_sem/eval/vocab.json").read_text()))
    ctxs = {}
    todo = sorted(((int(d.name.split("_")[1]), d) for d in live.glob("*/ckpt/step_*")
                   if (d / "COMPLETE").exists() and not (d / "AUDITED").exists()), key=lambda x: (x[0], str(x[1])))
    done = 0
    for step, d in todo[:max_n]:
        meta = json.loads((d.parents[1] / "meta.json").read_text())
        pop = meta["population"]
        ctx = ctxs.setdefault(pop, A.AuditContext(vocab, pop))
        m = SM.SemStudent(len(vocab.itos))
        m.load_state_dict(load_file(d / "model.safetensors"))
        aux = None
        if (d / "aux.safetensors").exists():
            aux = nn.Linear(m.d, C.P)
            aux.load_state_dict(load_file(d / "aux.safetensors"))
        t0 = time.process_time()
        summ, tens = A.audit(m, aux, ctx)
        summ["updates"] = json.loads((d / "updates.json").read_text())
        summ.update(step=step, run=d.parents[1].name, population=pop, wd=meta["wd"], seed=meta["seed"],
                    audit_cpu_s=round(time.process_time() - t0, 2))
        save_file(tens, d / "audit.safetensors")
        (d / "audit.json").write_text(json.dumps(summ))
        (d / "AUDITED").write_text("")
        done += 1
    return done


# ------------------------------------------------------------------ final evaluation (reserved test, once)
@torch.no_grad()
def final(live=LIVE, out=REPO_RUNS / "final_test.json"):
    if out.exists():
        raise SystemExit("reserved-test evaluation already run")
    vocab = D.Vocab.from_json(json.loads((ROOT / "runs/zoo_sem/eval/vocab.json").read_text()))
    split = C.make_split()
    rng = random.Random(77)
    canon = dict(num="num", unit="full")
    test = [C.example(r, v, f, canon, rng) for r, v in split["test"] for f in C.TRAIN_FRAMES]
    test_nf = [C.example(r, v, C.NEWFRAME, canon, rng) for r, v in split["test"]]
    res = {}
    for p, w, s in all_runs():
        d = live / run_name(p, w, s) / "ckpt" / f"step_{STEPS:05d}"
        assert (d / "COMPLETE").exists(), d
        m = SM.SemStudent(len(vocab.itos))
        m.load_state_dict(load_file(d / "model.safetensors"))
        m.eval()
        b = D.encode_batch(test, vocab)
        ok = (m(b["tok"]).argmax(-1) == b["ans"]).int().tolist()
        res[run_name(p, w, s)] = dict(population=p, wd=w, seed=s, test=sum(ok) / len(ok), test_items=ok,
                                      test_newframe=C.accuracy(m, test_nf, vocab))
    res["_meta"] = dict(n_items=len(test), best_constant=C.best_constant(test), time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return res


# ------------------------------------------------------------------ archives (outside Git) and Git-side records
def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def archive(name, live=LIVE, archive_dir=ARCHIVE, repo=REPO_RUNS, steps=STEPS):
    """Compress one finished run (all checkpoints, interval log) to <archive>/<run>.tar.xz, verify it, and write
    a per-file + archive checksum manifest and the gzipped audit summaries into Git."""
    src = live / name
    cks = sorted(src.glob("ckpt/step_*"))
    assert len(cks) == steps // BLOCK + 1 and all((c / "COMPLETE").exists() and (c / "AUDITED").exists() for c in cks), name
    files = {str(p.relative_to(src)): sha256_file(p) for p in sorted(src.rglob("*")) if p.is_file()}
    archive_dir.mkdir(parents=True, exist_ok=True)
    tgt = archive_dir / f"{name}.tar.xz"
    subprocess.run(["tar", "-C", str(live), "-cJf", str(tgt), name], check=True, env=dict(os.environ, XZ_OPT="-6 -T1"))
    listed = subprocess.run(["tar", "-tJf", str(tgt)], check=True, capture_output=True, text=True).stdout.split()
    assert sum(1 for x in listed if not x.endswith("/")) == len(files), "archive file count mismatch"
    rd = repo / name
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "meta.json").write_text((src / "meta.json").read_text())
    (rd / "manifest.json").write_text(json.dumps(dict(run=name, archive=tgt.name, archive_sha256=sha256_file(tgt),
                                                      archive_bytes=tgt.stat().st_size, files=files), indent=0))
    with gzip.open(rd / "audits.jsonl.gz", "wt") as f:
        for c in cks:
            f.write((c / "audit.json").read_text() + "\n")
    with gzip.open(rd / "interval.jsonl.gz", "wt") as f:
        f.write((src / "interval.jsonl").read_text())
    return tgt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "audit", "final", "archive"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    if a.cmd == "train":
        train_worker(int(a.args[0]), int(a.args[1]))
    elif a.cmd == "audit":
        while True:
            n = audit_pending()
            print(f"audited {n}", flush=True)
            if not a.loop:
                break
            if n == 0:
                time.sleep(30)
    elif a.cmd == "final":
        r = final()
        print({k: round(v["test"], 3) for k, v in r.items() if not k.startswith("_")})
    else:
        print(archive(a.args[0]))
