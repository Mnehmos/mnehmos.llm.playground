"""Independent verification of the registered concept factorial (Phase I.3).

Separately implemented from `sci/factorial_analysis.py`: numpy-based statistics, its own reserved-test item
construction and model evaluation from the archived (manifest-verified) step-6,000 weights, its own onset code.
It reads the original outputs only to compare against them. Nothing here alters original results.

    python -m sci.verify_factorial          # writes runs/concept_factorial/verification.json
"""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sem import data as D  # noqa: E402
from sem import lang as L  # noqa: E402
from sem import model as SM  # noqa: E402

RUNS = ROOT / "runs" / "concept_factorial"
WEIGHTS = Path("/home/user/nri_weights")
POPS, WDS, SEEDS = "ABCD", (0.1, 1.0), list(range(2001, 2013))
FRAMES = ("C01", "C02", "C04", "C05")
P = 13


def name(p, w, s):
    return f"wd{w}_{p}_s{s}"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- independent reserved-test evaluation
def reserved_items():
    """Rebuild the reserved test from the registered list in doc 37 (not from the training code's split)."""
    reg = {("acc", 0, 1): [8, 10], ("acc", 0, 2): [7, 9], ("acc", 1, 0): [8], ("acc", 2, 0): [9, 12], ("acc", 2, 1): [6],
           ("mass", 0, 1): [2], ("mass", 0, 2): [10], ("mass", 1, 0): [5, 12], ("mass", 1, 2): [1, 7, 8],
           ("mass", 2, 0): [5], ("mass", 2, 1): [2, 4]}
    items = []
    style = dict(num="num", unit="full")
    for (dim, u, t), vs in sorted(reg.items()):
        for v in vs:
            ast = dict(task="convert", dim=dim, v=v, u=u, t=t)
            ans = v * L.SCALES[dim][u] * L.INV[L.SCALES[dim][t]] % P          # independent answer formula
            for f in FRAMES:
                toks, roles = L.render(ast, f, style, None)
                items.append(dict(ast=ast, tokens=toks, roles=roles, answer=str(ans), combo=(dim, u, t, v)))
    return items


@torch.no_grad()
def reevaluate():
    vocab = D.Vocab.from_json(json.loads((ROOT / "runs/zoo_sem/eval/vocab.json").read_text()))
    items = reserved_items()
    b = D.encode_batch(items, vocab)
    out = {}
    for p in POPS:
        for w in WDS:
            for s in SEEDS:
                n = name(p, w, s)
                f = WEIGHTS / n / "ckpt" / "step_06000" / "model.safetensors"
                man = json.loads((RUNS / n / "manifest.json").read_text())["files"]
                assert sha(f) == man["ckpt/step_06000/model.safetensors"], f"weights do not match manifest: {n}"
                m = SM.SemStudent(len(vocab.itos))
                m.load_state_dict(load_file(f))
                m.eval()
                ok = (m(b["tok"]).argmax(-1) == b["ans"]).numpy()
                out[n] = ok.astype(int)
    return items, out


# ---------------------------------------------------------------- statistics (numpy, independent)
SIGNS = np.array(list(itertools.product([1, -1], repeat=12)), dtype=float)        # 4096 x 12


def signflip(d, two_sided=False):
    d = np.asarray(d, float)
    obs = d.mean()
    null = SIGNS @ d / len(d)
    if two_sided:
        return float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
    return float(np.mean(null >= obs - 1e-12))


def boot(d, n=20000, seed=12345):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj.tolist()


def contrasts(acc):
    a = lambda p, w: np.array([acc[name(p, w, s)] for s in SEEDS])
    spec = {"H1": (a("C", 1.0) - a("D", 1.0), False), "H2": ((a("C", 1.0) - a("D", 1.0)) - (a("C", 0.1) - a("D", 0.1)), False),
            "H3": (a("C", 1.0) - a("C", 0.1), False), "H4": (a("C", 1.0) - a("B", 1.0), False), "H5": (a("B", 1.0) - a("A", 1.0), True)}
    res = {k: dict(mean=float(d.mean()), ci95=boot(d), p=signflip(d, ts), per_seed=d.round(4).tolist()) for k, (d, ts) in spec.items()}
    for k, h in zip(res, holm([r["p"] for r in res.values()])):
        res[k]["p_holm"] = h
    return res


# ---------------------------------------------------------------- design checks
def audits(n):
    with gzip.open(RUNS / n / "audits.jsonl.gz", "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def intervals(n):
    with gzip.open(RUNS / n / "interval.jsonl.gz", "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def onsets(n):
    a = audits(n)
    st = [x["step"] for x in a]
    dec = [x["representation"]["decode_x1m_q_knn"] >= 0.5 for x in a]
    s1 = [x["causal"]["semantics_x1_src"] for x in a]
    util = [y["canonical"] - y["number"] >= 0.2 for y in s1]
    comp = [x["behaviour"]["heldout"] >= x["behaviour"]["heldout_best_constant"] + 0.2 for x in a]
    first = lambda c: next((st[i] for i in range(len(c) - 1) if c[i] and c[i + 1]), None)
    return dict(decodable=first(dec), utilised=first(util), compositional=first(comp))


def main():
    rep = {}
    # 1 completeness + manifests
    comp = {}
    for p in POPS:
        for w in WDS:
            for s in SEEDS:
                n = name(p, w, s)
                files = json.loads((RUNS / n / "manifest.json").read_text())["files"]
                steps = sorted({k.split("/")[1] for k in files if k.startswith("ckpt/")})
                need = ["model.safetensors", "optim.safetensors", "rng.safetensors", "state.json", "updates.safetensors",
                        "updates.json", "audit.json", "audit.safetensors", "COMPLETE", "AUDITED"] + (["aux.safetensors"] if p in "CD" else [])
                missing = [f"{st}/{f}" for st in steps for f in need if f"ckpt/{st}/{f}" not in files]
                comp[n] = dict(checkpoints=len(steps), missing=missing, audits=len(audits(n)), interval_steps=len(intervals(n)))
    rep["completeness"] = dict(all_61=all(v["checkpoints"] == 61 for v in comp.values()),
                               no_missing=all(not v["missing"] for v in comp.values()),
                               audits_61=all(v["audits"] == 61 for v in comp.values()),
                               interval_6000=all(v["interval_steps"] == 6000 for v in comp.values()),
                               total_checkpoints=sum(v["checkpoints"] for v in comp.values()))
    # 2 initial parameter matching (manifest hashes of step 0 weights; aux head excluded by construction)
    init = {s: {json.loads((RUNS / name(p, w, s) / "manifest.json").read_text())["files"]["ckpt/step_00000/model.safetensors"]
                for p in POPS for w in WDS} for s in SEEDS}
    rep["initial_matching"] = dict(identical_within_seed=all(len(v) == 1 for v in init.values()),
                                   distinct_across_seeds=len({next(iter(v)) for v in init.values()}) == 12)
    # 3 training-stream matching: same population + seed -> identical batch stream at both weight decays
    streams = {}
    for p in POPS:
        for s in SEEDS:
            a = [r["batch_sha"] for r in intervals(name(p, 0.1, s))]
            b = [r["batch_sha"] for r in intervals(name(p, 1.0, s))]
            streams[f"{p}_s{s}"] = a == b
    rep["stream_matching_across_wd"] = dict(all_identical=all(streams.values()), n=len(streams))
    # 4 partitions
    from sci import concept as C
    sp = C.make_split()
    tr, dv, te = set(sp["train"]), set(sp["dev"]), set(sp["test"])
    reg_test = {((d, u, t), v) for d, u, t, v in {it["combo"] for it in reserved_items()}}
    rep["partitions"] = dict(disjoint=not (tr & dv or tr & te or dv & te), sizes=[len(tr), len(dv), len(te)],
                             registered_test_equals_split_test=reg_test == te)
    # 5 reserved test re-evaluated from weights
    items, ok = reevaluate()
    acc = {n: float(v.mean()) for n, v in ok.items()}
    orig = json.loads((RUNS / "final_test.json").read_text())
    rep["reserved_reevaluation"] = dict(n_items=len(items), max_abs_diff_vs_original=max(abs(acc[n] - orig[n]["test"]) for n in acc),
                                        itemwise_identical=all(ok[n].tolist() == orig[n]["test_items"] for n in acc))
    counts = {}
    for it in items:
        counts[it["answer"]] = counts.get(it["answer"], 0) + 1
    rep["best_constant"] = max(counts.values()) / len(items)
    rep["condition_means"] = {f"wd{w}_{p}": float(np.mean([acc[name(p, w, s)] for s in SEEDS])) for p in POPS for w in WDS}
    rep["contrasts"] = contrasts(acc)
    oa = json.loads((RUNS / "analysis.json").read_text())["primary"]["contrasts"]
    rep["contrast_agreement"] = {k: dict(mean_orig=v["mean"], p_orig=v["p"], p_holm_orig=v["p_holm"], ci_orig=v["ci95"])
                                 for k, v in zip(rep["contrasts"], oa.values())}
    # 6 onsets recomputed
    on = {name("C", 1.0, s): onsets(name("C", 1.0, s)) for s in SEEDS}
    oo = json.loads((RUNS / "analysis.json").read_text())["mechanistic"]["onsets"]
    rep["onsets"] = dict(agree=all(on[n] == oo[n] for n in on), recomputed=on)
    (RUNS / "verification.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k not in ("onsets", "contrasts")}, indent=1))
    for k, v in r["contrasts"].items():
        print(k, round(v["mean"], 4), [round(x, 4) for x in v["ci95"]], v["p"], round(v["p_holm"], 5))
    print("onsets agree:", r["onsets"]["agree"])
