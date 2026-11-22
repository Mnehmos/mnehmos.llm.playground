"""Experiment B0/C0/D0 prototype: do ANLU-style typed priors make a network easier to
locate, edit, and transfer edits on?  CPU only, deterministic.

    python run_experiment.py --seeds 5 --steps 6000 --workers 4

Outputs (never overwrites existing checkpoints):
    runs/zoo/<condition>_s<seed>/       checkpoints, configs, training logs
    results/metrics.json                every per-model / per-pair measurement
    results/nri/<model>.nri.json        executed reference NRI documents
    results/summary.md                  tables (mean +- sd over seeds) + permutation tests
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from nri import analysis, tasks
from nri.nri_exec import Host, execute, sha256
from nri.reference_nri import reference_doc
from nri.train import load, set_determinism, train

ROOT = Path(__file__).parent
ZOO = ROOT / "runs" / "zoo"  # reset per regime in main()
RES = ROOT / "results"
WD = 0.01
PROGRAM = str(ROOT / "programs" / "op_swap.py")


def _train(args):
    cond, seed, steps, zoo, wd = args
    return str(train(cond, seed, Path(zoo), steps=steps, wd=wd))


@torch.no_grad()
def ablate_attn(model, cond, which):
    g = torch.Generator().manual_seed(analysis.TEST_SEED)
    b = tasks.make_batch(8192, cond, g)
    x = model.embed(b.tokens, b.types)
    x = x + (0 if which == 1 else model.attn1(x))
    x = x + (0 if which == 2 else model.attn2(x))
    pred = model.head_from_resid(x[:, -1]).argmax(-1)
    return (pred == b.y).float().mean().item()


def steps_to(run: Path, thr=0.95):
    for line in (run / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["eval_acc"] >= thr:
            return r["step"]
    return None


def per_model(args):
    run, res = args
    global RES
    RES = Path(res)
    set_determinism()
    cfg = json.loads((run / "config.json").read_text())
    cond = cfg["condition"]
    m = load(run)
    out = dict(model=run.name, condition=cond, seed=cfg["seed"], n_params=cfg["n_params"],
               acc=cfg["final_eval_acc"], steps_to_95=steps_to(run))
    out["acc_no_attn1"] = ablate_attn(m, cond, 1)
    out["acc_no_attn2"] = ablate_attn(m, cond, 2)
    for v in ("op", "a", "b"):
        c = analysis.iia_curve(m, cond, v)
        n = analysis.neuron_iia(m, cond, v)
        out[f"iia_{v}"] = c["curve"][-1]
        out[f"iia_{v}_k90"] = c["k90"]
        out[f"iia_{v}_curve"] = c["curve"]
        out[f"neuron_{v}_m90"] = n["m90"]
        out[f"neuron_{v}_curve"] = n["curve"]
    for v in ("op", "a", "b"):
        out[f"probe_{v}"] = analysis.probe_acc(m, cond, v)
    Qop = None
    for v in ("op", "a"):
        d, Qs = analysis.das_curve(m, cond, v, ks=(2, 4, 8, 16))
        out[f"das_{v}"] = d["curve"][16]
        out[f"das_{v}_curve"] = d["curve"]
        out[f"das_{v}_k90"] = d["k90"]
        if v == "op":
            Qop = Qs[d["k90"] or 4]
    for v in ("op", "a", "b"):
        hi = analysis.head_iia(m, cond, v)
        out[f"head_{v}_best"] = hi["best_any_iia"]
        out[f"head_{v}_where"] = hi["best_any"]
    rp = analysis.role_purity(m, cond)
    out["pure_role_heads"] = rp["n_pure_role_heads"]
    out["mean_head_purity"] = st.mean(h["purity"] for h in rp["heads"].values())
    out["role_heads"] = {k: (h["role"], round(h["purity"], 3)) for k, h in rp["heads"].items()}
    das_sw = {}
    for x, y in itertools.permutations(range(len(tasks.OPS)), 2):
        dW, db = analysis.das_swap_program(m, cond, x, y, Qop)
        das_sw[f"{tasks.OPS[x]}->{tasks.OPS[y]}"] = analysis.eval_edit(m, cond, x, y, dW, db)
    out["das_edit_success"] = st.mean(s["success"] for s in das_sw.values())
    out["das_edit_collateral"] = st.mean(s["collateral"] for s in das_sw.values())
    swaps = analysis.all_swaps(m, cond)
    out["edit"] = swaps
    out["edit_success"] = st.mean(s["success"] for s in swaps.values())
    out["edit_preserve"] = st.mean(s["preserve"] for s in swaps.values())
    out["edit_collateral"] = st.mean(s["collateral"] for s in swaps.values())
    # executed NRI document
    host = Host(m, lambda n, g: (lambda b: (b.tokens, b.types))(tasks.make_batch(n, cond, g)),
                tasks.P)
    doc = reference_doc(run.name, sha256(run / "model.pt"), cond, PROGRAM)
    ex = execute(doc, host)
    (RES / "nri").mkdir(parents=True, exist_ok=True)
    (RES / "nri" / f"{run.name}.nri.json").write_text(json.dumps(ex, indent=2))
    out["nri_verified"] = ex["summary"]["verified"]
    out["nri_claims"] = ex["summary"]["claims"]
    out["nri_status"] = {c["id"]: c["status"] for c in ex["claims"]}
    return out


def per_pair(args):
    set_determinism()
    cond, sa, sb, zoo = args
    A, B = load(Path(zoo) / f"{cond}_s{sa}"), load(Path(zoo) / f"{cond}_s{sb}")
    rows = []
    for x, y in itertools.permutations(range(len(tasks.OPS)), 2):
        t = analysis.transfer(A, B, cond, x, y)
        rows.append(dict(condition=cond, A=sa, B=sb, swap=f"{tasks.OPS[x]}->{tasks.OPS[y]}", **t))
    return rows


def perm_test(xs, ys, n_perm=20000, seed=0):
    """Two-sided permutation test on difference of means (exact-ish for small n)."""
    g = torch.Generator().manual_seed(seed)
    allv = torch.tensor(xs + ys, dtype=torch.float64)
    obs = abs(st.mean(xs) - st.mean(ys))
    nx, hits = len(xs), 0
    for _ in range(n_perm):
        p = allv[torch.randperm(len(allv), generator=g)]
        hits += abs(p[:nx].mean().item() - p[nx:].mean().item()) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{st.mean(vals):.3f} ± {st.stdev(vals):.3f}"


def summarize(models, pairs, conds):
    L = ["# Prototype results (auto-generated by run_experiment.py)", ""]
    by = {c: [m for m in models if m["condition"] == c] for c in conds}
    cols = [("acc", "task acc"), ("steps_to_95", "steps→95%"), ("acc_no_attn1", "acc w/o attn1"),
            ("probe_a", "probe a"), ("iia_op", "IIA op cm r2"), ("iia_a", "IIA a cm r10"),
            ("das_op", "DAS op k16"), ("das_a", "DAS a k16"),
            ("head_op_best", "head-IIA op"), ("head_a_best", "head-IIA a"),
            ("mean_head_purity", "head purity"), ("pure_role_heads", "pure role heads"),
            ("edit_success", "cm-edit success"), ("das_edit_success", "DAS-edit success"),
            ("edit_collateral", "cm-edit collateral"), ("nri_verified", "NRI verified /8")]
    L += ["## Per-condition (mean ± sd over seeds)", "",
          "| condition | n | " + " | ".join(c[1] for c in cols) + " |",
          "|---|---|" + "---|" * len(cols)]
    for c in conds:
        row = [c, str(len(by[c]))]
        for k, _ in cols:
            vals = [m[k] for m in by[c]]
            if any(v is None for v in vals):
                row.append(fmt([v for v in vals if v is not None]) + f" ({sum(v is None for v in vals)} n/a)")
            else:
                row.append(fmt(vals))
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## Transfer of the op-swap edit across independently trained seeds", "",
          "| condition | CKA(A,B) | Procrustes resid | rerun program success | copy ΔW success | aligned ΔW success | aligned ΔW preserve |",
          "|---|---|---|---|---|---|---|"]
    for c in conds:
        P = [p for p in pairs if p["condition"] == c]
        if not P:
            continue
        L.append("| " + " | ".join([c, fmt([p["cka"] for p in P]),
                                     fmt([p["procrustes_rel_residual"] for p in P]),
                                     fmt([p["rerun_program"]["success"] for p in P]),
                                     fmt([p["copy_delta"]["success"] for p in P]),
                                     fmt([p["aligned_delta"]["success"] for p in P]),
                                     fmt([p["aligned_delta"]["preserve"] for p in P])]) + " |")
    L += ["", "## Permutation tests vs `none` (two-sided, per-seed means)", "",
          "| metric | condition | Δ mean vs none | p |", "|---|---|---|---|"]
    tests = {}
    for k in ("steps_to_95", "iia_op", "iia_a", "das_op", "das_a", "head_op_best",
              "head_a_best", "mean_head_purity", "edit_success", "das_edit_success"):
        for c in conds:
            if c == "none":
                continue
            xs = [m[k] for m in by[c] if m[k] is not None]
            ys = [m[k] for m in by["none"] if m[k] is not None]
            if len(xs) < 2 or len(ys) < 2:
                continue
            p = perm_test(xs, ys)
            tests[f"{k}:{c}"] = dict(delta=st.mean(xs) - st.mean(ys), p=p)
            L.append(f"| {k} | {c} | {st.mean(xs) - st.mean(ys):+.3f} | {p:.4f} |")
    L += ["", "Uncorrected p-values over many comparisons: treat as exploratory.", ""]
    return "\n".join(L), tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conditions", nargs="+", default=list(tasks.CONDITIONS))
    ap.add_argument("--wd", type=float, default=0.01)
    a = ap.parse_args()
    tag = f"wd{a.wd:g}"
    zoo = ROOT / "runs" / f"zoo_{tag}"
    global RES
    RES = ROOT / "results" / tag
    RES.mkdir(parents=True, exist_ok=True)
    jobs = [(c, s, a.steps, str(zoo), a.wd) for c in a.conditions for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        runs = [Path(r) for r in ex.map(_train, jobs)]
        models = list(ex.map(per_model, [(r, str(RES)) for r in runs]))
        pair_jobs = [(c, s, (s + 1) % a.seeds, str(zoo)) for c in a.conditions for s in range(a.seeds)]
        pairs = [r for rows in ex.map(per_pair, pair_jobs) for r in rows]
    (RES / "metrics.json").write_text(json.dumps(dict(models=models, pairs=pairs), indent=2))
    md, tests = summarize(models, pairs, a.conditions)
    md = md.replace("# Prototype results", f"# Prototype results, regime {tag} (steps={a.steps}, seeds={a.seeds})")
    (RES / "summary.md").write_text(md)
    (RES / "tests.json").write_text(json.dumps(tests, indent=2))
    print(md)


if __name__ == "__main__":
    main()
