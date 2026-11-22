"""Experiment B1: ANLU as specified -- TYPED TRAINING, UNTYPED INFERENCE.

The ANLU Foundation Specification and the Pedagogy paper state that type vectors are
training scaffolding, removed at inference, and internalized "through an auxiliary type
prediction objective". The first prototype (run_experiment.py) fed types at inference too,
i.e. it tested the runtime-dependency variant that the Pedagogy paper itself predicts
to be a crutch. This experiment tests the specified version.

Type ids: 0 = NONE, role + 1 otherwise (roles: DISTRACTOR, LEFT, RIGHT, OP, QUERY).
EVERY model is evaluated with all type ids = NONE (= tasks.build(..., "none")).

Conditions (identical architecture and parameter count: 6-row type table + 5-way aux head):
  none               no types, task loss only                         (baseline)
  typed_always       types on every training sequence                 (scaffold removed unprepared)
  typed_dropout      types on 50% of training sequences
  typed_dropout_aux  types on 50% + auxiliary role prediction from the layer-1 residual
  aux_only           no type input; auxiliary role prediction only    (types as a training target)
  aux_shuffled       no type input; auxiliary loss on within-sequence-SHUFFLED roles (control)
  aux_lexical        no type input; auxiliary target = token mod 5 (learnable, irrelevant control)
  aux_contextual     no type input; auxiliary target = LEFT-NEIGHBOUR token mod 5 (learnable,
                     needs the same neighbour lookup as roles, task-irrelevant control)

    python run_untyped_inference.py --seeds 5 --steps 3000 --workers 4
Outputs: runs/zoo_untyped_inf/, results/untyped_inf/{metrics.json, summary.md}
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from additivity import r2 as additivity_r2
from nri import analysis, tasks
from nri.model import TinyTyped
from nri.train import load, set_determinism
from run_experiment import fmt, perm_test

ROOT = Path(__file__).parent
ZOO = ROOT / "runs" / "zoo_untyped_inf"
RES = ROOT / "results" / "untyped_inf"
CONDS = ["none", "typed_always", "typed_dropout", "typed_dropout_aux", "aux_only", "aux_shuffled",
         "aux_lexical", "aux_contextual"]
MODEL_KW = dict(n_types=tasks.N_TYPES + 1, aux_classes=tasks.N_TYPES)
P_DROP, LAMBDA_AUX = 0.5, 1.0


def make_train_batch(cond, n, g):
    lat = tasks.sample_latents(n, g)
    bt = tasks.build(lat, "typed")
    roles = bt.types  # 0..4, contextual
    types = torch.zeros_like(roles)
    if cond == "typed_always":
        types = roles + 1
    elif cond in ("typed_dropout", "typed_dropout_aux"):
        keep = (torch.rand(n, 1, generator=g) >= P_DROP).long()
        types = (roles + 1) * keep
    aux = None
    if cond in ("typed_dropout_aux", "aux_only"):
        aux = roles
    elif cond == "aux_shuffled":
        aux = tasks.build(lat, "shuffled").types
    elif cond == "aux_lexical":  # learnable, context-free, task-irrelevant
        aux = bt.tokens % tasks.N_TYPES
    elif cond == "aux_contextual":  # learnable, needs a neighbour lookup like roles do, task-irrelevant
        prev = torch.cat([torch.full_like(bt.tokens[:, :1], tasks.EQ_TOKEN), bt.tokens[:, :-1]], 1)
        aux = prev % tasks.N_TYPES
    return bt.tokens, types, bt.y, aux


@torch.no_grad()
def eval_acc(model, typed: bool, n=4096, seed=999_983):
    g = torch.Generator().manual_seed(seed)
    lat = tasks.sample_latents(n, g)
    b = tasks.build(lat, "typed")
    types = b.types + 1 if typed else torch.zeros_like(b.types)
    return (model(b.tokens, types).argmax(-1) == b.y).float().mean().item()


def train_one(args):
    cond, seed, steps = args
    run = ZOO / f"{cond}_s{seed}"
    if (run / "model.pt").exists():
        return str(run)
    run.mkdir(parents=True, exist_ok=True)
    set_determinism()
    torch.manual_seed(seed)
    m = TinyTyped(cond, **MODEL_KW)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(10_000 + seed)
    cfg = dict(condition=cond, seed=seed, steps=steps, batch=512, lr=2e-3, wd=0.01, p_drop=P_DROP,
               lambda_aux=LAMBDA_AUX, n_params=m.n_params(), model_kw=MODEL_KW, torch=torch.__version__)
    t0 = time.time()
    with open(run / "log.jsonl", "w") as log:
        for step in range(1, steps + 1):
            tok, typ, y, aux = make_train_batch(cond, 512, g)
            x1 = m.resid_layer1(tok, typ)
            h = (x1 + m.attn2(x1))[:, -1]
            loss = F.cross_entropy(m.head_from_resid(h), y)
            aux_loss = torch.tensor(0.0)
            if aux is not None:
                aux_loss = F.cross_entropy(m.aux(x1).reshape(-1, tasks.N_TYPES), aux.reshape(-1))
                loss = loss + LAMBDA_AUX * aux_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if (step <= 1000 and step % 50 == 0) or step % 250 == 0 or step == steps:
                rec = dict(step=step, loss=loss.item(), aux_loss=aux_loss.item(),
                           eval_acc=eval_acc(m, typed=False), eval_acc_typed=eval_acc(m, typed=True),
                           t=round(time.time() - t0, 1))
                log.write(json.dumps(rec) + "\n")
                log.flush()
    cfg.update(final_untyped_acc=eval_acc(m, False, 8192), final_typed_acc=eval_acc(m, True, 8192),
               train_seconds=round(time.time() - t0, 1))
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.save(m.state_dict(), run / "model.pt")
    print(f"[done] {run.name} untyped={cfg['final_untyped_acc']:.4f} typed={cfg['final_typed_acc']:.4f}")
    return str(run)


@torch.no_grad()
def role_probe(model, n=8192):
    """Balanced accuracy of a linear probe for the contextual role (LEFT / RIGHT / DISTRACTOR)
    of NUMBER tokens, from the layer-1 residual, with UNTYPED input. Has the role
    structure been internalized into the computation?"""
    def feats(seed):
        lat = analysis.latents(n, seed)
        b = tasks.build(lat, "typed")
        x1 = model.resid_layer1(b.tokens, torch.zeros_like(b.types))[:, :-1]
        roles = b.types[:, :-1]
        isnum = roles != tasks.T_OP
        return x1[isnum], roles[isnum]
    X, y = feats(analysis.FIT_SEED)
    Xt, yt = feats(analysis.TEST_SEED)
    y = torch.where(y == tasks.T_DISTRACTOR, 0, y)  # classes 0 (distractor), 1 (left), 2 (right)
    yt = torch.where(yt == tasks.T_DISTRACTOR, 0, yt)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    torch.manual_seed(0)
    with torch.enable_grad():
        lin = torch.nn.Linear(X.shape[1], 3)
        opt = torch.optim.LBFGS(lin.parameters(), max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            w = torch.bincount(y, minlength=3).float()
            loss = F.cross_entropy(lin((X - mu) / sd), y, weight=w.sum() / (3 * w))
            loss.backward()
            return loss
        opt.step(closure)
    pred = lin((Xt - mu) / sd).argmax(-1)
    return st.mean((pred[yt == c] == c).float().mean().item() for c in range(3))


def steps_to(run, thr=0.95):
    for line in (run / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["eval_acc"] >= thr:
            return r["step"]
    return None


def analyse(run_str):
    set_determinism()
    run = Path(run_str)
    cfg = json.loads((run / "config.json").read_text())
    m = load(run)
    c = "none"  # untyped inference for every condition
    out = dict(model=run.name, condition=cfg["condition"], seed=cfg["seed"], n_params=cfg["n_params"],
               acc_untyped=cfg["final_untyped_acc"], acc_typed=cfg["final_typed_acc"],
               steps_to_95=steps_to(run))
    out["role_probe_L1"] = role_probe(m)
    out["iia_op"] = analysis.iia_curve(m, c, "op")["curve"][-1]
    out["iia_a"] = analysis.iia_curve(m, c, "a")["curve"][-1]
    d_op, Qs = analysis.das_curve(m, c, "op", ks=(4, 16))
    d_a, _ = analysis.das_curve(m, c, "a", ks=(16,))
    out["das_op"], out["das_a"] = d_op["curve"][16], d_a["curve"][16]
    sw = analysis.all_swaps(m, c)
    out["edit_success"] = st.mean(s["success"] for s in sw.values())
    out["edit_collateral"] = st.mean(s["collateral"] for s in sw.values())
    import itertools
    das_s = []
    for x, y in itertools.permutations(range(3), 2):
        dW, db = analysis.das_swap_program(m, c, x, y, Qs[4])
        das_s.append(analysis.eval_edit(m, c, x, y, dW, db)["success"])
    out["das_edit_success"] = st.mean(das_s)
    rp = analysis.role_purity(m, c)
    out["head_purity"] = st.mean(h["purity"] for h in rp["heads"].values())
    out["pure_role_heads"] = rp["n_pure_role_heads"]
    out["additive_r2"], out["cell_r2"] = additivity_r2(m, c)
    return out


def summarize(models, conds):
    cols = [("acc_untyped", "acc (untyped inference)"), ("acc_typed", "acc (if types given)"),
            ("steps_to_95", "steps→95% (untyped)"), ("role_probe_L1", "role probe L1 (bal.)"),
            ("iia_op", "IIA op"), ("das_op", "DAS op k16"), ("das_a", "DAS a k16"),
            ("head_purity", "head purity"), ("additive_r2", "additive R²"),
            ("edit_success", "cm-edit"), ("das_edit_success", "DAS-edit")]
    L = ["# Typed training, untyped inference (auto-generated by run_untyped_inference.py)", "",
         "All metrics use UNTYPED inputs (type id = NONE) unless the column says otherwise.", "",
         "| condition | n | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    by = {c: [m for m in models if m["condition"] == c] for c in conds}
    for c in conds:
        row = [c, str(len(by[c]))]
        for k, _ in cols:
            vals = [m[k] for m in by[c]]
            nn = sum(v is None for v in vals)
            row.append(fmt([v for v in vals if v is not None]) + (f" ({nn} never)" if nn else ""))
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## Permutation tests vs `none` (two-sided, 5 vs 5 seeds, uncorrected)", "",
          "| metric | condition | Δ mean | p |", "|---|---|---|---|"]
    tests = {}
    for k in ("acc_untyped", "steps_to_95", "role_probe_L1", "iia_op", "das_op", "das_a",
              "additive_r2", "edit_success", "das_edit_success"):
        ys = [m[k] for m in by["none"] if m[k] is not None]
        for c in conds[1:]:
            xs = [m[k] for m in by[c] if m[k] is not None]
            if len(xs) < 2 or len(ys) < 2:
                continue
            p = perm_test(xs, ys)
            tests[f"{k}:{c}"] = dict(delta=st.mean(xs) - st.mean(ys), p=p)
            L.append(f"| {k} | {c} | {st.mean(xs) - st.mean(ys):+.3f} | {p:.4f} |")
    return "\n".join(L) + "\n", tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    jobs = [(c, s, a.steps) for c in CONDS for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        runs = list(ex.map(train_one, jobs))
        models = list(ex.map(analyse, runs))
    (RES / "metrics.json").write_text(json.dumps(models, indent=2))
    md, tests = summarize(models, CONDS)
    (RES / "summary.md").write_text(md)
    (RES / "tests.json").write_text(json.dumps(tests, indent=2))
    print(md)


if __name__ == "__main__":
    main()
