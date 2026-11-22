"""Experiment R6: enforce variable structure INSIDE the network with interchange intervention
training (IIT; Geiger et al., "Inducing Causal Structure for Interpretable Neural Networks",
ICML 2022), then test whether the resulting computation is easier to locate and to edit.

Design. At the site h = resid_pre_mlp we designate fixed coordinate blocks:
    op -> dims [0, 8),  a -> dims [8, 24),  b -> dims [24, 40)   (24 dims left free)
Every step: task loss on a base batch, plus for each variable V an interchange: copy V's block
from an independent source input into the base activation and train the output to equal the
counterfactual OP(a, b) with V taken from the source.

Conditions (untyped inference; same architecture and parameter count as doc 08):
  none, aux_only      reused from runs/zoo_untyped_inf (doc 08) -- identical setup
  iit                 task + IIT on {op, a, b}
  iit_aux             task + IIT + auxiliary role loss (doc 08's aux_only objective)

Every measurement tool is BLIND to the designed blocks (class-mean subspaces, DAS, closed-form
edits), except the explicitly labelled "designed-block" columns.

New, held-out edit family: OPERAND SHIFT. Make the network compute OP(a+1, b): an affine edit
inside the subspace located for `a`, folded into W_in. IIT never trains an edit of this kind.

    python run_iit.py --seeds 5 --steps 3000 --workers 4
Outputs: runs/zoo_iit/, results/iit/{metrics.json, summary.md}
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

import run_untyped_inference as UI
from nri import analysis, tasks
from nri.model import TinyTyped
from nri.train import load, set_determinism
from run_experiment import fmt, perm_test

ROOT = Path(__file__).parent
ZOO = ROOT / "runs" / "zoo_iit"
RES = ROOT / "results" / "iit"
BLOCKS = {"op": (0, 8), "a": (8, 24), "b": (24, 40)}
LAMBDA_IIT = 1.0
NEW = ["iit", "iit_aux"]
REUSED = ["none", "aux_only"]


def block_mask(var, d=64):
    m = torch.zeros(d)
    lo, hi = BLOCKS[var]
    m[lo:hi] = 1
    return m


def train_one(args):
    cond, seed, steps = args
    run = ZOO / f"{cond}_s{seed}"
    if (run / "model.pt").exists():
        return str(run)
    run.mkdir(parents=True, exist_ok=True)
    set_determinism()
    torch.manual_seed(seed)
    m = TinyTyped(cond, **UI.MODEL_KW)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(10_000 + seed)
    masks = {v: block_mask(v) for v in BLOCKS}
    cfg = dict(condition=cond, seed=seed, steps=steps, batch=512, lr=2e-3, wd=0.01,
               lambda_iit=LAMBDA_IIT, lambda_aux=UI.LAMBDA_AUX if cond == "iit_aux" else 0.0,
               blocks=BLOCKS, n_params=m.n_params(), model_kw=UI.MODEL_KW, torch=torch.__version__)
    t0 = time.time()
    with open(run / "log.jsonl", "w") as log:
        for step in range(1, steps + 1):
            lat_b = tasks.sample_latents(512, g)
            lat_s = tasks.sample_latents(512, g)
            bb, bs = tasks.build(lat_b, "typed"), tasks.build(lat_s, "typed")
            zero = torch.zeros_like(bb.types)
            x1b = m.resid_layer1(bb.tokens, zero)
            hb = (x1b + m.attn2(x1b))[:, -1]
            x1s = m.resid_layer1(bs.tokens, zero)
            hs = (x1s + m.attn2(x1s))[:, -1]
            loss = F.cross_entropy(m.head_from_resid(hb), bb.y)
            iit = 0.0
            for v, mk in masks.items():
                hp = hb * (1 - mk) + hs * mk
                iit = iit + F.cross_entropy(m.head_from_resid(hp), analysis.counterfactual(lat_b, lat_s, v))
            iit = iit / len(masks)
            loss = loss + LAMBDA_IIT * iit
            if cond == "iit_aux":
                loss = loss + UI.LAMBDA_AUX * F.cross_entropy(
                    m.aux(x1b).reshape(-1, tasks.N_TYPES), bb.types.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if (step <= 1000 and step % 50 == 0) or step % 250 == 0 or step == steps:
                rec = dict(step=step, loss=loss.item(), iit=iit.item(), eval_acc=UI.eval_acc(m, typed=False),
                           eval_acc_typed=UI.eval_acc(m, typed=True), t=round(time.time() - t0, 1))
                log.write(json.dumps(rec) + "\n")
                log.flush()
    cfg.update(final_untyped_acc=UI.eval_acc(m, False, 8192), final_typed_acc=UI.eval_acc(m, True, 8192),
               train_seconds=round(time.time() - t0, 1))
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.save(m.state_dict(), run / "model.pt")
    print(f"[done] {run.name} untyped={cfg['final_untyped_acc']:.4f} ({cfg['train_seconds']}s)")
    return str(run)


# ------------------------------------------------------------------ held-out edit: operand shift
@torch.no_grad()
def operand_shift_edit(model, basis, var="a", shift=1, n_fit=8192):
    """Affine edit inside span(basis) mapping each `var` class to class (var+shift), folded into
    W_in. Fitted by least squares on activations (no gradients). Returns (dW, db)."""
    lat = analysis.latents(n_fit, analysis.FIT_SEED)
    _, h = analysis.collect(model, lat, "none")
    mu = analysis.class_means(h, lat[var], analysis.VARS[var])
    target = mu[(lat[var] + shift) % analysis.VARS[var]] - mu[lat[var]]  # desired per-sample move
    m0 = h.mean(0)
    z = (h - m0) @ basis
    X = torch.cat([z, torch.ones(len(z), 1)], 1)
    W = torch.linalg.lstsq(X, target @ basis).solution  # (k+1) x k
    W1, w0 = W[:-1], W[-1]
    D = basis @ W1.T @ basis.T  # dh = D (h - m0) + basis w0
    dW = model.W_in.weight @ D
    db = model.W_in.weight @ (basis @ w0 - D @ m0)
    return dW, db


@torch.no_grad()
def eval_shift(model, dW, db, var="a", shift=1, n=8192):
    lat = analysis.latents(n, analysis.TEST_SEED)
    b, h = analysis.collect(model, lat, "none")
    pred = model.head_from_resid(h, model.W_in.weight + dW, model.W_in.bias + db).argmax(-1)
    lat2 = tasks.with_latent(lat, **{var: (lat[var] + shift) % analysis.VARS[var]})
    tgt = tasks.apply_op(lat2["op"], lat2["a"], lat2["b"])
    return (pred == tgt).float().mean().item()


@torch.no_grad()
def designed_block_iia(model, var, n=4096):
    base, src = analysis.latents(n, analysis.PAIR_SEED), analysis.latents(n, analysis.PAIR_SEED + 1)
    _, hb = analysis.collect(model, base, "none")
    _, hs = analysis.collect(model, src, "none")
    mk = block_mask(var)
    hp = hb * (1 - mk) + hs * mk
    return (model.head_from_resid(hp).argmax(-1) == analysis.counterfactual(base, src, var)).float().mean().item()


def analyse(run_str):
    out = UI.analyse(run_str)  # doc-08 metric set (untyped inference, blind to blocks)
    set_determinism()
    m = load(Path(run_str))
    for v in ("b",):
        d, _ = analysis.das_curve(m, "none", v, ks=(16,))
        out["das_b"] = d["curve"][16]
    out["iia_b"] = analysis.iia_curve(m, "none", "b")["curve"][-1]
    # held-out operand-shift edit, three localizations
    lat = analysis.latents(8192, analysis.FIT_SEED)
    _, h = analysis.collect(m, lat, "none")
    U, _ = analysis.mean_subspace(h, lat["a"], analysis.VARS["a"])
    out["shift_edit_cm"] = eval_shift(m, *operand_shift_edit(m, U))
    _, Qa = analysis.das(m, "none", "a", 16)
    out["shift_edit_das"] = eval_shift(m, *operand_shift_edit(m, Qa))
    lo, hi = BLOCKS["a"]
    E = torch.eye(64)[:, lo:hi]
    out["shift_edit_designed"] = eval_shift(m, *operand_shift_edit(m, E))
    out["shift_noop"] = eval_shift(m, torch.zeros_like(m.W_in.weight), torch.zeros_like(m.W_in.bias))
    for v in BLOCKS:
        out[f"designed_iia_{v}"] = designed_block_iia(m, v)
    return out


def summarize(models, conds):
    cols = [("acc_untyped", "acc"), ("steps_to_95", "steps→95%"),
            ("iia_op", "IIA op (cm)"), ("iia_a", "IIA a (cm)"),
            ("das_op", "DAS op"), ("das_a", "DAS a"), ("das_b", "DAS b"),
            ("edit_success", "op-swap edit (cm)"), ("das_edit_success", "op-swap edit (DAS)"),
            ("shift_edit_cm", "a+1 edit (cm)"), ("shift_edit_das", "a+1 edit (DAS)"),
            ("shift_edit_designed", "a+1 edit (designed block)"),
            ("designed_iia_a", "designed-block IIA a"), ("additive_r2", "additive R²")]
    L = ["# R6: interchange intervention training (auto-generated by run_iit.py)", "",
         "Untyped inference throughout. Columns marked (cm) / (DAS) use localizations found by "
         "tools that do not know the designed blocks. The no-op baseline for the a+1 edit is "
         f"{st.mean(m['shift_noop'] for m in models):.3f}.", "",
         "| condition | n | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    by = {c: [m for m in models if m["condition"] == c] for c in conds}
    for c in conds:
        row = [c, str(len(by[c]))]
        for k, _ in cols:
            vals = [m[k] for m in by[c]]
            nn = sum(v is None for v in vals)
            row.append(fmt([v for v in vals if v is not None]) + (f" ({nn} never)" if nn else ""))
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## Permutation tests (two-sided, 5 vs 5 seeds, uncorrected; min attainable p ≈ 0.008)", "",
          "| metric | comparison | Δ mean | p |", "|---|---|---|---|"]
    tests = {}
    for k in ("steps_to_95", "iia_op", "iia_a", "das_op", "das_a", "das_b", "edit_success",
              "das_edit_success", "shift_edit_cm", "shift_edit_das", "additive_r2"):
        for a_, b_ in (("iit", "none"), ("iit", "aux_only"), ("iit_aux", "none"), ("iit_aux", "aux_only"),
                       ("iit_aux", "iit")):
            xs = [m[k] for m in by[a_] if m[k] is not None]
            ys = [m[k] for m in by[b_] if m[k] is not None]
            if len(xs) < 2 or len(ys) < 2:
                continue
            p = perm_test(xs, ys)
            tests[f"{k}:{a_}-{b_}"] = dict(delta=st.mean(xs) - st.mean(ys), p=p)
            L.append(f"| {k} | {a_} − {b_} | {st.mean(xs) - st.mean(ys):+.3f} | {p:.4f} |")
    return "\n".join(L) + "\n", tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    jobs = [(c, s, a.steps) for c in NEW for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        runs = list(ex.map(train_one, jobs))
        reused = [str(UI.ZOO / f"{c}_s{s}") for c in REUSED for s in range(a.seeds)]
        models = list(ex.map(analyse, reused + runs))
    (RES / "metrics.json").write_text(json.dumps(models, indent=2))
    md, tests = summarize(models, REUSED + NEW)
    (RES / "summary.md").write_text(md)
    (RES / "tests.json").write_text(json.dumps(tests, indent=2))
    print(md)


if __name__ == "__main__":
    main()
