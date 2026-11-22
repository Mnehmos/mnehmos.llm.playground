"""Variable-specific edit test (post hoc; proposed for the next registration).

Problem with the registered edit (doc 11 §11.4): with F = M·A·S, doubling M and doubling A both
give exactly 2F, so "the output became 2F" cannot show WHICH variable the edit changed. An edit
that simply doubled the output would also score as a success.

Fix: an ADDITIVE edit. M → M+1 changes the answer by A·S; A → A+1 changes it by M·S (S ≠ 0).
On the discriminating set D, where the intended target, the wrong-variable target and the
original answer are pairwise distinct (for M+1: A ≠ 0, M ≠ 0, A ≠ M), every edited output is
classified as
    intended | wrong-variable | unchanged | other.
specificity = P(intended | D) - P(wrong | D); selectivity = P(intended) / (P(intended) + P(wrong)).
Chance for any single target is 1/13.

The same flaw partly affects R6's operand-shift edit (doc 09): for ADD, a+1 and b+1 give the
same answer. The R6 networks are re-scored on their own discriminating set (SUB rows, and MUL
rows with a ≠ b, a ≠ 0, b ≠ 0); ADD rows are excluded.

    python edit_specificity.py      -> results/edit_specificity.json, results/edit_specificity.md
"""
from __future__ import annotations

import json
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

import run_dimensional as D
from nri import analysis, dimtask as T, tasks
from nri.train import load as load_r6, set_determinism

ROOT = Path(__file__).parent


# ------------------------------------------------------------------ dimensional task
INV = torch.tensor([0] + [pow(x, -1, T.P) for x in range(1, T.P)])  # 0 -> 0 by convention
PERMS = {"add1": lambda y: (y + 1) % T.P, "inv": lambda y: INV[y]}


@torch.no_grad()
def additive_edit(m, basis, var, n=8192, kind="add1"):
    """var -> perm(var): affine map inside span(basis), least squares, folded into W_in.
    kind='add1': var+1 (mod 13). kind='inv': multiplicative inverse (0 -> 0). Inversion is
    negation in a discrete-log code, so it is affine-expressible if the code is multiplicative;
    unlike doubling, it is output-discriminating (M^-1 A S != M A^-1 S whenever A != +-M)."""
    lat = D.lats(n, D.FIT)
    _, h = D.site(m, lat)
    y = D.label(lat, var)
    mu = D.class_means(h, y, T.P)
    move = mu[PERMS[kind](y)] - mu[y]
    m0 = h.mean(0)
    z = (h - m0) @ basis
    X = torch.cat([z, torch.ones(len(z), 1)], 1)
    W = torch.linalg.lstsq(X, move @ basis).solution
    Dm = basis @ W[:-1].T @ basis.T
    return m.W_in.weight @ Dm, m.W_in.weight @ (basis @ W[-1] - Dm @ m0)


@torch.no_grad()
def classify_dim(m, dW, db, var, n=16384, kind="add1"):
    lat = D.lats(n, D.TEST)
    _, h = D.site(m, lat)
    pred = m.head_from_resid(h, m.W_in.weight + dW, m.W_in.bias + db).argmax(-1)
    M, A, S = T.canon(lat)
    F = T.target(M, A, S)
    tM = T.target(PERMS[kind](M), A, S)
    tA = T.target(M, PERMS[kind](A), S)
    t_int, t_wrong = (tM, tA) if var == "M" else (tA, tM)
    disc = (t_int != t_wrong) & (t_int != F) & (t_wrong != F)
    p = pred[disc]
    out = dict(n_disc=int(disc.sum()),
               p_intended=(p == t_int[disc]).float().mean().item(),
               p_wrong=(p == t_wrong[disc]).float().mean().item(),
               p_unchanged=(p == F[disc]).float().mean().item())
    out["p_other"] = 1 - out["p_intended"] - out["p_wrong"] - out["p_unchanged"]
    out["specificity"] = out["p_intended"] - out["p_wrong"]
    s = out["p_intended"] + out["p_wrong"]
    out["selectivity"] = out["p_intended"] / s if s > 0 else None
    return out


def analyse_dim(run):
    set_determinism()
    m, cfg = D.load(run)
    lat = D.lats(8192, D.FIT)
    _, h = D.site(m, lat)
    res = dict(model=Path(run).name, arm=cfg["arm"], seed=cfg["seed"], converged=None)
    res["acc_id"] = json.loads((Path(run) / "config.json").read_text())["acc_id"]
    res["converged"] = res["acc_id"] >= 0.95
    for var in ("M", "A"):
        U, _ = D.mean_basis(h, D.label(lat, var), T.P)
        _, Q = D.das(m, var)
        lo, hi = D.BLOCKS_FULL[var]
        for name, B in (("das", Q), ("cm", U), ("block", torch.eye(64)[:, lo:hi])):
            r = classify_dim(m, *additive_edit(m, B, var), var)
            for k, v in r.items():
                res[f"{var}_{name}_{k}"] = v
            r = classify_dim(m, *additive_edit(m, B, var, kind="inv"), var, kind="inv")
            for k, v in r.items():
                res[f"inv_{var}_{name}_{k}"] = v
    z = torch.zeros_like(m.W_in.weight), torch.zeros_like(m.W_in.bias)
    noop = classify_dim(m, *z, "M")
    res["noop_p_intended"], res["noop_p_unchanged"] = noop["p_intended"], noop["p_unchanged"]
    return res


# ------------------------------------------------------------------ R6 (modular arithmetic)
@torch.no_grad()
def classify_r6(m, dW, db, n=16384):
    lat = analysis.latents(n, analysis.TEST_SEED)
    b, h = analysis.collect(m, lat, "none")
    pred = m.head_from_resid(h, m.W_in.weight + dW, m.W_in.bias + db).argmax(-1)
    op, a, bb = lat["op"], lat["a"], lat["b"]
    F = tasks.apply_op(op, a, bb)
    t_int = tasks.apply_op(op, (a + 1) % tasks.P, bb)
    t_wrong = tasks.apply_op(op, a, (bb + 1) % tasks.P)
    disc = (t_int != t_wrong) & (t_int != F) & (t_wrong != F)
    p = pred[disc]
    out = dict(n_disc=int(disc.sum()), frac_add_in_disc=(op[disc] == 0).float().mean().item(),
               p_intended=(p == t_int[disc]).float().mean().item(),
               p_wrong=(p == t_wrong[disc]).float().mean().item(),
               p_unchanged=(p == F[disc]).float().mean().item())
    out["specificity"] = out["p_intended"] - out["p_wrong"]
    # the old registered-style score on ALL rows (includes the ambiguous ADD rows)
    out["old_success_all_rows"] = (pred == t_int).float().mean().item()
    return out


def analyse_r6(run):
    import run_iit as R6
    set_determinism()
    m = load_r6(Path(run))
    _, Qa = analysis.das(m, "none", "a", 16)
    r = classify_r6(m, *R6.operand_shift_edit(m, Qa))
    cfg = json.loads((Path(run) / "config.json").read_text())
    return dict(model=Path(run).name, condition=cfg["condition"], **r)


# ------------------------------------------------------------------ report
def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def main():
    dim_runs = sorted(str(p) for p in (ROOT / "runs" / "zoo_dim").iterdir() if (p / "model.pt").exists())
    r6_runs = [str(ROOT / "runs" / "zoo_untyped_inf" / f"{c}_s{s}") for c in ("none", "aux_only") for s in range(5)]
    r6_runs += [str(ROOT / "runs" / "zoo_iit" / f"iit_s{s}") for s in range(5)]
    with ProcessPoolExecutor(4) as ex:
        dim = list(ex.map(analyse_dim, dim_runs))
        r6 = list(ex.map(analyse_r6, r6_runs))
    (ROOT / "results" / "edit_specificity.json").write_text(json.dumps(dict(dim=dim, r6=r6), indent=2))

    L = ["# Variable-specific (additive) edit test, post hoc (auto-generated by edit_specificity.py)", "",
         "Scored only on the discriminating set, where intended target, wrong-variable target and "
         "original answer are pairwise distinct. Chance for any one target = 1/13 = 0.077.", "",
         "## Dimensional task, DAS-located (blind) edits, all 10 seeds per arm", "",
         "| arm | M+1: intended | M+1: wrong (A+1) | M+1: unchanged | **M+1 specificity** | "
         "A+1: intended | A+1: wrong (M+1) | **A+1 specificity** |", "|---|---|---|---|---|---|---|---|"]
    for a in D.ARMS:
        g = [r for r in dim if r["arm"] == a]
        L.append(f"| {a} | {mean(r['M_das_p_intended'] for r in g):.3f} | {mean(r['M_das_p_wrong'] for r in g):.3f} | "
                 f"{mean(r['M_das_p_unchanged'] for r in g):.3f} | **{mean(r['M_das_specificity'] for r in g):+.3f}** | "
                 f"{mean(r['A_das_p_intended'] for r in g):.3f} | {mean(r['A_das_p_wrong'] for r in g):.3f} | "
                 f"**{mean(r['A_das_specificity'] for r in g):+.3f}** |")
    L += ["", "## Dimensional task, converged seeds only (acc ≥ 0.95), DAS-located", "",
          "| arm | n | M+1 intended | M+1 wrong | M+1 specificity | A+1 intended | A+1 wrong | A+1 specificity |",
          "|---|---|---|---|---|---|---|---|"]
    for a in D.ARMS:
        g = [r for r in dim if r["arm"] == a and r["converged"]]
        if not g:
            continue
        L.append(f"| {a} | {len(g)} | {mean(r['M_das_p_intended'] for r in g):.3f} | {mean(r['M_das_p_wrong'] for r in g):.3f} | "
                 f"{mean(r['M_das_specificity'] for r in g):+.3f} | {mean(r['A_das_p_intended'] for r in g):.3f} | "
                 f"{mean(r['A_das_p_wrong'] for r in g):.3f} | {mean(r['A_das_specificity'] for r in g):+.3f} |")
    L += ["", "## Dimensional task, M+1 specificity by localization method (all seeds)", "",
          "| arm | DAS (blind) | class-mean (blind) | designed block |", "|---|---|---|---|"]
    for a in D.ARMS:
        g = [r for r in dim if r["arm"] == a]
        L.append(f"| {a} | {mean(r['M_das_specificity'] for r in g):+.3f} | {mean(r['M_cm_specificity'] for r in g):+.3f} | "
                 f"{mean(r['M_block_specificity'] for r in g):+.3f} |")
    L += ["", "## Dimensional task, INVERSION edit (var -> var^-1), discriminating & multiplicative-code-expressible", "",
          "| arm | n | M⁻¹: intended | M⁻¹: wrong (A⁻¹) | M⁻¹: unchanged | **M⁻¹ specificity** | A⁻¹: intended | A⁻¹: wrong | **A⁻¹ specificity** | M⁻¹ spec. (designed block) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for conv in (False, True):
        if conv:
            L.append("| *converged seeds only* | | | | | | | | | |")
        for a in D.ARMS:
            g = [r for r in dim if r["arm"] == a and (r["converged"] or not conv)]
            if not g:
                continue
            L.append(f"| {a} | {len(g)} | {mean(r['inv_M_das_p_intended'] for r in g):.3f} | {mean(r['inv_M_das_p_wrong'] for r in g):.3f} | "
                     f"{mean(r['inv_M_das_p_unchanged'] for r in g):.3f} | **{mean(r['inv_M_das_specificity'] for r in g):+.3f}** | "
                     f"{mean(r['inv_A_das_p_intended'] for r in g):.3f} | {mean(r['inv_A_das_p_wrong'] for r in g):.3f} | "
                     f"**{mean(r['inv_A_das_specificity'] for r in g):+.3f}** | {mean(r['inv_M_block_specificity'] for r in g):+.3f} |")
    L += ["", "## R6 (modular arithmetic), a → a+1 edit, DAS-located; ADD rows excluded (a+1 ≡ b+1 there)", "",
          "| condition | n | old score, all rows (doc 09) | intended | wrong (b+1) | unchanged | **specificity** |",
          "|---|---|---|---|---|---|---|"]
    for c in ("none", "aux_only", "iit"):
        g = [r for r in r6 if r["condition"] == c]
        L.append(f"| {c} | {len(g)} | {mean(r['old_success_all_rows'] for r in g):.3f} | {mean(r['p_intended'] for r in g):.3f} | "
                 f"{mean(r['p_wrong'] for r in g):.3f} | {mean(r['p_unchanged'] for r in g):.3f} | "
                 f"**{mean(r['specificity'] for r in g):+.3f}** |")
    (ROOT / "results" / "edit_specificity.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
