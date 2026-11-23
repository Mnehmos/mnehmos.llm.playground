"""Permanent-edit study (registration: docs/18; results: docs/19).

docs/17 showed that a unit-aware STORED TABLE (the network's own activation picks a stored
adjustment on each question) edits the declared variable almost as well as copying a real
pattern, while a single LINEAR weight change gains only a little. This study asks:

  Q1  Can the unit-aware edit be INSTALLED ONCE, as a small added layer that implements the
      table (no outside information, no per-question lookup outside the network)?
  Q2  Can it be installed WITHIN THE ORIGINAL NETWORK, by changing only existing weights?

Design (fixed before analysis):
  networks   NEW: 24 networks, mass_only / acc_only x seeds 400-411, trained with the frozen
             docs/14 procedure (train_perm.py). Never used for development.
  questions  the 7,350 distinct in-distribution questions are split ONCE (seed RESERVE_S):
             30% are RESERVED for evaluation and never used by the location search, any
             editor fit, distillation or tuning. Each reserved question is scored once.
  location   DAS subspace (k=16) searched on non-reserved questions (same algorithm as docs/15)
  scorer     run_swap.score (frozen: intended / wrong-variable / unchanged / other)
  edit       inversion mod 13 of the DECLARED variable

Editors (all fitted on the same non-reserved fitting draw; route = unit x position):
  current     docs/15 editor: pooled affine map in the subspace, folded into W_in
  linear      docs/17 unit-aware linear weight edit (full-activation input), folded into W_in
  table_pooled / table   docs/17 stored tables (nearest centroid in the subspace)
  layer       ADDED LAYER implementing the unit-aware table: a softmax key-value layer with one
              unit per (value, route) cluster; keys = cluster centroids, values = the stored
              moves; written once, reads only the activation. Sharpness beta locked in docs/18.
  weights     EXISTING WEIGHTS ONLY: the post-site weights (W_in, b_in, W_out, b_out, W_U) are
              trained to reproduce the table-edited network's outputs (KL), on non-reserved
              questions only. No new units. Steps / lr locked in docs/18.
  per_question, copy   references (need the question's value and unit, or a real source)

    python run_perm.py dev       # tuning on the OLD docs/15 networks, non-reserved questions
    python run_perm.py analyse   # registered run on the NEW networks, reserved questions
    python run_perm.py report
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

import run_dimensional as D
import run_editors as E
import run_swap as S
from nri import dimtask as T
from nri.train import set_determinism

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "perm"
ZOO_NEW = ROOT / "runs" / "zoo_perm"
ZOO_OLD = ROOT / "runs" / "zoo_swap"
SEEDS_NEW = list(range(400, 412))
SEEDS_OLD = list(range(300, 312))
GROUPS = ["mass_only", "acc_only"]
VAR_OF = {"mass_only": "M", "acc_only": "A"}
KIND = "inv"
RESERVE_S, RESERVE_FRAC = 13_131, 0.30
FIT_S2, PAIR_S2, DAS_S2, DISTILL_S, DEVCHK_S = 24_242, 35_353, 46_464, 57_575, 68_686
N_FIT, N_PAIR, N_DISTILL, N_DEVCHK = 8192, 8192, 65_536, 8192
# locked in docs/18 from the dev run (tuning on OLD networks, non-reserved questions only)
BETA = 100
DISTILL_STEPS = 3000
DISTILL_LR = 1e-3
BETA_GRID = (1, 3, 10, 30, 100)
DISTILL_GRID = ((1000, 1e-3), (1000, 3e-3), (3000, 1e-3), (3000, 3e-3))
KEYS = ("m", "mu", "a", "au", "fu", "order")


# ------------------------------------------------------------------ question split
def all_questions():
    rows = [(m, mu, a, au, fu, o)
            for m in range(T.P) for mu in range(3) for a in range(T.P) for au in range(3)
            for fu in range(3) for o in range(2)
            if not (mu == T.OOD_UNIT and m >= T.OOD_VALUES[0])
            and not (au == T.OOD_UNIT and a >= T.OOD_VALUES[0])]
    assert len(rows) == 7350
    return rows


def reserved_set():
    q = all_questions()
    g = torch.Generator().manual_seed(RESERVE_S)
    perm = torch.randperm(len(q), generator=g).tolist()
    return {q[i] for i in perm[: int(round(RESERVE_FRAC * len(q)))]}


def as_lat(rows):
    cols = list(zip(*rows))
    return {k: torch.tensor(c) for k, c in zip(KEYS, cols)}


def keys(lat):
    return list(zip(*[lat[k].tolist() for k in KEYS]))


def pool_lats(n, seed, reserved):
    """n draws from the task distribution, rejecting reserved questions (non-reserved pool)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    while len(out) < n:
        lat = T.sample_latents(4 * n, g)
        out += [k for k in keys(lat) if k not in reserved]
    return as_lat(out[:n])


def eval_lat(reserved):
    """Every reserved question once, in a fixed order."""
    return as_lat(sorted(reserved))


# ------------------------------------------------------------------ location (DAS, docs/15 algorithm)
def das_basis(m, var, reserved, k=16, steps=400):
    torch.manual_seed(DAS_S2)
    base, src = pool_lats(N_PAIR, PAIR_S2, reserved), pool_lats(N_PAIR, PAIR_S2 + 1, reserved)
    with torch.no_grad():
        _, hb = D.site(m, base)
        _, hs = D.site(m, src)
    y = D.counterfactual(base, src, var)
    rot = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(64, k, bias=False))
    opt = torch.optim.Adam(rot.parameters(), lr=1e-2)
    for p in m.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(DAS_S2)
    for _ in range(steps):
        i = torch.randint(0, N_PAIR, (1024,), generator=g)
        Q = rot.weight.T
        loss = F.cross_entropy(m.head_from_resid(hb[i] + (hs[i] - hb[i]) @ Q @ Q.T), y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in m.parameters():
        p.requires_grad_(True)
    return rot.weight.T.detach().clone()


# ------------------------------------------------------------------ the added layer
class TableLayer:
    """Softmax key-value layer implementing the unit-aware table: one unit per valid
    (value, route) cluster. score_k = -beta * ||z - c_k||^2 / s2  (z = subspace coordinates,
    c_k = cluster centroid, s2 = mean squared within-cluster distance on the fitting data),
    which is linear in z up to a term shared by all units, so it is a standard layer
    (keys 2*beta*c_k/s2, biases -beta*||c_k||^2/s2) followed by softmax. Output: the
    softmax-weighted stored moves, added to the activation. As beta grows it equals the
    hard nearest-centroid table."""

    def __init__(self, tab, B, h_fit, y_fit, r_fit, beta):
        nr = tab.mu_c.shape[1]
        valid = tab.ok.reshape(-1)
        self.C = (tab.mu_c.reshape(T.P * nr, -1) @ B)[valid]
        moves = (tab.tgt - torch.where(tab.ok.unsqueeze(-1), tab.mu_c, tab.mu_p[:, None]))
        self.V = (moves.reshape(T.P * nr, -1) @ B @ B.T)[valid]
        z = h_fit @ B
        own = (tab.mu_c[y_fit, r_fit] @ B)
        self.s2 = ((z - own) ** 2).sum(1).mean()
        self.B, self.beta = B, beta
        self.n_units = int(valid.sum())

    def __call__(self, h):
        z = h @ self.B
        s = -self.beta * torch.cdist(z, self.C) ** 2 / self.s2
        return h + torch.softmax(s, 1) @ self.V


# ------------------------------------------------------------------ existing-weights edit (distillation)
def distill(m, teacher_logits, h, steps, lr, seed=DISTILL_S):
    """Train ONLY the existing post-site weights (W_in, b_in, W_out, b_out, W_U) so the
    unedited activation produces the table-edited network's outputs. No new units."""
    P = {k: v.detach().clone().requires_grad_(True) for k, v in
         dict(Wi=m.W_in.weight, bi=m.W_in.bias, Wo=m.W_out.weight, bo=m.W_out.bias,
              Wu=m.W_U.weight).items()}
    opt = torch.optim.Adam(P.values(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    tgt = torch.log_softmax(teacher_logits, 1)
    for _ in range(steps):
        i = torch.randint(0, len(h), (512,), generator=g)
        out = student(P, h[i])
        loss = F.kl_div(torch.log_softmax(out, 1), tgt[i], log_target=True, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: v.detach() for k, v in P.items()}


def student(P, h):
    return (h + (torch.relu(h @ P["Wi"].T + P["bi"]) @ P["Wo"].T + P["bo"])) @ P["Wu"].T


def table_logits(m, h, tab, B):
    c_p, (c_r, q_r) = E.table_lookup(h, B, tab, "sub")
    return m.head_from_resid(h + tab.move_route(c_r, q_r) @ B @ B.T)


# ------------------------------------------------------------------ one network
def build(m, var, reserved):
    """Everything fitted on NON-reserved questions only."""
    Q = das_basis(m, var, reserved)
    with torch.no_grad():
        lat_f = pool_lats(N_FIT, FIT_S2, reserved)
        _, h_f = D.site(m, lat_f)
        y_f = D.label(lat_f, var)
        r_f = E.route(lat_f, var, "uo")
        tabs = {k: E.Tables(h_f, y_f, lat_f, var, k) for k in ("uo",)}
        lat_d = pool_lats(N_DISTILL, DISTILL_S, reserved)
        _, h_d = D.site(m, lat_d)
    return Q, lat_f, h_f, y_f, r_f, tabs["uo"], h_d


def dev(args):
    """Tuning on an OLD network (docs/15 zoo), non-reserved questions only. Criterion:
    agreement with the hard table editor's predictions on a separate non-reserved check
    draw. The criterion never uses reserved questions or intended-answer rates."""
    run, = args
    set_determinism()
    m, cfg = D.load(run)
    var = VAR_OF[cfg["group"]]
    reserved = reserved_set()
    Q, lat_f, h_f, y_f, r_f, tab, h_d = build(m, var, reserved)
    out = dict(model=Path(run).name, group=cfg["group"], seed=cfg["seed"])
    with torch.no_grad():
        lat_c = pool_lats(N_DEVCHK, DEVCHK_S, reserved)
        _, h_c = D.site(m, lat_c)
        teach_c = table_logits(m, h_c, tab, Q).argmax(-1)
        for beta in BETA_GRID:
            lay = TableLayer(tab, Q, h_f, y_f, r_f, beta)
            out[f"layer_beta{beta}"] = (m.head_from_resid(lay(h_c)).argmax(-1) == teach_c).float().mean().item()
        teach_d = table_logits(m, h_d, tab, Q)
    for steps, lr in DISTILL_GRID:
        P = distill(m, teach_d, h_d, steps, lr)
        with torch.no_grad():
            out[f"distill_{steps}_{lr:g}"] = (student(P, h_c).argmax(-1) == teach_c).float().mean().item()
    return out


def analyse(args):
    run, = args
    set_determinism()
    m, cfg = D.load(run)
    var = VAR_OF[cfg["group"]]
    reserved = reserved_set()
    Q, lat_f, h_f, y_f, r_f, tab, h_d = build(m, var, reserved)
    out = dict(model=Path(run).name, group=cfg["group"], seed=cfg["seed"], var=var,
               acc_id=cfg["acc_id"], converged=cfg["acc_id"] >= S.CONV_ACC)
    with torch.no_grad():
        teach_d = table_logits(m, h_d, tab, Q)
    P = distill(m, teach_d, h_d, DISTILL_STEPS, DISTILL_LR)
    with torch.no_grad():
        lat = eval_lat(reserved)
        _, h = D.site(m, lat)
        sc = lambda logits: S.score(logits.argmax(-1), lat, var, KIND)
        mv_pool_f = tab.move_pooled(y_f)
        mv_route_f = tab.move_route(y_f, r_f)
        lay = TableLayer(tab, Q, h_f, y_f, r_f, BETA)
        preds = {
            "current": E.apply_weight(m, h, *E.fit_weight(m, Q, h_f, mv_pool_f, False)),
            "linear": E.apply_weight(m, h, *E.fit_weight(m, Q, h_f, mv_route_f, True)),
            "table_pooled": E.apply_move(m, h, tab.move_pooled(E.table_lookup(h, Q, tab, "sub")[0]), Q),
            "table": table_logits(m, h, tab, Q),
            "layer": m.head_from_resid(lay(h)),
            "weights": student(P, h),
            "per_question": E.apply_move(m, h, tab.move_route(D.label(lat, var), E.route(lat, var, "uo")), Q),
        }
        src = S.source_for(lat, var, KIND)
        _, hs = D.site(m, src)
        preds["copy"] = m.head_from_resid(h + (hs - h) @ Q @ Q.T)
        # preservation: reserved questions whose intended edited answer EQUALS the original
        # answer (e.g. V in {0, 1, 12}); a good edit leaves these answers unchanged
        F0, FM, FA = S.targets(lat, KIND)
        same = (FM == F0) if var == "M" else (FA == F0)
        orig = m.head_from_resid(h).argmax(-1)
        for k, lg in preds.items():
            r = sc(lg)
            p = lg.argmax(-1)
            r["preserve"] = (p[same] == F0[same]).float().mean().item()
            r["n_preserve"] = int(same.sum())
            r["orig_correct_on_preserve"] = (orig[same] == F0[same]).float().mean().item()
            out[k] = r
        out["layer_units"] = lay.n_units
        out["weights_rel_change"] = {k: ((P[k] - w).norm() / w.norm()).item() for k, w in
                                     dict(Wi=m.W_in.weight, bi=m.W_in.bias, Wo=m.W_out.weight,
                                          bo=m.W_out.bias, Wu=m.W_U.weight).items()}
        out["table_agreement"] = {k: (preds[k].argmax(-1) == preds["table"].argmax(-1)).float().mean().item()
                                  for k in ("layer", "weights", "linear")}
    return out


# ------------------------------------------------------------------ report
def report():
    res = json.loads((OUT / "metrics.json").read_text())
    R = {(r["group"], r["seed"]): r for r in res}
    missing = [(g, s) for g in GROUPS for s in SEEDS_NEW if (g, s) not in R]
    assert not missing, f"missing networks: {missing}"
    POP = [("mass declared", "mass_only"), ("acceleration declared", "acc_only")]
    get = lambda g, s, e, k: R[(g, s)][e][k]
    L = ["# Permanent-edit study (auto-generated by `run_perm.py report`; analysis frozen in docs/18)", "",
         f"New networks (seeds {SEEDS_NEW[0]}-{SEEDS_NEW[-1]}); reserved questions only, each scored once "
         f"({get('mass_only', SEEDS_NEW[0], 'table', 'n_eligible')} eligible per network).", ""]

    def line(name, d):
        lo, hi = S.boot_ci(d)
        p = S.paired_signflip(d)
        return f"| {name} | {st.mean(d):+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.4f} | {sum(x > 0 for x in d)}/{len(d)} |", p

    fam_def = [("replication: unit-aware table − pooled table", "table_pooled", "table"),
               ("Q1: added layer − linear weight edit", "linear", "layer"),
               ("Q2: existing-weights edit − linear weight edit", "linear", "weights")]
    L += ["## Primary family (Holm, 6 tests): Δ intended, paired by network, all 12 networks", "",
          "| comparison | population | mean Δ | 95% CI | p | networks > 0 |", "|---|---|---|---|---|---|"]
    fam = []
    for name, a, b in fam_def:
        for pn, g in POP:
            d = [get(g, s, b, "intended") - get(g, s, a, "intended") for s in SEEDS_NEW]
            txt, p = line(f"{name} | {pn}", d)
            L.append(txt)
            fam.append((p, f"{name} / {pn}"))
    order = sorted(range(len(fam)), key=lambda i: fam[i][0])
    rej, stop = {}, False
    for rank, i in enumerate(order):
        ok = (not stop) and fam[i][0] <= 0.05 / (len(fam) - rank)
        stop = stop or not ok
        rej[i] = ok
    L += ["", "Holm: " + "; ".join(f"{fam[i][1]}: {'rejects' if rej[i] else 'does not reject'}" for i in range(len(fam))), ""]
    L += ["## Wrong-variable Δ for the same pairs (a significant rise makes a gain non-specific)", "",
          "| comparison | population | mean Δ wrong | 95% CI | p | networks > 0 |", "|---|---|---|---|---|---|"]
    for name, a, b in fam_def:
        for pn, g in POP:
            L.append(line(f"{name} | {pn}", [get(g, s, b, "wrong") - get(g, s, a, "wrong") for s in SEEDS_NEW])[0])
    L += ["", "## Gap to the stored table (installed − table; 'matches' = mean ≥ −0.05 and CI lower bound ≥ −0.10)", "",
          "| installed edit | population | mean gap | 95% CI | p | networks > 0 | matches the table? |", "|---|---|---|---|---|---|---|"]
    for e, nm in (("layer", "added layer"), ("weights", "existing weights")):
        for pn, g in POP:
            d = [get(g, s, e, "intended") - get(g, s, "table", "intended") for s in SEEDS_NEW]
            lo, hi = S.boot_ci(d)
            L.append(line(f"{nm} | {pn}", d)[0][:-1] + f" | {'yes' if st.mean(d) >= -0.05 and lo >= -0.10 else 'no'} |")
    ED = [("current editor (docs/15)", "current"), ("unit-aware linear weight edit (docs/17)", "linear"),
          ("pooled table", "table_pooled"), ("unit-aware table", "table"), ("**added layer**", "layer"),
          ("**existing weights only**", "weights"), ("per question (reference)", "per_question"),
          ("copy (benchmark)", "copy")]
    L += ["", "## Outcomes (mean over all 12 networks per population)", "",
          "| editor | population | intended | wrong | unchanged | other | preserve |", "|---|---|---|---|---|---|---|"]
    for nm, e in ED:
        for pn, g in POP:
            f = lambda k: st.mean(get(g, s, e, k) for s in SEEDS_NEW)
            L.append(f"| {nm} | {pn} | {f('intended'):.3f} | {f('wrong'):.3f} | {f('unchanged'):.3f} | "
                     f"{f('other'):.3f} | {f('preserve'):.3f} |")
    L += ["", "'preserve' = share of reserved questions whose intended edited answer equals the original answer "
          "that keep the original answer (original network: " + ", ".join(
              f"{pn} {st.mean(get(g, s, 'table', 'orig_correct_on_preserve') for s in SEEDS_NEW):.3f}" for pn, g in POP) + ").", ""]
    L += ["## Per network (intended / wrong)", "",
          "| group | seed | converged | current | linear | table | added layer | existing weights | copy |",
          "|---|---|---|---|---|---|---|---|---|"]
    for pn, g in POP:
        for s in SEEDS_NEW:
            c = lambda e: f"{get(g, s, e, 'intended'):.2f} / {get(g, s, e, 'wrong'):.2f}"
            L.append(f"| {g} | {s} | {'y' if R[(g, s)]['converged'] else 'n'} | " +
                     " | ".join(c(e) for e in ("current", "linear", "table", "layer", "weights", "copy")) + " |")
    L += ["", "## Converged vs struggling networks (mean intended)", "",
          "| population | networks | n | current | linear | table | added layer | existing weights | copy |",
          "|---|---|---|---|---|---|---|---|---|"]
    for pn, g in POP:
        for lab, cond in (("converged", True), ("struggling", False)):
            ss = [s for s in SEEDS_NEW if R[(g, s)]["converged"] == cond]
            if ss:
                L.append(f"| {pn} | {lab} | {len(ss)} | " + " | ".join(
                    f"{st.mean(get(g, s, e, 'intended') for s in ss):.2f}"
                    for e in ("current", "linear", "table", "layer", "weights", "copy")) + " |")
    L += ["", "## Agreement with the table's answers on all reserved questions; size of the existing-weights change", "",
          "| population | added layer | existing weights | linear | rel. change W_in | W_out | W_U |", "|---|---|---|---|---|---|---|"]
    for pn, g in POP:
        a = lambda k: st.mean(R[(g, s)]["table_agreement"][k] for s in SEEDS_NEW)
        w = lambda k: st.mean(R[(g, s)]["weights_rel_change"][k] for s in SEEDS_NEW)
        L.append(f"| {pn} | {a('layer'):.3f} | {a('weights'):.3f} | {a('linear'):.3f} | {w('Wi'):.3f} | {w('Wo'):.3f} | {w('Wu'):.3f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("dev", "analyse", "report"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "report":
        report()
        return
    if a.mode == "dev":
        runs = [(str(ZOO_OLD / f"{g}_s{s}"),) for s in SEEDS_OLD for g in GROUPS]
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(dev, runs))
        (OUT / "dev.json").write_text(json.dumps(res, indent=1))
        for k in [k for k in res[0] if k.startswith(("layer_", "distill_"))]:
            print(k, " ".join(f"{g}={st.mean(r[k] for r in res if r['group'] == g):.4f}" for g in GROUPS),
                  f"all={st.mean(r[k] for r in res):.4f}")
        return
    assert BETA is not None and DISTILL_STEPS is not None, "lock BETA / DISTILL_* in docs/18 first"
    runs = [(str(ZOO_NEW / f"{g}_s{s}"),) for s in SEEDS_NEW for g in GROUPS]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(analyse, runs))
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))
    print(f"analysed {len(res)} networks -> {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
