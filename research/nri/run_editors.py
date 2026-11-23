"""Editor comparison on FIXED networks (registration: docs/16; results: docs/17).

Question: the role-swap networks (docs/14-15) carry the declared variable in a usable place
(copying a real pattern works 0.8-0.9), but the fitted editor reaches only 0.2-0.3. Does an
editor that accounts for HOW the value was expressed (its unit and position: its "route") build
better replacement patterns than the current editor, which learns one pooled adjustment?

Nothing about the networks, the target locations or the scorer changes:
  networks   runs/zoo_swap/* (36 networks, seeds 300-311), loaded read-only
  locations  the frozen DAS subspace (run_swap.das_basis, primary) and the known block (secondary)
  scorer     run_swap.score (intended / wrong-variable / unchanged / other; frozen eligible set)
  fitting    the same 8,192 fitting examples (run_swap.FIT_S) for every editor
  test       a FRESH evaluation sample (TEST_S below), never used before this registration

Editors come in pooled / route-aware pairs that differ ONLY in whether they use the route.
Three tiers, reported separately because they are different capabilities:
  tier 1  weight edit    one change folded into existing weights (W_in); nothing at run time
  tier 2  stored table   a fixed table added once; the network's own activation picks the entry
  tier 3  per-question   uses each question's known value and unit/position at run time
Benchmark (tier 3): copy the real pattern of a source example (run_swap.copy_test, frozen).

    python run_editors.py --split dev    # development sample (docs/16 discloses it)
    python run_editors.py --split test   # registered run, fresh sample
    python run_editors.py --report            # (--split dev --report: dev check of the report code)
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

import run_dimensional as D
import run_swap as S
from nri import dimtask as T
from nri.train import set_determinism

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "editors"
ZOO = ROOT / "runs" / "zoo_swap"
SEEDS = list(range(300, 312))
GROUPS = S.GROUPS
DEV_S = 91_919          # development sample (used before registration; disclosed)
TEST_S = 72_727         # FRESH registered evaluation sample (never used before registration)
N_EVAL = 16_384
MIN_CLUSTER = 5         # a (value, route) cluster needs >= this many fitting examples
KIND = "inv"            # primary edit (docs/14): inversion mod 13, 0 -> 0
TABLE_SPACE = "sub"     # tier-2 lookup space; locked in docs/16 by the dev classification rule


# ------------------------------------------------------------------ routes
def route(lat, var, key):
    """How the value of `var` was expressed. key 'uo': (unit, position) -> 6 routes (primary);
    'u': unit only -> 3; 'o': position only -> 2."""
    u = lat["mu"] if var == "M" else lat["au"]
    o = lat["order"]
    return {"uo": u * 2 + o, "u": u, "o": o}[key]


N_ROUTES = {"uo": 6, "u": 3, "o": 2}


def route_order(q, key):
    return {"uo": q % 2, "u": None, "o": q}[key]


class Tables:
    """Class means (pooled) and (value, route) cluster means, from the FITTING examples only."""

    def __init__(self, h, y, lat, var, key):
        self.key, nr = key, N_ROUTES[key]
        r = route(lat, var, key)
        self.mu_p = D.class_means(h, y, T.P)
        self.cnt = torch.zeros(T.P, nr)
        self.mu_c = torch.zeros(T.P, nr, h.shape[1])
        for c in range(T.P):
            for q in range(nr):
                s = (y == c) & (r == q)
                self.cnt[c, q] = s.sum()
                if s.sum() >= MIN_CLUSTER:
                    self.mu_c[c, q] = h[s].mean(0)
        self.ok = self.cnt >= MIN_CLUSTER
        f = S.EDITS[KIND]
        # target center for (value c, route q): the edited value's cluster on the SAME route; if
        # that cluster does not exist in training (held-out unit-value combination) or is too
        # small, the mean of the edited value's available clusters with the same position.
        self.tgt = torch.zeros_like(self.mu_c)
        for c in range(T.P):
            tc = int(f(torch.tensor(c)))
            for q in range(nr):
                if self.ok[tc, q]:
                    self.tgt[c, q] = self.mu_c[tc, q]
                else:
                    o = route_order(q, key)
                    alts = [q2 for q2 in range(nr) if self.ok[tc, q2]
                            and (o is None or route_order(q2, key) == o)]
                    self.tgt[c, q] = (torch.stack([self.mu_c[tc, q2] for q2 in alts]).mean(0)
                                      if alts else self.mu_p[tc])

    def move_pooled(self, y):
        return self.mu_p[S.EDITS[KIND](y)] - self.mu_p[y]

    def move_route(self, y, q):
        src = torch.where(self.ok[y, q].unsqueeze(1), self.mu_c[y, q], self.mu_p[y])
        return self.tgt[y, q] - src


# ------------------------------------------------------------------ tier 1: weight edits
def fold(m, B, W, m0, wide):
    Dm = B @ W[:-1].T @ (torch.eye(64) if wide else B.T)
    shift = B @ W[-1] - Dm @ m0
    return m.W_in.weight @ Dm, m.W_in.weight @ shift


def fit_weight(m, B, h, move, wide):
    """Least-squares affine map, output inside span(B), input = the subspace coordinates (narrow)
    or the full activation (wide), folded into W_in as (dW, db)."""
    m0 = h.mean(0)
    X = (h - m0) if wide else (h - m0) @ B
    X = torch.cat([X, torch.ones(len(X), 1)], 1)
    W = torch.linalg.lstsq(X, move @ B).solution
    return fold(m, B, W, m0, wide)


def apply_weight(m, h, dW, db):
    return m.head_from_resid(h, m.W_in.weight + dW, m.W_in.bias + db)


# ------------------------------------------------------------------ tiers 2-3: activation edits
def apply_move(m, h, move, B):
    """Add `move`, projected onto span(B), to the activation (same write path as copy_test)."""
    return m.head_from_resid(h + move @ B @ B.T)


def nearest(x, C, valid):
    d = torch.cdist(x, C)
    d[:, ~valid] = float("inf")
    return d.argmin(1)


def table_lookup(h, B, tab, space):
    """Tier 2: identify the example's (value) or (value, route) cluster from its OWN activation
    (nearest fitted centroid), then return the stored pooled and route moves for that entry."""
    proj = (lambda x: x) if space == "full" else (lambda x: x @ B)
    nr = tab.mu_c.shape[1]
    c_p = nearest(proj(h), proj(tab.mu_p), torch.ones(T.P, dtype=torch.bool))
    flat = tab.mu_c.reshape(T.P * nr, -1)
    idx = nearest(proj(h), proj(flat), tab.ok.reshape(-1))
    c_r, q_r = idx // nr, idx % nr
    return c_p, (c_r, q_r)


# ------------------------------------------------------------------ one network
def analyse(args):
    run, split = args
    set_determinism()
    m, cfg = D.load(run)
    out = dict(model=Path(run).name, group=cfg["group"], seed=cfg["seed"], acc_id=cfg["acc_id"],
               converged=cfg["acc_id"] >= S.CONV_ACC)
    eval_seed = TEST_S if split == "test" else DEV_S
    for var in ("M", "A"):
        Q = S.das_basis(m, var)                       # frozen location (same seeds as docs/15)
        with torch.no_grad():
            lat_f = S.lats(8192, S.FIT_S)
            _, h_f = D.site(m, lat_f)
            y_f = D.label(lat_f, var)
            tabs = {k: Tables(h_f, y_f, lat_f, var, k) for k in ("uo", "u", "o")}
            mv_pool_f = tabs["uo"].move_pooled(y_f)
            mv_route_f = tabs["uo"].move_route(y_f, route(lat_f, var, "uo"))
            lat = S.lats(N_EVAL, eval_seed)
            _, h = D.site(m, lat)
            y = D.label(lat, var)
            sc = lambda logits: S.score(logits.argmax(-1), lat, var, KIND)
            for loc, B in (("das", Q), ("block", torch.eye(64)[:, S.BLOCK[0]:S.BLOCK[1]])):
                R = {}
                # tier 1 -- weight edits (folded into W_in once)
                R["w_current"] = sc(apply_weight(m, h, *S.fit_edit(m, B, var, KIND)))  # frozen
                R["w_pooled"] = sc(apply_weight(m, h, *fit_weight(m, B, h_f, mv_pool_f, True)))
                R["w_route"] = sc(apply_weight(m, h, *fit_weight(m, B, h_f, mv_route_f, True)))
                # tier 2 -- stored table, entry chosen by the network's own activation
                tab = tabs["uo"]
                for space in ("full", "sub"):
                    c_p, (c_r, q_r) = table_lookup(h, B, tab, space)
                    R[f"t_pooled_{space}"] = sc(apply_move(m, h, tab.move_pooled(c_p), B))
                    R[f"t_route_{space}"] = sc(apply_move(m, h, tab.move_route(c_r, q_r), B))
                    q_true = route(lat, var, "uo")
                    R[f"t_lookup_{space}"] = dict(
                        class_acc_pooled=(c_p == y).float().mean().item(),
                        class_acc_route=(c_r == y).float().mean().item(),
                        cluster_acc_route=((c_r == y) & (q_r == q_true)).float().mean().item())
                # tier 3 -- per-question (uses the question's known value and route)
                R["q_pooled"] = sc(apply_move(m, h, tab.move_pooled(y), B))
                for k in ("uo", "u", "o"):
                    R[f"q_route_{k}"] = sc(apply_move(m, h, tabs[k].move_route(y, route(lat, var, k)), B))
                # benchmark: copy a real source pattern (frozen construction, fresh sample)
                src = S.source_for(lat, var, KIND)
                _, hs = D.site(m, src)
                P_ = B @ B.T
                R["copy"] = sc(m.head_from_resid(h + (hs - h) @ P_))
                z = (h + (hs - h) @ P_) @ m.W_in.weight.T + m.W_in.bias  # same path as weight edits
                R["copy_mlp_path"] = sc(m.W_U(h + m.W_out(torch.relu(z))))
                # cluster structure (descriptive): share of within-class variance in span(B)
                # explained by the route
                zf = (h_f - h_f.mean(0)) @ B
                r_f = route(lat_f, var, "uo")
                within = sum(((zf[y_f == c] - zf[y_f == c].mean(0)) ** 2).sum() for c in range(T.P))
                between_route = sum(((zf[(y_f == c) & (r_f == q)].mean(0) - zf[y_f == c].mean(0)) ** 2).sum()
                                    * ((y_f == c) & (r_f == q)).sum()
                                    for c in range(T.P) for q in range(6) if tab.ok[c, q])
                R["route_share_within_class_var"] = (between_route / within).item()
                out[f"{var}_{loc}"] = R
    return out


# ------------------------------------------------------------------ statistics
def report(split="test"):
    res = json.loads((OUT / f"metrics_{split}.json").read_text())
    Rk = {(r["group"], r["seed"]): r for r in res}
    missing = [(g, s) for g in GROUPS for s in SEEDS if (g, s) not in Rk]
    assert not missing, f"missing networks: {missing}"
    space = TABLE_SPACE
    assert space in ("full", "sub"), "TABLE_SPACE must be locked (docs/16) before the report"
    PAIRS = [("1 weight edit", "w_pooled", "w_route"),
             ("2 stored table", f"t_pooled_{space}", f"t_route_{space}"),
             ("3 per-question", "q_pooled", "q_route_uo")]
    POP = [("mass declared", "mass_only", "M"), ("acceleration declared", "acc_only", "A")]
    get = lambda g, s, v, e, k, loc="das": Rk[(g, s)][f"{v}_{loc}"][e][k]
    L = ["# Editor comparison (auto-generated by `run_editors.py --report`; analysis frozen in docs/16)", ""]

    def line(name, d):
        lo, hi = S.boot_ci(d)
        return (f"| {name} | {st.mean(d):+.3f} | [{lo:+.3f}, {hi:+.3f}] | {S.paired_signflip(d):.4f} | "
                f"{sum(x > 0 for x in d)}/{len(d)} |"), S.paired_signflip(d)

    # primary family: route - pooled intended rate, 3 tiers x 2 populations, Holm
    L += ["## Primary family: route-aware − pooled, intended-answer rate (DAS location, all 12 seeds)", "",
          "| tier | population | mean Δ intended | 95% CI | p (exact sign-flip) | seeds > 0 |", "|---|---|---|---|---|---|"]
    fam = []
    for tier, a, b in PAIRS:
        for pname, g, v in POP:
            d = [get(g, s, v, b, "intended") - get(g, s, v, a, "intended") for s in SEEDS]
            txt, p = line(f"{tier} | {pname}", d)
            L.append(txt)
            fam.append((p, tier, pname, st.mean(d)))
    order = sorted(range(len(fam)), key=lambda i: fam[i][0])
    rej, stop = {}, False
    for rank, i in enumerate(order):
        ok = (not stop) and fam[i][0] <= 0.05 / (len(fam) - rank)
        stop = stop or not ok
        rej[i] = ok
    L += ["", "Holm (6 tests, alpha 0.05): " + "; ".join(
        f"{fam[i][1]} / {fam[i][2]}: {'rejects' if rej[i] else 'does not reject'}" for i in range(len(fam))), ""]
    L += ["## Wrong-variable changes: route-aware − pooled (same pairs; a rise would make a gain non-specific)", "",
          "| tier | population | mean Δ wrong | 95% CI | p | seeds > 0 |", "|---|---|---|---|---|---|"]
    for tier, a, b in PAIRS:
        for pname, g, v in POP:
            d = [get(g, s, v, b, "wrong") - get(g, s, v, a, "wrong") for s in SEEDS]
            L.append(line(f"{tier} | {pname}", d)[0])
    # pre-specified secondaries (uncorrected)
    L += ["", "## Secondary tests (uncorrected): intended-answer rate, DAS, all 12 seeds", "",
          "| comparison | population | mean Δ | 95% CI | p | seeds > 0 |", "|---|---|---|---|---|---|"]
    for name, a, b in (("tier 1: route-aware − current (frozen)", "w_current", "w_route"),
                       ("tier 3: unit-only − pooled", "q_pooled", "q_route_u"),
                       ("tier 3: position-only − pooled", "q_pooled", "q_route_o"),
                       ("tier 3: copy − route-aware (gap left)", "q_route_uo", "copy")):
        for pname, g, v in POP:
            d = [get(g, s, v, b, "intended") - get(g, s, v, a, "intended") for s in SEEDS]
            L.append(line(f"{name} | {pname}", d)[0])
    # main table
    L += ["", "## Intended / wrong-variable rates (mean over all 12 seeds, DAS location)", "",
          "| editor | tier | mass declared: intended | wrong | acceleration declared: intended | wrong |",
          "|---|---|---|---|---|---|"]
    ED = [("current (frozen, docs/15)", "1", "w_current"), ("pooled, wide input", "1", "w_pooled"),
          ("route-aware, wide input", "1", "w_route"),
          ("pooled table", "2", f"t_pooled_{space}"), ("route-aware table", "2", f"t_route_{space}"),
          ("pooled, per-question", "3", "q_pooled"), ("route-aware (unit+position), per-question", "3", "q_route_uo"),
          ("unit-only, per-question", "3", "q_route_u"), ("position-only, per-question", "3", "q_route_o"),
          ("**copy real pattern (benchmark)**", "3", "copy"), ("copy, weight-edit write path", "3", "copy_mlp_path")]
    for name, tier, e in ED:
        f = lambda g, v, k: st.mean(get(g, s, v, e, k) for s in SEEDS)
        L.append(f"| {name} | {tier} | {f('mass_only', 'M', 'intended'):.3f} | {f('mass_only', 'M', 'wrong'):.3f} | "
                 f"{f('acc_only', 'A', 'intended'):.3f} | {f('acc_only', 'A', 'wrong'):.3f} |")
    L += ["", "Share of the pooled→copy gap closed by the route-aware editor (ratio of means): " + ", ".join(
        f"tier {t[0]} {pn}: {(st.mean(get(g, s, v, b, 'intended') for s in SEEDS) - st.mean(get(g, s, v, a, 'intended') for s in SEEDS)) / max(1e-9, st.mean(get(g, s, v, 'copy', 'intended') for s in SEEDS) - st.mean(get(g, s, v, a, 'intended') for s in SEEDS)):.2f}"
        for t, a, b in PAIRS for pn, g, v in POP), ""]
    # full outcome breakdown
    L += ["## Full outcome breakdown (DAS; mean over seeds)", "",
          "| population | editor | n eligible | intended | wrong | unchanged | other |", "|---|---|---|---|---|---|---|"]
    for pname, g, v in POP + [("mass, acceleration declared", "acc_only", "M"), ("acceleration, mass declared", "mass_only", "A"),
                               ("mass, neither", "neither", "M"), ("acceleration, neither", "neither", "A")]:
        for name, _, e in ED:
            f = lambda k: st.mean(get(g, s, v, e, k) for s in SEEDS)
            L.append(f"| {pname} | {name} | {int(f('n_eligible'))} | {f('intended'):.3f} | {f('wrong'):.3f} | "
                     f"{f('unchanged'):.3f} | {f('other'):.3f} |")
    # per seed
    L += ["", "## Per network (declared variable, DAS): intended / wrong", "",
          "| group | seed | converged | current | tier 1 pooled | tier 1 route | tier 2 pooled | tier 2 route | tier 3 pooled | tier 3 route | copy |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for pname, g, v in POP:
        for s in SEEDS:
            cell = lambda e: f"{get(g, s, v, e, 'intended'):.2f} / {get(g, s, v, e, 'wrong'):.2f}"
            L.append(f"| {g} | {s} | {'y' if Rk[(g, s)]['converged'] else 'n'} | " + " | ".join(
                cell(e) for e in ("w_current", "w_pooled", "w_route", f"t_pooled_{space}", f"t_route_{space}",
                                  "q_pooled", "q_route_uo", "copy")) + " |")
    # converged vs struggling
    L += ["", "## Converged vs struggling networks (declared variable, DAS, mean intended)", "",
          "| population | networks | n | tier 1 pooled → route | tier 2 pooled → route | tier 3 pooled → route | copy |",
          "|---|---|---|---|---|---|---|"]
    for pname, g, v in POP:
        for lab, cond in (("converged", True), ("struggling", False)):
            ss = [s for s in SEEDS if Rk[(g, s)]["converged"] == cond]
            if not ss:
                continue
            f = lambda e: st.mean(get(g, s, v, e, "intended") for s in ss)
            L.append(f"| {pname} | {lab} | {len(ss)} | " + " | ".join(
                f"{f(a):.2f} → {f(b):.2f}" for _, a, b in PAIRS) + f" | {f('copy'):.2f} |")
    # known block (secondary) and tier-2 lookup accuracy
    L += ["", "## Secondary: known block location (declared variable, mean intended / wrong)", "",
          "| population | current | tier 1 pooled | tier 1 route | tier 2 pooled | tier 2 route | tier 3 pooled | tier 3 route | copy |",
          "|---|---|---|---|---|---|---|---|---|"]
    for pname, g, v in POP:
        cell = lambda e: (f"{st.mean(get(g, s, v, e, 'intended', 'block') for s in SEEDS):.2f} / "
                          f"{st.mean(get(g, s, v, e, 'wrong', 'block') for s in SEEDS):.2f}")
        L.append(f"| {pname} | " + " | ".join(cell(e) for e in ("w_current", "w_pooled", "w_route", f"t_pooled_{space}",
                                                                 f"t_route_{space}", "q_pooled", "q_route_uo", "copy")) + " |")
    L += ["", "## Tier-2 lookup accuracy and route structure (declared variable, DAS)", "",
          "| population | value identified (pooled table) | value identified (route table) | value+route identified | route share of within-value variance |",
          "|---|---|---|---|---|"]
    for pname, g, v in POP:
        f = lambda k: st.mean(get(g, s, v, f"t_lookup_{space}", k) for s in SEEDS)
        L.append(f"| {pname} | {f('class_acc_pooled'):.3f} | {f('class_acc_route'):.3f} | {f('cluster_acc_route'):.3f} | "
                 f"{st.mean(Rk[(g, s)][f'{v}_das']['route_share_within_class_var'] for s in SEEDS):.2f} |")
    (OUT / ("summary.md" if split == "test" else f"summary_{split}.md")).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.report:
        report(a.split or "test")
        return
    runs = [(str(ZOO / f"{g}_s{s}"), a.split) for s in SEEDS for g in GROUPS]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(analyse, runs))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"metrics_{a.split}.json").write_text(json.dumps(res, indent=1))
    print(f"analysed {len(res)} networks -> {OUT / f'metrics_{a.split}.json'}")


if __name__ == "__main__":
    main()
