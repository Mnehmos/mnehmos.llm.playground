"""Role-swap experiment (registration: docs/14; results: docs/15).

Question: does giving a variable its own place in the network (declaring it with interchange
intervention training, IIT) make THAT variable easier to edit?

Groups (same dimensional task as docs/11-12; units and wording unchanged):
  mass_only   IIT declares M (mass, canonical) in block dims 0-15
  acc_only    IIT declares A (acceleration, canonical) in the SAME block dims 0-15
  neither     no IIT
Matching: for a given seed, all three groups share (i) the initialization (torch seed) and
(ii) the base training batches (dedicated generator). The IIT source batches come from a
separate generator, so they never shift the base data stream.

Primary edit: INVERSION mod 13 (V -> V^-1, with 0 -> 0), fitted as an affine map inside the
DAS-located (blind) subspace of V, folded into W_in. Frozen scorer: see score() and docs/14.

    python run_swap.py pilot  --schedules const warmup --seeds 200 201 202 203 204
    python run_swap.py train  --seeds 300 ... 311
    python run_swap.py analyse --seeds 300 ... 311
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

import run_dimensional as D
from nri import dimtask as T
from nri.model import TinyTyped
from nri.train import set_determinism

ROOT = Path(__file__).parent
GROUPS = ["mass_only", "acc_only", "neither"]
VAR_OF = {"mass_only": "M", "acc_only": "A"}
BLOCK = (0, 16)
STEPS, BATCH, LR, WD = 6000, 512, 2e-3, 0.01
LAMBDA = 3.0
WARMUP_STEPS = 2000
SCHEDULE = None  # locked in docs/14 after the pilot; set via --schedule on train
MODEL_KW = D.MODEL_KW
# frozen evaluation seeds (never used before in this repository)
FIT_S, PAIR_S, TEST_S, DAS_S = 51_515, 61_616, 71_717, 81_818
CONV_ACC = 0.95


# ------------------------------------------------------------------ training
def lam_at(step, schedule):
    if schedule == "warmup":
        return LAMBDA * min(1.0, step / WARMUP_STEPS)
    return LAMBDA


def train_one(args):
    group, seed, schedule, zoo = args
    run = Path(zoo) / f"{group}_s{seed}"
    if (run / "model.pt").exists():
        return str(run)
    run.mkdir(parents=True, exist_ok=True)
    set_determinism()
    torch.manual_seed(seed)                                   # matched initialization
    m = TinyTyped(group, **MODEL_KW)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    g_base = torch.Generator().manual_seed(10_000 + seed)     # matched base batches
    g_src = torch.Generator().manual_seed(20_000 + seed)      # IIT sources only
    var = VAR_OF.get(group)
    mk = D.mask(*BLOCK)
    cfg = dict(group=group, arm=group, condition=group, seed=seed, schedule=schedule, steps=STEPS,
               batch=BATCH, lr=LR, wd=WD, lambda_iit=LAMBDA if var else 0.0, declared=var,
               block=BLOCK, warmup_steps=WARMUP_STEPS if schedule == "warmup" else 0,
               n_params=m.n_params(), model_kw=MODEL_KW, torch=torch.__version__,
               encoder_forwards_per_step=2 if var else 1)
    t0 = time.time()
    with open(run / "log.jsonl", "w") as log:
        for step in range(1, STEPS + 1):
            lat_b = T.sample_latents(BATCH, g_base)
            bb = T.build(lat_b)
            _, hb = D.encode(m, bb.tokens)
            loss = F.cross_entropy(m.head_from_resid(hb), bb.y)
            if var:
                lat_s = T.sample_latents(BATCH, g_src)
                _, hs = D.encode(m, T.build(lat_s).tokens)
                iit = F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk),
                                      D.counterfactual(lat_b, lat_s, var))
                loss = loss + lam_at(step, schedule) * iit
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 250 == 0 or step == STEPS:
                log.write(json.dumps(dict(step=step, loss=loss.item(), acc=D.acc(m, n=4096))) + "\n")
                log.flush()
    cfg.update(acc_id=D.acc(m), acc_ood=D.acc(m, "ood"), train_seconds=round(time.time() - t0, 1))
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.save(m.state_dict(), run / "model.pt")
    print(f"[done] {run.name} ({schedule}) id={cfg['acc_id']:.4f} ({cfg['train_seconds']}s)", flush=True)
    return str(run)


# ------------------------------------------------------------------ frozen scorer
INV = torch.tensor([0] + [pow(x, -1, T.P) for x in range(1, T.P)])  # zero maps to zero
EDITS = {"inv": lambda y: INV[y], "add1": lambda y: (y + 1) % T.P}


def lats(n, seed):
    return T.sample_latents(n, torch.Generator().manual_seed(seed))


def targets(lat, kind):
    """Unchanged answer, and the answers if the edit hit M or hit A."""
    M, A, S = T.canon(lat)
    f = EDITS[kind]
    return T.target(M, A, S), T.target(f(M), A, S), T.target(M, f(A), S)


def eligible(lat, kind):
    """Examples where unchanged, M-edited and A-edited answers are pairwise distinct. The SAME
    set serves both variables (the M-edit's wrong answer is the A-edit's intended one)."""
    F0, FM, FA = targets(lat, kind)
    return (F0 != FM) & (F0 != FA) & (FM != FA)


def score(pred, lat, var, kind):
    F0, FM, FA = targets(lat, kind)
    ok = eligible(lat, kind)
    t_int, t_wrong = (FM, FA) if var == "M" else (FA, FM)
    p, n = pred[ok], int(ok.sum())
    r = dict(n_eligible=n,
             intended=(p == t_int[ok]).float().mean().item(),
             wrong=(p == t_wrong[ok]).float().mean().item(),
             unchanged=(p == F0[ok]).float().mean().item())
    r["other"] = 1 - r["intended"] - r["wrong"] - r["unchanged"]
    r["specificity"] = r["intended"] - r["wrong"]
    return r


@torch.no_grad()
def fit_edit(m, basis, var, kind, n=8192):
    """Affine map inside span(basis) sending each class mean of `var` to the class mean of its
    edited value; least squares on FIT data only; folded into W_in (dW, db)."""
    lat = lats(n, FIT_S)
    _, h = D.site(m, lat)
    y = D.label(lat, var)
    mu = D.class_means(h, y, T.P)
    move = mu[EDITS[kind](y)] - mu[y]
    m0 = h.mean(0)
    z = (h - m0) @ basis
    X = torch.cat([z, torch.ones(len(z), 1)], 1)
    W = torch.linalg.lstsq(X, move @ basis).solution
    Dm = basis @ W[:-1].T @ basis.T
    return m.W_in.weight @ Dm, m.W_in.weight @ (basis @ W[-1] - Dm @ m0)


@torch.no_grad()
def eval_edit(m, dW, db, var, kind, n=16384):
    lat = lats(n, TEST_S)
    _, h = D.site(m, lat)
    pred = m.head_from_resid(h, m.W_in.weight + dW, m.W_in.bias + db).argmax(-1)
    return score(pred, lat, var, kind)


def das_basis(m, var, k=16, steps=400, n=8192):
    torch.manual_seed(DAS_S)
    base, src = lats(n, PAIR_S), lats(n, PAIR_S + 1)
    with torch.no_grad():
        _, hb = D.site(m, base)
        _, hs = D.site(m, src)
    y = D.counterfactual(base, src, var)
    rot = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(64, k, bias=False))
    opt = torch.optim.Adam(rot.parameters(), lr=1e-2)
    for p in m.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(DAS_S)
    for _ in range(steps):
        i = torch.randint(0, n, (1024,), generator=g)
        Q = rot.weight.T
        loss = F.cross_entropy(m.head_from_resid(hb[i] + (hs[i] - hb[i]) @ Q @ Q.T), y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in m.parameters():
        p.requires_grad_(True)
    return rot.weight.T.detach().clone()


# ------------------------------------------------------------------ copy diagnostic
def source_for(lat, var, kind):
    """For each example, a REAL input with the same other quantity (value AND unit), the same
    output unit and order, whose `var` equals the edited value. Unit for the edited quantity is
    drawn from in-distribution options (never an OOD unit-value combination)."""
    rng = random.Random(TEST_S)
    out = {k: v.clone() for k, v in lat.items()}
    M, A, _ = T.canon(lat)
    tgt = EDITS[kind](M if var == "M" else A)
    scales = T.MASS_S if var == "M" else T.ACC_S
    vk, uk = ("m", "mu") if var == "M" else ("a", "au")
    for i in range(len(tgt)):
        opts = []
        for u, s in enumerate(scales):
            v = int(tgt[i]) * pow(s, -1, T.P) % T.P
            if not (u == T.OOD_UNIT and v >= T.OOD_VALUES[0]):
                opts.append((v, u))
        v, u = rng.choice(opts)
        out[vk][i], out[uk][i] = v, u
    return out


@torch.no_grad()
def copy_test(m, var, kind, basis=None, n=16384):
    """Copy the real internal pattern of the source example into the base example's block (or,
    with `basis`, into the searched subspace) and score with the frozen scorer."""
    lat = lats(n, TEST_S)
    src = source_for(lat, var, kind)
    _, hb = D.site(m, lat)
    _, hs = D.site(m, src)
    if basis is None:
        mk = D.mask(*BLOCK)
        hp = hb * (1 - mk) + hs * mk
    else:
        hp = hb + (hs - hb) @ basis @ basis.T
    return score(m.head_from_resid(hp).argmax(-1), lat, var, kind)


# ------------------------------------------------------------------ analysis
def analyse(run):
    set_determinism()
    m, cfg = D.load(run)
    out = dict(model=Path(run).name, group=cfg["group"], seed=cfg["seed"], acc_id=cfg["acc_id"],
               acc_ood=cfg["acc_ood"], converged=cfg["acc_id"] >= CONV_ACC)
    for var in ("M", "A"):
        Q = das_basis(m, var)
        out[f"designed_iia_{var}"] = D.designed_iia(m, var, BLOCK)
        blk = torch.eye(64)[:, BLOCK[0]:BLOCK[1]]
        for kind in ("inv", "add1"):
            for loc, B in (("das", Q), ("block", blk)):
                r = eval_edit(m, *fit_edit(m, B, var, kind), var, kind)
                for k, v in r.items():
                    out[f"{kind}_{var}_{loc}_{k}"] = v
            for loc, B in (("copyblock", None), ("copydas", Q)):
                r = copy_test(m, var, kind, B)
                for k, v in r.items():
                    out[f"{kind}_{var}_{loc}_{k}"] = v
    return out


# ------------------------------------------------------------------ statistics
def paired_signflip(d):
    """Exact two-sided sign-flip permutation p for mean(d) (d = per-seed paired differences)."""
    obs = abs(st.mean(d))
    hits = sum(abs(st.mean([x * s for x, s in zip(d, signs)])) >= obs - 1e-12
               for signs in itertools.product((1, -1), repeat=len(d)))
    return hits / 2 ** len(d)


def boot_ci(d, n=10_000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def report(seeds):
    """Frozen analysis (docs/14 section 14.6). Every scheduled seed counts."""
    res = json.loads((ROOT / "results" / "swap" / "metrics.json").read_text())
    R = {(r["group"], r["seed"]): r for r in res}
    missing = [(g, s) for g in GROUPS for s in seeds if (g, s) not in R]
    assert not missing, f"missing networks: {missing}"
    sp = lambda g, s, v, kind="inv", loc="das": R[(g, s)][f"{kind}_{v}_{loc}_specificity"]
    L = ["# Role-swap results (auto-generated by `run_swap.py report`; analysis frozen in docs/14)", ""]

    def test_line(name, d):
        lo, hi = boot_ci(d)
        return f"| {name} | {st.mean(d):+.3f} | [{lo:+.3f}, {hi:+.3f}] | {paired_signflip(d):.4f} | {sum(x > 0 for x in d)}/{len(d)} |"

    for kind, title in (("inv", "PRIMARY edit: inversion"), ("add1", "SECONDARY edit: add-one")):
        L += [f"## {title} (DAS-located, blind)", "",
              "| comparison (per-seed paired) | mean | 95% bootstrap CI | p (exact sign-flip) | seeds > 0 |",
              "|---|---|---|---|---|"]
        gap = {g: [sp(g, s, "M", kind) - sp(g, s, "A", kind) for s in seeds] for g in GROUPS}
        prim = [gm - ga for gm, ga in zip(gap["mass_only"], gap["acc_only"])]
        L.append(test_line("**PRIMARY: gap(mass_only) − gap(acc_only)**, gap = spec(M) − spec(A)" if kind == "inv"
                           else "gap(mass_only) − gap(acc_only)", prim))
        dirM = [sp("mass_only", s, "M", kind) - sp("acc_only", s, "M", kind) for s in seeds]
        dirA = [sp("acc_only", s, "A", kind) - sp("mass_only", s, "A", kind) for s in seeds]
        L.append(test_line("direction M: spec_M(mass_only) − spec_M(acc_only)", dirM))
        L.append(test_line("direction A: spec_A(acc_only) − spec_A(mass_only)", dirA))
        if kind == "inv":
            ps = sorted([(paired_signflip(dirM), "M"), (paired_signflip(dirA), "A")])
            holm = {ps[0][1]: ps[0][0] <= 0.025, ps[1][1]: ps[0][0] <= 0.025 and ps[1][0] <= 0.05}
            L.append(f"\nHolm (2 directional key-secondary tests, alpha 0.05): direction M "
                     f"{'rejects' if holm['M'] else 'does not reject'}; direction A "
                     f"{'rejects' if holm['A'] else 'does not reject'}.\n")
        L += ["", "Against the undeclared baseline (help = own variable; side effect = other variable):", "",
              "| comparison | mean | 95% CI | p | seeds > 0 |", "|---|---|---|---|---|"]
        for g, v, o in (("mass_only", "M", "A"), ("acc_only", "A", "M")):
            L.append(test_line(f"{g}: own {v} spec − neither", [sp(g, s, v, kind) - sp("neither", s, v, kind) for s in seeds]))
            L.append(test_line(f"{g}: other {o} spec − neither", [sp(g, s, o, kind) - sp("neither", s, o, kind) for s in seeds]))
        L += ["", f"Outcome breakdown ({kind}, DAS-located; mean over all {len(seeds)} seeds):", "",
              "| group | edited variable | n eligible | intended | wrong-variable | unchanged | other | specificity |",
              "|---|---|---|---|---|---|---|---|"]
        for g in GROUPS:
            for v in ("M", "A"):
                f = lambda k: st.mean(R[(g, s)][f"{kind}_{v}_das_{k}"] for s in seeds)
                L.append(f"| {g} | {v} | {int(f('n_eligible'))} | {f('intended'):.3f} | {f('wrong'):.3f} | "
                         f"{f('unchanged'):.3f} | {f('other'):.3f} | {f('specificity'):+.3f} |")
        L.append("")
    L += ["## Diagnostic: copy real pattern vs fitted edit (inversion; intended-answer rate, mean over seeds)", "",
          "| group | var | copy into block | fitted edit, known block | fitted edit, DAS location | copy into DAS location | designed-block IIA |",
          "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        for v in ("M", "A"):
            f = lambda loc: st.mean(R[(g, s)][f"inv_{v}_{loc}_intended"] for s in seeds)
            L.append(f"| {g} | {v} | {f('copyblock'):.3f} | {f('block'):.3f} | {f('das'):.3f} | {f('copydas'):.3f} | "
                     f"{st.mean(R[(g, s)][f'designed_iia_{v}'] for s in seeds):.3f} |")
    L += ["", "## Per-seed results (inversion specificity, DAS-located)", "",
          "| seed | mass_only M | mass_only A | acc_only M | acc_only A | neither M | neither A | primary Δ | converged (mass/acc/neither) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        d = (sp("mass_only", s, "M") - sp("mass_only", s, "A")) - (sp("acc_only", s, "M") - sp("acc_only", s, "A"))
        conv = "/".join("y" if R[(g, s)]["converged"] else "n" for g in GROUPS)
        L.append(f"| {s} | {sp('mass_only', s, 'M'):+.3f} | {sp('mass_only', s, 'A'):+.3f} | {sp('acc_only', s, 'M'):+.3f} | "
                 f"{sp('acc_only', s, 'A'):+.3f} | {sp('neither', s, 'M'):+.3f} | {sp('neither', s, 'A'):+.3f} | {d:+.3f} | {conv} |")
    L += ["", "## Convergence (ID accuracy ≥ 0.95) and accuracy", "", "| group | converged | mean ID acc | mean OOD acc |", "|---|---|---|---|"]
    for g in GROUPS:
        rs = [R[(g, s)] for s in seeds]
        L.append(f"| {g} | {sum(r['converged'] for r in rs)}/{len(rs)} | {st.mean(r['acc_id'] for r in rs):.3f} | "
                 f"{st.mean(r['acc_ood'] for r in rs):.3f} |")
    conv_seeds = [s for s in seeds if all(R[(g, s)]["converged"] for g in GROUPS)]
    if len(conv_seeds) >= 2:
        d = [(sp("mass_only", s, "M") - sp("mass_only", s, "A")) - (sp("acc_only", s, "M") - sp("acc_only", s, "A")) for s in conv_seeds]
        L += ["", "Secondary (not headline): primary statistic on seeds where all three groups converged:", "",
              "| comparison | mean | 95% CI | p | seeds > 0 |", "|---|---|---|---|---|", test_line(f"primary Δ, n={len(d)}", d)]
    (ROOT / "results" / "swap" / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pilot", "train", "analyse", "report"))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--schedules", nargs="+", default=["const"])
    ap.add_argument("--schedule", default=None)
    ap.add_argument("--groups", nargs="+", default=GROUPS)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "pilot":
        zoo = ROOT / "runs" / "zoo_swap_pilot"
        jobs = [(g, s, sch, str(zoo / sch)) for sch in a.schedules for g in ("mass_only", "acc_only")
                for s in a.seeds]
        with ProcessPoolExecutor(a.workers) as ex:
            runs = list(ex.map(train_one, jobs))
        rows = []
        for r in runs:
            m, cfg = D.load(r)
            v = cfg["declared"]
            rows.append(dict(model=Path(r).name, schedule=cfg["schedule"], acc_id=cfg["acc_id"],
                             designed_iia=D.designed_iia(m, v, BLOCK)))
        out = ROOT / "results" / "swap" / "pilot.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
        for r in rows:
            print(r)
        return
    zoo = ROOT / "runs" / "zoo_swap"
    if a.mode == "report":
        report(a.seeds)
        return
    if a.mode == "train":
        assert a.schedule in ("const", "warmup"), "pass the schedule locked in docs/14"
        jobs = [(g, s, a.schedule, str(zoo)) for s in a.seeds for g in a.groups]
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(train_one, jobs))
        return
    runs = [str(zoo / f"{g}_s{s}") for s in a.seeds for g in GROUPS]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(analyse, runs))
    out = ROOT / "results" / "swap"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(f"analysed {len(res)} networks -> {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
