"""Pre-registered experiment (docs/11_preregistration.md; results docs/12): does enforcing
causal structure make networks programmable when the specification is full, partial, noisy
or built on the wrong (non-invariant) variables, on a task with a real unit-rescaling symmetry?

Arms (identical architecture and parameter count; untyped inference; 10 seeds each):
  none          task loss, batch 512
  none_2x       task loss, batch 1024 (compute-matched: same encoder forward passes per step as IIT)
  aux_types     task + auxiliary contextual-role loss (ANLU functional supervision)
  iit_full      task + IIT on canonical M, A, S          (full, correct causal spec)
  iit_partial   task + IIT on M only                     (partial spec)
  iit_noisy     task + IIT on M, A, S, 25% of counterfactual targets replaced at random (noisy spec)
  iit_raw       task + IIT on raw values m, a and S      (valid but NOT unit-invariant spec)

    python run_dimensional.py --seeds 10 --steps 6000 --workers 4
Outputs: runs/zoo_dim/, results/dim/{metrics.json, summary.md, tests.json}
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

from nri import dimtask as T
from nri.model import TinyTyped
from nri.train import set_determinism

ROOT = Path(__file__).parent
ZOO = ROOT / "runs" / "zoo_dim"
RES = ROOT / "results" / "dim"
ARMS = ["none", "none_2x", "aux_types", "iit_full", "iit_partial", "iit_noisy", "iit_raw"]
MODEL_KW = dict(n_types=1, aux_classes=T.N_ROLES, vocab=T.VOCAB, seq_len=T.SEQ_LEN, n_out=T.P)
BLOCKS_FULL = {"M": (0, 16), "A": (16, 32), "S": (32, 36)}
SPEC = {
    "iit_full": {"M": (0, 16), "A": (16, 32), "S": (32, 36)},
    "iit_noisy": {"M": (0, 16), "A": (16, 32), "S": (32, 36)},
    "iit_partial": {"M": (0, 16)},
    "iit_raw": {"m": (0, 16), "a": (16, 32), "S": (32, 36)},
}
NOISE = 0.25
LAMBDA = 3.0
N_CLASSES = {"M": T.P, "A": T.P, "S": 3, "m": T.P, "a": T.P}


# ------------------------------------------------------------------ causal model
def variables(lat):
    M, A, S = T.canon(lat)
    return dict(M=M, A=A, S=S, m=lat["m"], a=lat["a"], S_idx=lat["fu"])


def counterfactual(base, src, var):
    """Target when `var` takes its value from `src` and everything else from `base`."""
    M, A, S = T.canon(base)
    Ms, As, Ss = T.canon(src)
    if var == "M":
        M = Ms
    elif var == "A":
        A = As
    elif var == "S":
        S = Ss
    elif var == "m":  # raw mass value from source, base's unit
        M = src["m"] * torch.tensor(T.MASS_S)[base["mu"]] % T.P
    elif var == "a":
        A = src["a"] * torch.tensor(T.ACC_S)[base["au"]] % T.P
    return T.target(M, A, S)


def label(lat, var):
    v = variables(lat)
    return v["S_idx"] if var == "S" else v[var]


def mask(lo, hi, d=64):
    mk = torch.zeros(d)
    mk[lo:hi] = 1
    return mk


# ------------------------------------------------------------------ training
def encode(m, tok):
    zero = torch.zeros_like(tok)
    x1 = m.resid_layer1(tok, zero)
    return x1, (x1 + m.attn2(x1))[:, -1]


@torch.no_grad()
def acc(m, split="test", n=8192, seed=999_983):
    g = torch.Generator().manual_seed(seed)
    lat = T.sample_latents(n, g, "ood" if split == "ood" else "train")
    b = T.build(lat)
    return (m(b.tokens, torch.zeros_like(b.tokens)).argmax(-1) == b.y).float().mean().item()


def train_one(args):
    arm, seed, steps = args
    run = ZOO / f"{arm}_s{seed}"
    if (run / "model.pt").exists():
        return str(run)
    run.mkdir(parents=True, exist_ok=True)
    set_determinism()
    torch.manual_seed(seed)
    m = TinyTyped(arm, **MODEL_KW)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(10_000 + seed)
    spec = SPEC.get(arm, {})
    masks = {v: mask(*blk) for v, blk in spec.items()}
    bs = 1024 if arm == "none_2x" else 512
    cfg = dict(arm=arm, condition=arm, seed=seed, steps=steps, batch=bs, lr=2e-3, wd=0.01,
               spec=spec, noise=NOISE if arm == "iit_noisy" else 0.0, lambda_iit=LAMBDA,
               encoder_forwards_per_step=2 if (spec or arm == "none_2x") else 1,
               n_params=m.n_params(), model_kw=MODEL_KW, torch=torch.__version__)
    t0 = time.time()
    with open(run / "log.jsonl", "w") as log:
        for step in range(1, steps + 1):
            lat_b = T.sample_latents(bs, g)
            bb = T.build(lat_b)
            x1b, hb = encode(m, bb.tokens)
            loss = F.cross_entropy(m.head_from_resid(hb), bb.y)
            if arm == "aux_types":
                loss = loss + F.cross_entropy(m.aux(x1b).reshape(-1, T.N_ROLES), bb.roles.reshape(-1))
            if spec:
                lat_s = T.sample_latents(bs, g)
                _, hs = encode(m, T.build(lat_s).tokens)
                iit = 0.0
                for v, mk in masks.items():
                    y_cf = counterfactual(lat_b, lat_s, v)
                    if arm == "iit_noisy":
                        flip = torch.rand(bs, generator=g) < NOISE
                        y_cf = torch.where(flip, torch.randint(0, T.P, (bs,), generator=g), y_cf)
                    iit = iit + F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk), y_cf)
                loss = loss + LAMBDA * iit / len(masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if (step <= 1000 and step % 50 == 0) or step % 250 == 0 or step == steps:
                log.write(json.dumps(dict(step=step, loss=loss.item(), acc=acc(m, n=4096),
                                          acc_ood=acc(m, "ood", n=4096), t=round(time.time() - t0, 1))) + "\n")
                log.flush()
    cfg.update(acc_id=acc(m), acc_ood=acc(m, "ood"), train_seconds=round(time.time() - t0, 1))
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.save(m.state_dict(), run / "model.pt")
    print(f"[done] {run.name} id={cfg['acc_id']:.4f} ood={cfg['acc_ood']:.4f} ({cfg['train_seconds']}s)")
    return str(run)


def load(run):
    cfg = json.loads((Path(run) / "config.json").read_text())
    m = TinyTyped(cfg["arm"], **cfg["model_kw"])
    m.load_state_dict(torch.load(Path(run) / "model.pt", weights_only=True))
    return m.eval(), cfg


# ------------------------------------------------------------------ blind analysis
FIT, TEST, PAIR = 4242, 7777, 31337


def lats(n, seed, split="train"):
    return T.sample_latents(n, torch.Generator().manual_seed(seed), split)


@torch.no_grad()
def site(m, lat):
    b = T.build(lat)
    return b, m.resid_pre_mlp(b.tokens, torch.zeros_like(b.tokens))


def class_means(h, y, k):
    return torch.stack([h[y == c].mean(0) for c in range(k)])


def mean_basis(h, y, k):
    mu = class_means(h, y, k)
    _, S, Vh = torch.linalg.svd(mu - mu.mean(0), full_matrices=False)
    r = int((S > 1e-6 * S[0]).sum())
    return Vh[:r].T, mu


@torch.no_grad()
def iia(m, var, basis, n=4096):
    base, src = lats(n, PAIR), lats(n, PAIR + 1)
    _, hb = site(m, base)
    _, hs = site(m, src)
    hp = hb + (hs - hb) @ basis @ basis.T
    return (m.head_from_resid(hp).argmax(-1) == counterfactual(base, src, var)).float().mean().item()


@torch.no_grad()
def designed_iia(m, var, blk, n=4096):
    base, src = lats(n, PAIR), lats(n, PAIR + 1)
    _, hb = site(m, base)
    _, hs = site(m, src)
    mk = mask(*blk)
    hp = hb * (1 - mk) + hs * mk
    return (m.head_from_resid(hp).argmax(-1) == counterfactual(base, src, var)).float().mean().item()


def das(m, var, k=16, steps=400, n=8192, seed=0):
    torch.manual_seed(seed)
    base, src = lats(n, PAIR + 10), lats(n, PAIR + 11)
    with torch.no_grad():
        _, hb = site(m, base)
        _, hs = site(m, src)
    y = counterfactual(base, src, var)
    rot = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(64, k, bias=False))
    opt = torch.optim.Adam(rot.parameters(), lr=1e-2)
    for p in m.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, n, (1024,), generator=g)
        Q = rot.weight.T
        loss = F.cross_entropy(m.head_from_resid(hb[i] + (hs[i] - hb[i]) @ Q @ Q.T), y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in m.parameters():
        p.requires_grad_(True)
    Q = rot.weight.T.detach().clone()
    return iia(m, var, Q), Q


@torch.no_grad()
def remap_edit(m, basis, var, mult=2, n=8192):
    """Held-out 'new unit system' edit: var -> mult*var (mod 13), an affine map inside
    span(basis) fitted by least squares on activations, folded into W_in."""
    lat = lats(n, FIT)
    _, h = site(m, lat)
    y = label(lat, var)
    mu = class_means(h, y, T.P)
    move = mu[(y * mult) % T.P] - mu[y]
    m0 = h.mean(0)
    z = (h - m0) @ basis
    X = torch.cat([z, torch.ones(len(z), 1)], 1)
    W = torch.linalg.lstsq(X, move @ basis).solution
    D = basis @ W[:-1].T @ basis.T
    return m.W_in.weight @ D, m.W_in.weight @ (basis @ W[-1] - D @ m0)


@torch.no_grad()
def eval_remap(m, dW, db, var, mult=2, n=8192):
    lat = lats(n, TEST)
    _, h = site(m, lat)
    pred = m.head_from_resid(h, m.W_in.weight + dW, m.W_in.bias + db).argmax(-1)
    M, A, S = T.canon(lat)
    if var == "M":
        M = M * mult % T.P
    else:
        A = A * mult % T.P
    return (pred == T.target(M, A, S)).float().mean().item()


@torch.no_grad()
def additive_r2(m, n=16384):
    lat = lats(n, FIT)
    _, h = site(m, lat)
    M, A, S = T.canon(lat)
    X = torch.cat([F.one_hot(M, T.P), F.one_hot(A, T.P), F.one_hot(lat["fu"], 3)], 1).float()
    tot = ((h - h.mean(0)) ** 2).sum()
    return (1 - ((h - X @ torch.linalg.lstsq(X, h).solution) ** 2).sum() / tot).item()


def steps_to(run, thr=0.95):
    for line in (Path(run) / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["acc"] >= thr:
            return r["step"]
    return None


def analyse(run):
    set_determinism()
    m, cfg = load(run)
    out = dict(model=Path(run).name, arm=cfg["arm"], seed=cfg["seed"], n_params=cfg["n_params"],
               acc_id=cfg["acc_id"], acc_ood=cfg["acc_ood"], steps_to_95=steps_to(run))
    s95 = out["steps_to_95"]
    out["fwd_to_95"] = None if s95 is None else s95 * cfg["encoder_forwards_per_step"] * cfg["batch"]
    lat = lats(8192, FIT)
    _, h = site(m, lat)
    for var in ("M", "A"):
        U, _ = mean_basis(h, label(lat, var), T.P)
        out[f"iia_cm_{var}"] = iia(m, var, U)
        d, Q = das(m, var)
        out[f"das_{var}"] = d
        out[f"edit2x_{var}_das"] = eval_remap(m, *remap_edit(m, Q, var), var)
        out[f"edit2x_{var}_cm"] = eval_remap(m, *remap_edit(m, U, var), var)
        lo, hi = BLOCKS_FULL[var]
        out[f"edit2x_{var}_block"] = eval_remap(m, *remap_edit(m, torch.eye(64)[:, lo:hi], var), var)
        out[f"designed_iia_{var}"] = designed_iia(m, var, BLOCKS_FULL[var])
    z = torch.zeros_like(m.W_in.weight), torch.zeros_like(m.W_in.bias)
    out["edit2x_noop"] = eval_remap(m, *z, "M")
    out["additive_r2"] = additive_r2(m)
    return out


# ------------------------------------------------------------------ statistics
def perm_p(x, y, n=200_000, seed=0):
    """Two-sided permutation test on the difference of means (vectorized)."""
    g = torch.Generator().manual_seed(seed)
    v = torch.tensor(list(x) + list(y), dtype=torch.float64)
    nx, obs = len(x), abs(st.mean(x) - st.mean(y))
    hits = 0
    for _ in range(n // 20_000):
        idx = torch.rand(20_000, len(v), generator=g).argsort(1)
        pv = v[idx]
        d = (pv[:, :nx].mean(1) - pv[:, nx:].mean(1)).abs()
        hits += int((d >= obs - 1e-12).sum())
    return (hits + 1) / (n + 1)


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank):
            rej[i] = True
        else:
            break
    return rej


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    return f"{st.mean(vals):.3f} ± {st.stdev(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"


PRIMARY = [("edit2x_M_das", "iit_full", "none"), ("acc_ood", "iit_full", "none")]
SECONDARY = [
    ("edit2x_M_das", "iit_partial", "none"), ("edit2x_A_das", "iit_partial", "none"),
    ("edit2x_M_das", "iit_noisy", "none"), ("edit2x_M_das", "iit_noisy", "iit_full"),
    ("acc_ood", "iit_raw", "iit_full"), ("edit2x_M_das", "iit_raw", "iit_full"),
    ("acc_ood", "aux_types", "none"), ("das_M", "aux_types", "none"),
    ("edit2x_M_das", "none_2x", "none"), ("acc_ood", "none_2x", "none"),
    ("fwd_to_95", "iit_full", "none_2x"), ("acc_ood", "iit_partial", "none"),
    ("acc_ood", "iit_noisy", "none"),
]


def summarize(models):
    by = {a: [m for m in models if m["arm"] == a] for a in ARMS}
    cols = [("acc_id", "acc (ID)"), ("acc_ood", "acc (OOD units)"), ("steps_to_95", "steps→95%"),
            ("fwd_to_95", "samples encoded →95%"), ("das_M", "DAS M"), ("das_A", "DAS A"),
            ("iia_cm_M", "IIA M (cm)"), ("edit2x_M_das", "M→2M edit (DAS)"),
            ("edit2x_A_das", "A→2A edit (DAS)"), ("edit2x_M_cm", "M→2M edit (cm)"),
            ("edit2x_M_block", "M→2M edit (designed block)"), ("additive_r2", "additive R²")]
    L = ["# Dimensional reasoning: structure specification quality vs programmability "
         "(auto-generated by run_dimensional.py)", "",
         f"Edit no-op baseline: {st.mean(m['edit2x_noop'] for m in models):.3f}. "
         "All localizations except '(designed block)' are blind to the training specification.", "",
         "| arm | n | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for a in ARMS:
        row = [a, str(len(by[a]))]
        for k, _ in cols:
            vals = [m[k] for m in by[a]]
            nn = sum(v is None for v in vals)
            row.append(fmt(vals) + (f" ({nn} never)" if nn else ""))
        L.append("| " + " | ".join(row) + " |")
    tests = {}

    def run_tests(fam, title, correct):
        L.extend(["", f"## {title}", "", "| metric | comparison | Δ mean | p (two-sided) |" +
                  (" Holm reject @0.05 |" if correct else ""), "|---|---|---|---|" + ("---|" if correct else "")])
        ps, rows = [], []
        for k, a, b in fam:
            xs = [m[k] for m in by[a] if m[k] is not None]
            ys = [m[k] for m in by[b] if m[k] is not None]
            if len(xs) < 2 or len(ys) < 2:
                rows.append((k, a, b, None, None))
                ps.append(1.0)
                continue
            p = perm_p(xs, ys)
            ps.append(p)
            rows.append((k, a, b, st.mean(xs) - st.mean(ys), p))
        rej = holm(ps) if correct else [None] * len(ps)
        for (k, a, b, d, p), r in zip(rows, rej):
            tests[f"{k}:{a}-{b}"] = dict(delta=d, p=p, holm_reject=r)
            ds = "n/a" if d is None else f"{d:+.3f}"
            ps_ = "n/a" if p is None else f"{p:.5f}"
            L.append(f"| {k} | {a} − {b} | {ds} | {ps_} |" + (f" {'yes' if r else 'no'} |" if correct else ""))
    run_tests(PRIMARY, "Primary family (pre-registered; Holm-corrected)", True)
    run_tests(SECONDARY, "Secondary comparisons (pre-registered as exploratory; uncorrected)", False)
    return "\n".join(L) + "\n", tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    jobs = [(arm, s, a.steps) for arm in a.arms for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        runs = list(ex.map(train_one, jobs))
        models = list(ex.map(analyse, runs))
    (RES / "metrics.json").write_text(json.dumps(models, indent=2))
    md, tests = summarize(models)
    (RES / "summary.md").write_text(md)
    (RES / "tests.json").write_text(json.dumps(tests, indent=2))
    print(md)


if __name__ == "__main__":
    main()
