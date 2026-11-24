"""Birth of a Concept (docs 35-36): four training populations on modular unit conversion and an automated
100-step Concept Formation Scorecard. Stage 2 = qualification pilot (exploratory).

Task: convert v units-u into units-t, answer v * s_u * inv(s_t) mod 13 (fictional units; mod-13 toy semantics,
not real-world arithmetic). Routes = (dim, u, t) with u != t: 12 routes x 13 values = 156 combinations.

Split (purpose-built, seed-independent): value 0 always trained; on every route 3 non-zero values are withheld,
balanced so each value is withheld on 2-3 routes and trained on the rest. Withheld combinations are split into
`dev` (pilot-visible) and `test` (reserved for the registered experiment; never evaluated by the pilot).

Populations (same architecture, optimizer, steps, batch):
  A narrow   : 1/3 of the trained combinations, one frame (C01), canonical surface form
  B diverse  : all trained combinations, four frames (C01 C02 C04 C05), word numbers / abbreviations at 0.3
  C concept  : B + auxiliary head on x1 at the source-number token predicting the canonical quantity v*s_u
  D control  : B + the same head predicting a source-unit-specific permutation of the canonical quantity
               (same 13 classes, entropy, determinism; destroys cross-unit equivalence)
Auxiliary heads are discarded at inference.

    python -m sci.concept run <population> <seed> <out_dir> [--steps N] [--wd W]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sem import data as D  # noqa: E402
from sem import lang as L  # noqa: E402
from sem import model as SM  # noqa: E402
from sem import observe as O  # noqa: E402

P = 13
DIMS = ("mass", "acc")
ROUTES = [(d, u, t) for d in DIMS for u in range(3) for t in range(3) if u != t]
TRAIN_FRAMES = ("C01", "C02", "C04", "C05")
NEWFRAME = "C03"
POPULATIONS = ("A", "B", "C", "D")
STEPS, BATCH, LR, WARMUP, WD, AUX_W = 6000, 128, 1e-3, 100, 0.1, 1.0
AUDIT_EVERY = 100                     # scorecard AND complete checkpoint every 100 steps
SPLIT_SEED = 2811


def answer(dim, v, u, t):
    s = L.SCALES[dim]
    return v * s[u] * L.INV[s[t]] % P


def canonical(dim, v, u):
    return v * L.SCALES[dim][u] % P


def dlog_table():
    g, x, out = 2, 1, {}
    for i in range(12):
        out[x] = i
        x = x * g % P
    return out


DLOG = dlog_table()


# ------------------------------------------------------------------ split
def make_split(seed=SPLIT_SEED):
    """Each route withholds 3 non-zero values; each value is withheld on 2-3 of 12 routes (36 withheld).
    Withheld combos alternate dev/test in a seeded order."""
    rng = random.Random(seed)
    vals = list(range(1, P))
    held = {}
    load = {v: 0 for v in vals}
    for r in ROUTES:
        rng.shuffle(vals)
        pick = sorted(vals, key=lambda v: load[v])[:3]
        for v in pick:
            load[v] += 1
        held[r] = pick
    withheld = [(r, v) for r in ROUTES for v in held[r]]
    rng.shuffle(withheld)
    dev = sorted(withheld[0::2])
    test = sorted(withheld[1::2])
    train = sorted((r, v) for r in ROUTES for v in range(P) if v not in held[r])
    return dict(train=train, dev=dev, test=test)


def narrow_subset(train):
    """Population A: one third of the trained combinations, chosen so that every value and every route still
    occurs (components trained; only coverage of combinations is narrow)."""
    sub = [(r, v) for (r, v) in train if (v + ROUTES.index(r)) % 3 == 0]
    assert {v for _, v in sub} == set(range(P)) and {r for r, _ in sub} == set(ROUTES)
    return sub


def perms(seed=SPLIT_SEED + 1):
    """Population D: a fixed random permutation of Z13 per source unit (dim, u)."""
    rng = random.Random(seed)
    out = {}
    for d in DIMS:
        for u in range(3):
            p = list(range(P))
            rng.shuffle(p)
            out[(d, u)] = p
    return out


PERM = perms()


def aux_target(pop, dim, v, u):
    q = canonical(dim, v, u)
    if pop == "C":
        return q
    if pop == "D":
        return PERM[(dim, u)][q]
    return -1


# ------------------------------------------------------------------ examples
def example(r, v, fid, style, rng):
    dim, u, t = r
    ast = dict(task="convert", dim=dim, v=v, u=u, t=t)
    toks, roles = L.render(ast, fid, style, rng)
    ok, why = L.validate(ast, toks)
    assert ok, (ast, toks, why)
    return dict(ast=ast, fid=fid, style=style, tokens=toks, roles=roles, answer=str(answer(dim, v, u, t)))


def population_spec(pop, split):
    if pop == "A":
        return dict(combos=narrow_subset(split["train"]), frames=("C01",), p_word=0.0, p_abbr=0.0)
    return dict(combos=split["train"], frames=TRAIN_FRAMES, p_word=0.3, p_abbr=0.3)


def batch(spec, n, rng, pop):
    exs, aux = [], []
    for _ in range(n):
        r, v = rng.choice(spec["combos"])
        ex = example(r, v, rng.choice(spec["frames"]), D.draw_style(rng, spec["p_word"], spec["p_abbr"]), rng)
        exs.append(ex)
        aux.append(aux_target(pop, r[0], v, r[1]))
    return exs, torch.tensor(aux)


def src_pos(b):
    return (b["role"] == L.ROLE_ID["src_val"]).long().argmax(1)


# ------------------------------------------------------------------ scorecard
def eval_sets(split, rng_seed=5):
    rng = random.Random(rng_seed)
    canon = dict(num="num", unit="full")
    out = {"practiced": [example(r, v, "C01", canon, rng) for r, v in split["train"]],
           "heldout": [example(r, v, f, canon, rng) for r, v in split["dev"] for f in TRAIN_FRAMES],
           "heldout_newframe": [example(r, v, NEWFRAME, canon, rng) for r, v in split["dev"]],
           "practiced_paraphrase": [example(r, v, f, canon, rng) for r, v in split["train"] for f in TRAIN_FRAMES[1:]]}
    return out


def best_constant(exs):
    c = {}
    for e in exs:
        c[e["answer"]] = c.get(e["answer"], 0) + 1
    return max(c.values()) / len(exs)


@torch.no_grad()
def accuracy(model, exs, vocab):
    b = D.encode_batch(exs, vocab)
    return (model(b["tok"]).argmax(-1) == b["ans"]).float().mean().item()


def finish_from(model, x, keymask, site):
    """Complete the forward pass from the residual stream after block `site` (1 or 2)."""
    if site == 1:
        x = x + model.attn2(model.ln2a(x), keymask)
        x = x + model.mlp2(model.ln2m(x))
    x = x + model.attn3(model.ln3a(x), keymask)
    return model.head(x[:, -1])


@torch.no_grad()
def number_patch(model, vocab, bases, donors, site=1, n=400, seed=0):
    """Copy the residual at the source-number token from donor to base (after block `site`). Intended answer =
    donor number converted along the BASE route. Returns intended / unchanged / other rates over discriminating
    pairs (intended != base answer)."""
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < n:
        bi, di = rng.randrange(len(bases)), rng.randrange(len(donors))
        b, d = bases[bi]["ast"], donors[di]["ast"]
        tgt = answer(b["dim"], d["v"], b["u"], b["t"])
        if tgt != int(bases[bi]["answer"]):
            pairs.append((bi, di, tgt))
    bb, bd = D.encode_batch([bases[i] for i, _, _ in pairs], vocab), D.encode_batch([donors[j] for _, j, _ in pairs], vocab)
    cb, cd = model.trunk(bb["tok"], ret=True), model.trunk(bd["tok"], ret=True)
    key = "x1" if site == 1 else "x2"
    x = cb[key].clone()
    a = torch.arange(len(pairs))
    x[a, src_pos(bb)] = cd[key][a, src_pos(bd)]
    pred = finish_from(model, x, bb["tok"] != D.PAD, site).argmax(-1)
    tgt = torch.tensor([L.ANS_ID[str(t)] for _, _, t in pairs])
    base = bb["ans"]
    return dict(intended=round((pred == tgt).float().mean().item(), 4), unchanged=round((pred == base).float().mean().item(), 4),
                other=round(((pred != tgt) & (pred != base)).float().mean().item(), 4), n=len(pairs))


@torch.no_grad()
def novel_combo_patch(model, vocab, split, site=1, seed=3):
    """THE causal composition test: for every withheld combination (route r, value v), take a trained example on
    route r as base and a trained example carrying value v on another route as donor; patch the source-number
    residual. Intended = v converted along r, a combination never seen in training. Discriminating pairs only."""
    rng = random.Random(seed)
    canon = dict(num="num", unit="full")
    train = split["train"]
    bases, donors, tgts = [], [], []
    for r, v in split["dev"]:
        tgt = answer(r[0], v, r[1], r[2])
        for _ in range(8):
            rb, vb = rng.choice([(rr, vv) for rr, vv in train if rr == r and vv != v])
            if answer(r[0], vb, r[1], r[2]) == tgt:
                continue
            rd = rng.choice([rr for rr, vv in train if vv == v and rr != r])
            bases.append(example(rb, vb, "C01", canon, rng))
            donors.append(example(rd, v, "C01", canon, rng))
            tgts.append(tgt)
    bb, bd = D.encode_batch(bases, vocab), D.encode_batch(donors, vocab)
    key = "x1" if site == 1 else "x2"
    cb, cd = model.trunk(bb["tok"], ret=True), model.trunk(bd["tok"], ret=True)
    x = cb[key].clone()
    a = torch.arange(len(bases))
    x[a, src_pos(bb)] = cd[key][a, src_pos(bd)]
    pred = finish_from(model, x, bb["tok"] != D.PAD, site).argmax(-1)
    tgt = torch.tensor([L.ANS_ID[str(t)] for t in tgts])
    return dict(intended=round((pred == tgt).float().mean().item(), 4), unchanged=round((pred == bb["ans"]).float().mean().item(), 4),
                n=len(bases))


@torch.no_grad()
def patch_semantics(model, vocab, split, site=1, n=600, seed=4, positions="src"):
    """Which quantity does the patched code carry? Base and donor share the dimension and have DIFFERENT source
    units. After copying the donor residual (after block `site`) at `positions` ('src' = source-number token,
    'src+unit' = number and unit tokens), score two competing predictions on pairs where they differ from each
    other and from the base answer:
      number    : donor value v_d converted along the base route  (code = raw number)
      canonical : donor canonical quantity v_d * s_ud expressed in the base target unit (code = quantity)
    `unchanged` = base answer kept (code ignored / inaccessible at this site)."""
    rng = random.Random(seed)
    canon = dict(num="num", unit="full")
    tr = split["train"]
    bases, donors, tn, tc = [], [], [], []
    while len(bases) < n:
        (rb, vb), (rd, vd) = rng.choice(tr), rng.choice(tr)
        if rb[0] != rd[0] or rb[1] == rd[1]:
            continue
        dim, ub, tb = rb
        a_base = answer(dim, vb, ub, tb)
        a_num = answer(dim, vd, ub, tb)
        a_can = canonical(dim, vd, rd[1]) * L.INV[L.SCALES[dim][tb]] % P
        if len({a_base, a_num, a_can}) < 3:
            continue
        bases.append(example(rb, vb, "C01", canon, rng))
        donors.append(example(rd, vd, "C01", canon, rng))
        tn.append(a_num)
        tc.append(a_can)
    bb, bd = D.encode_batch(bases, vocab), D.encode_batch(donors, vocab)
    key = {0: "x0", 1: "x1", 2: "x2"}[site]
    cb, cd = model.trunk(bb["tok"], ret=True), model.trunk(bd["tok"], ret=True)
    x = cb[key].clone()
    a = torch.arange(len(bases))
    pos_b = [src_pos(bb)] + ([(bb["role"] == L.ROLE_ID["src_unit"]).long().argmax(1)] if positions == "src+unit" else [])
    pos_d = [src_pos(bd)] + ([(bd["role"] == L.ROLE_ID["src_unit"]).long().argmax(1)] if positions == "src+unit" else [])
    for pb, pd in zip(pos_b, pos_d):
        x[a, pb] = cd[key][a, pd]
    if site == 0:
        x = x + model.attn1(model.ln1a(x), bb["tok"] != D.PAD)
        x = x + model.mlp1(model.ln1m(x))
        site = 1
    pred = finish_from(model, x, bb["tok"] != D.PAD, site).argmax(-1)
    ids = lambda xs: torch.tensor([L.ANS_ID[str(t)] for t in xs])
    return dict(number=round((pred == ids(tn)).float().mean().item(), 4), canonical=round((pred == ids(tc)).float().mean().item(), 4),
                unchanged=round((pred == bb["ans"]).float().mean().item(), 4), n=len(bases))


def knn(Xf, yf, Xe, ye, k=10):
    mu = Xf.mean(0)
    idx = torch.cdist(Xe - mu, Xf - mu).topk(min(k, len(Xf)), dim=1, largest=False).indices
    votes = torch.zeros(len(Xe), P)
    votes.scatter_add_(1, yf[idx], torch.ones(idx.shape))
    return (votes.argmax(1) == ye).float().mean().item()


def r2_on(features, codes):
    X = torch.cat([features, torch.ones(len(features), 1)], 1).double()
    Y = (codes - codes.mean(0)).double()
    W = torch.linalg.lstsq(X, Y).solution
    res = Y - X @ W
    return 1 - (res.pow(2).sum() / Y.pow(2).sum()).item()


def circle_scores(codes_by_value):
    """codes_by_value: 12 x d (values 1..12). Max over frequencies k=1..6 of the variance explained by a
    discrete-log circle vs an additive circle; plus the mean over 50 shuffled value orders (chance)."""
    vals = list(range(1, P))

    def best(angle_of):
        out = []
        for k in range(1, 7):
            th = torch.tensor([2 * math.pi * k * angle_of[v] / 12 for v in vals])
            out.append(r2_on(torch.stack([th.cos(), th.sin()], 1), codes_by_value))
        return max(out), 1 + max(range(6), key=lambda i: out[i])
    dl, dk = best(DLOG)
    ad, ak = best({v: v * 12 / 13 for v in vals})
    rng = random.Random(0)
    sh = []
    for _ in range(50):
        p = vals[:]
        rng.shuffle(p)
        sh.append(best({v: DLOG[p[i]] for i, v in enumerate(vals)})[0])
    return dict(dlog=round(dl, 4), dlog_k=dk, additive=round(ad, 4), additive_k=ak, shuffled_mean=round(sum(sh) / len(sh), 4))


@torch.no_grad()
def scorecard(model, vocab, sets, split, prev_params=None, own=None):
    out = {k: round(accuracy(model, v, vocab), 4) for k, v in sets.items()}
    if own is not None:                                                   # the population's own training combos
        out["own_practiced"] = round(accuracy(model, own, vocab), 4)
    out["patch_x1_novel_combo"] = novel_combo_patch(model, vocab, split, 1)
    out["patch_x2_novel_combo"] = novel_combo_patch(model, vocab, split, 2)
    for site in (0, 1, 2):
        out[f"semantics_x{site}_src"] = patch_semantics(model, vocab, split, site)
        out[f"semantics_x{site}_src_unit"] = patch_semantics(model, vocab, split, site, positions="src+unit")
    out["heldout_best_constant"] = round(best_constant(sets["heldout"]), 4)
    canon = [e for e in sets["practiced"] if e["ast"]["v"] != 0]
    held = [e for e in sets["heldout"] if e["fid"] == "C01"]
    out["patch_x1_trained"] = number_patch(model, vocab, canon, canon, 1)
    out["patch_x1_heldout_base"] = number_patch(model, vocab, held, canon, 1, n=200, seed=1)
    out["patch_x2_trained"] = number_patch(model, vocab, canon, canon, 2, n=200, seed=2)
    # decodability of the canonical quantity at the source-number token (x1): fit trained, score held-out
    bt, bh = D.encode_batch(sets["practiced"] + sets["practiced_paraphrase"], vocab), D.encode_batch(sets["heldout"], vocab)
    ct, ch = model.trunk(bt["tok"], ret=True), model.trunk(bh["tok"], ret=True)
    at = torch.arange(len(bt["tok"]))
    ah = torch.arange(len(bh["tok"]))
    xt, xh = ct["x1"][at, src_pos(bt)], ch["x1"][ah, src_pos(bh)]
    qt = torch.tensor([canonical(e["ast"]["dim"], e["ast"]["v"], e["ast"]["u"]) for e in sets["practiced"] + sets["practiced_paraphrase"]])
    qh = torch.tensor([canonical(e["ast"]["dim"], e["ast"]["v"], e["ast"]["u"]) for e in sets["heldout"]])
    out["decode_canon_x1_linear"] = round(O.ridge(xt, qt, xh, qh, P), 4)
    out["decode_canon_x1_knn"] = round(knn(xt, qt, xh, qh), 4)
    # concept geometry: mean x1 code of each non-zero number across all routes/frames, on circles
    vs = torch.tensor([e["ast"]["v"] for e in sets["practiced"] + sets["practiced_paraphrase"]])
    codes = torch.stack([xt[vs == v].mean(0) for v in range(1, P)])
    out["circle_x1"] = circle_scores(codes)
    emb = model.E.weight[[vocab.stoi[str(v)] for v in range(1, P)]]
    out["circle_embed"] = circle_scores(emb)
    # weight dynamics since the previous audit (relative change per parameter group)
    cur = {n: p.detach().clone() for n, p in model.named_parameters()}
    if prev_params is not None:
        groups = {}
        for n, p in cur.items():
            g = n.split(".")[0]
            d, b = (p - prev_params[n]).pow(2).sum().item(), prev_params[n].pow(2).sum().item()
            groups.setdefault(g, [0.0, 0.0])
            groups[g][0] += d
            groups[g][1] += b
        out["rel_change"] = {g: round(math.sqrt(d / max(b, 1e-12)), 5) for g, (d, b) in groups.items()}
    out["weight_norm"] = round(math.sqrt(sum(p.pow(2).sum().item() for p in cur.values())), 3)
    return out, cur


# ------------------------------------------------------------------ training
def probe_panel(sets, n_each=16):
    """Fixed probe set for token-level gradient traces: practiced, paraphrased and held-out items."""
    return sets["practiced"][:n_each] + sets["practiced_paraphrase"][:n_each] + sets["heldout"][:n_each]


def token_traces(model, panel_batch):
    """Per-token gradient of the answer loss w.r.t. the token embedding input (x0) on the probe panel: L2 norm and
    signed gradient x input (attribution). Shapes [n_probe, MAX_LEN]. Parameter gradients are not left behind."""
    model.zero_grad(set_to_none=True)
    tok = panel_batch["tok"]
    x0 = (model.E(tok) + model.P(torch.arange(tok.shape[1]))).detach().requires_grad_(True)
    keymask = tok != D.PAD
    a1 = model.attn1(model.ln1a(x0), keymask)
    x1 = x0 + a1
    x1m = x1 + model.mlp1(model.ln1m(x1))
    x2 = x1m + model.attn2(model.ln2a(x1m), keymask)
    x2m = x2 + model.mlp2(model.ln2m(x2))
    x3 = x2m + model.attn3(model.ln3a(x2m), keymask)
    loss = F.cross_entropy(model.head(x3[:, -1]), panel_batch["ans"], reduction="sum")
    g, = torch.autograd.grad(loss, x0)
    model.zero_grad(set_to_none=True)
    return dict(grad_norm=g.norm(dim=-1), grad_x_input=(g * x0.detach()).sum(-1), loss_per_item=torch.zeros(len(tok)) + loss.detach() / len(tok))


def save_checkpoint(d, model, aux_head, opt, rng, step, traces=None):
    """Complete training state, safe formats only: weights (safetensors), AdamW state (safetensors + JSON),
    torch + python RNG (safetensors + JSON), token traces (safetensors). Restorable bit-exactly."""
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, d / "model.safetensors")
    if aux_head is not None:
        save_file({k: v.contiguous() for k, v in aux_head.state_dict().items()}, d / "aux.safetensors")
    names = [n for n, _ in model.named_parameters()] + ([f"auxhead.{n}" for n, _ in aux_head.named_parameters()] if aux_head else [])
    params = list(model.parameters()) + (list(aux_head.parameters()) if aux_head else [])
    assert len(set(names)) == len(names), "parameter names must be unique for optimizer-state persistence"
    tens, steps_ = {}, {}
    for n, p in zip(names, params):
        st = opt.state.get(p, {})
        if st:
            tens[n + "::exp_avg"] = st["exp_avg"].contiguous()
            tens[n + "::exp_avg_sq"] = st["exp_avg_sq"].contiguous()
            steps_[n] = float(st["step"])
    save_file(tens, d / "optim.safetensors")
    save_file(dict(torch_rng=torch.get_rng_state()), d / "rng.safetensors")
    s0 = rng.getstate()
    (d / "state.json").write_text(json.dumps(dict(step=step, adam_steps=steps_, lr=opt.param_groups[0]["lr"],
                                                 py_rng=[s0[0], list(s0[1]), s0[2]])))
    if traces is not None:
        save_file({k: v.contiguous().to(torch.float32) for k, v in traces.items()}, d / "traces.safetensors")


def load_checkpoint(d, model, aux_head, opt, rng):
    from safetensors.torch import load_file
    model.load_state_dict(load_file(d / "model.safetensors"))
    if aux_head is not None:
        aux_head.load_state_dict(load_file(d / "aux.safetensors"))
    meta = json.loads((d / "state.json").read_text())
    tens = load_file(d / "optim.safetensors")
    names = [n for n, _ in model.named_parameters()] + ([f"auxhead.{n}" for n, _ in aux_head.named_parameters()] if aux_head else [])
    params = list(model.parameters()) + (list(aux_head.parameters()) if aux_head else [])
    for n, p in zip(names, params):
        if n in meta["adam_steps"]:
            opt.state[p] = dict(step=torch.tensor(meta["adam_steps"][n]), exp_avg=tens[n + "::exp_avg"].clone(),
                                exp_avg_sq=tens[n + "::exp_avg_sq"].clone())
    torch.set_rng_state(load_file(d / "rng.safetensors")["torch_rng"])
    r = meta["py_rng"]
    rng.setstate((r[0], tuple(r[1]), r[2]))
    return meta["step"]


def state_digest(model, aux_head, opt):
    """Order-independent digest of weights and AdamW moments, keyed by parameter name."""
    h = hashlib.sha256()
    named = list(model.named_parameters()) + ([("auxhead." + n, p) for n, p in aux_head.named_parameters()] if aux_head else [])
    for n, p in sorted(named, key=lambda x: x[0]):
        h.update(n.encode())
        h.update(p.detach().contiguous().numpy().tobytes())
        st = opt.state.get(p, {})
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st:
                h.update(st[k].contiguous().numpy().tobytes())
    return h.hexdigest()


def run(pop, seed, out_dir, steps=STEPS, wd=WD, threads=1, resume=False, audit=True):
    """Train one network. Every AUDIT_EVERY steps: scorecard + COMPLETE checkpoint (ckpt/step_NNNNN/) + token
    traces; every step: a line in interval.jsonl. resume=True continues from the latest checkpoint bit-exactly."""
    assert pop in POPULATIONS
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    vocab = D.Vocab.from_json(json.loads((ROOT / "runs/zoo_sem/eval/vocab.json").read_text()))
    split = make_split()
    spec = population_spec(pop, split)
    model = SM.SemStudent(len(vocab.itos))
    aux_head = nn.Linear(model.d, P) if pop in ("C", "D") else None
    params = list(model.parameters()) + (list(aux_head.parameters()) if aux_head else [])
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=wd)
    rng = random.Random(seed)
    sets = eval_sets(split)
    own = [e for e in sets["practiced"] if (tuple((e["ast"]["dim"], e["ast"]["u"], e["ast"]["t"])), e["ast"]["v"]) in set(spec["combos"])]
    panel = D.encode_batch(probe_panel(sets), vocab)
    out = Path(out_dir)
    ck = out / "ckpt"
    start = 0
    if resume and ck.exists() and any(ck.iterdir()):
        start = load_checkpoint(max(ck.iterdir()), model, aux_head, opt, rng)
    else:
        out.mkdir(parents=True, exist_ok=True)
        meta = dict(population=pop, seed=seed, steps=steps, wd=wd, lr=LR, batch=BATCH, aux_weight=AUX_W if aux_head else 0,
                    split_sha=hashlib.sha256(json.dumps(split).encode()).hexdigest()[:16], n_train_combos=len(spec["combos"]),
                    frames=list(spec["frames"]), n_params_inference=model.n_params(False), checkpoint_every=AUDIT_EVERY,
                    checkpoint_contents=["model", "aux", "AdamW exp_avg/exp_avg_sq/step", "torch+python RNG", "token traces"])
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        model.eval()
        if audit:
            card, _ = scorecard(model, vocab, sets, split, own=own)
            with open(out / "scorecard.jsonl", "w") as f:
                f.write(json.dumps(dict(step=0, **card)) + "\n")
        save_checkpoint(ck / "step_00000", model, aux_head, opt, rng, 0, token_traces(model, panel))
    prev = {n: p.detach().clone() for n, p in model.named_parameters()}
    t0 = time.process_time()
    interval = open(out / "interval.jsonl", "a")
    for step in range(start + 1, steps + 1):
        model.train()
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, step / WARMUP)
        exs, aux = batch(spec, BATCH, rng, pop)
        b = D.encode_batch(exs, vocab)
        h, x1 = model.trunk(b["tok"])
        loss = F.cross_entropy(model.head(h), b["ans"])
        parts = dict(task=loss.item())
        if aux_head is not None:
            la = F.cross_entropy(aux_head(x1[torch.arange(len(exs)), src_pos(b)]), aux)
            parts["aux"] = la.item()
            loss = loss + AUX_W * la
        opt.zero_grad()
        loss.backward()
        gn = math.sqrt(sum(p.grad.pow(2).sum().item() for p in params if p.grad is not None))
        opt.step()
        interval.write(json.dumps(dict(step=step, grad_norm=round(gn, 5), lr=opt.param_groups[0]["lr"],
                                       batch_sha=hashlib.sha256(b["tok"].numpy().tobytes()).hexdigest()[:12],
                                       **{k: round(v, 5) for k, v in parts.items()})) + "\n")
        if step % AUDIT_EVERY == 0:
            interval.flush()
            model.eval()
            if audit:
                card, prev = scorecard(model, vocab, sets, split, prev, own=own)
                with open(out / "scorecard.jsonl", "a") as f:
                    f.write(json.dumps(dict(step=step, loss=parts, train_cpu_s=round(time.process_time() - t0, 1), **card)) + "\n")
            save_checkpoint(ck / f"step_{step:05d}", model, aux_head, opt, rng, step, token_traces(model, panel))
    interval.close()
    return out, state_digest(model, aux_head, opt)


def rescore(run_dir):
    """Re-measure saved checkpoints of a finished run with the current scorecard (writes rescore.jsonl)."""
    from safetensors.torch import load_file
    d = Path(run_dir)
    meta = json.loads((d / "meta.json").read_text())
    vocab = D.Vocab.from_json(json.loads((ROOT / "runs/zoo_sem/eval/vocab.json").read_text()))
    split = make_split()
    spec = population_spec(meta["population"], split)
    sets = eval_sets(split)
    own = [e for e in sets["practiced"] if (tuple((e["ast"]["dim"], e["ast"]["u"], e["ast"]["t"])), e["ast"]["v"]) in set(spec["combos"])]
    prev = None
    with open(d / "rescore.jsonl", "w") as f:
        cks = [(int(c.stem.split("_")[1]), c) for c in d.glob("model_*.safetensors")]
        cks += [(int(c.name.split("_")[1]), c / "model.safetensors") for c in (d / "ckpt").glob("step_*")] if (d / "ckpt").exists() else []
        for step, ck in sorted(cks):
            m = SM.SemStudent(len(vocab.itos))
            m.load_state_dict(load_file(ck))
            m.eval()
            card, prev = scorecard(m, vocab, sets, split, prev, own=own)
            f.write(json.dumps(dict(step=step, **card)) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "split", "rescore"))
    ap.add_argument("pop", nargs="?")
    ap.add_argument("seed", nargs="?", type=int)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--wd", type=float, default=WD)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "split":
        s = make_split()
        print({k: len(v) for k, v in s.items()}, "A:", len(narrow_subset(s["train"])))
    elif a.cmd == "rescore":
        torch.set_num_threads(1)
        rescore(a.pop)
    else:
        run(a.pop, a.seed, a.out, a.steps, a.wd, resume=a.resume)
