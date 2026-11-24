"""T002_swap_reconciler: why does block-swap IIA read 1.0 while exemplar copy (T001) reads ~0.55?

Swaps h[0:16] from a SOURCE example into a BASE example and scores whether the prediction follows the
source's canonical mass (target = M_src * A_base * S_base mod 13). Success is stratified by source and
base attributes (number style, unit style, family split, distractor, source pool), so the gap between
the two measurements can be attributed to a specific attribute rather than guessed.
"""
import random

import torch


def swap_success(model_head, h_base, h_src, target, block=(0, 16)):
    hp = h_base.clone()
    hp[:, block[0]:block[1]] = h_src[:, block[0]:block[1]]
    return (model_head(hp).argmax(-1) == target)


def stratify(ok, attrs):
    """ok: bool tensor; attrs: list of dicts (one per pair). Returns {attr: {value: [rate, n]}}."""
    out = {}
    for k in attrs[0]:
        vals = {}
        for i, a in enumerate(attrs):
            vals.setdefault(str(a[k]), []).append(bool(ok[i]))
        out[k] = {v: [round(sum(x) / len(x), 4), len(x)] for v, x in sorted(vals.items())}
    return out


def _force_examples(D, rng, n, fams, tuple_splits, p_word, p_abbr):
    out = []
    while len(out) < n:
        ex = D.make_example(D.sample_ast("force", rng, tuple_splits=tuple_splits), rng.choice(fams),
                            D.draw_style(rng, p_word, p_abbr), rng)
        if ex["valid"]:
            out.append(ex)
    return out


def _attrs(src, base, pool):
    return dict(src_num=src["style"]["num"], src_unit=src["style"]["unit"], base_num=base["style"]["num"],
                base_unit=base["style"]["unit"], src_canonical_style=src["style"] == dict(num="num", unit="full"),
                same_family=src["fid"] == base["fid"], src_pool=pool, src_mass_unit=src["ast"]["mu"])


def _pairs(ctx, model, bases, sources, pool, rng):
    O = ctx.observe
    bb, bs = ctx.encode(bases), ctx.encode(sources)
    with torch.no_grad():
        hb, _ = model.trunk(bb["tok"])
        hs, _ = model.trunk(bs["tok"])
    Mb, Ab, Sb = O.canon(bases)
    Ms, _, _ = O.canon(sources)
    perm = torch.tensor([rng.randrange(len(sources)) for _ in bases])
    target = Ms[perm] * Ab * Sb % 13
    base_ans = Mb * Ab * Sb % 13
    with torch.no_grad():
        ok = swap_success(model.head, hb, hs[perm], target)
    el = target != base_ans
    attrs = [_attrs(sources[int(perm[i])], bases[i], pool) for i in range(len(bases))]
    keep = el.nonzero().squeeze(1).tolist()
    return ok[keep], [attrs[i] for i in keep]


def controls(ctx):
    """Exact known-answer checks on a real checkpoint copy plus bookkeeping checks."""
    cases = []
    m = ctx.load_model(ctx.checkpoints[0])
    D = ctx.data
    rng = random.Random(5)
    exs = _force_examples(D, rng, 200, list(D.PRETRAIN_FAMILIES), ("dev",), 0.3, 0.3)
    b = ctx.encode(exs)
    with torch.no_grad():
        h, _ = m.trunk(b["tok"])
        p0 = m.head(h).argmax(-1)
        # 1. self-swap leaves every prediction unchanged
        same = swap_success(m.head, h, h, p0).float().mean().item()
        cases.append(dict(name="self-swap is identity", expected=1.0, observed=same, pass_=same == 1.0))
        # 2. swapping ALL 64 dims (block = whole h) reproduces the source's own prediction exactly
        perm = torch.randperm(len(exs), generator=torch.Generator().manual_seed(0))
        full = swap_success(m.head, h, h[perm], p0[perm], block=(0, h.shape[1])).float().mean().item()
        cases.append(dict(name="full-h swap = source prediction", expected=1.0, observed=full, pass_=full == 1.0))
        # 3. an empty block (0 dims) never changes the prediction
        none = swap_success(m.head, h, h[perm], p0, block=(0, 0)).float().mean().item()
        cases.append(dict(name="empty-block swap is identity", expected=1.0, observed=none, pass_=none == 1.0))
    # 4. stratification bookkeeping: weighted strata recover the overall rate for every attribute
    ok = torch.tensor([i % 3 == 0 for i in range(90)])
    attrs = [dict(a=i % 2, b=i % 5) for i in range(90)]
    st = stratify(ok, attrs)
    tot = [sum(r * n for r, n in st[k].values()) / 90 for k in ("a", "b")]
    good = all(abs(t - ok.float().mean().item()) < 1e-3 for t in tot)
    cases.append(dict(name="strata reweight to the total", expected=True, observed=good, pass_=good))
    return dict(cases=[dict(name=c["name"], expected=c["expected"], observed=c["observed"], **{"pass": c["pass_"]}) for c in cases])


def run(ctx):
    D = ctx.data
    a = ctx.args
    fams = a.get("families", list(D.PRETRAIN_FAMILIES))
    n = a.get("n", 800)
    out = {}
    for name in a.get("checkpoints", ctx.checkpoints):
        m = ctx.load_model(name)
        rng = random.Random(23)
        bases = _force_examples(D, rng, n, fams, ("dev",), 0.0, 0.0)            # canonical-style bases
        dev_src = _force_examples(D, rng, n, fams, ("dev",), 0.3, 0.3)
        fit_src = [e for e in ctx.observe.fit_pool(4096, seed=99) if e["fid"] in fams]
        ok1, at1 = _pairs(ctx, m, bases, dev_src, "dev", random.Random(1))
        ok2, at2 = _pairs(ctx, m, bases, fit_src, "fit", random.Random(2))
        ok = torch.cat([ok1, ok2])
        at = at1 + at2
        canon_src = torch.tensor([x["src_canonical_style"] for x in at])
        out[name] = dict(overall=round(ok.float().mean().item(), 4), n=len(at),
                         canonical_style_sources=round(ok[canon_src].float().mean().item(), 4),
                         noncanonical_style_sources=round(ok[~canon_src].float().mean().item(), 4),
                         strata=stratify(ok, at))
    return out
