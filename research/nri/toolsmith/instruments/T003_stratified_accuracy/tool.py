"""T003_stratified_accuracy: accuracy on development sets stratified by surface and content attributes
(number style, unit style, phrasing family, mass unit, distractor, task), per checkpoint. Distinguishes
'a skill is lost' from 'a surface form was never bound' and localises regressions."""
import json

import torch


def attrs(ex):
    a = ex["ast"]
    return dict(num=ex["style"]["num"], unit=ex["style"]["unit"], family=ex["fid"],
                canonical_style=ex["style"] == dict(num="num", unit="full"),
                distractor=ex.get("distractor") is not None, mass_unit=a.get("mu", a.get("u1", -1)))


def stratify(correct, rows):
    out = {}
    for k in rows[0]:
        vals = {}
        for c, r in zip(correct, rows):
            vals.setdefault(str(r[k]), []).append(bool(c))
        out[k] = {v: [round(sum(x) / len(x), 4), len(x)] for v, x in sorted(vals.items())}
    return out


def predict_correct(model, enc):
    with torch.no_grad():
        return (model(enc["tok"]).argmax(-1) == enc["ans"]).tolist()


def controls(ctx):
    cases = []
    rows = [dict(a=i % 2, b=i % 3) for i in range(60)]
    corr = [i % 4 == 0 for i in range(60)]
    st = stratify(corr, rows)
    tot = sum(r * n for r, n in st["a"].values()) / 60
    cases.append(dict(name="strata reweight to total", expected=sum(corr) / 60, observed=round(tot, 4), ok=abs(tot - sum(corr) / 60) < 1e-3))
    # a constant predictor scores exactly the share of its answer in every stratum (oracle bookkeeping)
    D = ctx.data
    m = ctx.load_model(ctx.checkpoints[0])
    import random
    rng = random.Random(0)
    exs = []
    while len(exs) < 120:
        ex = D.make_example(D.sample_ast("force", rng, tuple_splits=("dev",)), D.PRETRAIN_FAMILIES[0], D.draw_style(rng, 0.5, 0.5), rng)
        if ex["valid"]:
            exs.append(ex)
    enc = ctx.encode(exs)
    with torch.no_grad():
        p = m(enc["tok"]).argmax(-1)
    const = int(p[0])
    corr2 = (enc["ans"] == const).tolist()
    st2 = stratify(corr2, [attrs(e) for e in exs])
    share = sum(corr2) / len(corr2)
    w = sum(r * n for r, n in st2["num"].values()) / len(corr2)
    cases.append(dict(name="constant-predictor strata = answer share", expected=round(share, 4), observed=round(w, 4), ok=abs(w - share) < 1e-3))
    # the model's own accuracy equals the reweighted strata
    c3 = predict_correct(m, enc)
    st3 = stratify(c3, [attrs(e) for e in exs])
    w3 = sum(r * n for r, n in st3["unit"].values()) / len(c3)
    cases.append(dict(name="model accuracy = reweighted strata", expected=round(sum(c3) / len(c3), 4), observed=round(w3, 4), ok=abs(w3 - sum(c3) / len(c3)) < 1e-3))
    return dict(cases=[dict(name=c["name"], expected=c["expected"], observed=c["observed"], **{"pass": c["ok"]}) for c in cases])


def run(ctx):
    sets = ctx.dev_sets()
    names = ctx.args.get("sets", sorted(sets))
    keys = ctx.args.get("keys", ["num", "unit", "canonical_style", "family"])
    out = {}
    for ck in ctx.args.get("checkpoints", ctx.checkpoints):
        m = ctx.load_model(ck)
        out[ck] = {}
        for n in names:
            exs = sets[n]
            c = predict_correct(m, ctx.encode(exs))
            st = stratify(c, [attrs(e) for e in exs])
            out[ck][n] = dict(acc=round(sum(c) / len(c), 4), **{k: st[k] for k in keys if k in st})
    return out
