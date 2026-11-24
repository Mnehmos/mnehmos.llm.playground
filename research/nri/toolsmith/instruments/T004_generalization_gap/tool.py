"""T004_generalization_gap: per task, accuracy on FRESH training-pool items (train tuples, training phrasing
families and contexts, training values; canonical and styled) vs the matching development sets.
Separates 'not learned' (train-pool accuracy low) from 'learned but not generalised' (gap large)."""
import random

import torch

DEV_FOR = {"force": ["dev/force_id_trainfam", "dev/force_newfam"], "convert": ["dev/convert_trainfam", "dev/convert_newfam"],
           "equiv": ["dev/equiv_trainfam", "dev/equiv_newfam"], "inverse": ["dev/inverse_trainfam", "dev/inverse_newfam"],
           "sense": ["dev/sense_trainframe_newctx", "dev/sense_newframe_trainctx"]}


def train_pool(D, task, n, seed, p_word, p_abbr):
    rng = random.Random(seed)
    fams = D.families(task, D.TRAIN_FAMILY_SPLITS)
    ctx = D.contexts(("train",)) if task == "sense" else None
    out = []
    while len(out) < n:
        ex = D.make_example(D.sample_ast(task, rng, sense_ctx=ctx), rng.choice(fams), D.draw_style(rng, p_word, p_abbr), rng)
        if ex["valid"]:
            out.append(ex)
    return out


def accuracy(model, enc):
    with torch.no_grad():
        pred = model(enc["tok"]).argmax(-1)
    return round((pred == enc["ans"]).float().mean().item(), 4), pred


def answer_histogram(pred, itos):
    h = {}
    for p in pred.tolist():
        h[itos[p]] = h.get(itos[p], 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1])[:4])


def controls(ctx):
    D, L = ctx.data, ctx.lang
    cases = []
    # 1. pools never leave the training splits
    ok = True
    for t in D.TASKS:
        for ex in train_pool(D, t, 60, 1, 0.5, 0.5):
            ok &= L.FAMILY[ex["fid"]].split in ("pretrain", "train") and ex["valid"]
            if t != "sense":
                ok &= L.tuple_split(ex["ast"]) == "train" and L.numeric_split(ex["ast"]) == "train"
    cases.append(dict(name="training pools only", expected=True, observed=bool(ok), ok=bool(ok)))
    # 2. a model forced to answer gold (oracle logits) scores 1.0; a constant predictor scores the answer share
    exs = train_pool(D, "equiv", 100, 2, 0.0, 0.0)
    enc = ctx.encode(exs)

    class Oracle:
        def __call__(self, tok):
            return torch.nn.functional.one_hot(enc["ans"], 17).float()
    a, _ = accuracy(Oracle(), enc)
    cases.append(dict(name="oracle scores 1.0", expected=1.0, observed=a, ok=a == 1.0))

    class Const:
        def __call__(self, tok):
            return torch.nn.functional.one_hot(torch.full((len(tok),), 13), 17).float()   # always "yes"
    c, _ = accuracy(Const(), enc)
    share = sum(e["answer"] == "yes" for e in exs) / len(exs)
    cases.append(dict(name="constant 'yes' = yes share", expected=round(share, 4), observed=c, ok=abs(c - share) < 1e-3))
    return dict(cases=[dict(name=x["name"], expected=x["expected"], observed=x["observed"], **{"pass": x["ok"]}) for x in cases])


def run(ctx):
    D, L = ctx.data, ctx.lang
    dev = ctx.dev_sets()
    n = ctx.args.get("n", 384)
    out = {}
    for ck in ctx.args.get("checkpoints", ctx.checkpoints):
        m = ctx.load_model(ck)
        out[ck] = {}
        for t in ctx.args.get("tasks", list(D.TASKS)):
            canon = train_pool(D, t, n, 11, 0.0, 0.0)
            styled = train_pool(D, t, n, 12, 0.5, 0.5)
            a_c, p_c = accuracy(m, ctx.encode(canon))
            a_s, _ = accuracy(m, ctx.encode(styled))
            r = dict(train_canonical=a_c, train_styled=a_s, train_canonical_top_answers=answer_histogram(p_c, L.ANSWERS))
            for s in DEV_FOR[t]:
                enc = ctx.encode(dev[s])
                r[s], p_d = accuracy(m, enc)
                r[s + " top_answers"] = answer_histogram(p_d, L.ANSWERS)                     # v2
                r[s + " gold_answers"] = answer_histogram(enc["ans"], L.ANSWERS)
            out[ck][t] = r
    return out
