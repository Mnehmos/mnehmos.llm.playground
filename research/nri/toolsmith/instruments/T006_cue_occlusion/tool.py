"""T006_cue_occlusion: which question-frame words drive the TASK-TYPE decision? For sense items, the
'type mass' is the softmax probability on {phys, other}. Each frame word is occluded (replaced by <unk>)
and the drop in type mass is recorded, per word and frame, on fresh TRAINING-pool sense items and on the
development new-frame set."""
import random

import torch


def type_mass(model, tok, ids):
    with torch.no_grad():
        p = torch.softmax(model(tok), -1)
    return p[:, ids].sum(1)


def occlusion(model, tok, positions_of_word, unk, ids):
    """Mean drop in type mass when each occurrence of a word is replaced by <unk>."""
    base = type_mass(model, tok, ids)
    t2 = tok.clone()
    rows = []
    for i, pos in positions_of_word:
        t2[i, pos] = unk
        rows.append(i)
    if not rows:
        return None
    after = type_mass(model, t2, ids)
    r = torch.tensor(sorted(set(rows)))
    return round((base[r] - after[r]).mean().item(), 4)


def frame_words(tokens_list, vocab_stoi, skip):
    words = {}
    for i, toks in enumerate(tokens_list):
        start = max(j for j, w in enumerate(toks) if w == ".") + 1 if "." in toks else 0
        for j in range(start, len(toks)):
            w = toks[j]
            if w not in skip and w in vocab_stoi:
                words.setdefault(w, []).append((i, j))
    return words


def controls(ctx):
    cases = []
    # synthetic: a model whose type logit depends ONLY on token 7 at any position
    class Toy:
        def __call__(self, tok):
            z = torch.zeros(len(tok), 17)
            z[:, 15] = 8.0 * (tok == 7).any(1).float()
            return z
    tok = torch.full((4, 6), 3)
    tok[:, 2] = 7
    tok[:, 4] = 9
    ids = [15, 16]
    d7 = occlusion(Toy(), tok, [(i, 2) for i in range(4)], 1, ids)
    d9 = occlusion(Toy(), tok, [(i, 4) for i in range(4)], 1, ids)
    cases.append(dict(name="occluding the causal token drops type mass", expected="> 0.5", observed=d7, ok=d7 > 0.5))
    cases.append(dict(name="occluding an irrelevant token changes nothing", expected=0.0, observed=d9, ok=abs(d9) < 1e-6))
    # the real model: occluding nothing is a no-op
    m = ctx.load_model(ctx.checkpoints[0])
    D = ctx.data
    rng = random.Random(0)
    ex = [D.make_example(D.sample_ast("sense", rng, sense_ctx=D.contexts(("train",))), "S01", D.draw_style(rng), rng) for _ in range(20)]
    enc = ctx.encode(ex)
    z = occlusion(m, enc["tok"], [], 1, ids)
    cases.append(dict(name="empty occlusion is undefined (no rows)", expected=None, observed=z, ok=z is None))
    return dict(cases=[dict(name=c["name"], expected=c["expected"], observed=c["observed"], **{"pass": c["ok"]}) for c in cases])


def run(ctx):
    D, L = ctx.data, ctx.lang
    ids = [L.ANS_ID["phys"], L.ANS_ID["other"]]
    unk = D.UNK
    skip = {"?", "<q>", "force", "mass", "work"}
    out = {}
    rng = random.Random(3)
    pools = {}
    for fid in ("S01", "S02"):
        pools["train " + fid] = [D.make_example(D.sample_ast("sense", rng, sense_ctx=D.contexts(("train",))), fid, D.draw_style(rng), rng)
                                 for _ in range(200)]
    pools["dev newframe"] = ctx.dev_sets()["dev/sense_newframe_trainctx"]
    for ck in ctx.args.get("checkpoints", ctx.checkpoints):
        m = ctx.load_model(ck)
        out[ck] = {}
        for name, exs in pools.items():
            enc = ctx.encode(exs)
            # positions in the encoded (left-padded) sequence: offset = MAX_LEN - 1 - len(tokens)
            offs = [D.MAX_LEN - 1 - min(len(e["tokens"]), D.MAX_LEN - 1) for e in exs]
            fw = frame_words([e["tokens"][-(D.MAX_LEN - 1):] for e in exs], ctx.vocab.stoi, skip)
            res = dict(type_mass=round(type_mass(m, enc["tok"], ids).mean().item(), 4))
            for w, pos in fw.items():
                res["drop " + w] = occlusion(m, enc["tok"], [(i, offs[i] + j) for i, j in pos], unk, ids)
            out[ck][name] = res
    return out
