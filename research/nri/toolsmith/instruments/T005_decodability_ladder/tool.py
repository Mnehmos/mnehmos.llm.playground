"""T005_decodability_ladder: for a task, which ingredients of the answer are linearly decodable, and where?
Ladder for convert: source value v, source unit u, target unit t, canonical source quantity v*s_u, answer.
Sites: h (query position, input of the readout) and x1/x2 at the source-value token. Positive control on
the same model: canonical mass M at h for canonical pretrain-family force items. Fit and score on disjoint
fresh TRAINING-pool samples (in-distribution decodability; training pools only)."""
import random

import torch


def pool(D, task, n, seed, fams=None, p=(0.0, 0.0)):
    rng = random.Random(seed)
    fams = fams or D.families(task, D.TRAIN_FAMILY_SPLITS)
    out = []
    while len(out) < n:
        ex = D.make_example(D.sample_ast(task, rng), rng.choice(fams), D.draw_style(rng, *p), rng)
        if ex["valid"]:
            out.append(ex)
    return out


def labels(L, exs):
    lab = {"v": [], "u": [], "t": [], "canon": [], "answer": []}
    for e in exs:
        a = e["ast"]
        s = L.SCALES[a["dim"]]
        lab["v"].append(a["v"])
        lab["u"].append(a["u"])
        lab["t"].append(a["t"])
        lab["canon"].append(a["v"] * s[a["u"]] % 13)
        lab["answer"].append(int(L.answer(a)))
    return {k: torch.tensor(v) for k, v in lab.items()}


def knn(Xf, yf, Xe, ye, k=10):
    """Euclidean k-nearest-neighbour vote on centred, UNSCALED vectors (class separability without a
    linear-code assumption; no per-dimension standardisation, which would inflate pure-noise dimensions)."""
    mu = Xf.mean(0)
    d = torch.cdist(Xe - mu, Xf - mu)
    nn_idx = d.topk(k, dim=1, largest=False).indices
    votes = torch.zeros(len(Xe), int(yf.max()) + 1)
    votes.scatter_add_(1, yf[nn_idx], torch.ones(nn_idx.shape))
    return (votes.argmax(1) == ye).float().mean().item()


def site_vectors(model, enc, role_id):
    with torch.no_grad():
        c = model.trunk(enc["tok"], ret=True)
    idx = (enc["role"] == role_id).float().argmax(1)
    ar = torch.arange(len(idx))
    return dict(h=c["h"], h_block=c["h"][:, 0:16], x1_srcval=c["x1"][ar, idx], x2_srcval=c["x2"][ar, idx])


def controls(ctx):
    O = ctx.observe
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 13, (800,), generator=g)
    emb = torch.randn(13, 64, generator=g)
    X = emb[y] + 0.1 * torch.randn(800, 64, generator=g)
    lin = O.ridge(X[:400], y[:400], X[400:], y[400:], 13)
    rnd = O.ridge(torch.randn(800, 64, generator=g)[:400], y[:400], torch.randn(400, 64, generator=g), y[400:], 13)
    cases = [dict(name="linearly encoded label decodes", expected=">= 0.95", observed=round(lin, 3), ok=lin >= 0.95),
             dict(name="random features at chance", expected="<= 0.15", observed=round(rnd, 3), ok=rnd <= 0.15)]
    # planted curved code: discrete-log circle per class, rotated per route (9 routes), plus noise in 16-D
    dlog = {1: 0}
    x = 1
    for i in range(1, 12):
        x = x * 2 % 13
        dlog[x] = i
    yc = torch.randint(1, 13, (1800,), generator=g)
    route = torch.randint(0, 9, (1800,), generator=g)
    ang = torch.tensor([dlog[int(v)] for v in yc]) * (2 * 3.14159265 / 12) + route * 0.7
    Z = 0.05 * torch.randn(1800, 16, generator=g)                          # noise distance ~0.28 < class chord 0.52
    Z[:, 0], Z[:, 1] = Z[:, 0] + torch.cos(ang), Z[:, 1] + torch.sin(ang)
    Z[:, 2:11] += torch.nn.functional.one_hot(route, 9).float() * 2          # route fully identified
    kc = knn(Z[:1200], yc[:1200], Z[1200:], yc[1200:])
    lc = O.ridge(Z[:1200], yc[:1200], Z[1200:], yc[1200:], 13)
    cases.append(dict(name="planted route-rotated circle: kNN decodes", expected=">= 0.9", observed=round(kc, 3), ok=kc >= 0.9))
    cases.append(dict(name="planted circle: linear probe under-reads (blind spot documented)", expected="< kNN", observed=round(lc, 3), ok=lc < kc))
    kr = knn(torch.randn(800, 64, generator=g)[:400], y[:400], torch.randn(400, 64, generator=g), y[400:])
    cases.append(dict(name="kNN on random features at chance", expected="<= 0.15", observed=round(kr, 3), ok=kr <= 0.15))
    return dict(cases=[dict(name=c["name"], expected=c["expected"], observed=c["observed"], **{"pass": c["ok"]}) for c in cases])


def run(ctx):
    D, L, O = ctx.data, ctx.lang, ctx.observe
    rid = L.ROLE_ID["src_val"]
    n = ctx.args.get("n", 1500)
    out = {}
    for ck in ctx.args.get("checkpoints", ctx.checkpoints):
        m = ctx.load_model(ck)
        tr, te = pool(D, "convert", n, 3), pool(D, "convert", n // 2, 4)
        etr, ete = ctx.encode(tr), ctx.encode(te)
        Str, Ste = site_vectors(m, etr, rid), site_vectors(m, ete, rid)
        ltr, lte = labels(L, tr), labels(L, te)
        res = {}
        for site in Str:
            res[site] = {k: dict(linear=round(O.ridge(Str[site], ltr[k], Ste[site], lte[k], 13), 3),
                                 knn=round(knn(Str[site], ltr[k], Ste[site], lte[k]), 3)) for k in ltr}
        with torch.no_grad():
            res["model_accuracy"] = round((m(ete["tok"]).argmax(-1) == ete["ans"]).float().mean().item(), 3)
        out[ck] = res
    return out
