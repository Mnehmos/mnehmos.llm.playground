"""T001: editability decomposer (see manifest.json). Pure decision logic is in decompose(), which the
controls exercise on synthetic gold cases with known answers before the tool may run on a network."""
import math
import torch

INV13 = [0] + [pow(x, 11, 13) for x in range(1, 13)]          # x^-1 = x^(p-2) mod 13


def edit_fn(kind):
    if kind == "inv":
        return lambda y: torch.tensor([INV13[int(v)] for v in y])
    return lambda y: (y + 1) % 13


def _affine(Z, target):
    X = torch.cat([Z, torch.ones(len(Z), 1)], 1)
    W = torch.linalg.lstsq(X, target).solution
    return lambda Zq: torch.cat([Zq, torch.ones(len(Zq), 1)], 1) @ W


def decompose(Zf, yf, rf, Ze, ye, re, readout, kind, min_cell=3, k=2):
    """Zf/Ze: block codes (fit / eval); y: mass class; r: route id; readout(Z_edited, index) -> predicted
    mass class (or None to score by the caller). Returns intended rates for the three edit families."""
    f = edit_fn(kind)
    target_y = f(ye)
    # class means: pooled and per route (fit data only)
    pooled = {c: Zf[yf == c].mean(0) for c in range(13) if (yf == c).any()}
    cell = {}
    for c in range(13):
        for q in set(rf.tolist()):
            sel = (yf == c) & (rf == q)
            if int(sel.sum()) >= min_cell:
                cell[(c, q)] = Zf[sel].mean(0)
    # (a) pooled affine map sending each class mean to the edited class mean
    move = torch.stack([pooled.get(int(f(torch.tensor([c]))[0]), Zf.mean(0)) - pooled.get(c, Zf.mean(0)) for c in range(13)])
    mp = _affine(Zf, Zf + move[yf])
    out = {}
    Zp = mp(Ze)
    out["pooled"] = readout(Zp)
    # (b) per-route affine maps (fit per route)
    Zr = Ze.clone()
    for q in set(re.tolist()):
        sf, se = rf == q, re == q
        if int(sf.sum()) < 20 or not se.any():
            continue
        tgt = torch.stack([cell.get((int(f(torch.tensor([int(c)]))[0]), q), pooled.get(int(f(torch.tensor([int(c)]))[0]), Zf.mean(0)))
                           - cell.get((int(c), q), pooled.get(int(c), Zf.mean(0))) for c in yf[sf]])
        mq = _affine(Zf[sf], Zf[sf] + tgt)
        Zr[se] = mq(Ze[se])
    out["per_route_full"] = readout(Zr)
    # (b') per-route affine maps restricted to the route's top-k class-mean subspace (capacity-limited:
    # succeeds only if each route has LOW-DIMENSIONAL LINEAR structure in which the edit is affine)
    Zk = Ze.clone()
    for q in set(re.tolist()):
        sf, se = rf == q, re == q
        if int(sf.sum()) < 20 or not se.any():
            continue
        cms = torch.stack([v for (c, qq), v in cell.items() if qq == q and c != 0])   # zero class is special
        mu_q = cms.mean(0)
        U, S, Vh = torch.linalg.svd(cms - mu_q, full_matrices=False)
        B = Vh[:k].T
        tgt = torch.stack([cell.get((int(f(torch.tensor([int(c)]))[0]), q), pooled.get(int(f(torch.tensor([int(c)]))[0]), Zf.mean(0)))
                           - cell.get((int(c), q), pooled.get(int(c), Zf.mean(0))) for c in yf[sf]])
        nz = yf[sf] != 0
        zf = (Zf[sf] - mu_q) @ B
        mq = _affine(zf[nz], (zf + tgt @ B)[nz])
        ze = (Ze[se] - mu_q) @ B
        Zk[se] = Ze[se] + (mq(ze) - ze) @ B.T
    out["per_route_lowrank"] = readout(Zk)
    # (c) copy the route-specific target class mean (an activation-level upper bound, not a weight edit)
    Zc = Ze.clone()
    for i in range(len(Ze)):
        key = (int(target_y[i]), int(re[i]))
        Zc[i] = cell[key] if key in cell else pooled.get(int(target_y[i]), Zc[i])
    out["copy_target_mean"] = readout(Zc)
    # (d) v2: copy a REAL training exemplar's block with the target class from the same route
    g = torch.Generator().manual_seed(17)
    Zx = Ze.clone()
    for i in range(len(Ze)):
        cand = ((yf == target_y[i]) & (rf == re[i])).nonzero().squeeze(1)
        if len(cand) == 0:
            cand = (yf == target_y[i]).nonzero().squeeze(1)
        if len(cand):
            Zx[i] = Zf[cand[torch.randint(len(cand), (1,), generator=g)]]
    out["copy_target_exemplar"] = readout(Zx)
    return out


def verdict(r, hi=0.6, lo=0.3):
    """editable: one pooled affine edit works (route-additive code, edit linear in it).
    H1: per-route LOW-RANK affine edits work (route-specific but linearly structured geometry).
    H2: only lookup-like edits work (full per-route affine or copying the target mean): no
        low-dimensional linear structure in which this edit is affine.
    H4 (v2): only copying a REAL exemplar works: the block is sufficient but each class occupies an
        extended region whose mean is off the manifold the readout was trained on.
    H3: even copying a real target exemplar fails: the block is not sufficient for the answer."""
    if r["pooled"] >= hi:
        return "editable"
    if r["per_route_lowrank"] >= hi:
        return "H1"
    if r["copy_target_mean"] >= hi or r["per_route_full"] >= hi:
        return "H2"
    if r.get("copy_target_exemplar", 0) >= hi:
        return "H4"            # v2: sufficient block, but class means lie off the code manifold
    if r.get("copy_target_exemplar", r["copy_target_mean"]) < lo:
        return "H3"
    return "undetermined"


def _synthetic(mode, n=1500, seed=0):
    """Gold cases: 13 classes on a circle (2-D, embedded in 16-D), 3 routes. readout = nearest TRUE
    class-route centroid -> class."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 13, (n,), generator=g)
    r = torch.randint(0, 3, (n,), generator=g)
    dlog = {1: 0}
    x = 1
    for e in range(1, 12):
        x = x * 2 % 13
        dlog[x] = e
    ang = torch.tensor([2 * math.pi * dlog.get(int(v), 0) / 12 for v in y])
    base = torch.zeros(n, 16)
    base[:, 0], base[:, 1] = torch.cos(ang), torch.sin(ang)
    base[y == 0] = 0.0
    base[y == 0, 15] = 3.0                                        # zero class off the circle
    if mode == "additive":
        Z = base + torch.stack([torch.eye(16)[2 + int(q)] * 2.0 for q in r])
    elif mode == "route_rotated":                                 # each route its own rotation of the circle
        Z = base.clone()
        for q in range(3):
            th = 2 * math.pi * q / 3 + 0.7
            R = torch.eye(16)
            R[0, 0], R[0, 1], R[1, 0], R[1, 1] = math.cos(th), -math.sin(th), math.sin(th), math.cos(th)
            Z[r == q] = base[r == q] @ R.T
        Z = Z + torch.stack([torch.eye(16)[2 + int(q)] * 2.0 for q in r])
    elif mode == "scrambled":                                     # arbitrary (non-linear) class placement per route
        P = torch.randn(3, 13, 16, generator=g) * 2
        Z = P[r, y]
    elif mode == "ring":                                          # class = ring around its centre: mean off-manifold
        th = torch.rand(n, generator=g) * 2 * math.pi
        Z = base * 3.0
        rad = 1.0 + 0.3 * y.float()                               # class-dependent ring radius
        Z[:, 5], Z[:, 6] = torch.cos(th) * rad, torch.sin(th) * rad
        Z = Z + torch.stack([torch.eye(16)[2 + int(q)] * 2.0 for q in r])
    else:                                                         # random code: the block carries nothing
        Z = torch.randn(n, 16, generator=g)
    Z = Z + 0.05 * torch.randn(n, 16, generator=g)
    return Z, y, r


def _readout_factory(Z, y, r, ignore_block=False, ring=False):
    if ring:                                                      # defined only ON the class's own ring (dims 5-6)
        inner = _readout_factory(Z, y, r)

        def rd(Zq):
            pred = inner(Zq)
            rad = Zq[:, 5:7].norm(dim=1)
            return torch.where((rad - (1.0 + 0.3 * pred.float())).abs() < 0.1, pred, (pred + 5) % 13)
        return rd
    if ignore_block:
        y_fixed = y.clone()
        return lambda Zq: y_fixed[-len(Zq):]
    cents, keys = [], []
    for c in range(13):
        for q in range(3):
            s = (y == c) & (r == q)
            if s.any():
                cents.append(Z[s].mean(0))
                keys.append(c)
    C = torch.stack(cents)
    keys = torch.tensor(keys)
    return lambda Zq: keys[torch.cdist(Zq, C).argmin(1)]


def controls(ctx):
    cases = []
    expected = {"additive": "editable", "route_rotated": "H1", "scrambled": "H2", "ring": "H4", "random": "H3"}
    for mode, exp in expected.items():
        Z, y, r = _synthetic(mode)
        n = len(y) // 2
        rd = _readout_factory(Z, y, r, ignore_block=(mode == "random"), ring=(mode == "ring"))
        kind = "inv"
        tgt = edit_fn(kind)(y[n:])
        res = decompose(Z[:n], y[:n], r[:n], Z[n:], y[n:], r[n:], lambda Zq: (rd(Zq) == tgt).float().mean().item(), kind)
        v = verdict(res)
        cases.append(dict(name=f"synthetic {mode} -> {exp}", expected=exp, observed=v, rates={k: round(x, 3) for k, x in res.items()},
                          **{"pass": v == exp}))
    return dict(cases=cases)


def _route(ex, how):
    a = ex["ast"]
    if how == "unit":
        return a["mu"]
    fams = ["F01", "F04", "F11", "F02", "F03", "F05", "F06", "F08", "F07", "F10"]
    if how == "family":
        return fams.index(ex["fid"])
    return a["mu"] * 16 + fams.index(ex["fid"])


def run(ctx):
    D, L, O = ctx.data, ctx.lang, ctx.observe
    kind = ctx.args.get("kind", "inv")
    how = ctx.args.get("route", "unit_family")
    fams = ctx.args.get("families", ["F01", "F04", "F11"])
    fit = [e for e in O.fit_pool(4096, seed=99) if e["fid"] in fams]
    import random
    rng = random.Random(41)
    evl = []
    while len(evl) < 600:
        ex = D.make_example(D.sample_ast("force", rng, tuple_splits=("dev",)), rng.choice(fams), D.draw_style(rng), rng)
        if ex["valid"]:
            evl.append(ex)
    out = {}
    for name in ctx.args.get("checkpoints", ctx.checkpoints):
        m = ctx.load_model(name)
        with torch.no_grad():
            bf, be = ctx.encode(fit), ctx.encode(evl)
            hf, _ = m.trunk(bf["tok"])
            he, _ = m.trunk(be["tok"])
            Mf, Af, Sf = O.canon(fit)
            Me, Ae, Se = O.canon(evl)
            f = edit_fn(kind)
            F0, FM, FA = Me * Ae * Se % 13, f(Me) * Ae * Se % 13, Me * f(Ae) * Se % 13
            el = (F0 != FM) & (F0 != FA) & (FM != FA)

            def readout(Zq):
                hq = he.clone()
                hq[:, 0:16] = Zq
                pred = m.head(hq).argmax(-1)
                return (pred == FM)[el].float().mean().item()
            rf = torch.tensor([_route(e, how) for e in fit])
            re = torch.tensor([_route(e, how) for e in evl])
            res = decompose(hf[:, 0:16], Mf, rf, he[:, 0:16], Me, re, readout, kind)
            base_pred = m.head(he).argmax(-1)
            res["unedited_acc"] = (base_pred == F0).float().mean().item()
        out[name] = dict({k: round(v, 4) for k, v in res.items()}, verdict=verdict(res), n_eligible=int(el.sum()))
    return out
