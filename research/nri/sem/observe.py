"""Per-checkpoint observation for the semantic student (Issue #25; docs/30 §30.6).

Extends the concept lineage of docs 24-29 along: surface expression -> contextual sense -> semantic
role -> canonical quantity -> circuit -> output. All measurements use FIT (training pool) and
development sets only; sealed sets are never read here.
Repairs of known instrument weaknesses (doc 29): head identities are reported with a best-vs-runner-up
margin; status thresholds use hysteresis in the report; no k-means; unfamiliar value-unit combinations
are tracked at every checkpoint (dev OOD sets and the binding probe).
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from sem import data as D
from sem import lang as L
from sem.model import BLOCK

_CACHE = {}


def dev_sets(eval_dir):
    """Frozen development sets (written once by run_sem.py freeze)."""
    if "dev" not in _CACHE:
        out = {}
        for p in sorted((eval_dir / "dev").glob("*.jsonl")):
            out["dev/" + p.stem] = D.read_jsonl(p)
        _CACHE["dev"] = out
    return _CACHE["dev"]


def fit_pool(n=1024, seed=99):
    """Training-pool force examples for fitting edits and probes (never dev/sealed)."""
    key = ("fit", n, seed)
    if key not in _CACHE:
        rng = random.Random(seed)
        out = []
        fams = D.families("force", D.TRAIN_FAMILY_SPLITS)
        while len(out) < n:
            ex = D.make_example(D.sample_ast("force", rng), rng.choice(fams), D.draw_style(rng, 0.3, 0.3), rng)
            if ex["valid"]:
                out.append(ex)
        _CACHE[key] = out
    return _CACHE[key]


def canon(exs):
    M = torch.tensor([e["ast"]["m"] * L.MASS_S[e["ast"]["mu"]] % 13 for e in exs])
    A = torch.tensor([e["ast"]["a"] * L.ACC_S[e["ast"]["au"]] % 13 for e in exs])
    S = torch.tensor([L.OUT_FACTOR[e["ast"]["fu"]] for e in exs])
    return M, A, S


@torch.no_grad()
def acc(model, exs, vocab):
    b = D.encode_batch(exs, vocab)
    return (model(b["tok"]).argmax(-1) == b["ans"]).float().mean().item()


@torch.no_grad()
def iia_block(model, exs, vocab, seed=3):
    """Swap the declared block between force examples of DIFFERENT phrasing families; the answer should
    follow the source's canonical mass (mass representation shared across phrasings)."""
    b = D.encode_batch(exs, vocab)
    h, _ = model.trunk(b["tok"])
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(exs), generator=g)
    M, A, S = canon(exs)
    cf = M[perm] * A * S % 13
    y0 = M * A * S % 13
    hp = h.clone()
    hp[:, BLOCK[0]:BLOCK[1]] = h[perm][:, BLOCK[0]:BLOCK[1]]
    pred = model.head(hp).argmax(-1)
    fams = [e["fid"] for e in exs]
    diff = torch.tensor([fams[i] != fams[int(perm[i])] for i in range(len(exs))])
    el = (cf != y0) & diff
    return (pred == cf)[el].float().mean().item() if el.any() else float("nan")


EDITS = {"inv": lambda y: torch.tensor([0] + [pow(int(x), -1, 13) for x in range(1, 13)])[y], "add1": lambda y: (y + 1) % 13}


@torch.no_grad()
def pooled_edit(model, fit, evl, vocab, kind):
    """doc-15 pooled editor (affine map in the block, folded into W_in), frozen scorer."""
    bf, be = D.encode_batch(fit, vocab), D.encode_batch(evl, vocab)
    hf, _ = model.trunk(bf["tok"])
    he, _ = model.trunk(be["tok"])
    Mf, _, _ = canon(fit)
    mu = torch.stack([hf[Mf == c].mean(0) if (Mf == c).any() else hf.mean(0) for c in range(13)])
    move = mu[EDITS[kind](Mf)] - mu[Mf]
    B = torch.eye(model.d)[:, BLOCK[0]:BLOCK[1]]
    m0 = hf.mean(0)
    X = torch.cat([(hf - m0) @ B, torch.ones(len(hf), 1)], 1)
    W = torch.linalg.lstsq(X, move @ B).solution
    Dm = B @ W[:-1].T @ B.T
    dW, db = model.W_in.weight @ Dm, model.W_in.weight @ (B @ W[-1] - Dm @ m0)
    pred = model.head(he, model.W_in.weight + dW, model.W_in.bias + db).argmax(-1)
    M, A, S = canon(evl)
    f = EDITS[kind]
    F0, FM, FA = M * A * S % 13, f(M) * A * S % 13, M * f(A) * S % 13
    el = (F0 != FM) & (F0 != FA) & (FM != FA)
    return dict(intended=(pred == FM)[el].float().mean().item(), wrong=(pred == FA)[el].float().mean().item())


def ridge(Xf, yf, Xe, ye, k, lam=1e-2):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    A = torch.cat([(Xf - mu) / sd, torch.ones(len(Xf), 1)], 1).double()
    E = torch.cat([(Xe - mu) / sd, torch.ones(len(Xe), 1)], 1).double()
    W = torch.linalg.solve(A.T @ A + lam * len(A) * torch.eye(A.shape[1], dtype=torch.float64), A.T @ F.one_hot(yf, k).double())
    return ((E @ W).argmax(1) == ye).float().mean().item()


def role_vectors(model, exs, vocab, role, site="x1"):
    b = D.encode_batch(exs, vocab)
    c = model.trunk(b["tok"], ret=True)
    idx = (b["role"] == L.ROLE_ID[role]).float().argmax(1)       # first token with that role
    has = (b["role"] == L.ROLE_ID[role]).any(1)
    return c[site][torch.arange(len(exs)), idx], has


@torch.no_grad()
def binding_probe(model, fit, evl_id, evl_ood, vocab):
    """Canonical mass decodable at the MASS-VALUE token (layer-1 output): fit on training-pool force
    examples, score on development ID vs held-out-value examples (the doc-29 upstream locus)."""
    xf, hf_ = role_vectors(model, fit, vocab, "mass_val")
    Mf, _, _ = canon(fit)
    out = {}
    for name, ev in (("id", evl_id), ("ood", evl_ood)):
        xe, he_ = role_vectors(model, ev, vocab, "mass_val")
        Me, _, _ = canon(ev)
        out[name] = ridge(xf[hf_], Mf[hf_], xe[he_], Me[he_], 13)
    return out


@torch.no_grad()
def sense_probe(model, vocab, seed=5, n=512):
    """Sense decodable at the polysemous word token (layer-2 output); fit on training contexts, scored on
    development contexts (descriptive)."""
    rng = random.Random(seed)

    def mk(ctxs, fams):
        out = []
        while len(out) < n:
            ex = D.make_example(D.sample_ast("sense", rng, sense_ctx=ctxs), rng.choice(fams), D.draw_style(rng), rng)
            if ex["valid"]:
                out.append(ex)
        return out
    tr = mk(D.contexts(("train",)), D.families("sense", ("train",)))
    dv = mk(D.contexts(("dev",)), D.families("sense", ("train",)))
    res = {}
    for site in ("x1", "x2"):
        vecs = []
        for exs in (tr, dv):
            b = D.encode_batch(exs, vocab)
            c = model.trunk(b["tok"], ret=True)
            r = b["role"]
            idx = ((r == L.ROLE_ID["poly_phys"]) | (r == L.ROLE_ID["poly_other"])).float().argmax(1)
            y = torch.tensor([0 if e["ast"]["sense"] == "phys" else 1 for e in exs])
            vecs.append((c[site][torch.arange(len(exs)), idx], y))
        res[site] = ridge(vecs[0][0], vecs[0][1], vecs[1][0], vecs[1][1], 2)
    return res


def linear_cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    return ((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()).clamp_min(1e-12)).item()


@torch.no_grad()
def paraphrase_invariance(model, panel, vocab):
    """For ASTs rendered in many families: share of ASTs whose prediction is the same in every family;
    accuracy; and mean cosine of h within an AST vs between ASTs."""
    b = D.encode_batch(panel, vocab)
    h, _ = model.trunk(b["tok"])
    pred = model.head(h).argmax(-1)
    groups = {}
    for i, e in enumerate(panel):
        groups.setdefault(e["group"], []).append(i)
    same = sum(1 for g in groups.values() if len(set(pred[g].tolist())) == 1) / len(groups)
    hn = F.normalize(h - h.mean(0), dim=1)
    C = hn @ hn.T
    gid = torch.tensor([e["group"] for e in panel])
    within = (gid[:, None] == gid[None, :]) & ~torch.eye(len(panel), dtype=torch.bool)
    between = gid[:, None] != gid[None, :]
    return dict(all_families_agree=same, acc=(pred == b["ans"]).float().mean().item(),
                cos_within=C[within].mean().item(), cos_between=C[between].mean().item())


@torch.no_grad()
def head_roles(model, exs, vocab):
    """For each attention head, the share of the query position's attention on each semantic role;
    reported with the margin to the most similar OTHER head (ambiguous identity if < 0.05)."""
    b = D.encode_batch(exs, vocab)
    c = model.trunk(b["tok"], ret=True)
    roles = ["mass_val", "mass_unit", "acc_val", "acc_unit", "out_unit", "distract_num", "other"]
    sig = {}
    for li, key in enumerate(("pat1", "pat2", "pat3")):
        pat = c[key][:, :, -1, :]                                   # (n, heads, T) from the query position
        for hd in range(pat.shape[1]):
            v = [((pat[:, hd] * (b["role"] == L.ROLE_ID[r]).float()).sum(1)).mean().item() for r in roles]
            sig[f"L{li + 1}.h{hd}"] = v
    names = list(sig)
    S = F.normalize(torch.tensor([sig[n] for n in names]), dim=1)
    sim = S @ S.T - 2 * torch.eye(len(names))
    margin = {n: round(1 - sim[i].max().item(), 4) for i, n in enumerate(names)}
    return dict(signature={n: [round(x, 3) for x in v] for n, v in sig.items()}, roles=roles, distinctness=margin)


def observe(model, vocab, eval_dir, prev=None):
    """Return (card, carry). carry holds h on a fixed panel and weights for the next checkpoint."""
    model.eval()
    ds = dev_sets(eval_dir)
    fit = fit_pool()
    card = dict(dev_acc={k: round(acc(model, v, vocab), 4) for k, v in ds.items()})
    fid = ds["dev/force_id_trainfam"]
    card["iia_block_cross_family"] = round(iia_block(model, fid + ds["dev/force_newfam"], vocab), 4)
    card["iia_block_pretrainfam"] = round(iia_block(model, ds["dev/force_id_pretrainfam"], vocab), 4)
    card["edit_inv"] = {k: round(v, 4) for k, v in pooled_edit(model, fit, fid, vocab, "inv").items()}
    card["edit_add1"] = {k: round(v, 4) for k, v in pooled_edit(model, fit, fid, vocab, "add1").items()}
    card["binding_probe"] = {k: round(v, 4) for k, v in binding_probe(model, fit, fid, ds["dev/force_ood_value"], vocab).items()}
    card["sense_probe"] = {k: round(v, 4) for k, v in sense_probe(model, vocab).items()}
    if "panel" not in _CACHE:
        _CACHE["panel"] = D.paraphrase_panel(("pretrain", "train", "dev"))
    card["paraphrase"] = {k: round(v, 4) for k, v in paraphrase_invariance(model, _CACHE["panel"], vocab).items()}
    card["heads"] = head_roles(model, fid, vocab)
    with torch.no_grad():
        h, _ = model.trunk(D.encode_batch(fid, vocab)["tok"])
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if prev is not None:
        card["cka_h_prev"] = round(linear_cka(prev["h"], h), 4)
        card["weight_rel_change"] = {k: round(((sd[k] - prev["sd"][k]).norm() / prev["sd"][k].norm().clamp_min(1e-12)).item(), 5)
                                     for k in sd if sd[k].dtype.is_floating_point}
    model.train()
    return card, dict(h=h, sd=sd)
