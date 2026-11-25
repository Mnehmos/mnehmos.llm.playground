"""Complete per-checkpoint audit for the Birth-of-a-Concept factorial (doc 37).

For one checkpoint (weights + auxiliary head) it measures, on the frozen development material only (the reserved
test combinations are never touched here):

  behaviour     practiced / own-trained / paraphrase / held-out (dev) / held-out in a new frame, best constant
  traces        a manual forward through EVERY site (x0, attn1, x1, mlp1, x1m, attn2, x2, mlp2, x2m, attn3, x3, h,
                readout hidden u, r, logits) on a fixed 48-item probe panel; per-item task loss and aux loss; per-token
                gradient norms of the TASK loss and, separately, of the AUXILIARY loss at every residual/component site;
                per-token task-aux gradient cosine (alignment / conflict); gradient x activation attribution; per
                parameter-group task vs aux gradient norms and cosine
  geometry      discrete-log and additive circle scores for RAW values (codes grouped by v) and CANONICAL quantities
                (grouped by q = v * s_u) at the source-number token after x0, x1m, x2m; shuffled-order chance
  decoding      canonical q, raw v and source unit decodable at x1m (kNN and ridge), fit on trained, scored held-out
  pathways      mean-ablation of the canonical-quantity subspace and of the raw-value subspace at x1m (source-number
                token): accuracy on trained and held-out combinations after each ablation (competing-pathway test);
                principal-angle overlap of the two subspaces
  lineage       orthonormal bases of both subspaces (stored, compared across checkpoints by the cohort analysis)
  causal        patch_semantics at x0/x1/x2 (number token; number + unit tokens): number vs canonical vs unchanged;
                novel-combination patch at x1
Returns (summary: JSON-serialisable dict, tensors: dict of tensors for safetensors).
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from sci import concept as C
from sem import data as D
from sem import lang as L
from sem import observe as O

RESIDUAL = ("x0", "x1", "x1m", "x2", "x2m", "x3")
COMPONENT = ("a1", "m1", "a2", "m2", "a3")
VECTOR = ("h", "u", "r", "logits")
PANEL_EACH = 16


class AuditContext:
    """Everything that does not depend on the weights (built once per process)."""

    def __init__(self, vocab, pop):
        self.vocab, self.pop = vocab, pop
        self.split = C.make_split()
        self.spec = C.population_spec(pop, self.split)
        self.sets = C.eval_sets(self.split)
        combos = set(self.spec["combos"])
        self.own = [e for e in self.sets["practiced"] if ((e["ast"]["dim"], e["ast"]["u"], e["ast"]["t"]), e["ast"]["v"]) in combos]
        self.panel_ex = C.probe_panel(self.sets, PANEL_EACH)
        self.panel = D.encode_batch(self.panel_ex, vocab)
        self.panel_aux = torch.tensor([max(C.aux_target(pop if pop in ("C", "D") else "C", e["ast"]["dim"], e["ast"]["v"], e["ast"]["u"]), 0)
                                       for e in self.panel_ex])
        self.fit_ex = self.sets["practiced"] + self.sets["practiced_paraphrase"]
        self.fit = D.encode_batch(self.fit_ex, vocab)
        self.ho = D.encode_batch(self.sets["heldout"], vocab)
        self.tr = D.encode_batch(self.sets["practiced"], vocab)
        lab = lambda exs: dict(v=torch.tensor([e["ast"]["v"] for e in exs]),
                               q=torch.tensor([C.canonical(e["ast"]["dim"], e["ast"]["v"], e["ast"]["u"]) for e in exs]),
                               u=torch.tensor([e["ast"]["u"] + 3 * C.DIMS.index(e["ast"]["dim"]) for e in exs]))
        self.fit_lab, self.ho_lab, self.tr_lab = lab(self.fit_ex), lab(self.sets["heldout"]), lab(self.sets["practiced"])


def sites(model, tok, keep_grad=False):
    """Manual forward identical to SemStudent.forward, exposing every site."""
    keymask = tok != D.PAD
    s = {}
    s["x0"] = model.E(tok) + model.P(torch.arange(tok.shape[1]))   # not detached: embedding parameters keep gradients
    s["a1"] = model.attn1(model.ln1a(s["x0"]), keymask)
    s["x1"] = s["x0"] + s["a1"]
    s["m1"] = model.mlp1(model.ln1m(s["x1"]))
    s["x1m"] = s["x1"] + s["m1"]
    s["a2"] = model.attn2(model.ln2a(s["x1m"]), keymask)
    s["x2"] = s["x1m"] + s["a2"]
    s["m2"] = model.mlp2(model.ln2m(s["x2"]))
    s["x2m"] = s["x2"] + s["m2"]
    s["a3"] = model.attn3(model.ln3a(s["x2m"]), keymask)
    s["x3"] = s["x2m"] + s["a3"]
    s["h"] = s["x3"][:, -1]
    s["u"] = F.relu(s["h"] @ model.W_in.weight.T + model.W_in.bias)
    s["r"] = s["h"] + model.W_out(s["u"])
    s["logits"] = model.W_U(s["r"])
    if keep_grad:
        for v in s.values():
            v.retain_grad()
    return s


def _cos(a, b, dim=-1):
    return F.cosine_similarity(a, b, dim=dim, eps=1e-12)


def traces(model, aux_head, ctx):
    """Task and auxiliary gradients through every site and parameter group on the probe panel."""
    tok, ans = ctx.panel["tok"], ctx.panel["ans"]
    src = C.src_pos(ctx.panel)
    a = torch.arange(len(tok))
    names = list(SITE_ORDER)
    # ---- task loss (per item) ----
    s = sites(model, tok, keep_grad=True)
    task_items = F.cross_entropy(s["logits"], ans, reduction="none")
    gt = torch.autograd.grad(task_items.sum(), [s[k] for k in names], retain_graph=True, allow_unused=True)
    gtask = {k: (g if g is not None else torch.zeros_like(s[k])) for k, g in zip(names, gt)}
    pnames = [n for n, _ in model.named_parameters()]
    gp_task = torch.autograd.grad(task_items.sum(), list(model.parameters()), retain_graph=True, allow_unused=True)
    # ---- auxiliary loss (per item; C/D only, read at x1m source-number token) ----
    if aux_head is not None:
        aux_items = F.cross_entropy(aux_head(s["x1m"][a, src]), ctx.panel_aux, reduction="none")
        ga = torch.autograd.grad(aux_items.sum(), [s[k] for k in names], retain_graph=True, allow_unused=True)
        gaux = {k: (g if g is not None else torch.zeros_like(s[k])) for k, g in zip(names, ga)}
        gp_aux = torch.autograd.grad(aux_items.sum(), list(model.parameters()), allow_unused=True)
    else:
        aux_items, gaux, gp_aux = None, None, None
    tens, summ = {}, {}
    tens["loss_task_per_item"] = task_items.detach()
    if aux_items is not None:
        tens["loss_aux_per_item"] = aux_items.detach()
    for k in names:
        act = s[k].detach()
        g = gtask[k]
        if act.dim() == 3:                                                 # [n, T, d] token sites
            tens[f"{k}::act_norm"] = act.norm(dim=-1)
            tens[f"{k}::task_gnorm"] = g.norm(dim=-1)
            tens[f"{k}::task_gxa"] = (g * act).sum(-1)
            tens[f"{k}::act_src"] = act[a, src]
            tens[f"{k}::act_last"] = act[:, -1]
            row = dict(task_g_src=g[a, src].norm(dim=-1).mean().item(), task_g_last=g[:, -1].norm(dim=-1).mean().item(),
                       task_gxa_src=(g[a, src] * act[a, src]).sum(-1).mean().item())
            if gaux is not None:
                ga_ = gaux[k]
                tens[f"{k}::aux_gnorm"] = ga_.norm(dim=-1)
                c = _cos(g, ga_)
                tens[f"{k}::task_aux_cos"] = c
                row.update(aux_g_src=ga_[a, src].norm(dim=-1).mean().item(), task_aux_cos_src=c[a, src].mean().item(),
                           conflict_frac_src=(c[a, src] < 0).float().mean().item())
            summ[k] = {kk: round(v, 6) for kk, v in row.items()}
        else:                                                              # [n, d] vector sites
            tens[f"{k}::act_norm"] = act.norm(dim=-1)
            tens[f"{k}::task_gnorm"] = g.norm(dim=-1)
            summ[k] = dict(task_g=round(g.norm(dim=-1).mean().item(), 6))
    groups = {}
    for i, n in enumerate(pnames):
        grp = n.split(".")[0]
        gt_, ga_ = gp_task[i], (gp_aux[i] if gp_aux is not None else None)
        e = groups.setdefault(grp, dict(t2=0.0, a2=0.0, ta=0.0))
        if gt_ is not None:
            e["t2"] += gt_.pow(2).sum().item()
        if ga_ is not None:
            e["a2"] += ga_.pow(2).sum().item()
            if gt_ is not None:
                e["ta"] += (gt_ * ga_).sum().item()
    pg = {}
    for grp, e in groups.items():
        r = dict(task_norm=round(math.sqrt(e["t2"]), 6))
        if gp_aux is not None:
            r["aux_norm"] = round(math.sqrt(e["a2"]), 6)
            r["task_aux_cos"] = round(e["ta"] / max(math.sqrt(e["t2"] * e["a2"]), 1e-12), 4) if e["a2"] > 0 and e["t2"] > 0 else None
        pg[grp] = r
    summ["param_groups"] = pg
    summ["loss_task_mean"] = round(task_items.mean().item(), 5)
    third = len(task_items) // 3
    summ["loss_task_by_panel"] = dict(practiced=round(task_items[:third].mean().item(), 5),
                                      paraphrase=round(task_items[third:2 * third].mean().item(), 5),
                                      heldout=round(task_items[2 * third:].mean().item(), 5))
    if aux_items is not None:
        summ["loss_aux_mean"] = round(aux_items.mean().item(), 5)
    return summ, tens


SITE_ORDER = RESIDUAL[:1] + ("a1", "x1", "m1", "x1m", "a2", "x2", "m2", "x2m", "a3", "x3") + VECTOR


@torch.no_grad()
def src_codes(model, b, site):
    s = sites(model, b["tok"])
    return s[site][torch.arange(len(b["tok"])), C.src_pos(b)]


def class_basis(X, y, k=None):
    """Orthonormal basis of the span of centred class means (rank <= classes - 1)."""
    cls = sorted(set(y.tolist()))
    M = torch.stack([X[y == c].mean(0) for c in cls])
    M = M - M.mean(0)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = int((S > 1e-6 * S.max()).sum()) if k is None else k
    return Vh[:r].T                                                         # [d, r]


def specific(B1, B2, tol=0.1):
    """Directions of span(B1) orthogonal to span(B2) (orthonormal; singular values of the residual above tol)."""
    R = B1 - B2 @ (B2.T @ B1)
    U, S, _ = torch.linalg.svd(R, full_matrices=False)
    return U[:, S > tol]


@torch.no_grad()
def ablate_accuracy(model, b, basis, mean_code):
    """Mean-ablate `basis` at the x1m source-number token and finish the forward pass."""
    s = sites(model, b["tok"])
    x = s["x1m"].clone()
    a = torch.arange(len(b["tok"]))
    p = C.src_pos(b)
    v = x[a, p]
    proj = (v - mean_code) @ basis @ basis.T
    x[a, p] = v - proj
    pred = C.finish_from(model, x, b["tok"] != D.PAD, 1).argmax(-1)
    return round((pred == b["ans"]).float().mean().item(), 4)


@torch.no_grad()
def representation(model, ctx):
    summ, tens = {}, {}
    for site in ("x0", "x1m", "x2m"):
        X = src_codes(model, ctx.fit, site)
        for var in ("v", "q"):
            y = ctx.fit_lab[var]
            codes = torch.stack([X[y == c].mean(0) for c in range(1, C.P)])
            summ[f"circle_{site}_{'raw' if var == 'v' else 'canonical'}"] = C.circle_scores(codes)
    Xf = src_codes(model, ctx.fit, "x1m")
    Xh = src_codes(model, ctx.ho, "x1m")
    for var in ("q", "v", "u"):
        summ[f"decode_x1m_{var}_knn"] = round(C.knn(Xf, ctx.fit_lab[var], Xh, ctx.ho_lab[var]), 4)
        summ[f"decode_x1m_{var}_linear"] = round(O.ridge(Xf, ctx.fit_lab[var], Xh, ctx.ho_lab[var], C.P), 4)
    Bq, Bv = class_basis(Xf, ctx.fit_lab["q"]), class_basis(Xf, ctx.fit_lab["v"])
    tens["basis_canonical_x1m"], tens["basis_raw_x1m"] = Bq, Bv
    sv = torch.linalg.svdvals(Bq.T @ Bv)
    summ["subspace_overlap_canonical_raw"] = round(sv.pow(2).mean().item(), 4)
    mean_code = Xf.mean(0)
    Bq_only, Bv_only = specific(Bq, Bv), specific(Bv, Bq)
    summ["dims"] = dict(canonical=Bq.shape[1], raw=Bv.shape[1], canonical_specific=Bq_only.shape[1], raw_specific=Bv_only.shape[1])
    for name, B in (("canonical", Bq), ("raw", Bv), ("canonical_specific", Bq_only), ("raw_specific", Bv_only)):
        if B.shape[1] == 0:
            summ[f"ablate_{name}_trained"] = summ[f"ablate_{name}_heldout"] = None
            continue
        summ[f"ablate_{name}_trained"] = ablate_accuracy(model, ctx.tr, B, mean_code)
        summ[f"ablate_{name}_heldout"] = ablate_accuracy(model, ctx.ho, B, mean_code)
    # random-subspace control with the canonical-specific dimension count
    g = torch.Generator().manual_seed(0)
    k = max(Bq_only.shape[1], 1)
    Rr = torch.linalg.qr(torch.randn(Xf.shape[1], k, generator=g))[0]
    summ["ablate_random_trained"] = ablate_accuracy(model, ctx.tr, Rr, mean_code)
    summ["ablate_random_heldout"] = ablate_accuracy(model, ctx.ho, Rr, mean_code)
    tens["mean_code_x1m"] = mean_code
    return summ, tens


@torch.no_grad()
def behaviour(model, ctx):
    out = {k: round(C.accuracy(model, v, ctx.vocab), 4) for k, v in ctx.sets.items()}
    out["own_practiced"] = round(C.accuracy(model, ctx.own, ctx.vocab), 4)
    out["heldout_best_constant"] = round(C.best_constant(ctx.sets["heldout"]), 4)
    return out


@torch.no_grad()
def causal(model, ctx, n=300):
    out = {}
    for site in (0, 1, 2):
        out[f"semantics_x{site}_src"] = C.patch_semantics(model, ctx.vocab, ctx.split, site, n=n)
        out[f"semantics_x{site}_src_unit"] = C.patch_semantics(model, ctx.vocab, ctx.split, site, n=n, positions="src+unit")
    out["novel_combo_x1"] = C.novel_combo_patch(model, ctx.vocab, ctx.split, 1)
    return out


def audit(model, aux_head, ctx):
    model.eval()
    summ = dict(behaviour=behaviour(model, ctx))
    t_s, t_t = traces(model, aux_head, ctx)
    r_s, r_t = representation(model, ctx)
    summ.update(traces=t_s, representation=r_s, causal=causal(model, ctx))
    summ["weight_norm"] = {g: round(math.sqrt(sum(p.detach().pow(2).sum().item() for n, p in model.named_parameters()
                                                   if n.split(".")[0] == g)), 5)
                           for g in sorted({n.split(".")[0] for n, _ in model.named_parameters()})}
    tens = {**t_t, **r_t}
    return summ, {k: v.detach().float().contiguous() for k, v in tens.items()}
