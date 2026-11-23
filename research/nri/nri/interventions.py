"""Safe intervention executor and validation / rollback controller (spec v1.0 §6.1, §7; docs/28).

A supervisor proposal is a JSON document. It is VALIDATED against a fixed schema and EXECUTED by
this module; no supervisor-written code runs. Allowed primitives:

  restricted_gradient_update  Adam steps on FIT over an allowed parameter scope, minimising a weighted
                              sum of registered objective terms (OBJECTIVES), with an L2 leash; the
                              resulting delta is scaled to the stated budget.
  low_rank_adjust             a rank-r correction of one allowed tensor (optionally a row/column slice,
                              e.g. one attention head), fitted by ridge least squares to a registered
                              activation TARGET, then truncated by SVD to rank r.
  counterfactual_supervision  restricted_gradient_update whose main term is interchange supervision of a
                              named variable at a named site (block, rest, one attn2 head's output).

Scope selectors: a tensor name from ALLOWED_TENSORS, optionally "name[head:k]" (that head's slice:
qkv rows of the head for attn*.qkv.*, columns for attn*.o.weight) or "name[rows:a-b]".

validate(): runs on a deep copy of the FULL training state (model, optimizer, scheduler, generators,
RNG). It never touches the live state; installation is a separate call. Checks are phase-aware and
include a deterministic 100-step LOOKAHEAD on the copy (the same batches the live run would see) to
catch post-edit divergence (precursor loop, docs/27).
"""
from __future__ import annotations

import copy
import json
import re
import time

import torch
import torch.nn.functional as F

import run_correction as C
import run_dimensional as D
import run_swap as S
from nri import atlas as AT
from nri import dimtask as T

ALLOWED_TENSORS = ("E_c.weight", "E_p.weight", "attn1.qkv.weight", "attn1.qkv.bias", "attn1.o.weight", "attn1.o.bias",
                   "attn2.qkv.weight", "attn2.qkv.bias", "attn2.o.weight", "attn2.o.bias", "W_in.weight", "W_in.bias",
                   "W_out.weight", "W_out.bias", "W_U.weight")
OBJECTIVES = ("kl", "task", "iit", "interaction", "translate", "noise_kl", "route_keep", "separate_MA")
LOW_RANK_TARGETS = ("block_interaction_removal", "head_canonical_mass", "head_interaction_removal")
LIMITS = dict(max_steps=600, max_lr=3e-3, max_rank=8, max_rel_model=0.03, max_rel_tensor=0.25,
              max_candidates_per_checkpoint=3, max_extra_grad_steps_per_checkpoint=1500, max_cum_rel_model=0.15)

# phase-aware thresholds (frozen in docs/28 §28.6)
PHASES = {
    "early":   dict(S1_acc=0.0, S2_keep=0.95, S3_ood=0.01, S4_iia=None, S5_info=0.02, S6_wrong=0.02),
    "forming": dict(S1_acc=0.005, S2_keep=0.98, S3_ood=0.01, S4_iia=0.01, S5_info=0.02, S6_wrong=0.02),
    "mature":  dict(S1_acc=0.005, S2_keep=0.99, S3_ood=0.01, S4_iia=0.01, S5_info=0.02, S6_wrong=0.02),
}
TARGETS = {"edit_inv": 0.01, "interaction": 0.02, "unit_transfer": 0.03, "ood_dev": 0.01, "iia": 0.01}
EARLY_TARGETS = ("iia", "ood_dev")           # before mass is causal, only formation targets are eligible
LOOKAHEAD = 100
LOOKAHEAD_MAX_ACC_DROP = 0.02                # lookahead end accuracy vs the unedited lookahead
LOOKAHEAD_MAX_LOSS_RATIO = 3.0               # max training loss in the edited lookahead vs unedited max
SHAM_N, SHAM_S = 3, 28_028


# ------------------------------------------------------------------ schema
class ProposalError(ValueError):
    pass


def parse_scope(sel, m):
    mt = re.fullmatch(r"([A-Za-z0-9_.]+)(?:\[(head|rows):([0-9]+)(?:-([0-9]+))?\])?", sel)
    if not mt:
        raise ProposalError(f"bad scope selector {sel!r}")
    name, kind, a, b = mt.groups()
    if name not in ALLOWED_TENSORS:
        raise ProposalError(f"tensor {name!r} not allowed")
    P = dict(m.named_parameters())[name]
    mask = torch.zeros_like(P, dtype=torch.bool)
    if kind is None:
        mask[...] = True
    elif kind == "rows":
        mask[int(a):int(b or a) + 1] = True
    else:
        hd = int(a)
        if not 0 <= hd < 4:
            raise ProposalError("head index out of range")
        if ".qkv." in name:
            rows = [p * 64 + 16 * hd + i for p in range(3) for i in range(16)]
            mask[rows] = True
        elif name.endswith(".o.weight"):
            mask[:, 16 * hd:16 * (hd + 1)] = True
        else:
            raise ProposalError(f"head selector not defined for {name}")
    return name, mask


def check_schema(p, m):
    req = ("primitive", "scope", "target_claim", "rationale", "mechanism", "predicted", "falsification", "collateral_risk")
    for k in req:
        if k not in p:
            raise ProposalError(f"missing field {k!r}")
    if p["primitive"] not in ("restricted_gradient_update", "low_rank_adjust", "counterfactual_supervision"):
        raise ProposalError("unknown primitive")
    if p["target_claim"] not in TARGETS:
        raise ProposalError("unknown target_claim")
    scopes = [parse_scope(s, m) for s in p["scope"]]
    if p.get("max_rel_model", LIMITS["max_rel_model"]) > LIMITS["max_rel_model"]:
        raise ProposalError("budget above limit")
    if p["primitive"] in ("restricted_gradient_update", "counterfactual_supervision"):
        if p.get("steps", 300) > LIMITS["max_steps"] or p.get("lr", 1e-3) > LIMITS["max_lr"]:
            raise ProposalError("steps / lr above limit")
        for k in p.get("objective", {}):
            if k not in OBJECTIVES:
                raise ProposalError(f"unknown objective term {k!r}")
    if p["primitive"] == "counterfactual_supervision":
        if p.get("variable") not in ("M", "A", "S", "m", "a") or p.get("site") not in ("block", "rest", "head0", "head1", "head2", "head3"):
            raise ProposalError("counterfactual_supervision needs variable and site")
    if p["primitive"] == "low_rank_adjust":
        if len(scopes) != 1 or p.get("rank", 1) > LIMITS["max_rank"] or p.get("target") not in LOW_RANK_TARGETS:
            raise ProposalError("low_rank_adjust needs one scope, rank <= max_rank and a registered target")
    return scopes


# ------------------------------------------------------------------ objective terms
def _forward(m, tok):
    x1 = m.resid_layer1(tok, torch.zeros_like(tok))
    a2, pat2, pre2 = AT._attn_parts(m.attn2, x1)
    h = x1[:, -1] + m.attn2.o(pre2[:, -1])
    return h, pre2[:, -1]


def _design(M, R):
    cells = sorted({(int(a), int(b)) for a, b in zip(M.tolist(), R.tolist())})
    X = torch.zeros(len(cells), 1 + T.P + 6)
    for i, (c, q) in enumerate(cells):
        X[i, 0], X[i, 1 + c], X[i, 1 + T.P + q] = 1, 1, 1
    idx = {cq: i for i, cq in enumerate(cells)}
    cell_of = torch.tensor([idx[(int(a), int(b))] for a, b in zip(M.tolist(), R.tolist())])
    return cell_of, torch.eye(len(cells)) - X @ torch.linalg.pinv(X), len(cells)


def _interaction(Z, cell_of, Pr, nc, M):
    cnt = torch.bincount(cell_of, minlength=nc).float()
    cm = torch.zeros(nc, Z.shape[1]).index_add(0, cell_of, Z) / cnt[:, None]
    inter = Pr @ cm
    mu = torch.zeros(T.P, Z.shape[1]).index_add(0, M, Z) / torch.bincount(M, minlength=T.P).float().clamp_min(1)[:, None]
    return (cnt[:, None] * inter ** 2).sum() / ((Z - mu[M]) ** 2).sum()


INV = [0] + [pow(x, -1, T.P) for x in range(1, T.P)]
ADD1 = [(x + 1) % T.P for x in range(T.P)]


def run_gradient(m, scopes, objective, steps, lr, l2, seed, variable="M", site="block"):
    mm = copy.deepcopy(m)
    P = dict(mm.named_parameters())
    names = [n for n, _ in scopes]
    masks = {n: mk for n, mk in scopes}
    W0 = {n: P[n].detach().clone() for n in names}
    for k, p in P.items():
        p.requires_grad_(k in names)
    opt = torch.optim.Adam([P[n] for n in names], lr=lr)
    d = AT.partition()
    fit, src = d["FIT"], d["FIT_src"]
    b = T.build(fit)
    tok, y = b.tokens, b.y
    fo = AT.factors(fit)
    cell_of, Pr, nc = _design(fo["M"], fo["route"])
    with torch.no_grad():
        p0 = torch.log_softmax(m(tok, torch.zeros_like(tok)), -1)
        h0, _ = _forward(m, tok)
        cmu = torch.stack([h0[fo["M"] == c].mean(0) for c in range(T.P)])
        blk_sd = (h0 - cmu[fo["M"]]).std(0)
        Zb = torch.cat([h0[:, AT.BLK], torch.ones(len(h0), 1)], 1)
        Wr = torch.linalg.lstsq(Zb, F.one_hot(fo["route"], 6).float()).solution
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    M_all, A_all, S_all = T.canon(fit)
    for _ in range(steps):
        h, pre = _forward(mm, tok)
        logits = mm.head_from_resid(h)
        L = 0.0
        w = objective
        if w.get("kl"):
            L = L + w["kl"] * F.kl_div(torch.log_softmax(logits, -1), p0, log_target=True, reduction="batchmean")
        if w.get("task"):
            L = L + w["task"] * F.cross_entropy(logits, y)
        if w.get("interaction"):
            L = L + w["interaction"] * _interaction(h[:, AT.BLK], cell_of, Pr, nc, fo["M"])
        if w.get("iit"):
            j = torch.randperm(n, generator=g)
            srcl = {k: v[j] for k, v in fit.items()}
            if site in ("block", "rest"):
                dims = AT.BLK if site == "block" else list(range(16, 64))
                hp = h.clone()
                hp[:, dims] = h[j][:, dims]
            else:
                hd = int(site[-1])
                pre_p = pre.clone()
                pre_p[:, 16 * hd:16 * (hd + 1)] = pre[j][:, 16 * hd:16 * (hd + 1)]
                x1 = mm.resid_layer1(tok, torch.zeros_like(tok))
                hp = x1[:, -1] + mm.attn2.o(pre_p)
            L = L + w["iit"] * F.cross_entropy(mm.head_from_resid(hp), D.counterfactual(fit, srcl, variable))
        if w.get("translate"):
            mu = torch.stack([h[M_all == c][:, AT.BLK].mean(0) for c in range(T.P)])
            while True:
                pi = torch.randperm(T.P, generator=g)
                if pi.tolist() not in (INV, ADD1, list(range(T.P))):
                    break
            hp = h.clone()
            hp[:, AT.BLK] = h[:, AT.BLK] + mu[pi[M_all]] - mu[M_all]
            L = L + w["translate"] * F.cross_entropy(mm.head_from_resid(hp), T.target(pi[M_all], A_all, S_all))
        if w.get("noise_kl"):
            with torch.no_grad():
                eps = torch.randn(h0.shape, generator=g) * blk_sd
                eps[:, 16:] = 0
                pn = torch.log_softmax(m.head_from_resid(h0 + eps), -1)
            L = L + w["noise_kl"] * F.kl_div(torch.log_softmax(mm.head_from_resid(h + eps), -1), pn, log_target=True,
                                             reduction="batchmean")
        if w.get("route_keep"):
            zr = torch.cat([h[:, AT.BLK], torch.ones(n, 1)], 1) @ Wr
            L = L + w["route_keep"] * F.cross_entropy(zr * 5.0, fo["route"])
        if w.get("separate_MA"):
            # mass block should not carry acceleration: penalise A's class-mean variance in the block
            muA = torch.zeros(T.P, 16).index_add(0, fo["A"], h[:, AT.BLK]) / torch.bincount(fo["A"], minlength=T.P).float().clamp_min(1)[:, None]
            L = L + w["separate_MA"] * ((muA - muA.mean(0)) ** 2).sum() / ((h[:, AT.BLK] - h[:, AT.BLK].mean(0)) ** 2).sum() * n
        if l2:
            L = L + l2 * sum(((P[k] - W0[k]) ** 2).sum() / (W0[k] ** 2).sum().clamp_min(1e-12) for k in names)
        opt.zero_grad()
        L.backward()
        for k in names:                                         # restrict to the selected slice
            P[k].grad[~masks[k]] = 0
        opt.step()
        with torch.no_grad():
            for k in names:
                P[k][~masks[k]] = W0[k][~masks[k]]
    return {k: (P[k].detach() - W0[k]).clone() for k in names}


@torch.no_grad()
def run_low_rank(m, scope, target, rank, ridge=1e-2):
    """Rank-r correction of one tensor slice fitted to a registered activation target on FIT."""
    name, mask = scope
    d = AT.partition()
    fit = d["FIT"]
    c = AT.capture(m, fit)
    fo = AT.factors(fit)
    P = dict(m.named_parameters())[name]
    if target == "block_interaction_removal":
        if name != "attn2.o.weight":
            raise ProposalError("block_interaction_removal acts on attn2.o.weight")
        _, tab = AT.interaction_decomp(c["h"][:, AT.BLK], fo["M"], fo["route"])
        resid = -tab[fo["M"], fo["route"]]
        X = c["pre2"]
        rows = AT.BLK
        W = torch.linalg.solve(X.T @ X + ridge * len(X) * torch.eye(64), X.T @ resid)     # (64, 16)
        full = torch.zeros_like(P)
        full[rows] = W.T
    elif target in ("head_canonical_mass", "head_interaction_removal"):
        if name != "attn2.qkv.weight":
            raise ProposalError("head targets act on attn2.qkv.weight[head:k]")
        hd = [h for h in range(4) if mask[128 + 16 * h].any()]
        if len(hd) != 1:
            raise ProposalError("select exactly one head")
        hd = hd[0]
        a = (c["pat2"][:, hd, -1, :, None] * c["x1"]).sum(1)
        out = c["pre2"][:, 16 * hd:16 * (hd + 1)]
        _, tab = AT.interaction_decomp(out, fo["M"], fo["route"])
        if target == "head_interaction_removal":
            resid = -tab[fo["M"], fo["route"]]
        else:
            sel = (fo["mu"] == 0) | (fo["mu"] == 2)
            canon = AT.class_means(out[sel], fo["M"][sel], T.P)
            cell = out - tab[fo["M"], fo["route"]]
            resid = canon[fo["M"]] - AT.class_means(out, fo["M"], T.P)[fo["M"]] + (cell - out)
        W = torch.linalg.solve(a.T @ a + ridge * len(a) * torch.eye(64), a.T @ resid)
        full = torch.zeros_like(P)
        full[128 + 16 * hd:128 + 16 * (hd + 1)] = W.T
    else:
        raise ProposalError("unknown low-rank target")
    U, Sv, Vh = torch.linalg.svd(full, full_matrices=False)
    r = min(rank, len(Sv))
    low = (U[:, :r] * Sv[:r]) @ Vh[:r]
    low[~mask] = 0
    low[full.abs().sum(1) == 0] = 0          # only the rows the target was fitted on (no SVD round-off)
    return {name: low}


def scale_delta(m, delta, max_rel_model, max_rel_tensor=LIMITS["max_rel_tensor"]):
    W = dict(m.named_parameters())
    mn = torch.cat([p.detach().flatten() for p in m.parameters()]).norm()
    tot = torch.cat([v.flatten() for v in delta.values()]).norm().clamp_min(1e-12)
    s = min(1.0, (max_rel_model * mn / tot).item())
    for k, v in delta.items():
        s = min(s, (max_rel_tensor * W[k].detach().norm() / v.norm().clamp_min(1e-12)).item())
    return {k: v * s for k, v in delta.items()}, s


def execute(p, m):
    """Compile a validated proposal into a parameter delta. Returns (delta, info)."""
    scopes = check_schema(p, m)
    t0 = time.process_time()
    extra_steps = 0
    if p["primitive"] == "low_rank_adjust":
        delta = run_low_rank(m, scopes[0], p["target"], p.get("rank", 1))
        delta = {k: v * p.get("alpha", 1.0) for k, v in delta.items()}
    else:
        obj = dict(p.get("objective", {}))
        if p["primitive"] == "counterfactual_supervision":
            obj.setdefault("iit", 1.0)
        steps = p.get("steps", 300)
        extra_steps = steps
        delta = run_gradient(m, scopes, obj, steps, p.get("lr", 1e-3), p.get("l2", 3.0), p.get("seed", 0),
                             variable=p.get("variable", "M"), site=p.get("site", "block"))
        delta = {k: v * p.get("scale", 1.0) for k, v in delta.items()}
    delta, s = scale_delta(m, delta, p.get("max_rel_model", LIMITS["max_rel_model"]))
    delta = {k: v for k, v in delta.items() if v.abs().sum() > 0}
    return delta, dict(compile_cpu_s=round(time.process_time() - t0, 1), extra_grad_steps=extra_steps, budget_scale=round(s, 4))


# ------------------------------------------------------------------ validation and rollback
@torch.no_grad()
def metrics(m, split="VAL"):
    d = AT.partition()
    fit, val, vs = d["FIT"], d[split], d[split + "_src"]
    cf, cv, cvs = AT.capture(m, fit), AT.capture(m, val), AT.capture(m, vs)
    ff, fv = AT.factors(fit), AT.factors(val)
    pe = AT.pooled_edit(m, fit, val, h_fit=cf["h"], h_eval=cv["h"])
    dec, _ = AT.interaction_decomp(cv["h"][:, AT.BLK], fv["M"], fv["route"])
    tr, wi = [], []
    for u in range(3):
        trn, tst = ff["mu"] != u, fv["mu"] == u
        tr.append(AT.ridge_probe(cf["h"][trn][:, AT.BLK], ff["M"][trn], cv["h"][tst][:, AT.BLK], fv["M"][tst], AT.P))
        wi.append(AT.ridge_probe(cf["h"][:, AT.BLK], ff["M"], cv["h"][tst][:, AT.BLK], fv["M"][tst], AT.P))
    y = T.build(val).y
    pred = cv["logits"].argmax(-1)
    return dict(edit_inv=pe["intended"], wrong=pe["wrong"], interaction=dec["interaction_share"],
                unit_transfer=sum(tr) / max(sum(wi), 1e-9), ood_dev=AT.accuracy(m, d["OOD_DEV"]),
                iia=AT.swap_h(m, val, vs, AT.BLK, "M", cv, cvs), acc=(pred == y).float().mean().item(),
                pred=pred, correct=pred == y,
                route_probe=AT.ridge_probe(cf["h"][:, AT.BLK], ff["route"], cv["h"][:, AT.BLK], fv["route"], 6),
                unit_probe=AT.ridge_probe(cf["h"], ff["mu"], cv["h"], fv["mu"], 3))


def phase_of(before):
    if before["iia"] < 0.5:
        return "early"
    if before["iia"] < 0.9 or before["acc"] < 0.95:
        return "forming"
    return "mature"


def gain(target, b, a):
    return b["interaction"] - a["interaction"] if target == "interaction" else a[target] - b[target]


def _apply(m, delta, sign=1.0):
    P = dict(m.named_parameters())
    with torch.no_grad():
        for k, v in delta.items():
            P[k].add_(sign * v)


def clone_state(state):
    """Exact copy of the full training state (model, optimizer, scheduler, generators, step) by
    serialization (generators cannot be deep-copied)."""
    import io
    buf = io.BytesIO()
    C.save_state(state, buf)
    buf.seek(0)
    rng = torch.get_rng_state()
    st = C.load_state(buf)
    torch.set_rng_state(rng)
    return st


def lookahead(state, delta, n=LOOKAHEAD):
    """Deterministic lookahead on COPIES of the full training state: the same batches the live run
    will see. Returns per-copy (max training loss, end accuracy on VAL) for unedited and edited."""
    res = {}
    rng = torch.get_rng_state()
    for name, dl in (("unedited", None), ("edited", delta)):
        st = clone_state(state)
        if dl:
            _apply(st["model"], dl)
        losses = []
        m, opt, sched = st["model"], st["opt"], st["sched"]
        mk = D.mask(*S.BLOCK)
        target_step = min(st["step"] + n, S.STEPS)
        for step in range(st["step"] + 1, target_step + 1):
            lat_b = T.sample_latents(S.BATCH, st["g_base"])
            bb = T.build(lat_b)
            _, hb = D.encode(m, bb.tokens)
            loss = F.cross_entropy(m.head_from_resid(hb), bb.y)
            lat_s = T.sample_latents(S.BATCH, st["g_src"])
            _, hs = D.encode(m, T.build(lat_s).tokens)
            loss = loss + S.lam_at(step, "warmup") * F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk),
                                                                      D.counterfactual(lat_b, lat_s, "M"))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            st["step"] = step
            losses.append(loss.item())
        res[name] = dict(max_loss=max(losses) if losses else 0.0, end_acc=AT.accuracy(m, AT.partition()["VAL"]),
                         end_metrics={k: v for k, v in metrics(m).items() if k not in ("pred", "correct")})
    torch.set_rng_state(rng)
    return res


def validate(state, delta, target, cum_used, run_lookahead=True):
    """All checks on copies of the state. Never mutates `state`."""
    m = state["model"]
    snap = state_fingerprint(state)
    mm = copy.deepcopy(m)
    before = metrics(mm)
    phase = phase_of(before)
    th = PHASES[phase]
    mn = torch.cat([p.detach().flatten() for p in m.parameters()]).norm().item()
    rel_m = torch.cat([v.flatten() for v in delta.values()]).norm().item() / mn
    _apply(mm, delta)
    after = metrics(mm)
    _apply(mm, delta, -1.0)
    g = gain(target, before, after)
    keep = after["correct"][before["correct"]].float().mean().item() if before["correct"].any() else 1.0
    v = dict(phase=phase, target=target, gain=g, threshold=TARGETS[target], rel_norm_model=rel_m, keep_rate=keep,
             before={k: x for k, x in before.items() if k not in ("pred", "correct")},
             after={k: x for k, x in after.items() if k not in ("pred", "correct")})
    v["E0_eligible"] = phase != "early" or target in EARLY_TARGETS
    v["G_target"] = g >= TARGETS[target] and (target != "interaction" or after["edit_inv"] >= before["edit_inv"])
    v["S1_acc"] = after["acc"] >= before["acc"] - th["S1_acc"]
    v["S2_keep"] = keep >= th["S2_keep"]
    v["S3_ood_dev"] = after["ood_dev"] >= before["ood_dev"] - th["S3_ood"]
    v["S4_iia"] = th["S4_iia"] is None or after["iia"] >= before["iia"] - th["S4_iia"]
    v["S5_unit_info"] = (after["route_probe"] >= before["route_probe"] - th["S5_info"]
                         and after["unit_probe"] >= before["unit_probe"] - th["S5_info"])
    v["S6_wrong"] = after["wrong"] <= before["wrong"] + th["S6_wrong"]
    v["S7_budget"] = rel_m <= LIMITS["max_rel_model"] + 1e-9 and cum_used + rel_m <= LIMITS["max_cum_rel_model"]
    checks = ["E0_eligible", "G_target", "S1_acc", "S2_keep", "S3_ood_dev", "S4_iia", "S5_unit_info", "S6_wrong", "S7_budget"]
    if all(v[k] for k in checks):
        g_ = torch.Generator().manual_seed(SHAM_S)
        sg = []
        for _ in range(SHAM_N):
            sham = {}
            for k, x in delta.items():
                r = torch.randn(x.shape, generator=g_)
                r[x == 0] = 0
                sham[k] = r * x.norm() / r.norm().clamp_min(1e-12)
            _apply(mm, sham)
            sg.append(gain(target, before, metrics(mm)))
            _apply(mm, sham, -1.0)
        v["S8_sham_gains"] = sg
        v["S8_sham"] = g - sum(sg) / len(sg) >= TARGETS[target] / 2
        checks.append("S8_sham")
        if v["S8_sham"] and run_lookahead:
            la = lookahead(state, delta)
            v["lookahead"] = la
            v["S9_stability"] = (la["edited"]["max_loss"] <= LOOKAHEAD_MAX_LOSS_RATIO * max(la["unedited"]["max_loss"], 0.05)
                                 and la["edited"]["end_acc"] >= la["unedited"]["end_acc"] - LOOKAHEAD_MAX_ACC_DROP)
            checks.append("S9_stability")
    v["passed"] = all(v.get(k, False) for k in checks)
    v["failed"] = [k for k in checks if not v.get(k, False)]
    assert state_fingerprint(state) == snap, "validation mutated the live training state"
    return v


def state_fingerprint(state):
    """Hash of model, optimizer, scheduler, generators and step (rollback check)."""
    import hashlib
    h = hashlib.sha256()
    for k, v in state["model"].state_dict().items():
        h.update(k.encode())
        h.update(v.detach().numpy().tobytes())
    for k, sd in state["opt"].state_dict()["state"].items():
        for kk, vv in sd.items():
            h.update(f"{k}{kk}".encode())
            h.update(vv.detach().numpy().tobytes() if torch.is_tensor(vv) else str(vv).encode())
    h.update(json.dumps(state["sched"].state_dict(), sort_keys=True, default=str).encode())
    h.update(state["g_base"].get_state().numpy().tobytes())
    h.update(state["g_src"].get_state().numpy().tobytes())
    h.update(str(state["step"]).encode())
    return h.hexdigest()


def install(state, delta):
    """Apply an ACCEPTED delta to the live model. Optimizer policy (docs/28): Adam moments unchanged."""
    _apply(state["model"], delta)
