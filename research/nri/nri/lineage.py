"""Concept Lineage Engine (spec v1.0 §3-4; registration docs/28).

Components (all read-only on the model):
  matched_panel()     a fixed diagnostic panel of matched examples: each base question with variants
                      that change exactly one factor (same canonical mass in another unit = EQUIVALENT;
                      another mass; another acceleration; another output unit; swapped order).
                      Variants are drawn only from non-sealed questions.
  trace()             token / example lineage on the panel: activations at registered sites, gradients of
                      the correct-answer log-probability w.r.t. input embeddings (per position) and the
                      block of h, equivalence similarity, and semantic-GROUP parameter gradients for
                      selected tensor groups with their pairwise agreement (cosine). Token-embedding
                      gradients and example-level parameter gradients are recorded separately (attention
                      makes them different quantities).
  candidates()        candidate features at a checkpoint: attention-head roles, MLP neuron clusters
                      (k-means on response signatures), and concept subspaces per site.
  LineageTracker      persistent IDs, matching across checkpoints (signatures, membership, subspace
                      overlap), lineage events (emergence, consolidation, specialization, splitting,
                      merging, relocation, disappearance, persistence) with match confidence, and
                      concept records with evidence status.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import run_dimensional as D
import run_perm as RP
import run_swap as S
from nri import atlas as AT
from nri import dimtask as T

KEYS = ("m", "mu", "a", "au", "fu", "order")
PANEL_S = 31_337
N_BASE = 48
TRACE_TENSORS = {"attn1": ["attn1.qkv.weight", "attn1.o.weight"], "attn2": ["attn2.qkv.weight", "attn2.o.weight"],
                 "mlp_in": ["W_in.weight"], "mlp_out": ["W_out.weight"], "unembed": ["W_U.weight"]}
K_CLUSTERS = 10
MATCH_STRONG, MATCH_WEAK = 0.8, 0.5


# ------------------------------------------------------------------ matched diagnostic panel
def _valid(q, sealed):
    m, mu, a, au, fu, o = q
    if mu == T.OOD_UNIT and m >= 9 or au == T.OOD_UNIT and a >= 9:
        return False
    return q not in sealed


def matched_panel():
    """Base questions from OBS; for each, single-factor variants kept only if they are non-sealed ID
    questions. Returns (lat, meta) where meta[i] = (base_index, variant_kind)."""
    d = AT.partition()
    sealed = RP.reserved_set()
    obs = list(zip(*[d["OBS"][k].tolist() for k in KEYS]))
    g = torch.Generator().manual_seed(PANEL_S)
    bases = [obs[i] for i in torch.randperm(len(obs), generator=g)[:N_BASE * 3].tolist()]
    rows, meta, nb = [], [], 0
    for q in bases:
        m, mu, a, au, fu, o = q
        M = m * T.MASS_S[mu] % T.P
        var = {}
        for u2 in range(3):                                         # same canonical mass, other unit
            if u2 != mu:
                m2 = M * pow(T.MASS_S[u2], -1, T.P) % T.P
                c = (m2, u2, a, au, fu, o)
                if _valid(c, sealed):
                    var.setdefault("equivalent", c)
        for dm in (1, 2, 3):
            c = ((m + dm) % T.P, mu, a, au, fu, o)
            if _valid(c, sealed) and M != 0:
                var.setdefault("other_mass", c)
        for da in (1, 2, 3):
            c = (m, mu, (a + da) % T.P, au, fu, o)
            if _valid(c, sealed):
                var.setdefault("other_acc", c)
        c = (m, mu, a, au, (fu + 1) % 3, o)
        if _valid(c, sealed):
            var["other_out_unit"] = c
        c = (m, mu, a, au, fu, 1 - o)
        if _valid(c, sealed):
            var["swapped_order"] = c
        if len(var) < 5:
            continue
        rows.append(q)
        meta.append((nb, "base"))
        for k, c in var.items():
            rows.append(c)
            meta.append((nb, k))
        nb += 1
        if nb == N_BASE:
            break
    return RP.as_lat(rows), meta


_PANEL = {}


def panel():
    if not _PANEL:
        lat, meta = matched_panel()
        _PANEL.update(lat=lat, meta=meta)
    return _PANEL["lat"], _PANEL["meta"]


# ------------------------------------------------------------------ token / example tracer
def _forward_traced(m, lat):
    tok = T.build(lat).tokens
    x0 = m.embed(tok, torch.zeros_like(tok))
    x0 = x0.detach().requires_grad_(True)
    x1 = x0 + m.attn1(x0)
    h = (x1 + m.attn2(x1))[:, -1]
    h.retain_grad()
    logits = m.head_from_resid(h)
    return x0, x1, h, logits


def trace(m):
    """Token and example lineage on the matched panel (see module docstring)."""
    lat, meta = panel()
    y = T.build(lat).y
    x0, x1, h, logits = _forward_traced(m, lat)
    lp = torch.log_softmax(logits, -1)[torch.arange(len(y)), y]
    lp.sum().backward()
    tok_grad = x0.grad.norm(dim=-1)                                # (n, 8) per-position saliency
    blk_grad = h.grad[:, AT.BLK].norm(dim=-1)
    rp = AT.role_pos(lat)
    idx = torch.arange(len(y))
    roles = {r: tok_grad[idx, rp[r]] for r in ("mval", "munit", "aval", "aunit")}
    roles["out_unit"] = tok_grad[:, 6]
    tot = tok_grad.sum(1).clamp_min(1e-12)
    saliency = {r: round((v / tot).mean().item(), 4) for r, v in roles.items()}
    # matched comparisons at the block and at the layer-1 value token
    with torch.no_grad():
        Z = h[:, AT.BLK].detach()
        V = x1.detach()[idx, rp["mval"]]
        base_of = {}
        for i, (b, k) in enumerate(meta):
            if k == "base":
                base_of[b] = i
        sims = {}
        for kind in ("equivalent", "other_mass", "other_acc", "other_out_unit", "swapped_order"):
            pb, pv, vb, vv = [], [], [], []
            for i, (b, k) in enumerate(meta):
                if k == kind:
                    pb.append(Z[base_of[b]])
                    pv.append(Z[i])
                    vb.append(V[base_of[b]])
                    vv.append(V[i])
            if pb:
                sims[kind] = dict(block_cos=round(F.cosine_similarity(torch.stack(pb), torch.stack(pv)).mean().item(), 4),
                                  block_dist=round((torch.stack(pb) - torch.stack(pv)).norm(dim=1).mean().item(), 4),
                                  value_token_cos=round(F.cosine_similarity(torch.stack(vb), torch.stack(vv)).mean().item(), 4))
        pred = logits.argmax(-1)
        eq_agree = []
        for i, (b, k) in enumerate(meta):
            if k == "equivalent":
                eq_agree.append(float(pred[i] == pred[base_of[b]]))
    # semantic-group parameter gradients (example-level, averaged within group)
    groups = {f"mass_unit_{u}": lat["mu"] == u for u in range(3)}
    groups.update({f"acc_unit_{u}": lat["au"] == u for u in range(3)})
    P = dict(m.named_parameters())
    gg = {}
    for gname, sel in groups.items():
        if sel.sum() < 4:
            continue
        sub = {k: v[sel] for k, v in lat.items()}
        loss = F.cross_entropy(m(T.build(sub).tokens, torch.zeros_like(T.build(sub).tokens)), T.build(sub).y)
        names = [n for ts in TRACE_TENSORS.values() for n in ts]
        grads = torch.autograd.grad(loss, [P[n] for n in names], allow_unused=True)
        gg[gname] = {n: (g_.flatten() if g_ is not None else torch.zeros(P[n].numel())) for n, g_ in zip(names, grads)}
    agreement = {}
    for grp, ts in TRACE_TENSORS.items():
        for a_, b_ in (("mass_unit_0", "mass_unit_1"), ("mass_unit_0", "mass_unit_2"), ("mass_unit_1", "mass_unit_2"),
                       ("mass_unit_0", "acc_unit_0")):
            if a_ in gg and b_ in gg:
                va = torch.cat([gg[a_][n] for n in ts])
                vb = torch.cat([gg[b_][n] for n in ts])
                if va.norm() > 0 and vb.norm() > 0:
                    agreement[f"{grp}:{a_}~{b_}"] = round(F.cosine_similarity(va, vb, dim=0).item(), 4)
    gnorm = {f"{grp}:{g_}": round(torch.cat([gg[g_][n] for n in ts]).norm().item(), 5)
             for grp, ts in TRACE_TENSORS.items() for g_ in gg}
    return dict(token_saliency=saliency, block_grad_norm=round(blk_grad.mean().item(), 5),
                matched_similarity=sims, equivalent_pairs_same_prediction=round(sum(eq_agree) / max(len(eq_agree), 1), 4),
                group_gradient_agreement=agreement, group_gradient_norm=gnorm,
                panel_size=len(meta), panel_bases=sum(1 for _, k in meta if k == "base"))


# ------------------------------------------------------------------ candidate features
def _kmeans(X, k, iters=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    k = min(k, n)
    C = X[torch.randperm(n, generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(0)
    return a, C


@torch.no_grad()
def candidates(m, card, importance=None):
    """Candidate features at one checkpoint (OBS panel)."""
    d = AT.partition()
    obs = d["OBS"]
    c = AT.capture(m, obs)
    fo = AT.factors(obs)
    cz = card["causal"]
    out = []
    # 1. attention-head roles (function may relocate across head indices)
    rp = AT.role_pos(obs)
    idx = torch.arange(len(obs["m"]))
    for L, pat, pre, o in (("attn1", c["pat1"], None, None), ("attn2", c["pat2"], c["pre2"], m.attn2.o)):
        for hd in range(4):
            att = [pat[idx, hd, -1, rp[r]].mean().item() for r in ("mval", "munit", "aval", "aunit")] + \
                  [pat[:, hd, -1, 6].mean().item()]
            ev = dict(attends=dict(zip(("mass_value", "mass_unit", "acc_value", "acc_unit", "out_unit"), [round(x, 3) for x in att])))
            sig = torch.tensor(att + [0.0, 0.0])
            if L == "attn2":
                ev["iia_M"] = round(cz[f"iia_M_head{hd}"], 3)
                ev["iia_A"] = round(cz[f"iia_A_head{hd}"], 3)
                contrib = pre[:, 16 * hd:16 * (hd + 1)] @ o.weight[:, 16 * hd:16 * (hd + 1)].T
                ev["block_write_rms"] = round(contrib[:, AT.BLK].pow(2).sum(1).mean().sqrt().item(), 3)
                sig[5], sig[6] = ev["iia_M"], ev["iia_A"]
            out.append(dict(kind="head", site=f"{L}.h{hd}", signature=sig, members=[hd], evidence=ev))
    # 2. MLP neuron clusters on response signatures (active neurons only)
    act = c["act"]
    lab, et = AT.families(act, fo)
    sig = F.normalize(AT.signatures(act, fo).nan_to_num(0), dim=1)
    active = torch.tensor([l != "dead" for l in lab])
    ids = active.nonzero().squeeze(1)
    if len(ids) >= K_CLUSTERS:
        a, C = _kmeans(sig[ids], K_CLUSTERS)
        imp = importance.float() if importance is not None else torch.zeros(act.shape[1])
        for j in range(len(C)):
            mem = ids[a == j].tolist()
            if not mem:
                continue
            em = et[mem].mean(0)
            top = int(em.argmax())
            out.append(dict(kind="neuron_cluster", site="mlp", signature=F.normalize(C[j], dim=0), members=mem,
                            evidence=dict(size=len(mem), dominant_factor=AT.ETA_FACTORS[top],
                                          dominant_eta2=round(em[top].item(), 3),
                                          selectivity=round((em.max() / em.sum().clamp_min(1e-9)).item(), 3),
                                          ablation_dloss=round(imp[mem].sum().item(), 4))))
    # 3. concept subspaces per site
    sites = dict(L1_mass_value=c["x1"][idx, rp["mval"]], block=c["h"][:, AT.BLK], h=c["h"], mlp=act)
    for conc in ("M", "A", "m", "mu", "F", "route"):
        for sname, Z in sites.items():
            Q, mu_ = AT.mean_subspace(Z, fo[conc], AT.NCLS[conc])
            et_ = AT.eta2(Z, fo[conc], AT.NCLS[conc])
            share = ((mu_ ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum(1).mean().clamp_min(1e-12)).item() / len(Z) * len(mu_)
            out.append(dict(kind="subspace", site=f"{conc}@{sname}", signature=None, members=[], Q=Q,
                            evidence=dict(rank90=Q.shape[1], kappa=card["decode_kappa"].get(f"{conc}@{ {'L1_mass_value': 'L1@mval', 'block': 'h_block', 'h': 'h', 'mlp': 'mlp'}[sname]}"),
                                          mean_eta2=round(et_.mean().item(), 3))))
    return out


# ------------------------------------------------------------------ lineage tracker
def _status(ev, causal):
    ab = ev.get("ablation_dloss")
    if ab is not None:                 # neuron clusters: summed single-neuron mean-ablation loss increase (nats)
        if ab >= 0.3:
            return "causally supported"
        if ab >= 0.05:
            return "causally supported (partial)"
    if causal is not None and causal >= 0.9:
        return "causally supported"
    if causal is not None and causal >= 0.5:
        return "causally supported (partial)"
    k = ev.get("kappa")
    if (k is not None and k >= 0.5) or ev.get("dominant_eta2", 0) >= AT.FAMILY_ETA:
        return "correlationally supported"
    return "hypothesized"


class LineageTracker:
    """Persistent feature IDs and lineage events across checkpoints for ONE network."""

    def __init__(self, name):
        self.name = name
        self.records = {}          # id -> record
        self.prev = []             # list of (id, candidate)
        self.counter = 0
        self.events = []

    def _new_id(self, kind):
        self.counter += 1
        return f"{self.name[0].upper()}-{kind[:2]}-{self.counter:04d}"

    def _sim(self, a, b):
        if a["kind"] != b["kind"]:
            return 0.0
        if a["kind"] == "subspace":
            return 1.0 if a["site"] == b["site"] else 0.0
        s = F.cosine_similarity(a["signature"], b["signature"], dim=0).item()
        if a["kind"] == "neuron_cluster":
            A, B = set(a["members"]), set(b["members"])
            jac = len(A & B) / max(len(A | B), 1)
            return 0.7 * s + 0.3 * jac
        return s

    def update(self, step, cands, card):
        from scipy.optimize import linear_sum_assignment
        evs = []
        cur = []
        prev = self.prev
        causal_of = self._causal_lookup(card)
        for kind in ("head", "neuron_cluster", "subspace"):
            P_ = [(i, c) for i, c in prev if c["kind"] == kind]
            C_ = [c for c in cands if c["kind"] == kind]
            if kind == "subspace":
                pm = {c["site"]: (i, c) for i, c in P_}
                for c in C_:
                    if c["site"] in pm:
                        fid, pc = pm[c["site"]]
                        ov = AT.subspace_overlap(pc["Q"], c["Q"])
                        cur.append((fid, c, ov, "subspace overlap"))
                    else:
                        cur.append((None, c, None, None))
                continue
            if P_ and C_:
                S_ = torch.tensor([[self._sim(p, c) for p in [x[1] for x in P_]] for c in C_])
                r, col = linear_sum_assignment(-S_.numpy())
                assigned = {}
                for ri, ci in zip(r.tolist(), col.tolist()):
                    assigned[ri] = (P_[ci][0], S_[ri, ci].item())
                for ri, c in enumerate(C_):
                    if ri in assigned and assigned[ri][1] >= MATCH_WEAK:
                        cur.append((assigned[ri][0], c, assigned[ri][1], "hungarian signature match"))
                    else:
                        cur.append((None, c, None, None))
                # splitting / merging: unmatched current candidates strongly similar to a matched previous
                for ri, c in enumerate(C_):
                    best = int(S_[ri].argmax())
                    if (ri not in assigned or assigned[ri][1] < MATCH_WEAK) and S_[ri, best] >= MATCH_STRONG:
                        evs.append(dict(event="splitting", parent=P_[best][0], site=c["site"], similarity=round(S_[ri, best].item(), 3)))
                for ci, (pid, pc) in enumerate(P_):
                    col_sim = S_[:, ci]
                    matched_here = [ri for ri in assigned if assigned[ri][0] == pid]
                    if not matched_here and col_sim.max() >= MATCH_STRONG:
                        tgt = int(col_sim.argmax())
                        evs.append(dict(event="merging", parent=pid, into_site=C_[tgt]["site"], similarity=round(col_sim.max().item(), 3)))
            else:
                cur += [(None, c, None, None) for c in C_]
        new_prev = []
        seen = set()
        for fid, c, sim, how in cur:
            caus = causal_of(c)
            status = _status(c["evidence"], caus)
            if fid is None:
                if status == "hypothesized" and c["kind"] == "subspace":
                    fid_ = self._new_id(c["kind"])
                else:
                    fid_ = self._new_id(c["kind"])
                self.records[fid_] = dict(id=fid_, kind=c["kind"], first_observed=step, site_history=[], evidence_history=[],
                                          status_history=[], parents=[], descendants=[], match_confidence=[])
                if status != "hypothesized":
                    evs.append(dict(event="emergence", id=fid_, site=c["site"], status=status))
                fid = fid_
            rec = self.records[fid]
            seen.add(fid)
            last_status = rec["status_history"][-1][1] if rec["status_history"] else None
            last_site = rec["site_history"][-1][1] if rec["site_history"] else None
            rec["site_history"].append((step, c["site"]))
            ev = {k: v for k, v in c["evidence"].items()}
            if caus is not None:
                ev["causal"] = round(caus, 3)
            rec["evidence_history"].append((step, ev))
            rec["status_history"].append((step, status))
            rec["match_confidence"].append((step, None if sim is None else round(sim, 3), how,
                                            "uncertain" if sim is not None and sim < MATCH_STRONG else "confident" if sim is not None else "new"))
            rec["members"] = c["members"]
            if last_site is not None and last_site != c["site"]:
                evs.append(dict(event="relocation", id=fid, from_site=last_site, to_site=c["site"]))
            if last_status is not None:
                order = ["refuted", "hypothesized", "correlationally supported", "causally supported (partial)", "causally supported"]
                if order.index(status) > order.index(last_status):
                    evs.append(dict(event="consolidation", id=fid, site=c["site"], from_status=last_status, to_status=status))
                elif order.index(status) < order.index(last_status) and last_status.startswith("causally"):
                    evs.append(dict(event="disappearance" if status == "hypothesized" else "weakening", id=fid, site=c["site"],
                                    from_status=last_status, to_status=status))
                else:
                    evs.append(dict(event="persistence", id=fid, site=c["site"], status=status))
            if c["kind"] == "neuron_cluster" and len(rec["evidence_history"]) >= 2:
                s0 = rec["evidence_history"][-2][1].get("selectivity", 0)
                s1 = ev.get("selectivity", 0)
                if s1 - s0 >= 0.1:
                    evs.append(dict(event="specialization", id=fid, site=c["site"], selectivity=[s0, s1]))
            new_prev.append((fid, c))
        for fid, c in self.prev:
            if fid not in seen:
                rec = self.records[fid]
                if rec["status_history"] and rec["status_history"][-1][1] != "hypothesized":
                    evs.append(dict(event="disappearance", id=fid, site=c["site"]))
                rec.setdefault("ended", step)
        for e in evs:
            e["step"] = step
        self.events += evs
        self.prev = new_prev
        return evs

    @staticmethod
    def _causal_lookup(card):
        cz = card["causal"]

        def f(c):
            if c["kind"] == "head" and c["site"].startswith("attn2"):
                hd = int(c["site"][-1])
                return max(cz[f"iia_M_head{hd}"], cz[f"iia_A_head{hd}"])
            if c["kind"] == "subspace":
                conc, site = c["site"].split("@")
                table = {("M", "block"): "iia_M_block", ("M", "L1_mass_value"): "iia_M_L1_val+unit",
                         ("m", "L1_mass_value"): "iia_m_L1_val", ("A", "block"): "iia_A_block"}
                k = table.get((conc, site))
                return cz[k] if k else None
            if c["kind"] == "neuron_cluster":
                return None          # neuron-cluster causal tests are run offline (ablation), not here
            return None
        return f

    def concept_records(self):
        out = {}
        for fid, r in self.records.items():
            st = r["status_history"][-1][1] if r["status_history"] else "unresolved"
            sites = [s for _, s in r["site_history"]]
            out[fid] = dict(id=fid, kind=r["kind"], first_observed=r["first_observed"], ended=r.get("ended"),
                            current_site=sites[-1] if sites else None, current_status=st,
                            ever_causally_supported=any(s.startswith("causally") for _, s in r["status_history"]),
                            n_checkpoints=len(sites), parents=r["parents"], descendants=r["descendants"],
                            evidence_last=r["evidence_history"][-1][1] if r["evidence_history"] else None,
                            match_confidence=r["match_confidence"][-5:])
        return out
