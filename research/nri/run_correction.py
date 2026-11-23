"""Checkpoint-guided causal correction pilot (Issue #20; registration docs/22; results docs/23).

Exploratory: 3 fresh networks (mass_only, the doc-14 recipe), checkpoints at 0 / 600 / 1,500 /
3,000 / 4,500 / 6,000 steps. At step 3,000 the full training state is saved and two branches
resume from it with identical batches:
  control     resume normally
  correction  apply the registered, validated DOWNSTREAM READOUT CORRECTION, then resume

The correction (registered in docs/22): an affine map on the declared block (dims 0-15) of the
MLP input h that sends each calibration example's block vector to its pooled M-class mean
(removing within-class route variance AS THE MLP READS IT), folded with strength alpha into the
existing W_in columns 0-15 and b_in. It does not change the activations at h when installed.
Applied only if V1-V5 pass on calibration data; alpha = the largest of (1.0, 0.5, 0.25) that passes.

The trainer reproduces run_swap.train_one's loop exactly (tests/test_correction.py checks this
bitwise) and adds snapshots and save/restore of the complete training state. run_swap.py is not
modified.

    python run_correction.py run            # 3 seeds + the resume-fidelity reference, in parallel
    python run_correction.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

import run_dimensional as D
import run_editors as E
import run_perm as RP
import run_swap as S
from nri import dimtask as T
from nri.model import TinyTyped
from nri.train import set_determinism

ROOT = Path(__file__).parent
ZOO = ROOT / "runs" / "zoo_correction"
OUT = ROOT / "results" / "correction"
SEEDS = (600, 601, 602)
GROUP, VAR, KIND = "mass_only", "M", "inv"
CKPTS = (0, 600, 1500, 3000, 4500, 6000)
BRANCH = 3000
ALPHAS = (1.0, 0.5, 0.25)            # tried largest first; the first that passes V1-V5 is used
CAL_FIT_S, CAL_VAL_S, EVAL_PAIR_S = 70_701, 70_702, 70_703   # fresh, never used before
N_CAL = 8192
V_EDIT_GAIN, V_IIA_TOL, V_ACC_TOL, V_AGREE = 0.05, 0.01, 0.005, 0.99


# ------------------------------------------------------------------ resumable trainer
def new_state(seed, group=GROUP):
    """Exactly the initial state of run_swap.train_one (same call order)."""
    set_determinism()
    torch.manual_seed(seed)
    m = TinyTyped(group, **S.MODEL_KW)
    opt = torch.optim.AdamW(m.parameters(), lr=S.LR, weight_decay=S.WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=S.LR, total_steps=S.STEPS, pct_start=0.1)
    return dict(model=m, opt=opt, sched=sched, g_base=torch.Generator().manual_seed(10_000 + seed),
                g_src=torch.Generator().manual_seed(20_000 + seed), step=0, seed=seed, group=group)


def save_state(st, path):
    torch.save(dict(model=st["model"].state_dict(), opt=st["opt"].state_dict(),
                    sched=st["sched"].state_dict(), g_base=st["g_base"].get_state(),
                    g_src=st["g_src"].get_state(), rng=torch.get_rng_state(), step=st["step"],
                    seed=st["seed"], group=st["group"]), path)


def load_state(path):
    d = torch.load(path, weights_only=False)
    st = new_state(d["seed"], d["group"])
    st["model"].load_state_dict(d["model"])
    st["opt"].load_state_dict(d["opt"])
    st["sched"].load_state_dict(d["sched"])
    st["g_base"].set_state(d["g_base"])
    st["g_src"].set_state(d["g_src"])
    torch.set_rng_state(d["rng"])
    st["step"] = d["step"]
    return st


def train(st, until, on_step=None, schedule="warmup"):
    """The loop body of run_swap.train_one, from st['step'] + 1 to `until` (inclusive).
    Returns per-step batch hashes (base and IIT-source tokens)."""
    m, opt, sched = st["model"], st["opt"], st["sched"]
    var = S.VAR_OF.get(st["group"])
    mk = D.mask(*S.BLOCK)
    hashes = []
    for step in range(st["step"] + 1, until + 1):
        lat_b = T.sample_latents(S.BATCH, st["g_base"])
        bb = T.build(lat_b)
        _, hb = D.encode(m, bb.tokens)
        loss = F.cross_entropy(m.head_from_resid(hb), bb.y)
        hsh = hashlib.sha256(bb.tokens.numpy().tobytes())
        if var:
            lat_s = T.sample_latents(S.BATCH, st["g_src"])
            tok_s = T.build(lat_s).tokens
            _, hs = D.encode(m, tok_s)
            iit = F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk), D.counterfactual(lat_b, lat_s, var))
            loss = loss + S.lam_at(step, schedule) * iit
            hsh.update(tok_s.numpy().tobytes())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        st["step"] = step
        hashes.append(hsh.hexdigest()[:16])
        if on_step:
            on_step(st)
    return hashes


def state_hash(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ frozen data
_DATA = {}


def data():
    """Calibration = non-reserved questions (run_perm split); evaluation = every reserved
    question once. Calibration and evaluation never share a question."""
    if not _DATA:
        res = RP.reserved_set()
        ev = RP.eval_lat(res)
        g = torch.Generator().manual_seed(EVAL_PAIR_S)
        src = {k: v[torch.randperm(len(ev["m"]), generator=g)] for k, v in ev.items()}
        _DATA.update(reserved=res, cal_fit=RP.pool_lats(N_CAL, CAL_FIT_S, res),
                     cal_val=RP.pool_lats(N_CAL, CAL_VAL_S, res), ev=ev, ev_src=src,
                     ood=D.lats(8192, 999_983, "ood"))
    return _DATA


BLK = list(range(*S.BLOCK))


def h_of(m, lat):
    return D.site(m, lat)[1]


@torch.no_grad()
def accuracy(m, lat, h=None):
    h = h_of(m, lat) if h is None else h
    return (m.head_from_resid(h).argmax(-1) == T.build(lat).y).float().mean().item()


@torch.no_grad()
def designed_iia(m, base, src, var=VAR):
    """Block interchange on pairs where the counterfactual differs from the base answer."""
    hb, hs = h_of(m, base), h_of(m, src)
    mk = D.mask(*S.BLOCK)
    pred = m.head_from_resid(hb * (1 - mk) + hs * mk).argmax(-1)
    cf, y0 = D.counterfactual(base, src, var), T.build(base).y
    el = cf != y0
    return (pred == cf)[el].float().mean().item()


@torch.no_grad()
def subspace_iia(m, base, src, Q, var):
    hb, hs = h_of(m, base), h_of(m, src)
    pred = m.head_from_resid(hb + (hs - hb) @ Q @ Q.T).argmax(-1)
    cf, y0 = D.counterfactual(base, src, var), T.build(base).y
    el = cf != y0
    return (pred == cf)[el].float().mean().item()


@torch.no_grad()
def pooled_edit(m, B, var, fit_lat, eval_lat):
    """The doc-15 editor (run_swap.fit_edit) with explicit fitting data; frozen scorer."""
    h = h_of(m, fit_lat)
    y = D.label(fit_lat, var)
    mu = D.class_means(h, y, T.P)
    move = mu[S.EDITS[KIND](y)] - mu[y]
    m0 = h.mean(0)
    z = (h - m0) @ B
    X = torch.cat([z, torch.ones(len(z), 1)], 1)
    W = torch.linalg.lstsq(X, move @ B).solution
    Dm = B @ W[:-1].T @ B.T
    dW, db = m.W_in.weight @ Dm, m.W_in.weight @ (B @ W[-1] - Dm @ m0)
    he = h_of(m, eval_lat)
    pred = m.head_from_resid(he, m.W_in.weight + dW, m.W_in.bias + db).argmax(-1)
    return S.score(pred, eval_lat, var, KIND)


@torch.no_grad()
def copy_block(m, lat, var=VAR):
    src = S.source_for(lat, var, KIND)
    hb, hs = h_of(m, lat), h_of(m, src)
    mk = D.mask(*S.BLOCK)
    return S.score(m.head_from_resid(hb * (1 - mk) + hs * mk).argmax(-1), lat, var, KIND)


def route_share(Z, y, r):
    """Share of within-class variance explained by the route (doc 17 measure)."""
    within, between = 0.0, 0.0
    for c in range(T.P):
        sc = y == c
        if sc.sum() < 2:
            continue
        zc = Z[sc]
        within += ((zc - zc.mean(0)) ** 2).sum().item()
        for q in r[sc].unique():
            sq = sc & (r == q)
            between += sq.sum().item() * ((Z[sq].mean(0) - zc.mean(0)) ** 2).sum().item()
    return between / max(within, 1e-12)


@torch.no_grad()
def info_necessity(m, d):
    """Diagnostic (activation-level, NOT persistent): replace the block at h by its M-class mean
    (calibration means); does ID / OOD accuracy survive without the route variance?"""
    hc = h_of(m, d["cal_fit"])
    mu = D.class_means(hc[:, BLK], D.label(d["cal_fit"], VAR), T.P)
    out = {}
    for name, lat in (("id", d["ev"]), ("ood", d["ood"])):
        h = h_of(m, lat).clone()
        h[:, BLK] = mu[D.label(lat, VAR)]
        out[name] = accuracy(m, lat, h)
    return out


def probe_acc(m, d):
    zf, ze = h_of(m, d["cal_fit"])[:, BLK], h_of(m, d["ev"])[:, BLK]
    A = torch.cat([zf, torch.ones(len(zf), 1)], 1).double()
    W = torch.linalg.lstsq(A, F.one_hot(D.label(d["cal_fit"], VAR), T.P).double()).solution
    pred = (torch.cat([ze, torch.ones(len(ze), 1)], 1).double() @ W).argmax(1)
    return (pred == D.label(d["ev"], VAR)).float().mean().item()


def audit(m, tag, folder):
    """Full checkpoint audit on frozen data. Returns the JSON card; stores acts.pt."""
    d = data()
    t0 = time.process_time()
    blk = torch.eye(64)[:, BLK]
    with torch.no_grad():
        he = h_of(m, d["ev"])
        y_ev = D.label(d["ev"], VAR)
        r_ev = E.route(d["ev"], VAR, "uo")
        z = he[:, BLK]
        card = dict(tag=tag, model_sha256=state_hash(m),
                    acc_id=accuracy(m, d["ev"], he), acc_ood=accuracy(m, d["ood"]),
                    acc_id_legacy=D.acc(m), designed_iia_M=designed_iia(m, d["ev"], d["ev_src"]),
                    designed_iia_M_legacy=D.designed_iia(m, VAR, S.BLOCK),
                    route_share_upstream=route_share(z, y_ev, r_ev),
                    route_share_readout=route_share(z @ m.W_in.weight[:, BLK].T, y_ev, r_ev),
                    probe_M_block=probe_acc(m, d), info_necessity=info_necessity(m, d),
                    edit_block=pooled_edit(m, blk, VAR, d["cal_fit"], d["ev"]),
                    copy_block=copy_block(m, d["ev"]))
    QM = RP.das_basis(m, "M", d["reserved"])
    QA = RP.das_basis(m, "A", d["reserved"])
    with torch.no_grad():
        card.update(das_iia_M=subspace_iia(m, d["ev"], d["ev_src"], QM, "M"),
                    das_block_overlap_M=((blk.T @ QM) ** 2).sum().item() / QM.shape[1],
                    edit_das_M=pooled_edit(m, QM, "M", d["cal_fit"], d["ev"]),
                    das_iia_A=subspace_iia(m, d["ev"], d["ev_src"], QA, "A"),
                    edit_das_A=pooled_edit(m, QA, "A", d["cal_fit"], d["ev"]))
        hc = h_of(m, d["cal_fit"])
        yc, rc = D.label(d["cal_fit"], VAR), E.route(d["cal_fit"], VAR, "uo")
        torch.save(dict(class_means=D.class_means(hc, yc, T.P),
                        route_means={f"{c}_{q}": hc[(yc == c) & (rc == q)].mean(0)
                                     for c in range(T.P) for q in range(6) if ((yc == c) & (rc == q)).any()},
                        sample_h=he[:512]), folder / "acts.pt")
    card["audit_cpu_s"] = round(time.process_time() - t0, 1)
    (folder / "audit.json").write_text(json.dumps(card, indent=1))
    return card


# ------------------------------------------------------------------ the registered correction
@torch.no_grad()
def canon_fold(m, cal, alpha):
    """Affine map on block coordinates -> pooled M-class means (least squares on calibration),
    folded with strength alpha into W_in[:, block] and b_in. Returns (new W_in, new b_in)."""
    h = h_of(m, cal)
    z = h[:, BLK]
    y = D.label(cal, VAR)
    target = D.class_means(z, y, T.P)[y]
    X = torch.cat([z, torch.ones(len(z), 1)], 1)
    W = torch.linalg.lstsq(X, target).solution            # z_row @ W[:-1] + W[-1]
    A, c = W[:-1], W[-1]
    Mx = torch.eye(len(BLK)) + alpha * (A - torch.eye(len(BLK)))
    Wi, bi = m.W_in.weight.clone(), m.W_in.bias.clone()
    Wi_new = Wi.clone()
    Wi_new[:, BLK] = Wi[:, BLK] @ Mx.T
    bi_new = bi + alpha * Wi[:, BLK] @ c
    return Wi_new, bi_new


@torch.no_grad()
def apply_fold(m, Wi, bi):
    m.W_in.weight.copy_(Wi)
    m.W_in.bias.copy_(bi)


def validate(m, alpha):
    """V1-V5 on calibration data only (fit on cal_fit, measure on cal_val)."""
    d = data()
    cal, val = d["cal_fit"], d["cal_val"]
    g = torch.Generator().manual_seed(CAL_VAL_S + 1)
    val_src = {k: v[torch.randperm(len(val["m"]), generator=g)] for k, v in val.items()}
    blk = torch.eye(64)[:, BLK]
    with torch.no_grad():
        before = dict(edit=pooled_edit(m, blk, VAR, cal, val)["intended"],
                      iia=designed_iia(m, val, val_src), acc=accuracy(m, val),
                      pred=m.head_from_resid(h_of(m, val)).argmax(-1))
        Wi, bi = canon_fold(m, cal, alpha)
        saved = (m.W_in.weight.clone(), m.W_in.bias.clone())
        apply_fold(m, Wi, bi)
        after = dict(edit=pooled_edit(m, blk, VAR, cal, val)["intended"],
                     iia=designed_iia(m, val, val_src), acc=accuracy(m, val),
                     pred=m.head_from_resid(h_of(m, val)).argmax(-1))
        apply_fold(m, *saved)
        dW = Wi - saved[0]
        outside = dW.clone()
        outside[:, BLK] = 0
        changed = int((dW != 0).sum()) + int((bi != saved[1]).sum())
        v = dict(alpha=alpha,
                 V1_edit_gain=after["edit"] - before["edit"], V1=after["edit"] - before["edit"] >= V_EDIT_GAIN,
                 V2_iia_change=after["iia"] - before["iia"], V2=after["iia"] >= before["iia"] - V_IIA_TOL,
                 V3_acc_change=after["acc"] - before["acc"], V3=after["acc"] >= before["acc"] - V_ACC_TOL,
                 V4_agreement=(after["pred"] == before["pred"]).float().mean().item(),
                 V5=bool(torch.isfinite(Wi).all() and torch.isfinite(bi).all() and (outside == 0).all()),
                 params_changed=changed,
                 rel_change_W_in=((Wi - saved[0]).norm() / saved[0].norm()).item(),
                 rel_change_b_in=((bi - saved[1]).norm() / saved[1].norm()).item(),
                 edit_before=before["edit"], edit_after=after["edit"],
                 iia_before=before["iia"], iia_after=after["iia"], acc_before=before["acc"], acc_after=after["acc"])
    v["V4"] = v["V4_agreement"] >= V_AGREE
    v["passed"] = all(v[k] for k in ("V1", "V2", "V3", "V4", "V5"))
    return v, (Wi, bi)


# ------------------------------------------------------------------ one network
def run_seed(seed):
    root = ZOO / f"s{seed}"
    ledger = {}
    cards = {}

    def ck(folder):
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    # trunk: init -> 3,000
    t = time.process_time()
    st = new_state(seed)
    f0 = ck(root / "trunk" / "step_0")
    torch.save(st["model"].state_dict(), f0 / "model.pt")
    save_state(st, f0 / "train_state.pt")
    snaps = {}

    def snap(s):
        if s["step"] in (600, 1500, 3000):
            f = ck(root / "trunk" / f"step_{s['step']}")
            torch.save(s["model"].state_dict(), f / "model.pt")
            save_state(s, f / "train_state.pt")   # full state at every trunk checkpoint (small)
            snaps[s["step"]] = f
    hashes_trunk = train(st, BRANCH, snap)
    ledger["train_trunk_cpu_s"] = round(time.process_time() - t, 1)

    t = time.process_time()
    for s_ in (0, 600, 1500, 3000):
        m = load_state(root / "trunk" / f"step_{s_}" / "train_state.pt")["model"]
        cards[f"trunk_{s_}"] = audit(m, f"s{seed}/trunk/{s_}", root / "trunk" / f"step_{s_}")
    ledger["audit_trunk_cpu_s"] = round(time.process_time() - t, 1)

    # correction decision at 3,000 (calibration only)
    t = time.process_time()
    m3 = load_state(root / "trunk" / "step_3000" / "train_state.pt")["model"]
    decisions, chosen = [], None
    for a in ALPHAS:
        v, fold = validate(m3, a)
        decisions.append(v)
        if v["passed"]:
            chosen = (a, fold)
            break
    # registered diagnostic (no extra training): the same calibration-only validation at the other
    # trunk checkpoints; it never triggers an edit
    fold_by_ckpt = {}
    for s_ in (600, 1500, 3000):
        ms = load_state(root / "trunk" / f"step_{s_}" / "train_state.pt")["model"]
        fold_by_ckpt[s_] = [validate(ms, a)[0] for a in ALPHAS]
    ledger["correction_fit_validate_cpu_s"] = round(time.process_time() - t, 1)

    # control branch
    t = time.process_time()
    stc = load_state(root / "trunk" / "step_3000" / "train_state.pt")

    def snap_branch(name):
        def f(s):
            if s["step"] in (4500, 6000):
                fd = ck(root / name / f"step_{s['step']}")
                torch.save(s["model"].state_dict(), fd / "model.pt")
                if s["step"] == 6000:
                    save_state(s, fd / "train_state.pt")
        return f
    hashes_ctrl = train(stc, 6000, snap_branch("control"))
    ledger["train_control_cpu_s"] = round(time.process_time() - t, 1)

    hashes_corr = None
    if chosen:
        t = time.process_time()
        stx = load_state(root / "trunk" / "step_3000" / "train_state.pt")
        apply_fold(stx["model"], *chosen[1])
        fe = ck(root / "correction" / "step_3000_edited")
        torch.save(stx["model"].state_dict(), fe / "model.pt")
        hashes_corr = train(stx, 6000, snap_branch("correction"))
        ledger["train_correction_cpu_s"] = round(time.process_time() - t, 1)

    t = time.process_time()
    for name in ("control", "correction") if chosen else ("control",):
        for s_ in ((3000, 4500, 6000) if name == "correction" else (4500, 6000)):
            sub = "step_3000_edited" if s_ == 3000 else f"step_{s_}"
            m = TinyTyped(GROUP, **S.MODEL_KW)
            m.load_state_dict(torch.load(root / name / sub / "model.pt", weights_only=True))
            cards[f"{name}_{s_}"] = audit(m, f"s{seed}/{name}/{sub}", root / name / sub)
    ledger["audit_branches_cpu_s"] = round(time.process_time() - t, 1)

    # persistence of the readout change: W_in(corr) - W_in(ctrl)
    drift = {}
    if chosen:
        for s_ in (4500, 6000):
            wc = torch.load(root / "control" / f"step_{s_}" / "model.pt", weights_only=True)["W_in.weight"]
            wx = torch.load(root / "correction" / f"step_{s_}" / "model.pt", weights_only=True)["W_in.weight"]
            drift[s_] = ((wx - wc).norm() / wc.norm()).item()
        w3 = torch.load(root / "trunk" / "step_3000" / "model.pt", weights_only=True)["W_in.weight"]
        w3e = torch.load(root / "correction" / "step_3000_edited" / "model.pt", weights_only=True)["W_in.weight"]
        drift[3000] = ((w3e - w3).norm() / w3.norm()).item()
    mc6 = TinyTyped(GROUP, **S.MODEL_KW)
    mc6.load_state_dict(torch.load(root / "control" / "step_6000" / "model.pt", weights_only=True))
    fold_by_ckpt["control_6000"] = [validate(mc6, a)[0] for a in ALPHAS]
    out = dict(seed=seed, decisions=decisions, applied_alpha=chosen[0] if chosen else None,
               fold_validation_by_checkpoint=fold_by_ckpt,
               cards=cards, w_in_divergence=drift,
               batches_identical=(hashes_corr == hashes_ctrl) if hashes_corr is not None else None,
               n_trunk_steps=len(hashes_trunk), n_branch_steps=len(hashes_ctrl), ledger=ledger,
               storage_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()))
    (root / "run.json").write_text(json.dumps(out, indent=1))
    return out


def fidelity_reference(seed=SEEDS[0]):
    """Uninterrupted run with the ORIGINAL trainer (run_swap.train_one) for the bitwise check."""
    t = time.process_time()
    run = S.train_one((GROUP, seed, "warmup", str(ZOO / "fidelity_reference")))
    return dict(run=str(Path(run).relative_to(ROOT)), cpu_s=round(time.process_time() - t, 1))


def job(arg):
    return fidelity_reference() if arg == "fidelity" else run_seed(arg)


# ------------------------------------------------------------------ report
def classify(ctrl, corr):
    de = corr["edit_block"]["intended"] - ctrl["edit_block"]["intended"]
    did = corr["acc_id"] - ctrl["acc_id"]
    dood = corr["acc_ood"] - ctrl["acc_ood"]
    if abs(de) < 0.05:
        return "no effect", de, did, dood
    if de >= 0.05 and did >= -V_ACC_TOL and dood >= -0.02:
        return "improved", de, did, dood
    if de >= 0.05:
        return "traded off", de, did, dood
    return "harmed", de, did, dood


def report():
    runs = {s: json.loads((ZOO / f"s{s}" / "run.json").read_text()) for s in SEEDS}
    fid = json.loads((OUT / "fidelity.json").read_text())
    L = ["# Checkpoint-guided correction pilot (auto-generated by `run_correction.py report`; procedure frozen in docs/22)", "",
         f"Resume fidelity (seed {SEEDS[0]}: control branch vs uninterrupted `run_swap.train_one`): "
         f"**{'bitwise identical' if fid['bitwise_identical'] else 'NOT identical'}**.", ""]
    keys = ("acc_id", "acc_ood", "designed_iia_M", "das_iia_M", "das_block_overlap_M", "route_share_upstream",
            "route_share_readout", "probe_M_block")
    L += ["## Trajectories (frozen evaluation; reserved questions)", "",
          "| seed | checkpoint | ID | OOD | block IIA(M) | DAS IIA(M) | DAS∩block | route share upstream | route share readout | probe | block edit intended / wrong | DAS edit intended | copy intended | ID / OOD without route variance | DAS IIA(A) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = ("trunk_0", "trunk_600", "trunk_1500", "trunk_3000", "correction_3000", "control_4500",
             "correction_4500", "control_6000", "correction_6000")
    for s, r in runs.items():
        for k in order:
            if k not in r["cards"]:
                continue
            c = r["cards"][k]
            L.append(f"| {s} | {k.replace('_', ' ')} | " + " | ".join(f"{c[x]:.3f}" for x in keys) +
                     f" | {c['edit_block']['intended']:.3f} / {c['edit_block']['wrong']:.3f} | {c['edit_das_M']['intended']:.3f}"
                     f" | {c['copy_block']['intended']:.3f} | {c['info_necessity']['id']:.3f} / {c['info_necessity']['ood']:.3f}"
                     f" | {c['das_iia_A']:.3f} |")
    L += ["", "## Correction decisions at step 3,000 (calibration data only)", "",
          "| seed | alpha | V1 edit gain | V2 IIA change | V3 acc change | V4 agreement | V5 | passed | params changed | ‖ΔW_in‖/‖W_in‖ |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in runs.items():
        for v in r["decisions"]:
            L.append(f"| {s} | {v['alpha']} | {v['V1_edit_gain']:+.3f} | {v['V2_iia_change']:+.3f} | {v['V3_acc_change']:+.4f} | "
                     f"{v['V4_agreement']:.4f} | {v['V5']} | **{v['passed']}** | {v['params_changed']} | {v['rel_change_W_in']:.3f} |")
    L += ["", "## Registered diagnostic: the same calibration-only fold validation at every trunk checkpoint (never applied)", "",
          "| seed | checkpoint | alpha | edit gain | IIA change | accuracy change | agreement | would pass |", "|---|---|---|---|---|---|---|---|"]
    for s, r in runs.items():
        for ck_, vs in r["fold_validation_by_checkpoint"].items():
            for v in vs:
                L.append(f"| {s} | {ck_} | {v['alpha']} | {v['V1_edit_gain']:+.3f} | {v['V2_iia_change']:+.3f} | "
                         f"{v['V3_acc_change']:+.3f} | {v['V4_agreement']:.3f} | {v['passed']} |")
    L += ["", "## Separated claims (per network)", "",
          "| seed | C1 identified (first checkpoint) | C2 causal change (copy ≥ 0.8) at 3,000 | C3 persistent readout correction | C4 final outcome | Δ edit / Δ ID / Δ OOD at 6,000 | C5 upstream route share ctrl → corr at 6,000 | W_in divergence 3,000 / 4,500 / 6,000 | identical batches |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s, r in runs.items():
        cs = r["cards"]
        c1 = next((k for k in order if k in cs and cs[k]["designed_iia_M"] >= 0.9 and cs[k]["acc_id"] >= 0.95), "never")
        c2 = cs["trunk_3000"]["copy_block"]["intended"] >= 0.8
        if r["applied_alpha"] is None:
            c3, c4, deltas, c5, drift = "no correction passed", "not run", "—", "—", "—"
        else:
            e0, e1 = cs["trunk_3000"], cs["correction_3000"]
            c3 = (e1["edit_block"]["intended"] - e0["edit_block"]["intended"] >= 0.05
                  and e1["acc_id"] >= e0["acc_id"] - V_ACC_TOL)
            c3 = f"{'yes' if c3 else 'no'} (α={r['applied_alpha']}; edit {e0['edit_block']['intended']:.3f}→{e1['edit_block']['intended']:.3f}; ID {e0['acc_id']:.3f}→{e1['acc_id']:.3f})"
            out, de, did, dood = classify(cs["control_6000"], cs["correction_6000"])
            c4, deltas = f"**{out}**", f"{de:+.3f} / {did:+.4f} / {dood:+.3f}"
            c5 = f"{cs['control_6000']['route_share_upstream']:.3f} → {cs['correction_6000']['route_share_upstream']:.3f}"
            w = r["w_in_divergence"]
            drift = " / ".join(f"{w[k]:.3f}" for k in ("3000", "4500", "6000"))
        L.append(f"| {s} | {c1} | {c2} | {c3} | {c4} | {deltas} | {c5} | {drift} | {r['batches_identical']} |")
    L += ["", "## Compute and storage ledger", "", "| seed | phase | CPU-seconds |", "|---|---|---|"]
    tot = 0.0
    for s, r in runs.items():
        for k, v in r["ledger"].items():
            L.append(f"| {s} | {k} | {v} |")
            tot += v
    L.append(f"| — | fidelity reference run | {fid['cpu_s']} |")
    tot += fid["cpu_s"]
    L += ["", f"Total CPU: {tot / 3600:.2f} CPU-hours. Storage: "
          f"{sum(r['storage_bytes'] for r in runs.values()) / 1e6:.1f} MB (+ fidelity reference)."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    with open(OUT / "audit_cards.jsonl", "w") as f:
        for s, r in runs.items():
            for k in order:
                if k in r["cards"]:
                    c = r["cards"][k]
                    f.write(json.dumps({"seed": s, "checkpoint": k, **{x: round(c[x], 4) for x in keys},
                                        "edit_block_intended": round(c["edit_block"]["intended"], 4),
                                        "edit_block_wrong": round(c["edit_block"]["wrong"], 4),
                                        "copy_block_intended": round(c["copy_block"]["intended"], 4),
                                        "id_without_route_variance": round(c["info_necessity"]["id"], 4),
                                        "ood_without_route_variance": round(c["info_necessity"]["ood"], 4),
                                        "das_iia_A": round(c["das_iia_A"], 4)}) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "report"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "report":
        report()
        return
    ZOO.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(job, ["fidelity", *SEEDS]))
    ref = torch.load(ROOT / res[0]["run"] / "model.pt", weights_only=True)
    ctl = torch.load(ZOO / f"s{SEEDS[0]}" / "control" / "step_6000" / "model.pt", weights_only=True)
    same = all(torch.equal(ref[k], ctl[k]) for k in ref)
    (OUT / "fidelity.json").write_text(json.dumps(dict(seed=SEEDS[0], bitwise_identical=same, **res[0]), indent=1))
    print("resume fidelity bitwise identical:", same)
    report()


if __name__ == "__main__":
    main()
