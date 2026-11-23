"""Two-trajectory concept-formation atlas and continuous upstream edit loop (Issue #22; registration
docs/24; results docs/25).

Exactly two networks, both `mass_only` under the doc-14 recipe (run_swap / run_correction trainer,
unchanged), from ONE seed (SEED): identical initialization, identical base and IIT-source batch
streams, identical optimizer and schedule.

  control     trains normally. At every 100-step checkpoint it is observed AND the registered edit
              policy is run in SHADOW mode (proposals and validation computed, never applied), so
              both trajectories spend matched audit compute.
  supervised  identical, except that a proposal that passes every registered check is installed
              in place (attn2.o rows 0-15 and bias only) before training continues.

Checkpoints: step 0 and every 100 steps to 6,000 (61 per trajectory). A checkpoint is the state
after that step's optimizer update and BEFORE any edit at that step. Rejected or skipped proposals
never touch the training model or optimizer (proposals are evaluated on a deep copy).

    python run_atlas.py train              # both trajectories in parallel (2 processes)
    python run_atlas.py analyse            # offline milestone analysis (registered milestones)
    python run_atlas.py final              # ONE-TIME sealed evaluation (refuses to run twice)
    python run_atlas.py report             # tables, atlas JSON, audit cards, ledger
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

import run_correction as C
import run_dimensional as D
import run_perm as RP
import run_swap as S
from nri import atlas as AT
from nri import dimtask as T
from nri.model import TinyTyped

ROOT = Path(__file__).parent
ZOO = ROOT / "runs" / "zoo_atlas"
OUT = ROOT / "results" / "atlas"
SEED = 700                                   # fresh; never used before in this repository
TRAJ = ("control", "supervised")
CKPTS = tuple(range(0, S.STEPS + 1, 100))    # 61 checkpoints
FULL_STATE_AT = tuple(range(0, S.STEPS + 1, 1000))
MILESTONES = (0, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 2000, 3000, 4500, 6000)
ANOMALY_DELTA = 0.15                         # |change| in acc_obs or block IIA(M) between checkpoints


def sd_clone(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def batch_digest(hashes):
    h = hashlib.sha256()
    for x in hashes:
        h.update(x.encode())
    return h.hexdigest()[:16]


def file_bytes(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


# ------------------------------------------------------------------ one trajectory
def run_trajectory(name, zoo=ZOO, ckpts=CKPTS, seed=SEED, comparators=True):
    assert name in TRAJ
    root = Path(zoo) / name
    root.mkdir(parents=True, exist_ok=True)
    st = C.new_state(seed)
    sd_init = sd_clone(st["model"])
    sd_prev, upd_prev, carry, g_prev = sd_init, None, None, None
    led = dict(train_cpu_s=0.0, observe_cpu_s=0.0, weights_cpu_s=0.0, policy_cpu_s=0.0,
               edit_apply_cpu_s=0.0, io_cpu_s=0.0)
    cum_used, cum_cap = 0.0, AT.RHO_CUM
    cards_f = open(root / "cards.jsonl", "w")
    dec_f = open(root / "decisions.jsonl", "w")
    deltas = {}
    for s in ckpts:
        if s > 0:
            t = time.process_time()
            hashes = C.train(st, s)
            led["train_cpu_s"] += time.process_time() - t
        else:
            hashes = []
        m = st["model"]
        pre_hash = C.state_hash(m)
        rng = torch.get_rng_state()
        # ---- persist the checkpoint (pre-edit)
        t = time.process_time()
        f = root / f"step_{s:04d}"
        f.mkdir(exist_ok=True)
        torch.save(m.state_dict(), f / "model.pt")
        if s in FULL_STATE_AT:
            C.save_state(st, f / "train_state.pt")
        led["io_cpu_s"] += time.process_time() - t
        # ---- observe (atlas)
        t = time.process_time()
        card, carry, rec = AT.observe(m, carry)
        led["observe_cpu_s"] += time.process_time() - t
        # ---- weights, gradients, attribution
        t = time.process_time()
        sd = sd_clone(m)
        wd, upd = AT.weight_dynamics(sd, sd_prev, sd_init, upd_prev)
        g_now = AT.grads(m, S.lam_at(max(s, 1), "warmup"))
        attr = AT.attribution(g_now, g_prev, sd, sd_prev) if g_prev is not None else None
        led["weights_cpu_s"] += time.process_time() - t
        # ---- policy (applied only in the supervised trajectory)
        t = time.process_time()
        if s > 0:
            dec, delta = AT.decide(m, cum_used, cum_cap, comparators=comparators)
        else:
            dec, delta = dict(status="skipped", reason="initialization (no policy at step 0)"), None
        led["policy_cpu_s"] += time.process_time() - t
        dec.update(step=s, trajectory=name, mode="applied" if name == "supervised" else "shadow")
        applied = False
        post, g_next = None, None
        if delta is not None and name == "supervised":
            t = time.process_time()
            AT.apply_delta(m, *delta)
            applied = True
            cum_used += dec["rel_norm"]
            deltas[s] = dict(dW=delta[0], db=delta[1])
            torch.save(m.state_dict(), f / "model_edited.pt")
            led["edit_apply_cpu_s"] += time.process_time() - t
            t = time.process_time()
            post, _, _ = AT.observe(m, None)
            led["observe_cpu_s"] += time.process_time() - t
            t = time.process_time()
            g_next = AT.grads(m, 0)           # gradients at the post-edit point for the next interval
            led["weights_cpu_s"] += time.process_time() - t
        dec["applied"] = applied
        dec["cum_rel_norm_used"] = cum_used
        torch.set_rng_state(rng)
        t = time.process_time()
        torch.save(rec, f / "acts.pt")
        full = dict(step=s, trajectory=name, state_hash=pre_hash,
                    state_hash_post_edit=C.state_hash(m) if applied else None, batch_digest=batch_digest(hashes), n_steps_since_prev=len(hashes),
                    lambda_iit=S.lam_at(max(s, 1), "warmup") if s else 0.0, lr=st["sched"].get_last_lr()[0],
                    observe=card, post_edit_observe=post, weights=wd, attribution=attr,
                    grad_norms={k: round(v.norm().item(), 6) for k, v in g_now["task"].items()},
                    grad_norms_iit={k: round(v.norm().item(), 6) for k, v in g_now["iit"].items()},
                    task_loss_fit=g_now["task_loss"], iit_loss_fit=g_now["iit_loss"],
                    decision={k: v for k, v in dec.items() if k not in ("candidates", "comparators")})
        cards_f.write(json.dumps(full) + "\n")
        cards_f.flush()
        dec_f.write(json.dumps(dec) + "\n")
        dec_f.flush()
        led["io_cpu_s"] += time.process_time() - t
        sd_prev, upd_prev, g_prev = sd_clone(m), upd, (g_next if g_next is not None else g_now)
        print(f"[{name}] step {s} acc {card['acc_obs']:.3f} iia {card['causal']['iia_M_block']:.3f} "
              f"edit {card['causal']['edit_block_intended']:.3f} -> {dec['status']}"
              f"{' (APPLIED)' if applied else ''}", flush=True)
    cards_f.close()
    dec_f.close()
    torch.save(deltas, root / "accepted_deltas.pt")
    led = {k: round(v, 1) for k, v in led.items()}
    summ = dict(trajectory=name, seed=seed, final_state_hash=C.state_hash(st["model"]), ledger=led,
                n_accepted=len(deltas), cum_rel_norm_used=cum_used, storage_bytes=file_bytes(root),
                n_params=st["model"].n_params(), torch=torch.__version__)
    (root / "run.json").write_text(json.dumps(summ, indent=1))
    return summ


def train_all():
    ZOO.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(run_trajectory, TRAJ))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ loading
def load_cards(name):
    return [json.loads(l) for l in open(ZOO / name / "cards.jsonl")]


def load_model(name, step, edited=False):
    m = TinyTyped("mass_only", **S.MODEL_KW)
    f = ZOO / name / f"step_{step:04d}" / ("model_edited.pt" if edited else "model.pt")
    m.load_state_dict(torch.load(f, weights_only=True))
    return m.eval()


def milestone_steps(name):
    cards = load_cards(name)
    steps = set(MILESTONES)
    for a, b in zip(cards, cards[1:]):
        if (abs(b["observe"]["acc_obs"] - a["observe"]["acc_obs"]) >= ANOMALY_DELTA or
                abs(b["observe"]["causal"]["iia_M_block"] - a["observe"]["causal"]["iia_M_block"]) >= ANOMALY_DELTA):
            steps |= {a["step"], b["step"]}
    return sorted(steps)


# ------------------------------------------------------------------ offline milestone analysis
def deep_analysis(m):
    """Expensive analyses (offline, never read by the policy): DAS for M and A on non-reserved pairs,
    DAS interchange on OBS pairs, DAS-block overlap, DAS-located persistent edit, D1 information
    necessity, family-level MLP ablations, held-out edit kinds at the block."""
    d = AT.partition()
    res = RP.reserved_set()
    obs, src = d["OBS"], d["OBS_src"]
    out = {}
    blk = torch.eye(64)[:, AT.BLK]
    with torch.no_grad():
        cb, cs = AT.capture(m, obs), AT.capture(m, src)
        cf = AT.capture(m, d["FIT"])
    for var in ("M", "A"):
        Q = RP.das_basis(m, var, res)
        with torch.no_grad():
            hp = cb["h"] + (cs["h"] - cb["h"]) @ Q @ Q.T
            out[f"das_iia_{var}"] = AT.iia(m.head_from_resid(hp).argmax(-1), obs, src, var)
            out[f"das_block_overlap_{var}"] = ((blk.T @ Q) ** 2).sum().item() / Q.shape[1]
            e = AT.pooled_edit(m, d["FIT"], obs, var=var, basis=Q, h_fit=cf["h"], h_eval=cb["h"])
            out[f"edit_das_{var}_intended"], out[f"edit_das_{var}_wrong"] = e["intended"], e["wrong"]
    with torch.no_grad():
        fo = AT.factors(obs)
        mu = AT.class_means(cf["h"][:, AT.BLK], AT.factors(d["FIT"])["M"], T.P)
        for nm, lat in (("obs", obs), ("ood_dev", d["OOD_DEV"])):
            c = AT.capture(m, lat)
            h = c["h"].clone()
            h[:, AT.BLK] = mu[AT.factors(lat)["M"]]
            out[f"D1_acc_without_route_variance_{nm}"] = (m.head_from_resid(h).argmax(-1) == T.build(lat).y).float().mean().item()
            out[f"acc_{nm}"] = AT.accuracy(m, lat, c)
        lab, _ = AT.families(cb["act"], fo)
        y = T.build(obs).y
        for fam in sorted(set(lab)):
            sel = torch.tensor([l == fam for l in lab])
            act = cb["act"].clone()
            act[:, sel] = act[:, sel].mean(0)
            r = cb["h"] + m.W_out(act)
            pred = m.W_U(r).argmax(-1)
            out[f"ablate_family_{fam}"] = dict(n=int(sel.sum()), acc=(pred == y).float().mean().item())
        for kind in ("add1", "inv"):
            e = AT.pooled_edit(m, d["FIT"], obs, kind=kind, h_fit=cf["h"], h_eval=cb["h"])
            out[f"edit_block_{kind}_intended"], out[f"edit_block_{kind}_wrong"] = e["intended"], e["wrong"]
    return out


def analyse():
    t0 = time.process_time()
    res = {}
    for name in TRAJ:
        steps = milestone_steps(name)
        rows = []
        decs = [json.loads(l) for l in open(ZOO / name / "decisions.jsonl")]
        edited = [d_["step"] for d_ in decs if d_.get("applied")]
        for s in steps:
            r = dict(step=s, **deep_analysis(load_model(name, s)))
            rows.append(r)
            print(name, s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k.startswith(("das_iia", "D1", "edit_das_M"))}, flush=True)
        post = [dict(step=s, edited=True, **deep_analysis(load_model(name, s, edited=True))) for s in edited
                if s in steps]
        res[name] = dict(milestones=steps, rows=rows, post_edit_rows=post)
    res["cpu_s"] = round(time.process_time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "milestones.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ sealed final evaluation
S.EDITS.setdefault("dbl", lambda y: 2 * y % T.P)   # held-out edit kind (runtime only; run_swap.py unchanged)


@torch.no_grad()
def edit_preservation(m, dW, db, lat, kind, var="M"):
    """Unedited-behaviour preservation of a persistent edit: agreement with the unedited predictions
    on examples whose edited variable is a fixed point of the edit (the answer must not change)."""
    yv = D.label(lat, var)
    fixed = S.EDITS[kind](yv) == yv
    h = AT.capture(m, lat)["h"]
    p0 = m.head_from_resid(h).argmax(-1)
    p1 = m.head_from_resid(h, m.W_in.weight + dW, m.W_in.bias + db).argmax(-1)
    return (p0 == p1)[fixed].float().mean().item(), int(fixed.sum())


@torch.no_grad()
def pooled_fold(m, fit_lat, kind, B):
    h = AT.capture(m, fit_lat)["h"]
    y = D.label(fit_lat, "M")
    mu = D.class_means(h, y, T.P)
    move = mu[S.EDITS[kind](y)] - mu[y]
    m0 = h.mean(0)
    z = (h - m0) @ B
    X = torch.cat([z, torch.ones(len(z), 1)], 1)
    W = torch.linalg.lstsq(X, move @ B).solution
    Dm = B @ W[:-1].T @ B.T
    return m.W_in.weight @ Dm, m.W_in.weight @ (B @ W[-1] - Dm @ m0)


def sealed_metrics(m):
    d = AT.partition()
    fin, fsrc, oodf = d["FINAL"], d["FINAL_src"], d["OOD_FINAL"]
    out = {}
    blk = torch.eye(64)[:, AT.BLK]
    with torch.no_grad():
        cb, cs = AT.capture(m, fin), AT.capture(m, fsrc)
        out.update(acc_id=AT.accuracy(m, fin, cb), acc_ood_final=AT.accuracy(m, oodf),
                   acc_ood_legacy=D.acc(m, "ood"), acc_ood_dev=AT.accuracy(m, d["OOD_DEV"]),
                   iia_M_block=AT.swap_h(m, fin, fsrc, AT.BLK, "M", cb, cs),
                   iia_M_rest=AT.swap_h(m, fin, fsrc, list(range(16, 64)), "M", cb, cs),
                   iia_A_block=AT.swap_h(m, fin, fsrc, AT.BLK, "A", cb, cs),
                   copy_block_intended=AT.copy_block(m, fin, cb=cb)["intended"])
        fo = AT.factors(fin)
        dec, _ = AT.interaction_decomp(cb["h"][:, AT.BLK], fo["M"], fo["route"])
        out.update(block_interaction_share=dec["interaction_share"], block_route_main_share=dec["route_main_share"])
    QM = RP.das_basis(m, "M", RP.reserved_set())
    with torch.no_grad():
        hp = cb["h"] + (cs["h"] - cb["h"]) @ QM @ QM.T
        out["das_iia_M"] = AT.iia(m.head_from_resid(hp).argmax(-1), fin, fsrc, "M")
        out["das_block_overlap_M"] = ((blk.T @ QM) ** 2).sum().item() / QM.shape[1]
        # persistent weight edits (folded into W_in), fitted on FIT, scored on FINAL
        for kind in ("inv", "add1", "dbl"):
            for loc, B in (("block", blk), ("das", QM)):
                dW, db = pooled_fold(m, d["FIT"], kind, B)
                pred = m.head_from_resid(cb["h"], m.W_in.weight + dW, m.W_in.bias + db).argmax(-1)
                sc = S.score(pred, fin, "M", kind)
                pres, nfix = edit_preservation(m, dW, db, fin, kind)
                out[f"edit_{kind}_{loc}"] = dict(sc, preserved_on_fixed_points=pres, n_fixed=nfix,
                                                  rel_change_W_in=(dW.norm() / m.W_in.weight.norm()).item())
    return out


def benchmarks(m):
    """Post-training, evaluation-only corrections (never part of either training trajectory)."""
    d = AT.partition()
    out = {}
    import copy
    # (i) doc-22 readout canonicalization fold (W_in columns 0-15), alpha chosen on VAL
    for a in (1.0, 0.5, 0.25):
        mm = copy.deepcopy(m)
        before = AT.policy_metrics(mm)
        Wi, bi = C.canon_fold(mm, d["FIT"], a)
        C.apply_fold(mm, Wi, bi)
        after = AT.policy_metrics(mm)
        ok = (after["edit"] - before["edit"] >= 0.05 and after["iia"] >= before["iia"] - 0.01 and
              after["acc"] >= before["acc"] - 0.005 and (after["pred"] == before["pred"]).float().mean() >= 0.99)
        if ok:
            out["readout_fold"] = dict(alpha=a, **{k: v for k, v in sealed_metrics(mm).items()
                                                    if k in ("acc_id", "acc_ood_final", "iia_M_block", "edit_inv_block",
                                                             "edit_add1_block", "edit_dbl_block")})
            break
    else:
        out["readout_fold"] = dict(alpha=None, note="no alpha passed the doc-22 checks on VAL")
    # (ii) the registered upstream policy applied ONCE after training
    mm = copy.deepcopy(m)
    dec, delta = AT.decide(mm, 0.0, None, comparators=False)
    out["upstream_once"] = dict(status=dec["status"], reason=dec.get("reason"))
    if delta is not None:
        AT.apply_delta(mm, *delta)
        sm = sealed_metrics(mm)
        out["upstream_once"].update({k: sm[k] for k in ("acc_id", "acc_ood_final", "iia_M_block", "edit_inv_block",
                                                         "edit_add1_block", "edit_dbl_block", "block_interaction_share")})
    # (iii) the full (alpha = 1, unbudgeted) interaction-removal solution, evaluation only
    mm = copy.deepcopy(m)
    dW, db = AT.interaction_direction(mm)
    rel = torch.cat([dW.flatten(), db]).norm().item() / AT.block_row_norm(mm)
    AT.apply_delta(mm, dW, db)
    sm = sealed_metrics(mm)
    out["upstream_full_unbudgeted"] = dict(rel_norm=rel, **{k: sm[k] for k in (
        "acc_id", "acc_ood_final", "iia_M_block", "edit_inv_block", "edit_add1_block", "edit_dbl_block",
        "block_interaction_share")})
    return out


def final():
    seal = OUT / "final_sealed.json"
    if seal.exists():
        raise SystemExit("sealed evaluation already run; refusing to run it twice")
    t0 = time.process_time()
    res = {}
    for name in TRAJ:
        m = load_model(name, S.STEPS)
        res[name] = dict(metrics=sealed_metrics(m), benchmarks=benchmarks(m))
        print(name, json.dumps(res[name]["metrics"], indent=0)[:2000], flush=True)
    res["cpu_s"] = round(time.process_time() - t0, 1)
    seal.write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "analyse", "final", "report"))
    a = ap.parse_args()
    if a.mode == "train":
        train_all()
    elif a.mode == "analyse":
        analyse()
    elif a.mode == "final":
        final()
    else:
        import report_atlas
        report_atlas.report()


if __name__ == "__main__":
    main()
