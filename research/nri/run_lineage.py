"""Adaptive Concept Lineage and Causal Supervision (spec v1.0; registration docs/28; results docs/29).

Exactly two networks, seed SEED, doc-14 `mass_only` recipe (run_correction trainer, unchanged):
  A  control      observed every 100 steps, never edited
  B  supervised   observed every 100 steps; the supervisor (an LLM) MUST file a review for every
                  checkpoint before training continues, may run read-only inspections, and may submit
                  declarative proposals (nri/interventions.py) that are installed only if validated.

    python run_lineage.py init
    python run_lineage.py advance                       # needs a review for B's current checkpoint
    python run_lineage.py packet                        # print the supervisor packet again
    python run_lineage.py inspect <tool> [json-args]    # Level-2 read-only investigation
    python run_lineage.py propose <proposal.json>       # Level-3: compile, validate, install if passed
    python run_lineage.py review <review.json>          # the mandatory checkpoint review
    python run_lineage.py final                         # ONE-TIME sealed evaluation of A and B at 6,000
    python run_lineage.py report

Separable components: checkpoint recorder (record()), token/gradient tracer and concept-lineage
tracker (nri/lineage.py), causal audit engine (nri/atlas.observe + inspection tools), supervisor
interface (packet / inspect / review), safe intervention executor and validation/rollback controller
(nri/interventions.py), journal and atlas generator (report_lineage.py).
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import pickle
import sys
import time
from pathlib import Path

import torch

import run_correction as C
import run_swap as S
from nri import atlas as AT
from nri import interventions as IV
from nri import lineage as LN

ROOT = Path(__file__).parent
ZOO = ROOT / "runs" / "zoo_lineage"
OUT = ROOT / "results" / "lineage"
SEED = 701                                   # fresh; never used before in this repository
NETS = ("A_control", "B_supervised")
FULL_STATE_EVERY = 500
REVIEW_FIELDS = ("what_changed", "hypothesis", "evidence", "key_question", "best_experiment", "intervene",
                 "intervention_rationale", "previous_interventions", "investigation_level")
INSPECT_LIMIT = 8                            # read-only inspection calls per checkpoint (budget)


# ------------------------------------------------------------------ sealed-data guard
class Sealed(dict):
    def __getitem__(self, k):
        if k.startswith("FINAL") or k.startswith("OOD_FINAL"):
            raise PermissionError(f"the supervisor may not access the sealed split {k!r}")
        return super().__getitem__(k)


def guard_sealed():
    """Supervisor-facing commands run with the sealed splits unreachable through atlas.partition()."""
    AT.partition()
    if not isinstance(AT._PART, Sealed):
        sealed = Sealed(AT._PART)
        AT._PART = sealed
        AT.partition.__globals__["_PART"] = sealed


# ------------------------------------------------------------------ state files
def ndir(net, s):
    return ZOO / net / f"step_{s:04d}"


def meta_path():
    return ZOO / "meta.json"


def load_meta():
    return json.loads(meta_path().read_text())


def save_meta(m):
    meta_path().write_text(json.dumps(m, indent=1))


def live_state_path(net):
    return ZOO / net / "live_state.pt"


def save_live(net, st):
    C.save_state(st, live_state_path(net))


def load_live(net):
    return C.load_state(live_state_path(net))


def save_tracker(net, tr):
    (ZOO / net / "tracker.pkl").write_bytes(pickle.dumps(tr))


def load_tracker(net):
    return pickle.loads((ZOO / net / "tracker.pkl").read_bytes())


def carry_path(net):
    return ZOO / net / "carry.pt"


# ------------------------------------------------------------------ checkpoint recorder
def record(net, st, prev_sd, init_sd, upd_prev, g_prev, carry, tracker, hashes):
    """Persist and observe one checkpoint of one network. Returns the record and new carries."""
    s = st["step"]
    m = st["model"]
    f = ndir(net, s)
    f.mkdir(parents=True, exist_ok=True)
    rng = torch.get_rng_state()
    led = {}
    t = time.process_time()
    torch.save(m.state_dict(), f / "model.pt")
    if s % FULL_STATE_EVERY == 0:
        C.save_state(st, f / "train_state.pt")
    led["io_cpu_s"] = time.process_time() - t
    t = time.process_time()
    card, carry, rec = AT.observe(m, carry)
    torch.save(rec, f / "acts.pt")
    led["observe_cpu_s"] = time.process_time() - t
    t = time.process_time()
    tr = LN.trace(m)
    led["trace_cpu_s"] = time.process_time() - t
    t = time.process_time()
    cands = LN.candidates(m, card, rec["mlp_importance"])
    events = tracker.update(s, cands, card)
    led["lineage_cpu_s"] = time.process_time() - t
    t = time.process_time()
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    wd, upd = AT.weight_dynamics(sd, prev_sd, init_sd, upd_prev)
    g_now = AT.grads(m, 0)
    attr = AT.attribution(g_now, g_prev, sd, prev_sd) if g_prev is not None else None
    led["weights_cpu_s"] = time.process_time() - t
    torch.set_rng_state(rng)
    r = dict(net=net, step=s, state_hash=C.state_hash(m), state_fingerprint=IV.state_fingerprint(st),
             batch_digest=__import__("run_atlas").batch_digest(hashes), lr=st["sched"].get_last_lr()[0],
             lambda_iit=S.lam_at(max(s, 1), "warmup") if s else 0.0, observe=card, trace=tr, lineage_events=events,
             weights=wd, attribution=attr, task_loss_fit=g_now["task_loss"], iit_loss_fit=g_now["iit_loss"],
             cost={k: round(v, 2) for k, v in led.items()})
    with open(ZOO / net / "records.jsonl", "a") as fh:
        fh.write(json.dumps(r) + "\n")
    return r, sd, upd, g_now, carry


def init():
    ZOO.mkdir(parents=True, exist_ok=True)
    states = {n: C.new_state(SEED) for n in NETS}
    hs = {n: C.state_hash(st["model"]) for n, st in states.items()}
    assert hs[NETS[0]] == hs[NETS[1]], "initializations differ"
    for n, st in states.items():
        (ZOO / n).mkdir(exist_ok=True)
        (ZOO / n / "records.jsonl").write_text("")
        tr = LN.LineageTracker(n)
        init_sd = {k: v.detach().clone() for k, v in st["model"].state_dict().items()}
        torch.save(init_sd, ZOO / n / "init_sd.pt")
        r, sd, upd, g, carry = record(n, st, init_sd, init_sd, None, None, None, tr, [])
        torch.save(dict(prev_sd=sd, upd_prev=upd, g_prev=g, carry=carry), carry_path(n))
        save_tracker(n, tr)
        save_live(n, st)
    save_meta(dict(step=0, seed=SEED, cum_rel_norm=0.0, installed=[], reviews={}, proposals={}, inspections={},
                   ledger=dict(train_cpu_s=0.0, supervisor_extra_grad_steps=0, proposal_cpu_s=0.0, inspect_cpu_s=0.0)))
    (ZOO / "journal.jsonl").write_text("")
    print("initialized both networks (identical hash", hs[NETS[0]][:16], ")")
    packet()


def advance():
    meta = load_meta()
    s = meta["step"]
    if s > 0 and str(s) not in meta["reviews"]:
        raise SystemExit(f"no supervisor review filed for B at step {s}: run `review` first")
    assert s < S.STEPS
    for n in NETS:
        st = load_live(n)
        t = time.process_time()
        hashes = C.train(st, s + 100)
        meta["ledger"]["train_cpu_s"] += time.process_time() - t
        cr = torch.load(carry_path(n), weights_only=False)
        tr = load_tracker(n)
        init_sd = torch.load(ZOO / n / "init_sd.pt", weights_only=True)
        r, sd, upd, g, carry = record(n, st, cr["prev_sd"], init_sd, cr["upd_prev"], cr["g_prev"], cr["carry"], tr, hashes)
        torch.save(dict(prev_sd=sd, upd_prev=upd, g_prev=g, carry=carry), carry_path(n))
        save_tracker(n, tr)
        save_live(n, st)
    meta["step"] = s + 100
    save_meta(meta)
    packet()


def last_records(net, k=2):
    rs = [json.loads(l) for l in open(ZOO / net / "records.jsonl")]
    return rs[-k:]


def packet():
    """The supervisor packet: B's audit, the matching A checkpoint, changes since the previous
    checkpoint, lineage events, previous decisions, and the remaining budget."""
    meta = load_meta()
    s = meta["step"]
    rb = last_records("B_supervised")
    ra = last_records("A_control", 1)[-1]
    cur, prev = rb[-1], (rb[0] if len(rb) > 1 else None)
    o, oa = cur["observe"], ra["observe"]

    def row(name, f):
        a, b = f(o), f(oa)
        d = f"  Δprev {a - f(prev['observe']):+.3f}" if prev else ""
        return f"  {name:32s} B {a:7.3f}  A {b:7.3f}  B−A {a - b:+.3f}{d}"
    L = [f"=================== step {s} ===================",
         f"  identical to A: {cur['state_hash'] == ra['state_hash']}; batches identical: {cur['batch_digest'] == ra['batch_digest']}; "
         f"installed edits: {[e['step'] for e in meta['installed']]}; cum rel norm {meta['cum_rel_norm']:.3f}/{IV.LIMITS['max_cum_rel_model']}"]
    for name, f in (("acc OBS", lambda x: x["acc_obs"]), ("acc OOD_DEV", lambda x: x["acc_ood_dev"]),
                    ("block IIA(M)", lambda x: x["causal"]["iia_M_block"]), ("IIA(M) dims16-63", lambda x: x["causal"]["iia_M_rest"]),
                    ("L1 tokens IIA(M)", lambda x: x["causal"]["iia_M_L1_val+unit"]), ("L1 tokens IIA(A)", lambda x: x["causal"]["iia_A_L1_val+unit"]),
                    ("pooled edit inv", lambda x: x["causal"]["edit_block_intended"]), ("copy block", lambda x: x["causal"]["copy_block_intended"]),
                    ("interaction (block)", lambda x: x["geometry"]["block"]["M_interaction_share"]),
                    ("route main (block)", lambda x: x["geometry"]["block"]["M_route_main_share"]),
                    ("unit-transfer ratio", lambda x: x["unit_transfer"]["M@h_block"]["transfer"] / max(x["unit_transfer"]["M@h_block"]["within"], 1e-9)),
                    ("per-unit CKA (block)", lambda x: x["geometry"]["block"]["M_unit_cka"]),
                    ("M mult-char (h)", lambda x: x["geometry"]["h"]["M_char"]["mult"]),
                    ("M kappa value token", lambda x: x["decode_kappa"]["M@L1@mval"]), ("M kappa block", lambda x: x["decode_kappa"]["M@h_block"]),
                    ("A kappa mlp", lambda x: x["decode_kappa"]["A@mlp"]), ("F kappa mlp", lambda x: x["decode_kappa"]["F@mlp"])):
        L.append(row(name, f))
    L.append(f"  attn2 heads IIA(M): B {[round(o['causal'][f'iia_M_head{i}'], 2) for i in range(4)]}  A {[round(oa['causal'][f'iia_M_head{i}'], 2) for i in range(4)]}")
    t, ta = cur["trace"], ra["trace"]
    L.append(f"  token saliency (share of input-gradient norm): B {t['token_saliency']}  A {ta['token_saliency']}")
    L.append(f"  matched panel block cosine: equivalent {t['matched_similarity'].get('equivalent', {}).get('block_cos')} / other mass "
             f"{t['matched_similarity'].get('other_mass', {}).get('block_cos')} / other acc {t['matched_similarity'].get('other_acc', {}).get('block_cos')}; "
             f"equivalent pairs same prediction {t['equivalent_pairs_same_prediction']}")
    ag = t["group_gradient_agreement"]
    L.append("  gradient agreement (cos) mass-unit groups: " + ", ".join(f"{k.split(':')[0]} {k.split(':')[1].replace('mass_unit_', 'u')} {v:+.2f}"
                                                                         for k, v in ag.items() if "acc" not in k))
    top = sorted(cur["weights"].items(), key=lambda kv: -kv[1]["rel_from_prev"])[:4]
    L.append("  largest relative weight updates: " + ", ".join(f"{k} {v['rel_from_prev']:.3f}" for k, v in top))
    if cur["attribution"]:
        ph = cur["attribution"]["iit"]["per_head"]
        L.append("  IIT-loss attribution by head (neg = helped): " + ", ".join(f"{k} {v:+.3f}" for k, v in sorted(ph.items())))
    L.append(f"  MLP families: {o['features']['mlp']['counts']}; match {o['features']['mlp'].get('match_status')}")
    evs = [e for e in cur["lineage_events"] if e["event"] not in ("persistence",)]
    L.append(f"  lineage events ({len(cur['lineage_events'])} total, {len(evs)} non-persistence): " +
             "; ".join(f"{e['event']} {e.get('id', e.get('parent', ''))} {e.get('site', e.get('into_site', ''))} {e.get('to_status', e.get('status', ''))}".strip()
                       for e in evs[:14]))
    prevrev = meta["reviews"].get(str(s - 100))
    if prevrev:
        L.append(f"  previous review question: {prevrev.get('key_question')}")
    pr = meta["proposals"].get(str(s - 100), [])
    for p in pr:
        L.append(f"  previous proposal {p['file']}: {p['verdict']}")
    L.append(f"  budget this checkpoint: {IV.LIMITS['max_candidates_per_checkpoint']} candidate validations, "
             f"{IV.LIMITS['max_extra_grad_steps_per_checkpoint']} extra gradient steps, {INSPECT_LIMIT} inspections")
    print("\n".join(L))


# ------------------------------------------------------------------ Level 2: read-only inspection tools
def _model_and_state():
    st = load_live("B_supervised")
    return st, st["model"]


def tool_heads(args):
    sys.path.insert(0, str(ROOT / "loop_edits"))
    from inspect_heads import inspect as ih
    out = {}
    for n in NETS:
        rows, extra = ih(load_live(n)["model"])
        out[n] = dict(heads=rows, extra=extra)
    return out


@torch.no_grad()
def tool_ablate(args):
    """Mean-ablate a set of MLP neurons or a lineage feature's members; report accuracy, IIA(M) and
    OOD_DEV change on OBS (causal test of a candidate feature)."""
    st, m = _model_and_state()
    if "feature" in args:
        rec = load_tracker("B_supervised").records[args["feature"]]
        neurons = rec.get("members", [])
    else:
        neurons = args["neurons"]
    d = AT.partition()
    out = {}
    for split in ("OBS", "OOD_DEV"):
        lat = d[split]
        c = AT.capture(m, lat)
        act = c["act"].clone()
        act[:, neurons] = act[:, neurons].mean(0)
        pred = m.W_U(c["h"] + m.W_out(act)).argmax(-1)
        y = __import__("nri.dimtask", fromlist=["x"]).build(lat).y
        out[f"acc_{split}"] = round((pred == y).float().mean().item(), 4)
        out[f"acc_{split}_unablated"] = round((c["logits"].argmax(-1) == y).float().mean().item(), 4)
    return dict(neurons=len(neurons), **out)


@torch.no_grad()
def tool_patch(args):
    """Interchange at a chosen site for a chosen variable on OBS pairs: site in block/rest/head0-3/
    L1:<roles>; variable in M/A/S/m/a/mu/au."""
    st, m = _model_and_state()
    d = AT.partition()
    obs, src, so = d["OBS"], d["OBS_src"], d["OBS_src_same_order"]
    site, var = args["site"], args["variable"]
    if site == "block":
        v = AT.swap_h(m, obs, src, AT.BLK, var)
    elif site == "rest":
        v = AT.swap_h(m, obs, src, list(range(16, 64)), var)
    elif site.startswith("head"):
        v = AT.swap_head(m, obs, src, int(site[-1]), var, AT.capture(m, src))
    elif site.startswith("L1:"):
        v = AT.swap_l1(m, obs, so, tuple(site[3:].split("+")), var, AT.capture(m, so))
    else:
        raise SystemExit("unknown site")
    return dict(site=site, variable=var, iia=round(v, 4))


def tool_sharpness(args):
    sys.path.insert(0, str(ROOT / "loop_edits"))
    from sharpness import sharpness
    out = {}
    for n in NETS:
        st = load_live(n)
        out[n] = dict(sharpness=round(sharpness(st, iters=args.get("iters", 12)), 2),
                      preconditioned=round(sharpness(st, iters=args.get("iters", 12), precond=True), 2),
                      edge=round(2 / st["sched"].get_last_lr()[0], 1))
    return out


def tool_feature(args):
    """Full lineage record of one feature (B or A)."""
    tr = load_tracker(args.get("net", "B_supervised"))
    rec = tr.records[args["id"]]
    return {k: (v[-8:] if isinstance(v, list) else v) for k, v in rec.items()}


def tool_features(args):
    """Current concept records of one network, optionally filtered by kind / status."""
    tr = load_tracker(args.get("net", "B_supervised"))
    recs = tr.concept_records()
    out = [r for r in recs.values() if r["ended"] is None and (args.get("kind") in (None, r["kind"]))
           and (args.get("status") in (None, r["current_status"]))]
    return [dict(id=r["id"], site=r["current_site"], status=r["current_status"], first=r["first_observed"],
                 evidence=r["evidence_last"]) for r in out]


def tool_dryrun(args):
    """Compile a proposal and score it on OBS (NOT the gate; installs nothing). Counts against the
    candidate budget."""
    st, m = _model_and_state()
    p = json.loads(Path(args["proposal"]).read_text())
    delta, info = IV.execute(p, m)
    mm = copy.deepcopy(m)
    b = IV.metrics(mm, "OBS")
    IV._apply(mm, delta)
    a = IV.metrics(mm, "OBS")
    keep = a["correct"][b["correct"]].float().mean().item()
    mn = torch.cat([q.detach().flatten() for q in m.parameters()]).norm().item()
    return dict(info=info, rel_norm=round(torch.cat([v.flatten() for v in delta.values()]).norm().item() / mn, 4),
                keep=round(keep, 4), **{f"d_{k}": round(a[k] - b[k], 4) for k in ("acc", "iia", "ood_dev", "edit_inv", "interaction",
                                                                                   "unit_transfer", "wrong", "route_probe", "unit_probe")})


TOOLS = dict(heads=tool_heads, ablate=tool_ablate, patch=tool_patch, sharpness=tool_sharpness, feature=tool_feature,
             features=tool_features, dryrun=tool_dryrun)


def inspect(tool, args):
    guard_sealed()
    meta = load_meta()
    s = str(meta["step"])
    used = meta["inspections"].setdefault(s, [])
    if len(used) >= INSPECT_LIMIT:
        raise SystemExit("inspection budget for this checkpoint used up")
    t = time.process_time()
    out = TOOLS[tool](args)
    dt = time.process_time() - t
    meta["ledger"]["inspect_cpu_s"] += dt
    used.append(dict(tool=tool, args=args, cpu_s=round(dt, 1)))
    save_meta(meta)
    with open(ZOO / "inspections.jsonl", "a") as fh:
        fh.write(json.dumps(dict(step=int(s), tool=tool, args=args, result=out, cpu_s=round(dt, 1)), default=str) + "\n")
    print(json.dumps(out, indent=1, default=str)[:6000])


# ------------------------------------------------------------------ Level 3: proposals
def propose(path):
    guard_sealed()
    meta = load_meta()
    s = str(meta["step"])
    prev = meta["proposals"].setdefault(s, [])
    if len(prev) >= IV.LIMITS["max_candidates_per_checkpoint"]:
        raise SystemExit("candidate budget for this checkpoint used up")
    if any(p["installed"] for p in prev):
        raise SystemExit("an edit was already installed at this checkpoint")
    p = json.loads(Path(path).read_text())
    st = load_live("B_supervised")
    fp0 = IV.state_fingerprint(st)
    t = time.process_time()
    try:
        delta, info = IV.execute(p, st["model"])
        steps_used = sum(q.get("extra_grad_steps", 0) for q in prev) + info["extra_grad_steps"]
        if steps_used > IV.LIMITS["max_extra_grad_steps_per_checkpoint"]:
            raise IV.ProposalError("extra-gradient-step budget for this checkpoint exceeded")
        v = IV.validate(st, delta, p["target_claim"], meta["cum_rel_norm"])
    except IV.ProposalError as e:
        v, info = dict(passed=False, failed=["schema"], error=str(e)), dict(extra_grad_steps=0)
        delta = None
    dt = time.process_time() - t
    assert IV.state_fingerprint(st) == fp0
    installed = False
    if v["passed"] and p.get("install", True):
        IV.install(st, delta)
        save_live("B_supervised", st)
        torch.save(st["model"].state_dict(), ndir("B_supervised", int(s)) / "model_edited.pt")
        torch.save(delta, ndir("B_supervised", int(s)) / f"delta_{len(prev)}.pt")
        meta["cum_rel_norm"] += v["rel_norm_model"]
        meta["installed"].append(dict(step=int(s), file=Path(path).name, target=p["target_claim"], rel_norm=v["rel_norm_model"]))
        installed = True
    meta["ledger"]["proposal_cpu_s"] += dt
    meta["ledger"]["supervisor_extra_grad_steps"] += info.get("extra_grad_steps", 0) + (IV.LOOKAHEAD * 2 if "lookahead" in v else 0)
    verdict = ("INSTALLED" if installed else "passed (not installed)" if v["passed"] else "rejected: " + ",".join(v["failed"]))
    prev.append(dict(file=Path(path).name, verdict=verdict, installed=installed, extra_grad_steps=info.get("extra_grad_steps", 0),
                     cpu_s=round(dt, 1)))
    save_meta(meta)
    with open(ZOO / "proposals.jsonl", "a") as fh:
        fh.write(json.dumps(dict(step=int(s), file=Path(path).name, proposal=p, compile=info, validation=v, installed=installed),
                            default=lambda x: x.tolist() if torch.is_tensor(x) else str(x)) + "\n")
    b, a = v.get("before", {}), v.get("after", {})
    print(f"[{verdict}] phase {v.get('phase')} target {p['target_claim']} gain {v.get('gain', 0):+.4f} rel {v.get('rel_norm_model', 0):.4f} keep {v.get('keep_rate', 0):.4f}")
    if b:
        print("  before:", {k: round(x, 3) for k, x in b.items()})
        print("  after: ", {k: round(x, 3) for k, x in a.items()})
    if "lookahead" in v:
        print("  lookahead:", {k: dict(max_loss=round(x["max_loss"], 3), end_acc=round(x["end_acc"], 3)) for k, x in v["lookahead"].items()})
    if "error" in v:
        print("  error:", v["error"])


# ------------------------------------------------------------------ supervisor review
def review(path):
    meta = load_meta()
    s = str(meta["step"])
    r = json.loads(Path(path).read_text())
    missing = [k for k in REVIEW_FIELDS if not str(r.get(k, "")).strip()]
    if missing:
        raise SystemExit(f"review missing fields: {missing}")
    if r["investigation_level"] not in (1, 2, 3):
        raise SystemExit("investigation_level must be 1, 2 or 3")
    rb = last_records("B_supervised", 1)[-1]
    ra = last_records("A_control", 1)[-1]
    entry = dict(model="B_supervised", step=int(s), state_hash=rb["state_hash"], state_fingerprint=rb["state_fingerprint"],
                 control_state_hash=ra["state_hash"], acc_obs=rb["observe"]["acc_obs"], acc_ood_dev=rb["observe"]["acc_ood_dev"],
                 iia_M_block=rb["observe"]["causal"]["iia_M_block"],
                 lineage_events=[e for e in rb["lineage_events"] if e["event"] != "persistence"],
                 inspections=meta["inspections"].get(s, []), proposals=meta["proposals"].get(s, []),
                 installed=[e for e in meta["installed"] if e["step"] == int(s)],
                 cost=rb["cost"], **r)
    meta["reviews"][s] = dict(key_question=r["key_question"], intervene=r["intervene"])
    save_meta(meta)
    with open(ZOO / "journal.jsonl", "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"review filed for step {s}")


# ------------------------------------------------------------------ sealed final evaluation
def final():
    seal = OUT / "final_sealed.json"
    if seal.exists():
        raise SystemExit("sealed evaluation already run")
    meta = load_meta()
    assert meta["step"] == S.STEPS and str(S.STEPS) in meta["reviews"]
    import run_atlas as RA
    t = time.process_time()
    res = {}
    for n in NETS:
        m = load_live(n)["model"]
        res[n] = dict(metrics=RA.sealed_metrics(m), benchmarks=RA.benchmarks(m))
    res["cpu_s"] = round(time.process_time() - t, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    seal.write_text(json.dumps(res, indent=1))
    print(json.dumps({n: res[n]["metrics"] for n in NETS}, indent=0)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("init", "advance", "packet", "inspect", "propose", "review", "final", "report"))
    ap.add_argument("arg", nargs="?")
    ap.add_argument("json", nargs="?", default="{}")
    a = ap.parse_args()
    if a.mode == "init":
        init()
    elif a.mode == "advance":
        advance()
    elif a.mode == "packet":
        packet()
    elif a.mode == "inspect":
        inspect(a.arg, json.loads(a.json))
    elif a.mode == "propose":
        propose(a.arg)
    elif a.mode == "review":
        review(a.arg)
    elif a.mode == "final":
        final()
    else:
        import report_lineage
        report_lineage.report()


if __name__ == "__main__":
    main()
