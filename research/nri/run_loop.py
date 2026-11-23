"""LLM-in-the-loop supervised training (follow-up to Issue #22; registration docs/26; results docs/27).

ONE new network: seed 700, the doc-14 `mass_only` recipe, the same initialization and batch streams
as the Issue #22 pilot. Its matched control is the pilot's control trajectory
(runs/zoo_atlas/control), which this network reproduces bitwise until the first installed edit
(checked at every checkpoint).

The loop is driven step by step by the research agent:

    python run_loop.py init                       # step 0
    python run_loop.py advance                    # train 100 steps, checkpoint, observe, print packet
    python run_loop.py gate loop_edits/step_1000_x.py [--install]
                                                  # fixed safety gate on a bespoke edit module
    python run_loop.py status
    python run_loop.py final                      # ONE-TIME sealed evaluation at step 6,000
    python run_loop.py report

An edit module defines
    TARGET  = one of GATE_TARGETS (the claim it must deliver on VAL)
    RATIONALE = str
    def propose(model, ctx) -> dict[str, Tensor]   # parameter name -> additive delta
`ctx` exposes only FIT, VAL, OBS and OOD_DEV (never FINAL / OOD_FINAL) and helper modules.
The gate runs on a deep copy; the training model and optimizer are touched only by --install of a
passing edit (AdamW moments are left as they are: stale, declared).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import run_atlas as RA
import run_correction as C
import run_dimensional as D
import run_swap as S
from nri import atlas as AT
from nri import dimtask as T
from nri.model import TinyTyped

ROOT = Path(__file__).parent
ZOO = ROOT / "runs" / "zoo_llm_loop"
CTRL = ROOT / "runs" / "zoo_atlas" / "control"
OUT = ROOT / "results" / "llm_loop"
SEED = RA.SEED

# ------------------------------------------------------------------ the fixed gate (docs/26 §26.4)
GATE_TARGETS = {
    # name: (description, minimum gain on VAL)
    "edit_inv": ("pooled inversion edit at the block, intended rate (fit FIT, score VAL)", 0.01),
    "interaction": ("block mass x route interaction share on VAL, DEcrease (edit_inv must not fall)", 0.02),
    "unit_transfer": ("mass unit-transfer probe ratio at the block (FIT -> VAL), increase", 0.03),
    "ood_dev": ("OOD_DEV accuracy (held-out values 9, 10), increase", 0.01),
    "iia": ("block IIA(M) on VAL pairs, increase", 0.01),
}
S_ACC, S_KEEP, S_OOD, S_IIA, S_INFO, S_WRONG = 0.005, 0.99, 0.01, 0.01, 0.02, 0.02
MAX_REL_TENSOR, MAX_REL_MODEL, MAX_REL_CUM = 0.5, 0.10, 0.5
N_SHAM, SHAM_S = 3, 9_900


def model_norm(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()]).norm().item()


@torch.no_grad()
def gate_metrics(m):
    d = AT.partition()
    fit, val, vs = d["FIT"], d["VAL"], d["VAL_src"]
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


def gain(target, before, after):
    if target == "interaction":
        return before["interaction"] - after["interaction"]
    return after[target] - before[target]


@torch.no_grad()
def apply(m, delta, sign=1.0):
    P = dict(m.named_parameters())
    for k, v in delta.items():
        P[k].add_(sign * v)


def run_gate(m, delta, target, cum_used):
    assert target in GATE_TARGETS, target
    P = dict(m.named_parameters())
    mm = copy.deepcopy(m)
    before = gate_metrics(mm)
    rel_t = {k: (v.norm() / P[k].detach().norm().clamp_min(1e-12)).item() for k, v in delta.items()}
    rel_m = torch.cat([v.flatten() for v in delta.values()]).norm().item() / model_norm(m)
    apply(mm, delta)
    after = gate_metrics(mm)
    apply(mm, delta, -1.0)
    g = gain(target, before, after)
    thr = GATE_TARGETS[target][1]
    kept = (after["correct"] | ~before["correct"]).float().mean().item()     # previously-correct answers kept
    keep_rate = after["correct"][before["correct"]].float().mean().item() if before["correct"].any() else 1.0
    v = dict(target=target, gain=g, threshold=thr, rel_norm_per_tensor=rel_t, rel_norm_model=rel_m,
             before={k: x for k, x in before.items() if k not in ("pred", "correct")},
             after={k: x for k, x in after.items() if k not in ("pred", "correct")}, keep_rate=keep_rate,
             agreement=(after["pred"] == before["pred"]).float().mean().item(), kept_or_fixed=kept)
    rep = before["iia"] >= 0.5
    v["G_target"] = g >= thr and (target != "interaction" or after["edit_inv"] >= before["edit_inv"])
    v["S1_acc"] = after["acc"] >= before["acc"] - S_ACC
    v["S2_keep_correct"] = keep_rate >= S_KEEP
    v["S3_ood_dev"] = after["ood_dev"] >= before["ood_dev"] - S_OOD
    v["S4_iia"] = (not rep) or after["iia"] >= before["iia"] - S_IIA
    v["S5_unit_info"] = (after["route_probe"] >= before["route_probe"] - S_INFO and
                         after["unit_probe"] >= before["unit_probe"] - S_INFO)
    v["S6_wrong"] = after["wrong"] <= before["wrong"] + S_WRONG
    v["S7_budget"] = bool(all(torch.isfinite(x).all() for x in delta.values()) and max(rel_t.values()) <= MAX_REL_TENSOR
                          and rel_m <= MAX_REL_MODEL and cum_used + rel_m <= MAX_REL_CUM)
    checks = [k for k in v if k[:2] in ("G_", "S1", "S2", "S3", "S4", "S5", "S6", "S7")]
    if all(v[k] for k in checks):
        g_ = torch.Generator().manual_seed(SHAM_S)
        sg = []
        for _ in range(N_SHAM):
            sham = {}
            for k, x in delta.items():
                r = torch.randn(x.shape, generator=g_)
                sham[k] = r * x.norm() / r.norm().clamp_min(1e-12)
            apply(mm, sham)
            sg.append(gain(target, before, gate_metrics(mm)))
            apply(mm, sham, -1.0)
        v["S8_sham_gains"] = sg
        v["S8_sham"] = g - sum(sg) / len(sg) >= thr / 2
        checks.append("S8_sham")
    v["passed"] = all(v[k] for k in checks)
    v["failed"] = [k for k in checks if not v[k]]
    return v


# ------------------------------------------------------------------ state handling
def ckpt_dir(s):
    return ZOO / f"step_{s:04d}"


def current_step():
    st = json.loads((ZOO / "state.json").read_text())
    return st["step"], st


def save_meta(meta):
    (ZOO / "state.json").write_text(json.dumps(meta, indent=1))


def init():
    ZOO.mkdir(parents=True, exist_ok=True)
    st = C.new_state(SEED)
    f = ckpt_dir(0)
    f.mkdir(exist_ok=True)
    torch.save(st["model"].state_dict(), f / "model.pt")
    C.save_state(st, f / "train_state.pt")
    ctrl_hash = json.loads(open(CTRL / "cards.jsonl").readline())["state_hash"]
    assert C.state_hash(st["model"]) == ctrl_hash, "initialization differs from the pilot control"
    save_meta(dict(step=0, cum_rel_norm=0.0, installed=[], diverged_at=None,
                   ledger=dict(train_cpu_s=0.0, observe_cpu_s=0.0, gate_cpu_s=0.0, design_cpu_s=0.0)))
    with open(ZOO / "cards.jsonl", "w") as fh:
        card, _, rec = AT.observe(st["model"], None)
        fh.write(json.dumps(dict(step=0, state_hash=C.state_hash(st["model"]), observe=card)) + "\n")
    print("initialized; matches pilot control at step 0")


def ctrl_card(s):
    for l in open(CTRL / "cards.jsonl"):
        c = json.loads(l)
        if c["step"] == s:
            return c
    return None


@torch.no_grad()
def extras(m):
    """Agent-facing extra measurements (OBS only): per-unit-pair CKA at the block, fusion at the value
    token, interaction sources, and the cost of full interaction removal."""
    d = AT.partition()
    obs, so = d["OBS"], d["OBS_src_same_order"]
    c = AT.capture(m, obs)
    fo = AT.factors(obs)
    Z = c["h"][:, AT.BLK]
    pair = {}
    for u1, u2 in ((0, 1), (0, 2), (1, 2)):
        cs = [k for k in range(AT.P) if ((fo["M"] == k) & (fo["mu"] == u1)).any() and ((fo["M"] == k) & (fo["mu"] == u2)).any()]
        A = torch.stack([Z[(fo["M"] == k) & (fo["mu"] == u1)].mean(0) for k in cs])
        B = torch.stack([Z[(fo["M"] == k) & (fo["mu"] == u2)].mean(0) for k in cs])
        pair[f"{u1}{u2}"] = round(AT.linear_cka(A, B), 3)
    cs_ = AT.capture(m, so)
    rp = AT.role_pos(obs)
    idx = torch.arange(len(obs["m"]))

    def f(x1):
        x1 = x1.clone()
        x1[idx, rp["mval"]] = cs_["x1"][idx, rp["mval"]]
        return x1
    pred = AT.capture(m, obs, {"x1": f})["logits"].argmax(-1)
    y0 = T.build(obs).y
    fus = {}
    for hyp in ("M", "m"):
        cf = AT.counterfactual(obs, so, hyp)
        el = cf != y0
        fus[hyp] = round((pred == cf)[el].float().mean().item(), 3)
    _, Tb = AT.interaction_decomp(Z, fo["M"], fo["route"])
    _, Ta = AT.interaction_decomp(c["a2"][:, AT.BLK], fo["M"], fo["route"])
    tot = (Tb ** 2).sum().clamp_min(1e-12)
    dW, db = AT.interaction_direction(m)
    return dict(unit_pair_cka=pair, value_token_swap_follows=fus,
                interaction_from_attn2=round(((Ta * Tb).sum() / tot).item(), 3),
                full_removal_rel_norm=round(torch.cat([dW.flatten(), db]).norm().item() / AT.block_row_norm(m), 2))


def packet(card, ex, s, prev):
    o, z, g = card, card["causal"], card["geometry"]["block"]
    cc = ctrl_card(s)
    co = cc["observe"] if cc else None
    t = o["unit_transfer"]["M@h_block"]
    L = [f"=== step {s} ==="]
    def row(name, f):
        a = f(o)
        b = f(co) if co else None
        L.append(f"  {name:34s} {a:7.3f}" + (f"   ctrl {b:7.3f}  Δ {a - b:+.3f}" if b is not None else ""))
    row("acc OBS", lambda x: x["acc_obs"])
    row("acc OOD_DEV", lambda x: x["acc_ood_dev"])
    row("block IIA(M)", lambda x: x["causal"]["iia_M_block"])
    row("IIA(M) dims16-63", lambda x: x["causal"]["iia_M_rest"])
    row("L1 val+unit IIA(M)", lambda x: x["causal"]["iia_M_L1_val+unit"])
    row("A via L1 tokens", lambda x: x["causal"]["iia_A_L1_val+unit"])
    row("pooled edit inv (OBS)", lambda x: x["causal"]["edit_block_intended"])
    row("copy block", lambda x: x["causal"]["copy_block_intended"])
    row("interaction share (block)", lambda x: x["geometry"]["block"]["M_interaction_share"])
    row("route main share (block)", lambda x: x["geometry"]["block"]["M_route_main_share"])
    row("unit-transfer ratio (block)", lambda x: x["unit_transfer"]["M@h_block"]["transfer"] / max(x["unit_transfer"]["M@h_block"]["within"], 1e-9))
    row("M per-unit CKA (block)", lambda x: x["geometry"]["block"]["M_unit_cka"])
    row("M mult-char (h)", lambda x: x["geometry"]["h"]["M_char"]["mult"])
    row("M kappa L1@mval", lambda x: x["decode_kappa"]["M@L1@mval"])
    row("M kappa h_block", lambda x: x["decode_kappa"]["M@h_block"])
    L.append(f"  heads IIA(M) attn2 0-3: {[round(z[f'iia_M_head{i}'], 2) for i in range(4)]}")
    L.append(f"  extras: {json.dumps(ex)}")
    L.append(f"  MLP families: {o['features']['mlp']['counts']}  match: {o['features']['mlp'].get('match_status')}")
    L.append(f"  top neurons (idx, ablation dloss, family): {o['mlp_top_importance'][:5]}")
    return "\n".join(L)


def advance(n=100):
    s0, meta = current_step()
    assert s0 < S.STEPS
    st = C.load_state(ckpt_dir(s0) / ("train_state_edited.pt" if (ckpt_dir(s0) / "train_state_edited.pt").exists() else "train_state.pt"))
    t = time.process_time()
    C.train(st, s0 + n)
    meta["ledger"]["train_cpu_s"] += time.process_time() - t
    s = s0 + n
    f = ckpt_dir(s)
    f.mkdir(exist_ok=True)
    m = st["model"]
    torch.save(m.state_dict(), f / "model.pt")
    C.save_state(st, f / "train_state.pt")
    h = C.state_hash(m)
    cc = ctrl_card(s)
    same = cc is not None and cc["state_hash"] == h
    if not same and meta["diverged_at"] is None:
        meta["diverged_at"] = s
        assert meta["installed"], "diverged from the control without any installed edit"
    t = time.process_time()
    prev = None
    card, _, rec = AT.observe(m, None)
    ex = extras(m)
    meta["ledger"]["observe_cpu_s"] += time.process_time() - t
    torch.save(rec, f / "acts.pt")
    with open(ZOO / "cards.jsonl", "a") as fh:
        fh.write(json.dumps(dict(step=s, state_hash=h, identical_to_control=same, observe=card, extras=ex)) + "\n")
    meta["step"] = s
    save_meta(meta)
    print(packet(card, ex, s, prev))
    print(f"  identical to control: {same}; installed edits so far: {[e['step'] for e in meta['installed']]}; "
          f"cum rel norm {meta['cum_rel_norm']:.3f}")


def load_current_model():
    s, meta = current_step()
    f = ckpt_dir(s)
    st = C.load_state(f / ("train_state_edited.pt" if (f / "train_state_edited.pt").exists() else "train_state.pt"))
    return st, s, meta


def ctx():
    d = AT.partition()
    return dict(FIT=d["FIT"], FIT_src=d["FIT_src"], VAL=d["VAL"], OBS=d["OBS"], OBS_src=d["OBS_src"],
                OOD_DEV=d["OOD_DEV"], AT=AT, D=D, T=T, S=S)


def gate(path, install=False):
    st, s, meta = load_current_model()
    if (ckpt_dir(s) / "train_state_edited.pt").exists():
        raise SystemExit(f"an edit was already installed at step {s}; advance first")
    spec = importlib.util.spec_from_file_location("edit_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = st["model"]
    rng = torch.get_rng_state()
    w0 = {k: v.clone() for k, v in m.state_dict().items()}
    t = time.process_time()
    delta = mod.propose(copy.deepcopy(m), ctx())
    meta["ledger"]["design_cpu_s"] += time.process_time() - t
    assert all(torch.equal(w0[k], v) for k, v in m.state_dict().items()), "propose() must not mutate the model"
    t = time.process_time()
    v = run_gate(m, delta, mod.TARGET, meta["cum_rel_norm"])
    meta["ledger"]["gate_cpu_s"] += time.process_time() - t
    torch.set_rng_state(rng)
    rec = dict(step=s, module=str(Path(path).relative_to(ROOT)), target=mod.TARGET, rationale=mod.RATIONALE,
               installed=False, **{k: x for k, x in v.items()})
    if install and v["passed"]:
        apply(m, delta)
        torch.save(m.state_dict(), ckpt_dir(s) / "model_edited.pt")
        C.save_state(st, ckpt_dir(s) / "train_state_edited.pt")
        torch.save(delta, ckpt_dir(s) / "delta.pt")
        meta["cum_rel_norm"] += v["rel_norm_model"]
        meta["installed"].append(dict(step=s, module=rec["module"], target=mod.TARGET, rel_norm_model=v["rel_norm_model"]))
        rec["installed"] = True
    with open(ZOO / "gate_log.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    save_meta(meta)
    b, a = v["before"], v["after"]
    print(f"[gate step {s}] {Path(path).name} target={mod.TARGET} gain={v['gain']:+.4f} (thr {v['threshold']}) "
          f"passed={v['passed']} failed={v['failed']} installed={rec['installed']}")
    print("  before:", {k: round(x, 3) for k, x in b.items()})
    print("  after: ", {k: round(x, 3) for k, x in a.items()})
    print(f"  rel norm model {v['rel_norm_model']:.4f}; per tensor { {k: round(x, 3) for k, x in v['rel_norm_per_tensor'].items()} }; "
          f"keep-correct {v['keep_rate']:.4f}; shams {v.get('S8_sham_gains')}")


def status():
    s, meta = current_step()
    print(json.dumps(meta, indent=1))


def final():
    seal = OUT / "final_sealed.json"
    if seal.exists():
        raise SystemExit("sealed evaluation already run")
    s, meta = current_step()
    assert s == S.STEPS
    m = TinyTyped("mass_only", **S.MODEL_KW)
    m.load_state_dict(torch.load(ckpt_dir(s) / "model.pt", weights_only=True))
    t = time.process_time()
    res = dict(llm_loop=dict(metrics=RA.sealed_metrics(m), benchmarks=RA.benchmarks(m)))
    res["cpu_s"] = round(time.process_time() - t, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    seal.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["llm_loop"]["metrics"], indent=0)[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("init", "advance", "gate", "status", "final"))
    ap.add_argument("edit", nargs="?")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    if a.mode == "init":
        init()
    elif a.mode == "advance":
        advance(a.n)
    elif a.mode == "gate":
        gate(a.edit, a.install)
    elif a.mode == "status":
        status()
    else:
        final()


if __name__ == "__main__":
    main()
