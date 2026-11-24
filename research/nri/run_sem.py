"""Issue #25 pilot driver: adaptive semantic curriculum vs fixed curriculum (registration docs/30).

    python run_sem.py freeze                 # vocab, dev sets on disk, sealed sets HASHED ONLY (not stored)
    python run_sem.py pretrain               # common student until the stabilization trigger (or cap)
    python run_sem.py cont-init              # A and B continuation states from the identical checkpoint
    python run_sem.py lesson FILE.json       # supervisor: lesson for B's next 100 steps (validated)
    python run_sem.py review FILE.json       # supervisor: review of B's current checkpoint (mandatory)
    python run_sem.py advance                # A (fixed curriculum) and B (chosen lesson) +100 steps, observe
    python run_sem.py packet                 # print the supervisor packet
    python run_sem.py final                  # ONE-TIME sealed evaluation (regenerates and hash-checks)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from sem import data as D
from sem import lang as L
from sem import observe as O
from sem import train as TR

ROOT = Path(__file__).parent
ZOO = ROOT / "runs" / "zoo_sem"
EVAL = ZOO / "eval"
OUT = ROOT / "results" / "sem"
SEED = 2501                       # fresh
CONT_SEED = 2502                  # continuation data stream (identical for A and B)
SEALED_SEEDS = {name: 9_000 + i for i, name in enumerate(n for n in D.EVAL_SPECS if n.startswith("sealed/"))}
DEV_SEEDS = {name: 8_000 + i for i, name in enumerate(n for n in D.EVAL_SPECS if n.startswith("dev/"))}

PRETRAIN_LESSON = dict(task_weights=dict(force=1.0), replay_weight=0.2, families=dict(force=list(D.PRETRAIN_FAMILIES)),
                       objectives=dict(aux_role=0.5, iit=3.0, consistency=0.0), note="pretraining: force basics")
TRIGGER = dict(min_step=1000, id_acc=0.95, id_range=0.02, ood_slope=0.01, cka=0.98, window=3)

# ---- the registered FIXED curriculum (arm A): one lesson per 100-step block, decided before training
_ALL_TRAIN = dict(sense_contexts=D.contexts(("train",)), distractor_contexts=D.contexts(("train",), "other", numbered=True))
_OBJ = dict(aux_role=0.5, iit=3.0, consistency=0.2)


def _stage(weights, distractor, hard, pair):
    return dict(task_weights=weights, replay_weight=0.3, style=dict(p_word=0.3, p_abbr=0.3), distractor_rate=distractor,
                hard_negative_rate=hard, paraphrase_pair_rate=pair, objectives=_OBJ, **_ALL_TRAIN)


FIXED_CURRICULUM = (
    [_stage(dict(force=0.4, convert=0.3, equiv=0.15, inverse=0.0, sense=0.15), 0.10, 0.2, 0.1)] * 5 +
    [_stage(dict(force=0.3, convert=0.2, equiv=0.15, inverse=0.2, sense=0.15), 0.20, 0.3, 0.1)] * 5 +
    [_stage(dict(force=0.2, convert=0.2, equiv=0.2, inverse=0.2, sense=0.2), 0.25, 0.3, 0.1)] * 5)


# ------------------------------------------------------------------ freeze
def gen_eval(name, seed):
    exs, q = D.make_eval_set(name, seed)
    return exs, q


def digest(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True).encode())
    return h.hexdigest()


def freeze():
    if (EVAL / "manifest.json").exists():
        raise SystemExit("already frozen")
    D.SEALED_LOCK[0] = False
    vocab = D.build_vocab()
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "vocab.json").write_text(json.dumps(vocab.to_json()))
    man = dict(vocab_size=len(vocab.itos), dev={}, sealed={}, max_len=D.MAX_LEN)
    for name, seed in DEV_SEEDS.items():
        exs, q = gen_eval(name, seed)
        h = D.write_jsonl(EVAL / "dev" / (name.split("/")[1] + ".jsonl"), exs)
        man["dev"][name] = dict(seed=seed, n=len(exs), quarantined=q, sha256=h, oov=oov(exs, vocab))
    for name, seed in SEALED_SEEDS.items():                  # generated in memory, hashed, NOT stored
        exs, q = gen_eval(name, seed)
        man["sealed"][name] = dict(seed=seed, n=len(exs), quarantined=q, content_sha256=digest(exs), oov=oov(exs, vocab))
    (EVAL / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (vv["n"], vv["quarantined"], vv["oov"]) for kk, vv in v.items()})
                      for k, v in man.items()}, indent=1))


def oov(exs, vocab):
    toks = [w for e in exs for w in e["tokens"]]
    return round(sum(1 for w in toks if w not in vocab.stoi) / max(len(toks), 1), 4)


def vocab():
    return D.Vocab.from_json(json.loads((EVAL / "vocab.json").read_text()))


# ------------------------------------------------------------------ pretraining with a prospective stabilization trigger
def trigger_ok(cards, lr_now):
    """Registered (docs/30 §30.5): over the last `window` checkpoints, ID dev accuracy >= 0.95 with range
    <= 0.02; held-out-value dev slope <= 0.01 per 100 steps; CKA of h to the previous checkpoint >= 0.98;
    learning rate past warm-up (constant phase)."""
    w = TRIGGER["window"]
    if len(cards) < w + 1 or cards[-1]["step"] < TRIGGER["min_step"] or lr_now < TR.PRE_LR:
        return False, "too early"
    last = cards[-w:]
    ids = [c["dev_acc"]["dev/force_id_pretrainfam"] for c in last]
    ood = [c["dev_acc"]["dev/force_ood_value"] for c in last]
    ck = [c.get("cka_h_prev", 0) for c in last]
    slope = (ood[-1] - ood[0]) / (w - 1)
    conds = dict(id=min(ids) >= TRIGGER["id_acc"] and max(ids) - min(ids) <= TRIGGER["id_range"],
                 ood=slope <= TRIGGER["ood_slope"], cka=min(ck) >= TRIGGER["cka"])
    return all(conds.values()), conds


def pretrain():
    V = vocab()
    out = ZOO / "pretrain"
    out.mkdir(parents=True, exist_ok=True)
    st = TR.new_state(V, SEED)
    cards, carry, log = [], None, []
    t0 = time.process_time()
    ledger = dict(train_cpu_s=0.0, observe_cpu_s=0.0)
    from safetensors.torch import save_file
    while True:
        if st["step"] > 0 or cards:
            t = time.process_time()
            TR.train_steps(st, PRETRAIN_LESSON, 100, log)
            ledger["train_cpu_s"] += time.process_time() - t
        t = time.process_time()
        card, carry = O.observe(st["model"], V, EVAL, carry)
        card["step"] = st["step"]
        ledger["observe_cpu_s"] += time.process_time() - t
        ok, why = trigger_ok(cards + [card], TR.lr_at("pretrain", st["step"]))
        card["trigger"] = dict(ok=ok, detail=why)
        cards.append(card)
        ck = out / f"step_{st['step']:04d}"
        ck.mkdir(exist_ok=True)
        save_file({k: v.contiguous() for k, v in st["model"].state_dict().items()}, ck / "model.safetensors")
        print(f"[pretrain] step {st['step']} pre-fam ID {card['dev_acc']['dev/force_id_pretrainfam']:.3f} "
              f"OOD {card['dev_acc']['dev/force_ood_value']:.3f} newfam {card['dev_acc']['dev/force_newfam']:.3f} "
              f"IIA {card['iia_block_cross_family']:.3f} cka {card.get('cka_h_prev', 0):.3f} trigger {ok} {why}", flush=True)
        if ok or st["step"] >= TR.PRE_CAP:
            break
    h = TR.save_state(st, ZOO / "common")
    ledger = {k: round(v, 1) for k, v in ledger.items()}
    (ZOO / "common" / "pretrain_summary.json").write_text(json.dumps(dict(stop_step=st["step"], trigger_fired=cards[-1]["trigger"]["ok"],
                                                                          state_hash=h, ledger=ledger), indent=1))
    with open(out / "cards.jsonl", "w") as f:
        for c in cards:
            f.write(json.dumps(c) + "\n")
    with open(out / "train_log.jsonl", "w") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")
    print("common checkpoint", st["step"], h[:16])


def pretrain_dev_set():
    """Pretraining-family ID development set (the trigger's ID measure; dev tuples, train numerics)."""
    import random
    rng = random.Random(8_888)
    out = []
    while len(out) < D.EVAL_N:
        ex = D.make_example(D.sample_ast("force", rng, tuple_splits=("dev",)), rng.choice(D.PRETRAIN_FAMILIES),
                            D.draw_style(rng), rng)
        if ex["valid"]:
            out.append(ex)
    return out


# ------------------------------------------------------------------ continuations
def arm_dir(arm):
    return ZOO / ("A_fixed" if arm == "A" else "B_adaptive")


def meta_path():
    return ZOO / "cont_meta.json"


def load_meta():
    return json.loads(meta_path().read_text())


def save_meta(m):
    meta_path().write_text(json.dumps(m, indent=1))


def cont_init():
    if meta_path().exists():
        raise SystemExit("continuations already initialised")
    V = vocab()
    common = TR.load_state(ZOO / "common")
    h0 = TR.state_hash(common)
    hashes = {}
    for arm in ("A", "B"):
        st = TR.load_state(ZOO / "common")
        TR.restart_for_continuation(st, CONT_SEED)
        TR.save_state(st, arm_dir(arm) / "live")
        card, carry = O.observe(st["model"], V, EVAL, None)
        card["step"] = 0
        torch.save(dict(h=carry["h"]), arm_dir(arm) / "carry.pt")
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in carry["sd"].items()}, arm_dir(arm) / "carry_sd.safetensors")
        with open(arm_dir(arm) / "cards.jsonl", "w") as f:
            f.write(json.dumps(card) + "\n")
        hashes[arm] = TR.state_hash(st)
    assert hashes["A"] == hashes["B"]
    save_meta(dict(step=0, common_hash=h0, start_hash=hashes["A"], lessons={}, reviews={}, tools={},
                   ledger=dict(A_train_cpu_s=0.0, B_train_cpu_s=0.0, observe_cpu_s=0.0, A_tokens=0, B_tokens=0)))
    print("A and B initialised from", h0[:16])
    packet()


def set_lesson(path):
    meta = load_meta()
    spec = json.loads(Path(path).read_text())
    D.validate_lesson(spec)
    meta["lessons"][str(meta["step"])] = spec
    save_meta(meta)
    print(f"lesson for B steps {meta['step'] + 1}-{meta['step'] + 100} accepted")


def review(path):
    meta = load_meta()
    r = json.loads(Path(path).read_text())
    need = ("what_changed", "knows", "fails", "explanation", "competing_explanation", "prediction", "tool_needed",
            "next_action", "why_this_checkpoint", "previous_outcome")
    miss = [k for k in need if not str(r.get(k, "")).strip()]
    if miss:
        raise SystemExit(f"review missing {miss}")
    meta["reviews"][str(meta["step"])] = r
    save_meta(meta)
    with open(ZOO / "journal.jsonl", "a") as f:
        f.write(json.dumps(dict(step=meta["step"], **r)) + "\n")
    print("review filed for step", meta["step"])


def advance():
    meta = load_meta()
    s = meta["step"]
    if str(s) not in meta["reviews"]:
        raise SystemExit(f"file the review for step {s} first")
    if str(s) not in meta["lessons"]:
        raise SystemExit(f"choose B's lesson for steps {s + 1}-{s + 100} first")
    if s >= TR.CONT_STEPS:
        raise SystemExit("continuation finished")
    V = vocab()
    lessons = dict(A=FIXED_CURRICULUM[s // 100], B=meta["lessons"][str(s)])
    for arm in ("A", "B"):
        st = TR.load_state(arm_dir(arm) / "live")
        log = []
        t = time.process_time()
        dig, ntok = TR.train_steps(st, lessons[arm], 100, log)
        meta["ledger"][f"{arm}_train_cpu_s"] += time.process_time() - t
        meta["ledger"][f"{arm}_tokens"] += ntok
        TR.save_state(st, arm_dir(arm) / "live")
        from safetensors.torch import load_file, save_file
        prev = dict(h=torch.load(arm_dir(arm) / "carry.pt", weights_only=True)["h"], sd=load_file(arm_dir(arm) / "carry_sd.safetensors"))
        t = time.process_time()
        card, carry = O.observe(st["model"], V, EVAL, prev)
        meta["ledger"]["observe_cpu_s"] += time.process_time() - t
        card["step"] = st["step"]
        card["lesson"] = lessons[arm]
        card["batch_digest"] = hashlib.sha256("".join(dig).encode()).hexdigest()[:16]
        card["train_log_tail"] = log[-1]
        card["state_hash"] = TR.state_hash(st)
        torch.save(dict(h=carry["h"]), arm_dir(arm) / "carry.pt")
        save_file({k: v.contiguous() for k, v in carry["sd"].items()}, arm_dir(arm) / "carry_sd.safetensors")
        ck = arm_dir(arm) / f"step_{st['step']:04d}"
        ck.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in st["model"].state_dict().items()}, ck / "model.safetensors")
        with open(arm_dir(arm) / "cards.jsonl", "a") as f:
            f.write(json.dumps(card) + "\n")
    meta["step"] = s + 100
    save_meta(meta)
    packet()


def cards(arm):
    return [json.loads(l) for l in open(arm_dir(arm) / "cards.jsonl")]


def packet():
    meta = load_meta()
    a, b = cards("A")[-1], cards("B")[-1]
    pb = cards("B")[-2] if len(cards("B")) > 1 else None
    print(f"=============== continuation step {meta['step']} ===============")
    keys = sorted(b["dev_acc"])
    for k in keys:
        d = f"  Δprev {b['dev_acc'][k] - pb['dev_acc'][k]:+.3f}" if pb else ""
        print(f"  {k:34s} B {b['dev_acc'][k]:.3f}  A {a['dev_acc'][k]:.3f}  B−A {b['dev_acc'][k] - a['dev_acc'][k]:+.3f}{d}")
    for k, f in (("IIA block (cross-family)", lambda c: c["iia_block_cross_family"]), ("edit inv intended", lambda c: c["edit_inv"]["intended"]),
                 ("edit add1 intended", lambda c: c["edit_add1"]["intended"]), ("binding probe ID", lambda c: c["binding_probe"]["id"]),
                 ("binding probe OOD", lambda c: c["binding_probe"]["ood"]), ("sense probe x2", lambda c: c["sense_probe"]["x2"]),
                 ("paraphrase all-agree", lambda c: c["paraphrase"]["all_families_agree"]),
                 ("paraphrase cos within", lambda c: c["paraphrase"]["cos_within"]), ("paraphrase cos between", lambda c: c["paraphrase"]["cos_between"]),
                 ("cka h prev", lambda c: c.get("cka_h_prev", float("nan")))):
        print(f"  {k:34s} B {f(b):.3f}  A {f(a):.3f}")
    print("  ledger", meta["ledger"])


PRIMARY_SETS = ("sealed/convert_newfam", "sealed/convert_ood_value", "sealed/equiv_trainfam", "sealed/equiv_newfam",
                "sealed/inverse_newfam", "sealed/inverse_ood_value", "sealed/sense_newframe_newctx",
                "sealed/sense_trainframe_newctx")
RETENTION_SET, RETENTION_TOL = "sealed/force_retention_pretrainfam", 0.05


def primary(correct, res, n_boot=2000, seed=0):
    """Registered primary endpoint (docs/30 §30.7): mean accuracy over PRIMARY_SETS, B − A, paired example-level
    bootstrap 95% CI (resampling within each set, sets equally weighted); retention guard for B."""
    g = torch.Generator().manual_seed(seed)
    A = {k: torch.tensor(correct["A_fixed"][k], dtype=torch.float) for k in PRIMARY_SETS}
    B = {k: torch.tensor(correct["B_adaptive"][k], dtype=torch.float) for k in PRIMARY_SETS}
    point = sum((B[k].mean() - A[k].mean()).item() for k in PRIMARY_SETS) / len(PRIMARY_SETS)
    boots = torch.zeros(n_boot)
    for k in PRIMARY_SETS:
        idx = torch.randint(0, len(A[k]), (n_boot, len(A[k])), generator=g)
        boots += (B[k][idx].mean(1) - A[k][idx].mean(1)) / len(PRIMARY_SETS)
    lo, hi = torch.quantile(boots, torch.tensor([0.025, 0.975])).tolist()
    ref = res["common (pre-curriculum reference)"][RETENTION_SET]
    guard = {n: res[n][RETENTION_SET] >= ref - RETENTION_TOL for n in ("A_fixed", "B_adaptive")}
    return dict(mean_A=sum(A[k].mean().item() for k in PRIMARY_SETS) / len(PRIMARY_SETS),
                mean_B=sum(B[k].mean().item() for k in PRIMARY_SETS) / len(PRIMARY_SETS),
                diff_B_minus_A=round(point, 4), ci95=[round(lo, 4), round(hi, 4)], retention_guard=guard,
                B_better_claimed=bool(lo > 0 and guard["B_adaptive"]), A_better=bool(hi < 0))


def final():
    seal = OUT / "final_sealed.json"
    if seal.exists():
        raise SystemExit("sealed evaluation already run")
    meta = load_meta()
    assert meta["step"] == TR.CONT_STEPS
    man = json.loads((EVAL / "manifest.json").read_text())
    D.SEALED_LOCK[0] = False                                  # the ONLY unlock after freezing
    V = vocab()
    nets = {"common (pre-curriculum reference)": TR.load_state(ZOO / "common")["model"],
            "A_fixed": TR.load_state(arm_dir("A") / "live")["model"], "B_adaptive": TR.load_state(arm_dir("B") / "live")["model"]}
    res = {n: {} for n in nets}
    correct = {n: {} for n in nets}
    t0 = time.process_time()
    for name, info in man["sealed"].items():
        exs, _ = gen_eval(name, info["seed"])
        assert digest(exs) == info["content_sha256"], f"sealed set {name} does not match its frozen hash"
        b = D.encode_batch(exs, V)
        for n, m in nets.items():
            m.eval()
            with torch.no_grad():
                ok = (m(b["tok"]).argmax(-1) == b["ans"]).tolist()
            res[n][name] = round(sum(ok) / len(ok), 4)
            correct[n][name] = [int(x) for x in ok]          # correctness bits only (no sealed content)
    res["primary"] = primary(correct, res)
    from sem import sealed_extras
    for n, m in nets.items():
        res[n].update(sealed_extras.extras(m, V, man, gen_eval))
    res["cpu_s"] = round(time.process_time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    seal.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("freeze", "pretrain", "cont-init", "lesson", "review", "advance", "packet", "final"))
    ap.add_argument("arg", nargs="?")
    a = ap.parse_args()
    TR.set_det()
    dict(freeze=freeze, pretrain=pretrain, **{"cont-init": cont_init}, packet=packet, advance=advance, final=final).get(a.mode, lambda: None)()
    if a.mode == "lesson":
        set_lesson(a.arg)
    if a.mode == "review":
        review(a.arg)


if __name__ == "__main__":
    main()
