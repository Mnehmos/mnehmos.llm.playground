"""Held-out causal test of the canonical code (doc 39). Retrospective, exploratory.

`patch_semantics_split`: copy the donor residual at the source-number token (after block `site`; optionally also
the unit token) into the base, exactly as `concept.patch_semantics`, but enumerate ALL eligible pairs and classify
each by (base pool: trained | heldout) x (combination implied by the canonical prediction: trained | heldout) and,
separately, the combination implied by the number prediction. Reserved test combinations are excluded everywhere.

    python -m sci.heldout_causal            # runs the doc-39 analyses on extracted weights, writes results
"""
from __future__ import annotations

import gzip
import json
import random
import statistics
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import concept as C  # noqa: E402
from sci import factorial as Fx  # noqa: E402
from sci import factorial_analysis as FA  # noqa: E402
from sem import data as D  # noqa: E402
from sem import lang as L  # noqa: E402
from sem import model as SM  # noqa: E402

WEIGHTS = Path("/home/user/nri_weights")
OUT = Fx.REPO_RUNS / "heldout_causal.json"


def pools(split):
    test = set(split["test"])
    return dict(trained=set(split["train"]), heldout=set(split["dev"]), test=test)


def classify(combo, P):
    return "trained" if combo in P["trained"] else "heldout" if combo in P["heldout"] else "test"


def enumerate_pairs(split):
    """All (base, donor) pairs with same dimension, different source units, donor trained, three distinct answers,
    nothing touching the reserved test; returns list of dicts with predictions and strata."""
    P = pools(split)
    out = []
    for rb, vb in sorted(P["trained"] | P["heldout"]):
        for rd, vd in sorted(P["trained"]):
            dim, ub, tb = rb
            if rd[0] != dim or rd[1] == ub:
                continue
            a_base = C.answer(dim, vb, ub, tb)
            a_num = C.answer(dim, vd, ub, tb)
            q = C.canonical(dim, vd, rd[1])
            a_can = q * L.INV[L.SCALES[dim][tb]] % C.P
            if len({a_base, a_num, a_can}) < 3:
                continue
            v_can = q * L.INV[L.SCALES[dim][ub]] % C.P                # base-route input whose answer is a_can
            base_c, num_c, can_c = classify((rb, vb), P), classify((rb, vd), P), classify((rb, v_can), P)
            if "test" in (base_c, num_c, can_c):
                continue
            out.append(dict(base=(rb, vb), donor=(rd, vd), a_base=a_base, a_num=a_num, a_can=a_can,
                            base_pool=base_c, num_implied=num_c, can_implied=can_c))
    return out


@torch.no_grad()
def run_pairs(model, vocab, pairs, site, positions="src"):
    canon = dict(num="num", unit="full")
    rng = random.Random(0)
    bases = [C.example(p["base"][0], p["base"][1], "C01", canon, rng) for p in pairs]
    donors = [C.example(p["donor"][0], p["donor"][1], "C01", canon, rng) for p in pairs]
    bb, bd = D.encode_batch(bases, vocab), D.encode_batch(donors, vocab)
    key = {0: "x0", 1: "x1", 2: "x2"}[site]
    cb, cd = model.trunk(bb["tok"], ret=True), model.trunk(bd["tok"], ret=True)
    x = cb[key].clone()
    a = torch.arange(len(pairs))
    pos_b = [C.src_pos(bb)] + ([(bb["role"] == L.ROLE_ID["src_unit"]).long().argmax(1)] if positions == "src+unit" else [])
    pos_d = [C.src_pos(bd)] + ([(bd["role"] == L.ROLE_ID["src_unit"]).long().argmax(1)] if positions == "src+unit" else [])
    for pb, pd in zip(pos_b, pos_d):
        x[a, pb] = cd[key][a, pd]
    s = site
    if s == 0:
        x = x + model.attn1(model.ln1a(x), bb["tok"] != D.PAD)
        x = x + model.mlp1(model.ln1m(x))
        s = 1
    pred = C.finish_from(model, x, bb["tok"] != D.PAD, s).argmax(-1)
    ans = lambda k: torch.tensor([L.ANS_ID[str(p[k])] for p in pairs])
    return dict(canonical=(pred == ans("a_can")), number=(pred == ans("a_num")), unchanged=(pred == ans("a_base")))


def strata_rates(model, vocab, pairs, site, positions="src", n=300, seed=0):
    rng = random.Random(seed)
    groups = {}
    for p in pairs:
        groups.setdefault((p["base_pool"], p["can_implied"]), []).append(p)
    res = {}
    for k, g in sorted(groups.items()):
        g = rng.sample(g, min(n, len(g)))
        r = run_pairs(model, vocab, g, site, positions)
        res[f"base_{k[0]}|can_implied_{k[1]}"] = dict(n=len(g), **{m: round(v.float().mean().item(), 4) for m, v in r.items()})
    return res


def load(run, step):
    vocab = D.Vocab.from_json(json.loads((ROOT / "runs/zoo_sem/eval/vocab.json").read_text()))
    m = SM.SemStudent(len(vocab.itos))
    m.load_state_dict(load_file(WEIGHTS / run / "ckpt" / f"step_{step:05d}" / "model.safetensors"))
    return m.eval(), vocab


def audits(run):
    with gzip.open(Fx.REPO_RUNS / run / "audits.jsonl.gz", "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def m2_clustered():
    """Seed-clustered re-analysis of M2 (doc 39 §39.3)."""
    fr = {}
    for s in Fx.SEEDS:
        n = Fx.run_name("C", 1.0, s)
        a = audits(n)
        held = [x["behaviour"]["heldout"] for x in a]
        can = [x["causal"]["semantics_x1_src"]["canonical"] for x in a]
        const = a[0]["behaviour"]["heldout_best_constant"]
        steps = [x["step"] for x in a]
        onset = next((steps[i] for i in range(len(a) - 1) if held[i] >= const + .2 and held[i + 1] >= const + .2), None)
        if onset is None:
            continue
        agree = tot = 0
        for i in range(1, len(a)):
            if steps[i - 1] < onset:
                continue
            dh = held[i] - held[i - 1]
            if abs(dh) >= 0.05:
                tot += 1
                agree += (can[i] - can[i - 1]) * dh > 0
        if tot:
            fr[n] = dict(agree=agree, total=tot, fraction=round(agree / tot, 4))
    f = [v["fraction"] for v in fr.values()]
    d = [x - 0.5 for x in f]
    return dict(per_network=fr, n_networks=len(f), mean_fraction=round(statistics.mean(f), 4),
                ci95_seed_bootstrap=[round(x + 0.5, 4) for x in FA.bootstrap_ci(d)], p_signflip=FA.sign_flip_p(d),
                pooled_registered=dict(agree=sum(v["agree"] for v in fr.values()), total=sum(v["total"] for v in fr.values())))


def main():
    split = C.make_split()
    pairs = enumerate_pairs(split)
    res = dict(n_pairs_by_stratum={}, controls={}, final={}, timing={})
    for p in pairs:
        k = f"base_{p['base_pool']}|can_implied_{p['can_implied']}"
        res["n_pairs_by_stratum"][k] = res["n_pairs_by_stratum"].get(k, 0) + 1
    torch.set_num_threads(4)
    # controls: input-level swaps (x0) on C wd 1.0 final weights; untrained (step 0) at x1
    for s in Fx.SEEDS[:4]:
        m, v = load(Fx.run_name("C", 1.0, s), 6000)
        res["controls"][f"x0_token_C1.0_s{s}"] = strata_rates(m, v, pairs, 0, "src")
        res["controls"][f"x0_token+unit_C1.0_s{s}"] = strata_rates(m, v, pairs, 0, "src+unit")
        m0, v = load(Fx.run_name("C", 1.0, s), 0)
        res["controls"][f"x1_step0_s{s}"] = strata_rates(m0, v, pairs, 1, "src")
    # final step, all populations and weight decays, site x1
    for p_, w, s in Fx.all_runs():
        n = Fx.run_name(p_, w, s)
        m, v = load(n, 6000)
        r = strata_rates(m, v, pairs, 1, "src")
        r["dev_heldout_acc"] = audits(n)[-1]["behaviour"]["heldout"]
        res["final"][n] = r
    # timing for C wd 1.0
    for s in Fx.SEEDS:
        n = Fx.run_name("C", 1.0, s)
        res["timing"][n] = {}
        for st in range(500, 6001, 500):
            m, v = load(n, st)
            r = strata_rates(m, v, pairs, 1, "src", n=200)
            res["timing"][n][st] = {k: r[k]["canonical"] for k in r if k.startswith("base_trained")}
    res["m2_clustered"] = m2_clustered()
    OUT.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
