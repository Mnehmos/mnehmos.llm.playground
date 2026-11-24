"""Registered analysis of the concept factorial (doc 37 §37.6-37.9) + declared exploratory analyses.

    python -m sci.factorial_analysis            # writes runs/concept_factorial/analysis.json and prints a summary
"""
from __future__ import annotations

import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import cohort as K  # noqa: E402
from sci import episodes as E  # noqa: E402
from sci import factorial as Fx  # noqa: E402

SEEDS = Fx.SEEDS


# ------------------------------------------------------------------ statistics
def sign_flip_p(d, alternative="greater"):
    """Exact sign-flip permutation p-value for the mean of paired differences d."""
    n, obs = len(d), statistics.mean(d)
    count = 0
    for signs in itertools.product((1, -1), repeat=n):
        m = sum(s * x for s, x in zip(signs, d)) / n
        if alternative == "greater":
            count += m >= obs - 1e-12
        else:
            count += abs(m) >= abs(obs) - 1e-12
    return count / 2 ** n


def bootstrap_ci(d, n=10_000, seed=0):
    rng = random.Random(seed)
    ms = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def binom_greater(k, n, p=0.5):
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


# ------------------------------------------------------------------ data
def trajectory(name):
    out = []
    for st in range(0, Fx.STEPS + 1, Fx.BLOCK):
        f = Fx.LIVE / name / "ckpt" / f"step_{st:05d}" / "audit.json"
        if not f.exists():
            break
        a = json.loads(f.read_text())
        m = K.metrics(a)
        m["step"], m["const"] = st, a["behaviour"]["heldout_best_constant"]
        m["upd"] = a["updates"]
        out.append(m)
    return out


def primary(final):
    acc = lambda p, w, s, key="test": final[Fx.run_name(p, w, s)][key]
    contrasts = {
        "H1 C-D @wd1.0": ([acc("C", 1.0, s) - acc("D", 1.0, s) for s in SEEDS], "greater"),
        "H2 (C-D)@1.0 - (C-D)@0.1": ([(acc("C", 1.0, s) - acc("D", 1.0, s)) - (acc("C", 0.1, s) - acc("D", 0.1, s)) for s in SEEDS], "greater"),
        "H3 C@1.0 - C@0.1": ([acc("C", 1.0, s) - acc("C", 0.1, s) for s in SEEDS], "greater"),
        "H4 C-B @wd1.0": ([acc("C", 1.0, s) - acc("B", 1.0, s) for s in SEEDS], "greater"),
        "H5 B-A @wd1.0": ([acc("B", 1.0, s) - acc("A", 1.0, s) for s in SEEDS], "two-sided"),
    }
    res = {}
    for k, (d, alt) in contrasts.items():
        res[k] = dict(mean=round(statistics.mean(d), 4), ci95=[round(x, 4) for x in bootstrap_ci(d)], p=sign_flip_p(d, alt), alternative=alt,
                      per_seed=[round(x, 4) for x in d])
    adj = holm([r["p"] for r in res.values()])
    for (k, r), a in zip(res.items(), adj):
        r["p_holm"] = a
        r["significant"] = a < 0.05
    means = {f"wd{w}_{p}": dict(test=round(statistics.mean(acc(p, w, s) for s in SEEDS), 4),
                                test_sd=round(statistics.pstdev(acc(p, w, s) for s in SEEDS), 4),
                                test_newframe=round(statistics.mean(acc(p, w, s, "test_newframe") for s in SEEDS), 4))
             for w in Fx.WDS for p in Fx.POPS}
    sec = {}
    for k, (d, alt) in {"H1nf": ([acc("C", 1.0, s, "test_newframe") - acc("D", 1.0, s, "test_newframe") for s in SEEDS], "greater"),
                        "H2nf": ([(acc("C", 1.0, s, "test_newframe") - acc("D", 1.0, s, "test_newframe")) - (acc("C", 0.1, s, "test_newframe") - acc("D", 0.1, s, "test_newframe")) for s in SEEDS], "greater")}.items():
        sec[k] = dict(mean=round(statistics.mean(d), 4), ci95=[round(x, 4) for x in bootstrap_ci(d)], p=sign_flip_p(d, alt))
    return dict(contrasts=res, condition_means=means, secondary_newframe=sec)


def mechanistic(trajs):
    ons = {n: K.onsets_so_far(n, Fx.STEPS) for n in trajs}
    # M1
    both = [(n, o["utilised"], o["compositional"]) for n, o in ons.items() if o["utilised"] is not None and o["compositional"] is not None]
    precede = sum(u <= c - 100 for _, u, c in both)
    follow = sum(u >= c for _, u, c in both)
    comp_never_util = [n for n, o in ons.items() if o["compositional"] is not None and o["utilised"] is None]
    m1 = dict(networks_with_both=len(both), precede=precede, follow_or_same=follow,
              p_sign=binom_greater(precede, precede + follow) if precede + follow else None,
              falsified=follow >= precede, detail=both, compositional_without_utilisation=comp_never_util)
    # M2 (C networks after their compositional onset)
    agree = total = 0
    for n, t in trajs.items():
        o = ons[n]
        if not n.split("_")[1] == "C" or o["compositional"] is None:
            continue
        seq = [m for m in t if m["step"] >= o["compositional"]]
        for a, b in zip(seq, seq[1:]):
            dh = b["held"] - a["held"]
            if abs(dh) >= 0.05:
                total += 1
                agree += (b["x1_can"] - a["x1_can"]) * dh > 0
    m2 = dict(changes=total, agree=agree, fraction=round(agree / total, 4) if total else None,
              p_binomial=binom_greater(agree, total) if total else None, falsified=(agree / total <= 0.5) if total else None)
    # M3
    d = [trajs[Fx.run_name("C", 1.0, s)][-1]["x1_cmn"] - trajs[Fx.run_name("C", 0.1, s)][-1]["x1_cmn"] for s in SEEDS]
    m3 = dict(mean=round(statistics.mean(d), 4), ci95=[round(x, 4) for x in bootstrap_ci(d)], p=sign_flip_p(d), per_seed=[round(x, 4) for x in d])
    # M4
    dec = sum(trajs[Fx.run_name("C", 0.1, s)][-1]["dec_q"] >= 0.5 for s in SEEDS)
    nocomp = sum(ons[Fx.run_name("C", 0.1, s)]["compositional"] is None for s in SEEDS)
    m4 = dict(decodable_at_end=dec, not_compositional=nocomp, holds=dec >= 9 and nocomp >= 9)
    return dict(onsets=ons, M1=m1, M2=m2, M3=m3, M4=m4)


def exploratory(trajs, ons):
    out = {}
    ep = E.count(Fx.STEPS)
    out["episodes"] = E.summary(ep)
    # collapse cycles in C wd 1.0 composers: held-out drop >= 0.15 between consecutive audits after onset
    cycles = []
    for n, t in trajs.items():
        o = ons[n]
        if o["compositional"] is None:
            continue
        eps = set(ep[n]["steps"])
        for i in range(1, len(t)):
            a, b = t[i - 1], t[i]
            if b["step"] <= o["compositional"] or a["held"] - b["held"] < 0.15:
                continue
            rec = next((t[j]["step"] for j in range(i + 1, len(t)) if t[j]["held"] >= a["held"] - 0.05), None)
            after = t[[m["step"] for m in t].index(rec) + 1]["held"] if rec is not None and rec + Fx.BLOCK <= t[-1]["step"] else None
            win_a = a["upd"].get("W_in", {}).get("cos_adam_wd")
            win_b = b["upd"].get("W_in", {}).get("cos_adam_wd")
            cycles.append(dict(run=n, step=b["step"], held_before=a["held"], held_at=b["held"], own_at=b["own"],
                               in_episode_block=b["step"] in eps, recovered_at=rec, held_one_after_recovery=after,
                               ratchet=(after is not None and after > a["held"]), x1_can_before=a["x1_can"], x1_can_at=b["x1_can"],
                               dec_before=a["dec_q"], dec_at=b["dec_q"], win_cos_before=win_a, win_cos_at=win_b))
    out["collapse_cycles"] = cycles
    # own-trained drop at audits of episode blocks after 2,500, per population at wd 1.0
    drops = {}
    for p in Fx.POPS:
        k = tot = 0
        for s in SEEDS:
            n = Fx.run_name(p, 1.0, s)
            t = {m["step"]: m for m in trajs[n]}
            for st in ep[n]["steps"]:
                if st > 2500 and st in t and st - Fx.BLOCK in t:
                    tot += 1
                    k += t[st - Fx.BLOCK]["own"] - t[st]["own"] >= 0.05
        drops[p] = dict(episode_blocks_after_2500=tot, with_own_drop=k, fraction=round(k / tot, 3) if tot else None)
    out["own_drop_in_episodes_wd1"] = drops
    # readout cosine: episode blocks vs calm blocks (all wd 1.0 networks)
    ce, cc = [], []
    for p in Fx.POPS:
        for s in SEEDS:
            n = Fx.run_name(p, 1.0, s)
            eps = set(ep[n]["steps"])
            for m in trajs[n]:
                c = m["upd"].get("W_in", {}).get("cos_adam_wd")
                if c is None or m["step"] <= 300:
                    continue
                (ce if m["step"] in eps else cc).append(c)
    out["readout_cos_episode_vs_calm_wd1"] = dict(episode_mean=round(statistics.mean(ce), 3) if ce else None, n_episode=len(ce),
                                                  calm_mean=round(statistics.mean(cc), 3) if cc else None, n_calm=len(cc))
    # auxiliary loss relative rise in episode blocks, C vs D (wd 1.0)
    rise = {}
    for p in ("C", "D"):
        r = []
        for s in SEEDS:
            n = Fx.run_name(p, 1.0, s)
            t = {m["step"]: m for m in trajs[n]}
            for st in ep[n]["steps"]:
                if st in t and st - Fx.BLOCK in t and t[st - Fx.BLOCK]["loss_aux"]:
                    r.append(t[st]["loss_aux"] / max(t[st - Fx.BLOCK]["loss_aux"], 1e-6))
        rise[p] = dict(n=len(r), median_ratio=round(statistics.median(r), 3) if r else None)
    out["aux_loss_rise_in_episodes_wd1"] = rise
    # composers: canonical vs mixed (mean x1 canonical-number after onset)
    comp = []
    for n, o in ons.items():
        if o["compositional"] is None:
            continue
        t = [m for m in trajs[n] if m["step"] >= o["compositional"]]
        below = sum(m["held"] < m["const"] + 0.2 for m in t)
        comp.append(dict(run=n, onset=o["compositional"], utilised=o["utilised"], mean_x1_cmn_after=round(statistics.mean(m["x1_cmn"] for m in t), 3),
                         final_held=t[-1]["held"], final_newframe=t[-1]["newf"], audits_after=len(t), audits_below_criterion=below))
    out["composers"] = comp
    return out


def main():
    trajs = {Fx.run_name(p, w, s): trajectory(Fx.run_name(p, w, s)) for p, w, s in Fx.all_runs()}
    res = {}
    fp = Fx.REPO_RUNS / "final_test.json"
    if fp.exists():
        res["primary"] = primary(json.loads(fp.read_text()))
    res["mechanistic"] = mechanistic(trajs)
    res["exploratory"] = exploratory(trajs, res["mechanistic"]["onsets"])
    (Fx.REPO_RUNS / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    r = main()
    if "primary" in r:
        for k, v in r["primary"]["contrasts"].items():
            print(k, {x: v[x] for x in ("mean", "ci95", "p", "p_holm", "significant")})
    for k in ("M1", "M2", "M3", "M4"):
        print(k, {x: y for x, y in r["mechanistic"][k].items() if x not in ("detail", "per_seed")})
