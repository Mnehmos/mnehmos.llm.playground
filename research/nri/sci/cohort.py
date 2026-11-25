"""Cohort tables and anomaly flags for the factorial's per-100-step supervisor reviews (doc 37 §37.5).

    python -m sci.cohort <step>            # print the cohort table + anomalies; write runs/concept_factorial/cohorts/
    python -m sci.cohort status            # which cohorts are fully audited
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import factorial as Fx  # noqa: E402

OUT = Fx.REPO_RUNS / "cohorts"


def metrics(a):
    b, r, c, t = a["behaviour"], a["representation"], a["causal"], a["traces"]
    s1, s2 = c["semantics_x1_src"], c["semantics_x2_src"]
    upd = a["updates"]
    m = dict(own=b["own_practiced"], prac=b["practiced"], para=b["practiced_paraphrase"], held=b["heldout"],
             newf=b["heldout_newframe"], dec_q=r["decode_x1m_q_knn"], dec_v=r["decode_x1m_v_knn"],
             x1_can=s1["canonical"], x1_num=s1["number"], x1_cmn=round(s1["canonical"] - s1["number"], 4),
             x2_can=s2["canonical"], x2_unch=s2["unchanged"],
             circ_raw=r["circle_x1m_raw"]["dlog"], circ_can=r["circle_x1m_canonical"]["dlog"],
             loss_held=t["loss_task_by_panel"]["heldout"], loss_aux=t.get("loss_aux_mean"),
             cos_ta_x1m=t["x1m"].get("task_aux_cos_src"), wnorm=round(sum(v ** 2 for v in a["weight_norm"].values()) ** 0.5, 3))
    if "W_in" in upd:
        m.update(win_adam=upd["W_in"]["adam_net"], win_wd=upd["W_in"]["wd_net"], win_cos=upd["W_in"]["cos_adam_wd"])
    return m


def load(step):
    out = {}
    for p, w, s in Fx.all_runs():
        f = Fx.LIVE / Fx.run_name(p, w, s) / "ckpt" / f"step_{step:05d}" / "audit.json"
        if f.exists():
            out[Fx.run_name(p, w, s)] = json.loads(f.read_text())
    return out


def interval_stats(name, step):
    f = Fx.LIVE / name / "interval.jsonl"
    if not f.exists() or step == 0:
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l]
    rows = [r for r in rows if step - Fx.BLOCK < r["step"] <= step]
    if not rows:
        return None
    g = sorted(r["grad_norm"] for r in rows)
    med = g[len(g) // 2]
    seq = [r["grad_norm"] for r in rows]
    run = best = 0
    for x in seq:                                       # longest run of consecutive steps above 3x median
        run = run + 1 if x > 3 * med else 0
        best = max(best, run)
    return dict(gmax=g[-1], gmed=med, spike=g[-1] > 5 * med, sustained=best >= 3, longest_run=best,
                task_max=max(r["task"] for r in rows), loss_item_max=max(r["loss_max_item"] for r in rows))


def onsets_so_far(name, upto):
    """Registered onsets (doc 37 §37.7) from all audits <= upto."""
    d = Fx.LIVE / name / "ckpt"
    seq = []
    for st in range(0, upto + 1, Fx.BLOCK):
        f = d / f"step_{st:05d}" / "audit.json"
        if not f.exists():
            break
        seq.append((st, metrics(json.loads(f.read_text())), json.loads(f.read_text())["behaviour"]["heldout_best_constant"]))
    crit = dict(decodable=lambda m, c: m["dec_q"] >= 0.5, utilised=lambda m, c: m["x1_cmn"] >= 0.2,
                compositional=lambda m, c: m["held"] >= c + 0.2)
    out = {}
    for k, f in crit.items():
        out[k] = next((a[0] for a, b in zip(seq, seq[1:]) if f(a[1], a[2]) and f(b[1], b[2])), None)
    return out


def cohort(step, write=True):
    cur, prev = load(step), (load(step - Fx.BLOCK) if step > 0 else {})
    n_all = len(Fx.all_runs())
    per = {n: metrics(a) for n, a in cur.items()}
    keys = ["own", "para", "held", "newf", "dec_q", "x1_cmn", "x2_can", "circ_raw", "circ_can", "loss_held", "cos_ta_x1m", "wnorm", "win_cos"]
    table = {}
    for w in Fx.WDS:
        for p in Fx.POPS:
            rows = [per[n] for n in per if n.startswith(f"wd{w}_{p}_")]
            if not rows:
                continue
            e = {}
            for k in keys:
                v = [r[k] for r in rows if r.get(k) is not None]
                if v:
                    e[k] = (round(statistics.mean(v), 3), round(statistics.pstdev(v), 3))
            e["n"] = len(rows)
            table[f"wd{w}_{p}"] = e
    anomalies = []
    for n, m in per.items():
        flags = []
        if n in prev:
            pm = metrics(prev[n])
            if abs(m["held"] - pm["held"]) >= 0.15:
                flags.append(f"held-out {pm['held']:.2f}->{m['held']:.2f}")
            if pm["own"] - m["own"] >= 0.05:
                flags.append(f"own-trained drop {pm['own']:.2f}->{m['own']:.2f}")
            if abs(m["x1_can"] - pm["x1_can"]) >= 0.15:
                flags.append(f"x1 canonical {pm['x1_can']:.2f}->{m['x1_can']:.2f}")
        iv = interval_stats(n, step)
        if iv and iv["spike"]:
            kind = "SUSTAINED" if iv["sustained"] else "isolated"
            flags.append(f"grad spike ({kind}, run {iv['longest_run']}) max {iv['gmax']:.2f} vs median {iv['gmed']:.2f}")
        ons = onsets_so_far(n, step + Fx.BLOCK)
        for k, v in ons.items():
            if v is not None and v == step - Fx.BLOCK:
                flags.append(f"ONSET {k} at {v}")
        if flags:
            anomalies.append(dict(run=n, flags=flags, metrics=m))
    res = dict(step=step, audited=len(cur), of=n_all, table=table, anomalies=anomalies)
    if write and len(cur) == n_all:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"cohort_{step:05d}.json").write_text(json.dumps(res))
    return res


def show(res):
    print(f"cohort {res['step']}: {res['audited']}/{res['of']} audited")
    keys = list(next(iter(res["table"].values())).keys()) if res["table"] else []
    keys = [k for k in keys if k != "n"]
    print(f"{'cond':9s} " + " ".join(f"{k:>11s}" for k in keys))
    for c, e in res["table"].items():
        print(f"{c:9s} " + " ".join(f"{e[k][0]:>6.2f}±{e[k][1]:<4.2f}" if k in e else f"{'-':>11s}" for k in keys))
    print(f"anomalies: {len(res['anomalies'])}")
    for a in res["anomalies"]:
        print("  ", a["run"], "; ".join(a["flags"]))


def status():
    done = []
    for st in range(0, Fx.STEPS + 1, Fx.BLOCK):
        n = sum(1 for p, w, s in Fx.all_runs() if (Fx.LIVE / Fx.run_name(p, w, s) / "ckpt" / f"step_{st:05d}" / "AUDITED").exists())
        if n:
            done.append((st, n))
    return done


if __name__ == "__main__":
    if sys.argv[1] == "status":
        print(status())
    else:
        show(cohort(int(sys.argv[1])))
