"""Phase III cohort reviews (doc 41 §41.6 Q5): one review per 100-step checkpoint over all reconstructed networks,
generated from the offline audits. Events since the previous review are flagged; written commentary is added
separately at phase transitions, only after reading the generated review.

    python -m sci.swap_cohort <zoo> <out_dir>
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

GROUPS = ("mass_only", "acc_only", "neither")
DECL = {"mass_only": "M", "acc_only": "A"}
FOCUS, COMP, MID = (307, 311), (304, 310), (305,)


def load(zoo):
    runs = {}
    for p in sorted(Path(zoo).iterdir()):
        f = p / "audits.jsonl"
        if f.exists():
            runs[p.name] = {r["step"]: r for r in map(json.loads, f.read_text().splitlines()) if r}
    return runs


def declared_ood(r, var):
    return statistics.mean(v for k, v in r["ood_combo_acc"].items() if k[0] == var)


def row(name, r):
    g, s = name.rsplit("_s", 1)
    var = DECL.get(g)
    tag = "focus" if int(s) in FOCUS else "comparator" if int(s) in COMP else "intermediate" if int(s) in MID else ""
    d = dict(net=name, tag=tag, id=r["acc_id"], ood=r["acc_ood"], oodM=declared_ood(r, "M"), oodA=declared_ood(r, "A"),
             iia=r[f"iia_{var}"] if var else None, iia_oodsrc=r[f"iia_{var}_oodsrc"] if var else None,
             rsiM=r["rsi_M"], rsiA=r["rsi_A"])
    return d


def events(prev, cur):
    ev = []
    if prev is None:
        return ev
    if prev["id"] < 0.95 <= cur["id"]:
        ev.append("ID >= 0.95")
    if prev["id"] >= 0.95 > cur["id"]:
        ev.append("ID fell below 0.95")
    if cur["iia"] is not None and prev["iia"] < 0.9 <= cur["iia"]:
        ev.append("declared block formed (IIA >= 0.9)")
    if cur["iia_oodsrc"] is not None and prev["iia_oodsrc"] < 0.5 <= cur["iia_oodsrc"]:
        ev.append("block carries unseen inputs (IIA-OOD >= 0.5)")
    if cur["iia_oodsrc"] is not None and prev["iia_oodsrc"] >= 0.5 > cur["iia_oodsrc"]:
        ev.append("block LOST unseen inputs (IIA-OOD < 0.5)")
    if abs(cur["ood"] - prev["ood"]) >= 0.15:
        ev.append(f"OOD jump {prev['ood']:.2f}->{cur['ood']:.2f}")
    return ev


def review(runs, step, prev_step):
    L = [f"# Cohort review — step {step:,}", "",
         "Generated from offline audits of the reconstructed networks (doc 41 §41.6). IIA = designed-block interchange "
         "accuracy for the declared variable; IIA-OOD = same with a never-trained source (validated causal measure); "
         "oodM/oodA = accuracy on never-trained combinations of mass / acceleration; RSI = route-specificity index.", ""]
    rows = {n: row(n, r[step]) for n, r in runs.items() if step in r}
    prevs = {n: row(n, r[prev_step]) for n, r in runs.items() if prev_step is not None and prev_step in r}
    L += ["| group | n | mean ID | mean OOD | ID ≥ 0.95 | mean IIA | mean IIA-OOD |", "|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        rs = [v for n, v in rows.items() if n.startswith(g)]
        if not rs:
            continue
        iia = [v["iia"] for v in rs if v["iia"] is not None]
        io = [v["iia_oodsrc"] for v in rs if v["iia_oodsrc"] is not None]
        L.append(f"| {g} | {len(rs)} | {statistics.mean(v['id'] for v in rs):.3f} | {statistics.mean(v['ood'] for v in rs):.3f} | "
                 f"{sum(v['id'] >= 0.95 for v in rs)} | {statistics.mean(iia):.3f} | {statistics.mean(io):.3f} |" if iia else
                 f"| {g} | {len(rs)} | {statistics.mean(v['id'] for v in rs):.3f} | {statistics.mean(v['ood'] for v in rs):.3f} | "
                 f"{sum(v['id'] >= 0.95 for v in rs)} | — | — |")
    L += ["", "## Focus, comparator and intermediate networks (mass_only)", "",
          "| network | role | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in FOCUS + COMP + MID:
        n = f"mass_only_s{s}"
        if n in rows:
            v = rows[n]
            L.append(f"| {n} | {v['tag']} | {v['id']:.3f} | {v['ood']:.3f} | {v['oodM']:.2f} | {v['oodA']:.2f} | {v['iia']:.2f} | "
                     f"{v['iia_oodsrc']:.2f} | {v['rsiM']:.2f} | {v['rsiA']:.2f} |")
    ev = {n: events(prevs.get(n), v) for n, v in rows.items()}
    ev = {n: e for n, e in ev.items() if e}
    L += ["", f"## Events since step {prev_step:,}" if prev_step is not None else "## Events", ""]
    L += [f"- **{n}**: {'; '.join(e)}" for n, e in sorted(ev.items())] or ["- none"]
    L += ["", "## All networks", "", "| network | ID | OOD | oodM | oodA | IIA | IIA-OOD | RSI-M | RSI-A |", "|---|---|---|---|---|---|---|---|---|"]
    for n, v in sorted(rows.items()):
        f = lambda x: "—" if x is None else f"{x:.2f}"
        L.append(f"| {n} | {v['id']:.3f} | {v['ood']:.3f} | {v['oodM']:.2f} | {v['oodA']:.2f} | {f(v['iia'])} | {f(v['iia_oodsrc'])} | "
                 f"{v['rsiM']:.2f} | {v['rsiA']:.2f} |")
    return "\n".join(L) + "\n", ev


def main(zoo, out):
    runs = load(zoo)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    all_ev = {}
    prev = None
    for step in range(0, 6001, 100):
        if not any(step in r for r in runs.values()):
            break
        txt, ev = review(runs, step, prev)
        (out / f"cohort_{step:05d}.md").write_text(txt)
        all_ev[step] = ev
        prev = step
    (out / "events.json").write_text(json.dumps(all_ev, indent=1))
    return all_ev


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
