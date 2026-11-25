"""Phase III/IV analysis of the 36 reconstructed role-swap trajectories (doc 41 §41.7; doc 42). Retrospective,
exploratory; the network is the unit. Definitions fixed here before the full set of audits is read:

  outcome_V     declared-variable OOD accuracy at step 6,000 (mean over its four never-trained combinations)
  escape step   first audit with ID accuracy >= 0.5           (end of the shared plateau)
  settle step   first audit with ID accuracy >= 0.95
  RSI_V@escape  route-specificity index of the declared variable at the escape step
  revised       |outcome_V(6000) - outcome_V(settle)| > 0.2    (is the outcome ever revised after fitting?)
  generaliser   outcome_V >= 0.9 ; failure outcome_V <= 0.2 (descriptive labels only)

Predictors (Phase IV.1), each correlated (Spearman) with outcome_V over the 24 declared networks, and the
leave-one-out accuracy of a one-threshold rule (threshold chosen on the other 23):
  RSI_V at escape, RSI_V at fixed steps 100..1,000, step-0 RSI_V, escape step, early task loss / grad norm
  (interval log, steps 1-500), declared-block IIA at escape.
Generalisation vs editability: Spearman of outcome_V (and acc_ood) with doc-15 edit specificity and copy-into-block.

    python -m sci.swap_analysis /home/user/swap_recon/zoo
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci.mechanism_sswd import spearman  # noqa: E402

DECL = {"mass_only": "M", "acc_only": "A"}
OUT = ROOT / "runs" / "swap_recon" / "analysis.json"


def oodv(r, v):
    return statistics.mean(x for k, x in r["ood_combo_acc"].items() if k[0] == v)


def load(zoo):
    out = {}
    for p in sorted(Path(zoo).iterdir()):
        f = p / "audits.jsonl"
        if f.exists():
            a = sorted((json.loads(l) for l in f.read_text().splitlines() if l.strip()), key=lambda r: r["step"])
            if a and a[-1]["step"] == 6000:
                out[p.name] = a
    return out


def interval_early(run_dir, upto=500):
    task, gn = [], []
    with open(Path(run_dir) / "interval.jsonl") as f:
        for l in f:
            r = json.loads(l)
            if r["step"] > upto:
                break
            task.append(r["task"])
            gn.append(math.sqrt(sum(v * v for v in r["grad_norm"].values())))
    return statistics.mean(task), statistics.mean(gn)


def loo_threshold(x, y, cut=0.5):
    """Leave-one-out accuracy of predicting y >= cut from x with a single threshold and direction fit on the rest."""
    hits = 0
    for i in range(len(x)):
        xs = [x[j] for j in range(len(x)) if j != i]
        ys = [y[j] >= cut for j in range(len(x)) if j != i]
        best = None
        for t in sorted(set(xs)):
            for sign in (1, -1):
                acc = sum(((sign * a) >= (sign * t)) == b for a, b in zip(xs, ys))
                if best is None or acc > best[0]:
                    best = (acc, t, sign)
        _, t, sign = best
        hits += ((sign * x[i]) >= (sign * t)) == (y[i] >= cut)
    return hits / len(x)


def main(zoo):
    runs = load(zoo)
    hist = {d["model"]: d for d in json.loads((ROOT / "results/swap/metrics.json").read_text())}
    net = {}
    for n, a in runs.items():
        g, s = n.rsplit("_s", 1)
        v = DECL.get(g)
        by = {r["step"]: r for r in a}
        esc = next((r["step"] for r in a if r["acc_id"] >= 0.5), None)
        set_ = next((r["step"] for r in a if r["acc_id"] >= 0.95), None)
        d = dict(group=g, seed=int(s), acc_id=a[-1]["acc_id"], acc_ood=a[-1]["acc_ood"], escape=esc, settle=set_,
                 oodM=oodv(a[-1], "M"), oodA=oodv(a[-1], "A"))
        tl, gn = interval_early(Path(zoo) / n)
        d.update(early_task_loss=tl, early_grad_norm=gn)
        if v:
            d["outcome"] = oodv(a[-1], v)
            d["rsi_escape"] = by[esc][f"rsi_{v}"] if esc is not None else None
            d["rsi_step0"] = by[0][f"rsi_{v}"]
            for st in range(100, 1001, 100):
                d[f"rsi_{st}"] = by[st][f"rsi_{v}"]
            d["iia_escape"] = by[esc][f"iia_{v}"] if esc is not None else None
            d["iia_oodsrc_6000"] = a[-1][f"iia_{v}_oodsrc"]
            if set_ is not None:
                d["outcome_at_settle"] = oodv(by[set_], v)
                d["revised"] = abs(d["outcome"] - d["outcome_at_settle"]) > 0.2
            h = hist.get(n, {})
            d["edit_spec_declared"] = h.get(f"inv_{v}_das_specificity")
            d["copy_block_declared"] = h.get(f"inv_{v}_copyblock_intended")
        net[n] = d
    dec = [d for d in net.values() if "outcome" in d]
    y = [d["outcome"] for d in dec]
    preds = ["rsi_escape", "rsi_step0", "escape", "early_task_loss", "early_grad_norm", "iia_escape"] + \
            [f"rsi_{s}" for s in range(100, 1001, 100)]
    pred_res = {}
    for p in preds:
        xs = [d[p] for d in dec]
        if any(x is None for x in xs):
            pred_res[p] = dict(note="undefined for some networks", n_missing=sum(x is None for x in xs))
            continue
        pred_res[p] = dict(spearman=spearman(xs, y), loo_threshold_acc=loo_threshold(xs, y))
    ed = [(d["edit_spec_declared"], d["outcome"], d["acc_ood"], d["copy_block_declared"]) for d in dec
          if d["edit_spec_declared"] is not None]
    gen_edit = dict(n=len(ed), spearman_outcome_vs_edit=spearman([e[0] for e in ed], [e[1] for e in ed]),
                    spearman_accood_vs_edit=spearman([e[0] for e in ed], [e[2] for e in ed]),
                    spearman_outcome_vs_copy=spearman([e[3] for e in ed], [e[1] for e in ed]))
    fitted = [d for d in dec if d["acc_id"] >= 0.95]
    summary = dict(
        n_networks=len(net), n_declared=len(dec),
        declared_fitted=len(fitted),
        generalisers=sorted(n for n, d in net.items() if d.get("outcome", 0) >= 0.9),
        failures=sorted(n for n, d in net.items() if "outcome" in d and d["outcome"] <= 0.2),
        revised_after_settle=sorted(n for n, d in net.items() if d.get("revised")),
        never_settled=sorted(n for n, d in net.items() if "outcome" in d and d["settle"] is None),
        outcome_bimodality=dict(le_0_2=sum(x <= 0.2 for x in y), mid=sum(0.2 < x < 0.9 for x in y), ge_0_9=sum(x >= 0.9 for x in y)),
        neither_ood_mean=statistics.mean(d["acc_ood"] for d in net.values() if d["group"] == "neither"),
    )
    res = dict(summary=summary, predictors=pred_res, generalisation_vs_editability=gen_edit, networks=net)
    OUT.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(sys.argv[1])
    print(json.dumps(r["summary"], indent=1))
    for k, v in r["predictors"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    print(r["generalisation_vs_editability"])
