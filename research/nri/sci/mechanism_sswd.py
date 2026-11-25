"""Phase II.2 (doc 41 §41.4): why does semantic supervision help only under strong weight decay? H-A .. H-E.
Retrospective, exploratory. Seed is the unit; exact paired sign-flip; Holm across the five primary contrasts.

Operationalisations (fixed here, before the analysis is run):
  H-A  participation ratio PR = (sum l)^2 / sum l^2 of the covariance eigenvalues of the 120 trained (route, value)
       class means of the x1 residual at the source-number token (canonical rendering, frame C01). Step 6,000.
       Prediction C1.0 < C0.1  (contrast d = C0.1 - C1.0 > 0).
  H-B  route-specificity index RSI = 1 - R^2 of a least-squares fit of those 120 class means on one-hot canonical q.
       Prediction C1.0 < C0.1 at 6,000 (d = C0.1 - C1.0 > 0); trajectory every 500 steps reported.
  H-C  mean task-aux gradient cosine at x1m, source token (audits traces.x1m.task_aux_cos_src), audits 0..2,000.
       Prediction C1.0 > C0.1 (d = C1.0 - C0.1 > 0).
  H-D  I1 canonical SWITCH rate in the stratum trained base | canonical-implied held-out at the first 100-step
       audit with q kNN decodability (decode_x1m_q_knn) >= 0.9 (two consecutive audits, like the registered onsets).
       Prediction C1.0 > C0.1. Networks never reaching decodability are scored at step 6,000 and flagged.
  H-E  total parameter norm sqrt(sum_g ||W_g||^2) at 6,000 (audits weight_norm). Prediction C1.0 < C0.1
       (d = C0.1 - C1.0 > 0). Knee = step of the maximum total norm; descriptive: knee precedes the registered
       compositional onset in >= 9/12 C1.0 networks.

    python -m sci.mechanism_sswd     # writes runs/instruments/mechanism_sswd.json
"""
from __future__ import annotations

import gzip
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import causal_instruments as CI  # noqa: E402
from sci import concept as C  # noqa: E402
from sci import factorial as Fx  # noqa: E402
from sci import factorial_analysis as FA  # noqa: E402

OUT = CI.OUT_DIR / "mechanism_sswd.json"
STRAT = "base_trained|can_implied_heldout"


def audits(run):
    with gzip.open(Fx.REPO_RUNS / run / "audits.jsonl.gz", "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


@torch.no_grad()
def geometry(run, step, split):
    ad = CI.load_student(run, step)
    combos = list(split["train"])
    enc = ad.encode(combos)
    x = enc["states"]["x1"][torch.arange(len(combos)), enc["pos"]["src"]].double()      # 120 x 64 (one item each)
    xc = x - x.mean(0)
    lam = torch.linalg.eigvalsh(xc.T @ xc / len(x)).clamp(min=0)
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    q = torch.tensor([C.canonical(r[0], v, r[1]) for r, v in combos])
    Q = torch.nn.functional.one_hot(q, C.P).double()
    beta = torch.linalg.lstsq(Q, xc).solution
    r2 = 1 - float(((xc - Q @ beta) ** 2).sum() / (xc ** 2).sum())
    return dict(pr=pr, rsi=1 - r2)


def onset_step(a):
    st = [x["step"] for x in a]
    comp = [x["behaviour"]["heldout"] >= x["behaviour"]["heldout_best_constant"] + 0.2 for x in a]
    return next((st[i] for i in range(len(a) - 1) if comp[i] and comp[i + 1]), None)


def per_network(run, split):
    a = audits(run)
    inst = json.loads((CI.OUT_DIR / "factorial" / f"{run}.json").read_text())
    geo = {s: geometry(run, s, split) for s in range(0, 6001, 500)}
    cos = [x["traces"]["x1m"].get("task_aux_cos_src") for x in a if x["step"] <= 2000]   # absent for A, B (no aux head)
    cos = [c for c in cos if c is not None]
    dec = [x["representation"]["decode_x1m_q_knn"] >= 0.9 for x in a]
    steps = [x["step"] for x in a]
    i_dec = next((i for i in range(len(a) - 1) if dec[i] and dec[i + 1]), None)
    s_dec = steps[i_dec] if i_dec is not None else 6000
    sw = inst["I1"][str(s_dec)][STRAT]["canonical_switch"]
    norms = [math.sqrt(sum(v ** 2 for v in x["weight_norm"].values() if v is not None)) for x in a]
    knee = steps[max(range(len(norms)), key=lambda i: norms[i])]
    return dict(PR=geo[6000]["pr"], RSI=geo[6000]["rsi"], geometry_traj=geo,
                cos_0_2000=statistics.mean(cos) if cos else None,
                decodable_step=s_dec if i_dec is not None else None, switch_at_decodable=sw,
                norm_6000=norms[-1], norm_knee=knee, comp_onset=onset_step(a),
                dev_heldout=a[-1]["behaviour"]["heldout"],
                i1_traj={s: inst["I1"][s][STRAT]["canonical_switch"] for s in inst["I1"]})


def spearman(x, y):
    rk = lambda v: [sorted(v).index(t) + (sorted(v).count(t) - 1) / 2 for t in v]
    rx, ry = rk(x), rk(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def main():
    torch.set_num_threads(2)
    split = C.make_split()
    conds = {"C1.0": ("C", 1.0), "C0.1": ("C", 0.1), "D1.0": ("D", 1.0), "B1.0": ("B", 1.0), "A1.0": ("A", 1.0),
             "D0.1": ("D", 0.1), "B0.1": ("B", 0.1), "A0.1": ("A", 0.1)}
    net = {k: {s: per_network(Fx.run_name(p, w, s), split) for s in Fx.SEEDS} for k, (p, w) in conds.items()}
    # primary: one contrast per hypothesis, direction as predicted, C1.0 vs C0.1
    spec = {"H-A": ("PR", -1), "H-B": ("RSI", -1), "H-C": ("cos_0_2000", +1), "H-D": ("switch_at_decodable", +1),
            "H-E": ("norm_6000", -1)}
    prim = {}
    for h, (m, sign) in spec.items():
        d = [sign * (net["C1.0"][s][m] - net["C0.1"][s][m]) for s in Fx.SEEDS]
        prim[h] = dict(measure=m, d_mean=statistics.mean(d), ci95=FA.bootstrap_ci(d), p=FA.sign_flip_p(d),
                       seeds_positive=sum(x > 0 for x in d),
                       means={k: statistics.mean(net[k][s][m] for s in Fx.SEEDS) for k in conds if
                              all(net[k][s][m] is not None for s in Fx.SEEDS)})
    for h, ph in zip(prim, FA.holm([prim[h]["p"] for h in prim])):
        prim[h]["p_holm"] = ph
    sec = {}
    for h, (m, sign) in spec.items():
        for other in ("D1.0", "B1.0"):
            if all(net[other][s][m] is not None for s in Fx.SEEDS):
                d = [sign * (net["C1.0"][s][m] - net[other][s][m]) for s in Fx.SEEDS]
                sec[f"{h}_vs_{other}"] = dict(d_mean=statistics.mean(d), ci95=FA.bootstrap_ci(d), p=FA.sign_flip_p(d))
    within = {h: spearman([net["C1.0"][s][m] for s in Fx.SEEDS], [net["C1.0"][s]["dev_heldout"] for s in Fx.SEEDS])
              for h, (m, _) in spec.items()}
    knee_before = sum(1 for s in Fx.SEEDS if net["C1.0"][s]["comp_onset"] is not None
                      and net["C1.0"][s]["norm_knee"] < net["C1.0"][s]["comp_onset"])
    res = dict(primary=prim, secondary=sec, within_C1_spearman_with_dev_heldout=within,
               HE_knee_before_onset=knee_before, networks=net)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    r = main()
    for h, v in r["primary"].items():
        print(h, v["measure"], "d=%.4f" % v["d_mean"], [round(x, 4) for x in v["ci95"]], "p=%.5f holm=%.5f" % (v["p"], v["p_holm"]),
              f"{v['seeds_positive']}/12", {k: round(x, 3) for k, x in v["means"].items()})
    for k, v in r["secondary"].items():
        print("  ", k, round(v["d_mean"], 4), round(v["p"], 4))
    print("within C1.0 spearman:", {k: None if v is None else round(v, 2) for k, v in r["within_C1_spearman_with_dev_heldout"].items()})
    print("H-E knee before onset:", r["HE_knee_before_onset"], "/ 12")
