"""Summarise Birth-of-a-Concept runs: per-run onsets and per-population final scorecards.

Onsets (PILOT-PROVISIONAL definitions; the registered experiment freezes its own): the first audited step at
which the criterion holds at two consecutive audits.
  practiced     practiced accuracy >= 0.95
  decodable     canonical quantity decodable at x1 on held-out combinations (kNN) >= 0.50
  causal        number patch at x1 into HELD-OUT bases follows the donor under the base route >= 0.50
  compositional held-out accuracy >= best constant + 0.20

    python -m sci.concept_report <runs_dir>
"""
import json
import statistics
import sys
from pathlib import Path

CRIT = {
    "practiced": lambda c: c["practiced"] >= 0.95,
    "decodable": lambda c: c["decode_canon_x1_knn"] >= 0.50,
    "causal": lambda c: c["patch_x1_heldout_base"]["intended"] >= 0.50,
    "compositional": lambda c: c["heldout"] >= c["heldout_best_constant"] + 0.20,
}


def onset(cards, f):
    for a, b in zip(cards, cards[1:]):
        if f(a) and f(b):
            return a["step"]
    return None


def load(d):
    return json.loads((d / "meta.json").read_text()), [json.loads(x) for x in open(d / "scorecard.jsonl")]


def summarise(root):
    rows = []
    for d in sorted(p for p in Path(root).iterdir() if (p / "scorecard.jsonl").exists()):
        meta, cards = load(d)
        last = cards[-1]
        rows.append(dict(run=d.name, pop=meta["population"], wd=meta["wd"], seed=meta["seed"], step=last["step"],
                         practiced=last["practiced"], paraphrase=last["practiced_paraphrase"], heldout=last["heldout"],
                         newframe=last["heldout_newframe"], const=last["heldout_best_constant"],
                         patch_tr=last["patch_x1_trained"]["intended"], patch_ho=last["patch_x1_heldout_base"]["intended"],
                         patch_x2=last["patch_x2_trained"]["intended"], dec_knn=last["decode_canon_x1_knn"],
                         dec_lin=last["decode_canon_x1_linear"], dlog=last["circle_x1"]["dlog"],
                         dlog_sh=last["circle_x1"]["shuffled_mean"], best_heldout=max(c["heldout"] for c in cards),
                         **{"on_" + k: onset(cards, f) for k, f in CRIT.items()}))
    return rows


def main(root):
    rows = summarise(root)
    keys = ["practiced", "paraphrase", "heldout", "best_heldout", "newframe", "patch_tr", "patch_ho", "patch_x2", "dec_knn", "dlog"]
    print(f"{'wd':>4} {'pop':3}  " + " ".join(f"{k:>12}" for k in keys) + "   onsets(prac/dec/causal/comp) per seed")
    groups = {}
    for r in rows:
        groups.setdefault((r["wd"], r["pop"]), []).append(r)
    for (wd, pop), rs in sorted(groups.items()):
        def ms(k):
            v = [r[k] for r in rs]
            return f"{statistics.mean(v):.2f}±{statistics.pstdev(v):.2f}"
        ons = " ".join(f"[{r['on_practiced']}/{r['on_decodable']}/{r['on_causal']}/{r['on_compositional']}]" for r in rs)
        print(f"{wd:>4} {pop:3}  " + " ".join(f"{ms(k):>12}" for k in keys) + f"   {ons}  (n={len(rs)}, const {rs[0]['const']:.2f}, step {rs[0]['step']})")
    return rows


if __name__ == "__main__":
    main(sys.argv[1])
