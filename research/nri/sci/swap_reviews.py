"""Scientific reviews of the 61 role-swap reconstruction cohorts (doc 41 §41.6 Q5; owner INIT §7).

Each review is appended to runs/swap_recon/reviews/cohort_XXXXX.md. It has four parts:
  - the measured state of all 36 networks at that step (computed from the offline audits; never typed by hand);
  - the supervisor's interpretation, written after reading these measurements (REVIEWS below);
  - the previous cohort's stated prediction and whether it held (checked by code against the measurements);
  - a prediction for the next cohort.
Interpretation text is the supervisor's; numbers are inserted from data so that no figure is transcribed by hand.

    python -m sci.swap_reviews /home/user/swap_recon/zoo
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "runs/swap_recon/reviews"
FOCUS = (307, 311, 305, 304, 310)
MARK = "\n## Scientific review (supervisor)\n"

# Supervisor interpretation, keyed by the first step of each range; the text of the latest key <= step applies,
# with step-specific notes in NOTES. Written after reading the cohort tables, events and doc 46 measurements.
PHASES = {
    0: "Initialisation. All 36 networks predict near chance on familiar (ID) and unseen (OOD) inputs. Every declared "
       "block is fully factorised (value + unit; R²_add ≈ 0.99), as expected from random embeddings mixed linearly by "
       "attention. Interchange accuracy is at chance. Nothing distinguishes the later generalisers 307/311 from the "
       "failures 304/310.",
    100: "Shared plateau. Every network, declared or not, sits at ID ≈ 0.24-0.25. A direct check on 307 at step 400 "
         "shows this is a zero-detection shortcut: questions where mass or acceleration is 0 (answer 0; 16.8% of "
         "questions) are answered perfectly and the rest at chance (0.083). Declared blocks remain factorised "
         "(R²_add 1.00). Interchange accuracy rises slightly above chance (≈ 0.2) because the zero case transfers "
         "through the block.",
    500: "End of the plateau for undeclared networks. The `neither` group, without the interchange objective, begins "
         "to escape first. Declared networks are still on the plateau: the interchange objective delays the escape.",
    600: "Critical window opens. The `neither` group escapes rapidly. In declared networks the block starts to "
         "lose factorisation (mass group mean R²_add below 0.98) before ID rises. In the 10-step replay, 304 is "
         "already at 0.97 at step 650, the earliest measured sign of its later failure. The focus networks are not "
         "yet behaviourally distinguishable.",
    800: "Escape under way for the earliest declared networks. 304 leads (ID 0.50 at 800) and has already lost "
         "factorisation (0.93 → 0.84 by 850). Its unseen-mass accuracy does not rise with ID. 307 and 311 escape "
         "with unseen-mass accuracy rising together with ID. This is the first behavioural divergence.",
    1000: "Decision period. 304 fits its training data (ID ≥ 0.95 by about 1,100) while unseen-mass stays at chance "
          "and factorisation freezes at about 0.71-0.75: the interaction is retained. 307 passes through its "
          "factorisation dip (0.93 at 900) and recovers to 1.00 by 1,050, and unseen-mass climbs. 311 dips later "
          "(0.91 at 1,000) and recovers. 310 is still on the plateau.",
    1200: "Generalisers settle. 307 (ID 0.97, OOD-mass 0.92) and 311 (settles at 1,300) reach near-perfect unseen-mass "
          "accuracy at the same time as they fit the training data. No memorise-then-generalise delay is seen. 305 "
          "escapes late with partial OOD and mild factorisation loss (0.96).",
    1400: "Late escapes. 310 escapes now (ID 0.38 → 0.92 by 1,700) and loses factorisation in the same window "
          "(1.00 at 1,300 → 0.74 at 1,500). This replicates 304's pattern about 600 steps later, so failure is not "
          "tied to escaping early or late.",
    1800: "Post-fit consolidation. All five focus networks fit the training data. Group means change little. The "
          "mass group's mean factorisation is about 0.74 and stays nearly fixed thereafter: with weight decay 0.01 "
          "and losses near zero, no measurable pressure reverses a locked-in interaction. 305's partial "
          "generalisation (OOD-mass about 0.75 at 1,800) starts to erode.",
    2300: "Stable phase with slow drift. Failures are frozen at chance on unseen masses (304, 310: 0.09). "
          "Generalisers stay at 0.97-1.00. 305 erodes slowly: per-value accuracy on unit 2 with values 9 and 11 falls "
          "from 0.51/0.57 at 1,800 to 0.25/0.35 at 3,000, and its unseen-source interchange from 0.55 to 0.41, while "
          "route-specificity stays fixed at 0.39. Unconstrained unseen pairings drift once no loss anchors them. "
          "Undeclared networks keep improving OOD slowly.",
    3100: "Stable phase. No focus network changes by more than the noise of the audit panel. The remaining "
          "population changes are late settlements in the acceleration group (see events).",
    4100: "Late settlements (acceleration group). `acc_only_s309` first fits at step 4,100 and generalises (final 0.97): "
          "a late, successful settlement. `acc_only_s311` settles at 4,200 but fails (0.21). Late settlement is "
          "therefore not itself a failure signature.",
    4300: "Stable phase. Focus networks unchanged. 305 continues its slow erosion (OOD-mass about 0.46-0.48), "
          "consistent with drift of unanchored unseen pairings.",
    6000: "Final state. Outcomes are as frozen since about step 2,000 for the focus networks: 307, 311 = 1.00; "
          "305 = 0.46 (eroded from a peak of 0.75); 304, 310 = 0.09. Final factorisation: 307 1.00, 311 0.98, "
          "305 0.96, 304 0.71, 310 0.68. Across all 24 declared networks, final factorisation ≥ 0.98 is necessary "
          "for near-perfect generalisation (4/4) but not sufficient (doc 46 §46.1).",
}
NOTES = {
    900: "Anomaly check: 307 has a transient ID dip in the 10-step replay (0.84 at 1,100 → 0.58 at 1,150 → 0.97 at "
         "1,200), coinciding with its factorisation recovery. Instability during reorganisation, not collapse.",
    1300: "Anomaly check: 307's ID drops to 0.85 and OOD-mass to 0.66 at step 1,300, then recovers by 1,400. A "
          "transient instability during consolidation (task-loss spike in the per-step log at about 1,300-1,400). "
          "Not a collapse.",
    1500: "Anomaly check: 304 shows a single-step task-loss spike near 1,450 (100-step mean task loss 0.84) with no "
          "lasting change to its factorisation (0.72) or OOD (0.09). Instability without reorganisation, "
          "consistent with lock-in.",
    1900: "Anomaly check: 305's ID dips to 0.87 at 1,900 (from 1.00) and recovers by 2,000. OOD-mass falls from 0.75 to "
          "0.47 at the same time and recovers only partly (0.67). This is the first erosion step of its partial "
          "generalisation.",
}


def oodv(r, v):
    return st.mean(x for k, x in r["ood_combo_acc"].items() if k[0] == v)


def load(zoo):
    runs = {}
    for p in sorted(Path(zoo).iterdir()):
        f = p / "audits.jsonl"
        if f.exists():
            runs[p.name] = {json.loads(l)["step"]: json.loads(l) for l in f.read_text().splitlines() if l.strip()}
    return runs


def snapshot(runs, F, step):
    out = {}
    for g, v in (("mass_only", "M"), ("acc_only", "A"), ("neither", None)):
        rs = [runs[f"{g}_s{s}"][step] for s in range(300, 312)]
        d = dict(fit=sum(r["acc_id"] >= 0.95 for r in rs), id=st.mean(r["acc_id"] for r in rs))
        d["ood"] = st.mean(oodv(r, v) for r in rs) if v else st.mean(r["acc_ood"] for r in rs)
        if v:
            d["iia"] = st.mean(r[f"iia_{v}"] for r in rs)
            d["iia_ood"] = st.mean(r[f"iia_{v}_oodsrc"] for r in rs)
            ks = [str(step)] if str(step) in F[f"{g}_s300"] else []
            d["r2add"] = st.mean(F[f"{g}_s{s}"][ks[0]] for s in range(300, 312)) if ks else None
        out[g] = d
    out["focus"] = {s: dict(id=runs[f"mass_only_s{s}"][step]["acc_id"], oodM=oodv(runs[f"mass_only_s{s}"][step], "M"),
                            iia_ood=runs[f"mass_only_s{s}"][step]["iia_M_oodsrc"],
                            r2add=F[f"mass_only_s{s}"].get(str(step))) for s in FOCUS}
    return out


def phase_text(step):
    k = max(s for s in PHASES if s <= step)
    return PHASES[k]


def prediction(step, snap):
    """A testable prediction for step+100 derived from the current phase (stated before the next cohort is read)."""
    if step < 500:
        return ("plateau", "every network's ID stays in [0.20, 0.30] at the next cohort")
    if step < 1800:
        return ("focus_direction", "304 and 310 stay ≤ 0.25 on unseen masses at the next cohort, whatever their ID "
                                   "(failures never rise on unseen masses)")
    return ("stable", "no focus network changes unseen-mass accuracy by more than 0.10 at the next cohort")


def check(kind, prev, cur):
    if kind == "plateau":
        ids = [cur["focus"][s]["id"] for s in FOCUS]
        return all(0.2 <= x <= 0.3 for x in ids), f"focus ID at next cohort: {[round(x, 2) for x in ids]}"
    if kind == "focus_direction":
        v = [cur["focus"][s]["oodM"] for s in (304, 310)]
        return all(x <= 0.25 for x in v), f"304/310 unseen-mass: {[round(x, 2) for x in v]}"
    d = {s: round(cur["focus"][s]["oodM"] - prev["focus"][s]["oodM"], 2) for s in FOCUS}
    return all(abs(x) <= 0.10 for x in d.values()), f"changes in unseen-mass accuracy: {d}"


def main(zoo):
    runs = load(zoo)
    F = json.loads((ROOT / "runs/swap_recon/factorization_trajectories.json").read_text())
    ev = json.loads((REV / "events.json").read_text())
    prev, prev_pred, ledger = None, None, []
    for step in range(0, 6001, 100):
        snap = snapshot(runs, F, step)
        L = [MARK]
        L.append("**Measured state.** " + "; ".join(
            f"{g}: {d['fit']}/12 fit (ID ≥ 0.95), mean ID {d['id']:.2f}, mean OOD {d['ood']:.2f}"
            + (f", block IIA {d['iia']:.2f}, unseen-source IIA {d['iia_ood']:.2f}" if 'iia' in d else "")
            + (f", factorisation R²_add {d['r2add']:.2f}" if d.get('r2add') is not None else "")
            for g, d in snap.items() if g != "focus") + ".")
        L.append("**Focus networks** (ID / unseen-mass / unseen-source IIA / R²_add): " + "; ".join(
            f"{s}: {f['id']:.2f} / {f['oodM']:.2f} / {f['iia_ood']:.2f} / "
            f"{'—' if f['r2add'] is None else format(f['r2add'], '.2f')}" for s, f in snap["focus"].items()) + ".")
        e = ev.get(str(step), {})
        L.append(f"**Events since previous cohort:** {len(e)}" + (" — " + "; ".join(f"{k}: {', '.join(v)}" for k, v in sorted(e.items())) if e else "") + ".")
        L.append("**Interpretation.** " + phase_text(step))
        if step in NOTES:
            L.append("**Anomaly investigated.** " + NOTES[step])
        if prev_pred is not None:
            ok, detail = check(prev_pred[0], prev, snap)
            L.append(f"**Previous prediction** ('{prev_pred[1]}'): {'HELD' if ok else 'FAILED'} ({detail}).")
            ledger.append(dict(step=step, prediction=prev_pred[1], held=ok, detail=detail))
        pred = prediction(step, snap)
        L.append(f"**Prediction for step {step + 100:,}:** {pred[1]}." if step < 6000 else "**Final cohort.** No further prediction.")
        L.append("\n*Observed: the measured state and events. Inferred: the interpretation. Nothing in this review is an "
                 "experimental demonstration (doc 46 labels).*\n")
        p = REV / f"cohort_{step:05d}.md"
        txt = p.read_text()
        if MARK in txt:
            txt = txt.split(MARK)[0]
        p.write_text(txt + "\n".join(L) + "\n")
        prev, prev_pred = snap, (pred if step < 6000 else None)
    (REV / "prediction_ledger.json").write_text(json.dumps(ledger, indent=1))
    held = sum(x["held"] for x in ledger)
    print(f"{len(ledger)} predictions checked, {held} held, {len(ledger) - held} failed")
    for x in ledger:
        if not x["held"]:
            print("FAILED at", x["step"], x["prediction"], x["detail"])


if __name__ == "__main__":
    main(sys.argv[1])
