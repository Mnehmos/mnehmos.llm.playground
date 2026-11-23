"""Review + advance for a stationary checkpoint: builds the review from the CURRENT records (actual
deltas) plus the supervisor's checkpoint-specific note; refuses if the packet shows a change beyond
the stationary bands (then the supervisor must write a full review instead)."""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
note, key_q, level = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1
rs = [json.loads(l) for l in open(ROOT / "runs/zoo_lineage/B_supervised/records.jsonl")][-2:]
a, b = rs[0]["observe"], rs[1]["observe"]
s = rs[1]["step"]
f = lambda o: dict(acc=o["acc_obs"], ood=o["acc_ood_dev"], iia=o["causal"]["iia_M_block"], edit=o["causal"]["edit_block_intended"],
                   inter=o["geometry"]["block"]["M_interaction_share"], A_L1=o["causal"]["iia_A_L1_val+unit"])
x, y = f(a), f(b)
d = {k: round(y[k] - x[k], 3) for k in x}
band = dict(acc=0.01, ood=0.08, iia=0.02, edit=0.04, inter=0.03, A_L1=0.05)
out = [k for k in d if abs(d[k]) > band[k]]
if out:
    sys.exit(f"step {s}: change beyond stationary band in {out} {d}: write a full review")
w = rs[1]["weights"]
top = max(w.items(), key=lambda kv: kv[1]["rel_from_prev"])
mt = b["features"]["mlp"].get("match_status", {})
rev = dict(
    what_changed=f"Stationary at step {s}: " + ", ".join(f"{k} {y[k]:.3f} ({d[k]:+.3f})" for k in y) +
                 f"; largest update {top[0]} {top[1]['rel_from_prev']:.3f}; MLP match {mt}.",
    hypothesis=note,
    evidence="Routine Level-1 audit (OBS interchange, persistent pooled edit, geometry, weight lineage); all deltas inside the stationary bands "
             "(acc 0.01, OOD_DEV 0.08 (~2 sd of the converged noise), IIA 0.02, editability 0.04, interaction 0.03, acceleration L1 0.05).",
    key_question=key_q,
    best_experiment="Continue routine recording; scheduled Level-2 checks at 4,000 and 5,000 (OOD locus, head anatomy).",
    intervene="no",
    intervention_rationale="Stationary network; editability already high; registered primitives shown (1,500 / 1,800 dry runs) to cost held-out accuracy.",
    previous_interventions="none",
    investigation_level=level)
p = ROOT / "runs/zoo_lineage/reviews" / f"review_{s:04d}.json"
p.write_text(json.dumps(rev, indent=1))
print(subprocess.run([sys.executable, "run_lineage.py", "review", str(p)], cwd=ROOT, capture_output=True, text=True).stdout.strip())
