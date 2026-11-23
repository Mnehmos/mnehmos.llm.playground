"""Report for the Adaptive Concept Lineage pilot (spec v1.0; registration docs/28; results docs/29).

Reads runs/zoo_lineage/ (records, journal, proposals, inspections, trackers) and
results/lineage/final_sealed.json; writes results/lineage/:
  summary.md, journal.jsonl (copy), concept_records_{A,B}.json, lineage_events_{A,B}.jsonl,
  lineage_event_audit.json (post hoc artifact filters, disclosed), history_long.csv, ledger.json,
  fig_lineage_atlas.png (concept formation atlas), fig_lineage_graph.png (feature lineage graph),
  fig_lineage_weights.png (weight development timeline), fig_lineage_circuit.png (causal circuit atlas),
  fig_lineage_interventions.png (intervention / review history), plus the doc-24 atlas figures.
"""
from __future__ import annotations

import csv
import json
import pickle
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import report_atlas as RA  # noqa: E402

ROOT = Path(__file__).parent
ZOO = ROOT / "runs" / "zoo_lineage"
OUT = ROOT / "results" / "lineage"
NETS = ("A_control", "B_supervised")
LABEL = {"A_control": "control", "B_supervised": "supervised"}
INK2, GRID = "#52514e", "#e4e3df"
STATUS_COL = {"hypothesized": "#d7d6d0", "correlationally supported": "#8fb8ea",
              "causally supported (partial)": "#2a78d6", "causally supported": "#0b3d91"}


def recs(n):
    return [json.loads(l) for l in open(ZOO / n / "records.jsonl")]


def tracker(n):
    return pickle.loads((ZOO / n / "tracker.pkl").read_bytes())


# ------------------------------------------------------------------ post hoc artifact audit (disclosed in docs/29)
def audit_events(tr):
    """Classify lineage events: head relocations between near-identical heads (ambiguous matches);
    status changes that revert within two checkpoints (threshold jitter); cluster events in checkpoints
    where the cluster's membership changed by > 50% (k-means repartition)."""
    out = dict(total=0, by_event={}, ambiguous_head_relocation=0, status_jitter=0, cluster_repartition=0, retained=0)
    recs_ = tr.records
    for e in tr.events:
        if e["event"] == "persistence":
            continue
        out["total"] += 1
        out["by_event"][e["event"]] = out["by_event"].get(e["event"], 0) + 1
        fid = e.get("id")
        r = recs_.get(fid) if fid else None
        flag = None
        if e["event"] == "relocation" and r and r["kind"] == "head":
            ev = {s: x for s, x in r["evidence_history"]}.get(e["step"], {})
            # a head whose attention profile is diffuse (max share < 0.4) is not identifiable by its role
            att = ev.get("attends", {})
            if not att or max(att.values()) < 0.4:
                flag = "ambiguous_head_relocation"
        if flag is None and e["event"] in ("consolidation", "weakening", "disappearance", "emergence") and r:
            hist = r["status_history"]
            idx = [i for i, (s, _) in enumerate(hist) if s == e["step"]]
            if idx:
                i = idx[0]
                before = hist[i - 1][1] if i > 0 else None
                later = [x for _, x in hist[i + 1:i + 3]]
                if before is not None and before in later:
                    flag = "status_jitter"
        if flag is None and r and r["kind"] == "neuron_cluster" and e["event"] in ("splitting", "merging", "consolidation", "weakening", "specialization"):
            sizes = {s: x.get("size") for s, x in r["evidence_history"]}
            steps = sorted(sizes)
            if e["step"] in sizes and steps.index(e["step"]) > 0:
                prev = sizes[steps[steps.index(e["step"]) - 1]]
                if prev and abs(sizes[e["step"]] - prev) / prev > 0.5:
                    flag = "cluster_repartition"
        if e["event"] in ("splitting", "merging") and flag is None:
            parent = recs_.get(e.get("parent"))
            if parent and parent["kind"] == "neuron_cluster":
                flag = "cluster_repartition"          # fixed-k k-means: splits/merges are partition-dependent
        if flag:
            out[flag] += 1
        else:
            out["retained"] += 1
    return out


# ------------------------------------------------------------------ figures
def fig_lineage_graph(tr, path, title):
    rows = [r for r in tr.records.values() if any(s.startswith("causally") for _, s in r["status_history"])
            or (r["kind"] == "head" and r["first_observed"] == 0)]
    rows.sort(key=lambda r: (r["kind"], r["first_observed"]))
    fig, ax = plt.subplots(figsize=(13, 0.28 * len(rows) + 1.5))
    for y, r in enumerate(rows):
        hist = r["status_history"]
        for (s, stt), nxt in zip(hist, hist[1:] + [(hist[-1][0] + 100, None)]):
            ax.plot([s, nxt[0]], [y, y], color=STATUS_COL.get(stt, "#cccccc"), lw=5, solid_capstyle="butt")
        sites = r["site_history"]
        for (s0, a), (s1, b) in zip(sites, sites[1:]):
            if a != b:
                ax.plot([s1], [y], marker="|", color="#e34948", ms=8)
        ax.text(-60, y, f"{r['id']} {sites[-1][1]}", ha="right", va="center", fontsize=6, color=INK2)
    for e in tr.events:
        if e["event"] in ("splitting", "merging") and e.get("parent") in [r["id"] for r in rows]:
            y = [r["id"] for r in rows].index(e["parent"])
            ax.plot([e["step"]], [y], marker="v" if e["event"] == "splitting" else "^", color="#eda100", ms=6)
    ax.set_yticks([])
    ax.set_xlim(-50, 6100)
    ax.set_xlabel("training step", fontsize=8, color=INK2)
    for s, c in STATUS_COL.items():
        ax.plot([], [], color=c, lw=5, label=s)
    ax.plot([], [], marker="|", color="#e34948", ls="", label="site change (relocation)")
    ax.plot([], [], marker="v", color="#eda100", ls="", label="split / merge (partition-dependent)")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    ax.set_title(title, fontsize=9, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_circuit(rb, path):
    """Causal circuit atlas at the final checkpoint: nodes = token inputs, layer-1 value token, attn2
    heads, the mass block, MLP clusters, output; edge width = causal evidence (interchange / ablation);
    solid = verified (>= 0.9 or >= 0.3 nats), dashed = partial (>= 0.5 / >= 0.05), dotted = decodable only."""
    o = rb[-1]["observe"]
    cz = o["causal"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    nodes = {"mass value\ntoken": (0, 3), "mass unit\ntoken": (0, 2), "acc value\ntoken": (0, 1), "acc unit\ntoken": (0, 0),
             "L1 value-token\nresidual": (1.6, 2.5), "attn2 head 0": (3.2, 3.2), "attn2 head 3": (3.2, 2.3),
             "attn2 heads 1, 2\n(acc, out unit)": (3.2, 0.8), "mass block\nh[0:16]": (4.8, 2.7), "h dims 16-63": (4.8, 1.0),
             "MLP (mass-family\nneurons)": (6.4, 2.0), "answer F": (8.0, 2.0)}
    for k, (x, y) in nodes.items():
        ax.text(x, y, k, ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="#f4f3ef", ec="#b5b4ae"))

    def edge(a, b, val, kind):
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        ls = {"verified": "-", "partial": "--", "decodable": ":"}[kind]
        ax.annotate("", xy=(x1 - 0.45, y1), xytext=(x0 + 0.45, y0),
                    arrowprops=dict(arrowstyle="->", lw=0.8 + 3 * min(val, 1), ls=ls, color="#2a78d6" if kind != "decodable" else "#8a8984"))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, f"{val:.2f}", fontsize=6.5, color=INK2, ha="center")

    def kind(v):
        return "verified" if v >= 0.9 else "partial" if v >= 0.5 else "decodable"
    edge("mass value\ntoken", "L1 value-token\nresidual", o["decode_kappa"]["M@L1@mval"], "decodable")
    edge("mass unit\ntoken", "L1 value-token\nresidual", o["decode_kappa"]["mu@L1@mval"], "decodable")
    edge("L1 value-token\nresidual", "attn2 head 0", cz["iia_M_L1_val+unit"], kind(cz["iia_M_L1_val+unit"]))
    edge("L1 value-token\nresidual", "attn2 head 3", cz["iia_M_L1_val+unit"], kind(cz["iia_M_L1_val+unit"]))
    edge("attn2 head 0", "mass block\nh[0:16]", cz["iia_M_head0"], kind(cz["iia_M_head0"]))
    edge("attn2 head 3", "mass block\nh[0:16]", cz["iia_M_head3"], kind(cz["iia_M_head3"]))
    edge("acc value\ntoken", "attn2 heads 1, 2\n(acc, out unit)", cz["iia_A_L1_val+unit"], kind(cz["iia_A_L1_val+unit"]))
    edge("attn2 heads 1, 2\n(acc, out unit)", "h dims 16-63", cz["iia_A_rest"], kind(cz["iia_A_rest"]))
    edge("mass block\nh[0:16]", "MLP (mass-family\nneurons)", cz["iia_M_block"], kind(cz["iia_M_block"]))
    edge("h dims 16-63", "MLP (mass-family\nneurons)", cz["iia_A_rest"], kind(cz["iia_A_rest"]))
    edge("MLP (mass-family\nneurons)", "answer F", o["acc_obs"], "verified")
    ax.set_xlim(-0.7, 8.7)
    ax.set_ylim(-0.6, 3.8)
    ax.axis("off")
    ax.set_title("Causal circuit atlas, step 6,000 (OBS). Edge labels: interchange accuracy (solid >= 0.9 verified, dashed >= 0.5 partial) "
                 "or probe kappa (dotted, decodable only).\nSingle-head interchange is low because mass is written jointly by heads 0 and 3.",
                 fontsize=8, loc="left")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_interventions(rb, journal, path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax = axes[0]
    steps = [r["step"] for r in rb]
    for key, lab, col in (("acc_obs", "ID accuracy (OBS)", "#2a78d6"), ("acc_ood_dev", "held-out (OOD_DEV)", "#eb6834")):
        ax.plot(steps, [r["observe"][key] for r in rb], color=col, lw=1.6, label=lab)
    ax.plot(steps, [r["observe"]["causal"]["edit_block_intended"] for r in rb], color="#1baf7a", lw=1.6, label="persistent pooled edit (inversion)")
    ax.plot(steps, [r["observe"]["causal"]["iia_M_block"] for r in rb], color="#6250d6", lw=1.2, ls="--", label="block IIA(M)")
    RA.style(ax, "Supervised network (identical to control: no edit installed). Grey markers: dry runs (never installed).")
    ax.legend(fontsize=7, frameon=False, ncol=4)
    for j in journal:
        if any("dryrun" in str(i.get("tool")) for i in j.get("inspections", [])):
            ax.axvline(j["step"], color="#b5b4ae", lw=1)
    ax = axes[1]
    lv = [j["investigation_level"] for j in journal]
    ax.bar([j["step"] for j in journal], lv, width=70, color=["#d7d6d0" if v == 1 else "#8fb8ea" if v == 2 else "#0b3d91" for v in lv])
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["1 routine", "2 targeted", "3 intervention"], fontsize=7)
    RA.style(ax, "Supervisor investigation level per checkpoint (60 reviews)")
    ax.set_xlabel("training step", fontsize=8, color=INK2)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ report
def report():
    OUT.mkdir(parents=True, exist_ok=True)
    R = {n: recs(n) for n in NETS}
    TR = {n: tracker(n) for n in NETS}
    journal = [json.loads(l) for l in open(ZOO / "journal.jsonl")]
    meta = json.loads((ZOO / "meta.json").read_text())
    insp = [json.loads(l) for l in open(ZOO / "inspections.jsonl")] if (ZOO / "inspections.jsonl").exists() else []
    props = [json.loads(l) for l in open(ZOO / "proposals.jsonl")] if (ZOO / "proposals.jsonl").exists() else []
    # reuse the doc-24 atlas figures (A -> 'control', B -> 'supervised')
    RA.OUT = OUT
    C = {"control": (R["A_control"], [], None), "supervised": (R["B_supervised"], [], None)}
    for fn in (RA.fig_decode, RA.fig_causal, RA.fig_geometry, RA.fig_features, RA.fig_weights):
        fn(C)
    for old, new in (("fig_atlas_decode.png", "fig_lineage_atlas_decode.png"), ("fig_atlas_causal.png", "fig_lineage_atlas_causal.png"),
                     ("fig_atlas_geometry.png", "fig_lineage_geometry.png"), ("fig_atlas_features.png", "fig_lineage_features.png"),
                     ("fig_atlas_weights.png", "fig_lineage_weights.png")):
        (OUT / old).replace(OUT / new)
    fig_lineage_graph(TR["B_supervised"], OUT / "fig_lineage_graph.png",
                      "Feature lineage graph (supervised = control): features that were ever causally supported, plus the 8 heads")
    fig_circuit(R["B_supervised"], OUT / "fig_lineage_circuit.png")
    fig_interventions(R["B_supervised"], journal, OUT / "fig_lineage_interventions.png")
    # machine-readable outputs
    for n in NETS:
        (OUT / f"concept_records_{n}.json").write_text(json.dumps(TR[n].concept_records(), indent=1, default=str))
        with open(OUT / f"lineage_events_{n}.jsonl", "w") as f:
            for e in TR[n].events:
                f.write(json.dumps(e, default=str) + "\n")
    audit = {n: audit_events(TR[n]) for n in NETS}
    (OUT / "lineage_event_audit.json").write_text(json.dumps(audit, indent=1))
    with open(OUT / "journal.jsonl", "w") as f:
        for j in journal:
            f.write(json.dumps(j) + "\n")
    with open(OUT / "history_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network", "step", "key", "value"])
        for n in NETS:
            for r in R[n]:
                for k, v in RA.flatten({kk: vv for kk, vv in r.items() if kk not in ("lineage_events",)}):
                    w.writerow([n, r["step"], k, v])
    # ledger
    cost = {n: {} for n in NETS}
    for n in NETS:
        for r in R[n]:
            for k, v in r["cost"].items():
                cost[n][k] = round(cost[n].get(k, 0) + v, 1)
    led = dict(base_training_cpu_s_both=round(meta["ledger"]["train_cpu_s"], 1), observation_cpu_s=cost,
               supervisor_inspect_cpu_s=round(meta["ledger"]["inspect_cpu_s"], 1),
               supervisor_proposal_cpu_s=round(meta["ledger"]["proposal_cpu_s"], 1),
               supervisor_extra_grad_steps=meta["ledger"]["supervisor_extra_grad_steps"],
               dry_run_extra_grad_steps=sum(i.get("result", {}).get("info", {}).get("extra_grad_steps", 0) for i in insp
                                            if i.get("tool") == "dryrun"),
               inspections=len(insp), proposals_submitted=len(props), installed=len(meta["installed"]),
               storage_bytes=sum(p.stat().st_size for p in ZOO.rglob("*") if p.is_file()))
    fs = OUT / "final_sealed.json"
    if fs.exists():
        led["sealed_eval_cpu_s"] = json.loads(fs.read_text())["cpu_s"]
    (OUT / "ledger.json").write_text(json.dumps(led, indent=1))
    # summary
    ra, rb = R["A_control"], R["B_supervised"]
    L = ["# Adaptive Concept Lineage pilot (auto-generated by `run_lineage.py report`; registration docs/28)", "",
         "## Integrity", "",
         f"- checkpoints: A {len(ra)}, B {len(rb)}; reviews filed: {len(journal)} (steps {journal[0]['step']}-{journal[-1]['step']})",
         f"- identical state fingerprints at every checkpoint: **{all(a['state_fingerprint'] == b['state_fingerprint'] for a, b in zip(ra, rb))}**; "
         f"identical batch digests: **{all(a['batch_digest'] == b['batch_digest'] for a, b in zip(ra, rb))}**",
         f"- proposals submitted to the gate: {len(props)}; installed: {len(meta['installed'])}; read-only inspections: {len(insp)} "
         f"(dry runs: {sum(1 for i in insp if i.get('tool') == 'dryrun')})",
         f"- investigation levels: " + ", ".join(f"L{k}: {sum(1 for j in journal if j['investigation_level'] == k)}" for k in (1, 2, 3)), ""]
    ev = RA.AT.registered_events([dict(step=r["step"], observe=r["observe"]) for r in rb])
    L += ["## Registered timings (doc-24 criteria; A = B)", "", "| event | step |", "|---|---|"]
    L += [f"| {k} | {RA.fmt(v)} |" for k, v in ev.items()]
    L += ["", "## Trajectory (B = A; every 100 steps to 2,000, then every 500)", "",
          "| step | ID | OOD_DEV | block IIA | L1 tokens IIA(M) | pooled edit | copy | interaction | mult-char (h) | M-family neurons | stable identities | review level |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lvl = {j["step"]: j["investigation_level"] for j in journal}
    for r in rb:
        s = r["step"]
        if s > 2000 and s % 500:
            continue
        o = r["observe"]
        L.append(f"| {s} | {o['acc_obs']:.3f} | {o['acc_ood_dev']:.3f} | {o['causal']['iia_M_block']:.3f} | {o['causal']['iia_M_L1_val+unit']:.3f} | "
                 f"{o['causal']['edit_block_intended']:.3f} | {o['causal']['copy_block_intended']:.3f} | {o['geometry']['block']['M_interaction_share']:.3f} | "
                 f"{o['geometry']['h']['M_char']['mult']:.3f} | {o['features']['mlp']['counts'].get('M', 0)} | "
                 f"{o['features']['mlp'].get('match_status', {}).get('stable', '—')} | {lvl.get(s, '—')} |")
    conv = [r["observe"] for r in rb if r["step"] >= 1500]
    L += ["", f"Post-convergence noise floor (1,500-6,000, n = {len(conv)}): OOD_DEV mean {st.mean(o['acc_ood_dev'] for o in conv):.3f}, "
          f"sd {st.stdev(o['acc_ood_dev'] for o in conv):.3f}; pooled edit mean {st.mean(o['causal']['edit_block_intended'] for o in conv):.3f}, "
          f"sd {st.stdev(o['causal']['edit_block_intended'] for o in conv):.3f}.", ""]
    L += ["## Lineage events and post-hoc artifact audit (disclosed)", "", "| network | non-persistence events | ambiguous head relocations | status jitter | cluster repartition | retained |", "|---|---|---|---|---|---|"]
    for n in NETS:
        a = audit[n]
        L.append(f"| {n} | {a['total']} | {a['ambiguous_head_relocation']} | {a['status_jitter']} | {a['cluster_repartition']} | {a['retained']} |")
    L += ["", "Event types (B): " + ", ".join(f"{k} {v}" for k, v in sorted(audit['B_supervised']['by_event'].items())), ""]
    recsB = TR["B_supervised"].concept_records()
    caus = [r for r in recsB.values() if r["ever_causally_supported"]]
    L += [f"Concept records (B): {len(recsB)} lineage IDs; {len(caus)} ever causally supported "
          f"({sum(1 for r in caus if r['kind'] == 'subspace')} subspaces, {sum(1 for r in caus if r['kind'] == 'neuron_cluster')} neuron clusters, "
          f"{sum(1 for r in caus if r['kind'] == 'head')} heads).", ""]
    if fs.exists():
        Fz = json.loads(fs.read_text())
        ma, mb = Fz["A_control"]["metrics"], Fz["B_supervised"]["metrics"]
        L += ["## Sealed final evaluation (step 6,000; FINAL / OOD_FINAL; once)", "", "| measure | A control | B supervised | B−A |", "|---|---|---|---|"]
        for k in ("acc_id", "acc_ood_final", "acc_ood_legacy", "acc_ood_dev", "iia_M_block", "iia_M_rest", "iia_A_block", "das_iia_M",
                  "das_block_overlap_M", "copy_block_intended", "block_interaction_share", "block_route_main_share"):
            L.append(f"| {k} | {ma[k]:.3f} | {mb[k]:.3f} | {mb[k] - ma[k]:+.3f} |")
        for kind in ("inv", "add1"):
            for loc in ("block", "das"):
                a, b = ma[f"edit_{kind}_{loc}"], mb[f"edit_{kind}_{loc}"]
                L.append(f"| persistent edit {kind} @ {loc}: intended / wrong / ‖ΔW_in‖/‖W_in‖ | {a['intended']:.3f} / {a['wrong']:.3f} / "
                         f"{a['rel_change_W_in']:.2f} | {b['intended']:.3f} / {b['wrong']:.3f} / {b['rel_change_W_in']:.2f} | {b['intended'] - a['intended']:+.3f} |")
        L += ["", f"**Registered outcome (docs/28 §28.7): {'not tested (no edit installed)' if not meta['installed'] else 'see docs/29'}.**", "",
              "Post-training benchmarks (evaluation only):", ""]
        for k, v in Fz["B_supervised"]["benchmarks"].items():
            vv = {kk: (round(x, 3) if isinstance(x, float) else (round(x.get("intended"), 3) if isinstance(x, dict) and x.get("intended") == x.get("intended") else x))
                  for kk, x in v.items() if kk != "edit_dbl_block"}
            L.append(f"- {k}: {vv}")
    L += ["", "## Compute and storage ledger", "", "```", json.dumps(led, indent=1), "```"]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    report()
