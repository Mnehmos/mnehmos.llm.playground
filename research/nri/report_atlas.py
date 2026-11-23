"""Report for the Issue #22 atlas pilot (docs/24 registration; docs/25 results).

Reads runs/zoo_atlas/{control,supervised}/{cards,decisions}.jsonl, results/atlas/milestones.json and
results/atlas/final_sealed.json, and writes to results/atlas/:
  summary.md            tables (auto-generated)
  atlas.json            concept-to-site atlas per trajectory x checkpoint (registered criteria)
  events.json           registered emergence / identification / coalescence timings
  audit_cards.jsonl     concise per-checkpoint supervisor audit cards
  decisions.jsonl       every proposal (both trajectories; shadow on control)
  history_long.csv      every scalar of every card: trajectory, step, key, value (queryable)
  ledger.json           compute and storage ledger
  fig_atlas_*.png       figures
Applies only the registered definitions in nri/atlas.py and docs/24; nothing here is tuned.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nri import atlas as AT  # noqa: E402

ROOT = Path(__file__).parent
ZOO = ROOT / "runs" / "zoo_atlas"
OUT = ROOT / "results" / "atlas"
TRAJ = ("control", "supervised")
COL = {"control": "#2a78d6", "supervised": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FORMATION = (500, 2000)


def load(name):
    cards = [json.loads(l) for l in open(ZOO / name / "cards.jsonl")]
    decs = [json.loads(l) for l in open(ZOO / name / "decisions.jsonl")]
    run = json.loads((ZOO / name / "run.json").read_text()) if (ZOO / name / "run.json").exists() else None
    return cards, decs, run


def flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from flatten(v, key + ".")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield key, v


def style(ax, title=None):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")


def shade_formation(ax):
    ax.axvspan(*FORMATION, color="#efeeea", zorder=0)


# ------------------------------------------------------------------ retention of accepted edits
def retention(decs):
    out = []
    by = {d["step"]: d for d in decs}
    for d in decs:
        if not d.get("applied"):
            continue
        acc = [c for c in d["candidates"] if c.get("passed")][-1]
        nxt = by.get(d["step"] + 100)
        imm = acc["edit_after"] - acc["edit_before"]
        r = dict(step=d["step"], step_rel=d["step_rel"], alpha_equiv=d["alpha_equiv"], rel_norm=d["rel_norm"],
                 val_edit_before=acc["edit_before"], val_edit_after=acc["edit_after"], immediate_gain=imm,
                 interaction_before=acc["interaction_before"], interaction_after=acc["interaction_after"],
                 sham_mean_gain=acc.get("E9_sham_mean_gain"), iit_grad_gain=d.get("comparators", {}).get("iit_grad", {}).get("edit_gain"))
        if nxt and "before" in nxt:
            r["val_edit_next_checkpoint"] = nxt["before"]["edit"]
            r["retained_fraction"] = (nxt["before"]["edit"] - acc["edit_before"]) / imm if abs(imm) > 1e-9 else None
            r["interaction_next_checkpoint"] = nxt["before"]["interaction"]
        out.append(r)
    return out


# ------------------------------------------------------------------ figures
def fig_decode(C):
    sites = ["L1@mval", "L1@munit", "L1@last", "A2@last", "h_block", "h_rest", "mlp"]
    facs = [("M", "mass M"), ("m", "raw mass value m"), ("mu", "mass unit"), ("A", "acceleration A"),
            ("MA", "product M·A"), ("F", "answer F")]
    fig, axes = plt.subplots(len(facs), 2, figsize=(11, 1.55 * len(facs) + 0.8), sharex=True, sharey=True)
    for j, name in enumerate(TRAJ):
        cards = C[name][0]
        steps = [c["step"] for c in cards]
        for i, (f, lab) in enumerate(facs):
            ax = axes[i, j]
            Z = [[c["observe"]["decode_kappa"][f"{f}@{s}"] for c in cards] for s in sites]
            im = ax.imshow(Z, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
                           extent=(steps[0] - 50, steps[-1] + 50, len(sites) - 0.5, -0.5))
            ax.set_yticks(range(len(sites)))
            ax.set_yticklabels(sites, fontsize=7)
            ax.tick_params(labelsize=7, colors=INK2)
            ax.set_title(f"{name}: {lab}", fontsize=8, loc="left", color=INK)
            ax.set_xscale("symlog", linthresh=1000)
    for ax in axes[-1]:
        ax.set_xlabel("training step (linear to 1,000, log after)", fontsize=8, color=INK2)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("probe κ (chance-adjusted; descriptive, not causal)", fontsize=8)
    fig.suptitle("Where each concept is linearly decodable, by step (OBS panel)", fontsize=10, x=0.01, ha="left")
    fig.savefig(OUT / "fig_atlas_decode.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_causal(C):
    panels = [("iia_M_block", "mass M: swap block (dims 0–15)"), ("iia_M_rest", "mass M: swap dims 16–63"),
              ("iia_M_L1_val+unit", "mass M: swap layer-1 value+unit tokens"), ("iia_m_L1_val", "raw m: swap value token"),
              ("iia_mu_L1_unit", "mass unit: swap unit token"), ("iia_A_L1_val+unit", "acceleration A: swap layer-1 tokens"),
              ("edit_block_intended", "persistent pooled inversion edit (intended)"), ("copy_block_intended", "copy real block (intended)")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4), sharex=True, sharey=True)
    for ax, (k, lab) in zip(axes.flat, panels):
        shade_formation(ax)
        for name in TRAJ:
            cards = C[name][0]
            ax.plot([c["step"] for c in cards], [c["observe"]["causal"][k] for c in cards], color=COL[name], lw=1.6,
                    label=name)
        ax.axhline(0.5, color=INK2, lw=0.6, ls=":")
        ax.axhline(0.9, color=INK2, lw=0.6, ls="--")
        ax.set_xscale("symlog", linthresh=1000)
        style(ax, lab)
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[0, 0].set_ylim(-0.02, 1.02)
    fig.suptitle("Causal use (activation interchange accuracy, OBS pairs). Dotted 0.5 = causal; dashed 0.9 = verified. Grey = steps 500–2,000",
                 fontsize=9, x=0.01, ha="left")
    fig.savefig(OUT / "fig_atlas_causal.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_geometry(C):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharex=True)
    keys = [(lambda o: o["acc_obs"], "ID accuracy (OBS)"), (lambda o: o["acc_ood_dev"], "held-out combos (OOD_DEV)"),
            (lambda o: o["geometry"]["block"]["M_interaction_share"], "block: mass × route interaction share"),
            (lambda o: o["unit_transfer"]["M@h_block"]["transfer"] / max(o["unit_transfer"]["M@h_block"]["within"], 1e-9),
             "block: mass unit-transfer ratio")]
    for ax, (f, lab) in zip(axes, keys):
        shade_formation(ax)
        for name in TRAJ:
            cards = C[name][0]
            ax.plot([c["step"] for c in cards], [f(c["observe"]) for c in cards], color=COL[name], lw=1.6, label=name)
        ax.set_xscale("symlog", linthresh=1000)
        style(ax, lab)
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(OUT / "fig_atlas_geometry.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_features(C):
    fams = ["M", "A", "F", "MA", "m", "a", "mu", "au", "fu", "order", "route", "mixed", "dead"]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
            "#8a8984", "#b5b4ae", "#5c5b57", "#d7d6d0", "#efeeea"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
    for ax, name in zip(axes, TRAJ):
        cards = C[name][0]
        steps = [c["step"] for c in cards]
        ys = [[c["observe"]["features"]["mlp"]["counts"].get(f, 0) for c in cards] for f in fams]
        ax.stackplot(steps, ys, labels=fams, colors=cols, edgecolor="white", linewidth=0.3)
        ax.set_xscale("symlog", linthresh=1000)
        style(ax, f"{name}: MLP neuron families (argmax η² ≥ 0.25)")
    axes[1].legend(fontsize=7, frameon=False, ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.savefig(OUT / "fig_atlas_features.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_weights(C):
    tensors = ["E_c.weight", "E_p.weight", "attn1.qkv.weight", "attn1.o.weight", "attn2.qkv.weight", "attn2.o.weight",
               "W_in.weight", "W_out.weight", "W_U.weight"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    for ax, name in zip(axes[:2], TRAJ):
        cards = C[name][0][1:]
        steps = [c["step"] for c in cards]
        Z = [[c["weights"][t]["rel_from_prev"] for c in cards] for t in tensors]
        im = ax.imshow(Z, aspect="auto", cmap="Blues", interpolation="nearest",
                       extent=(steps[0] - 50, steps[-1] + 50, len(tensors) - 0.5, -0.5))
        ax.set_yticks(range(len(tensors)))
        ax.set_yticklabels(tensors, fontsize=7)
        ax.tick_params(labelsize=7, colors=INK2)
        ax.set_xscale("symlog", linthresh=1000)
        ax.set_title(f"{name}: relative weight change per 100 steps", fontsize=8, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.03)
    ax = axes[2]
    shade_formation(ax)
    for name in TRAJ:
        cards = C[name][0][1:]
        ax.plot([c["step"] for c in cards], [-c["attribution"]["iit"]["per_tensor"]["attn2.o.weight"] for c in cards],
                color=COL[name], lw=1.4, label=f"{name}: attn2.o")
        ax.plot([c["step"] for c in cards], [-c["attribution"]["iit"]["per_tensor"]["attn1.qkv.weight"] for c in cards],
                color=COL[name], lw=1.0, ls="--", label=f"{name}: attn1.qkv")
    ax.set_xscale("symlog", linthresh=1000)
    style(ax, "IIT-loss reduction attributed per interval")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(OUT / "fig_atlas_weights.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_policy(C):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4), sharey=True)
    marks = {"accepted": ("o", "#1baf7a"), "rejected": ("x", "#e34948"), "skipped": (".", "#b5b4ae")}
    for ax, name in zip(axes, TRAJ):
        decs = C[name][1]
        shade_formation(ax)
        for st, (mk, cl) in marks.items():
            xs = [d["step"] for d in decs if d["status"] == st and "before" in d]
            ys = [d["before"]["edit"] for d in decs if d["status"] == st and "before" in d]
            ax.scatter(xs, ys, marker=mk, color=cl, s=24, label=f"{st} ({sum(d['status'] == st for d in decs)})", zorder=3)
        ax.plot([d["step"] for d in decs if "before" in d], [d["before"]["edit"] for d in decs if "before" in d],
                color=COL[name], lw=1.0, zorder=2)
        ax.set_xscale("symlog", linthresh=1000)
        style(ax, f"{name} ({'applied' if name == 'supervised' else 'shadow, never applied'}): VAL pooled edit before proposal")
        ax.legend(fontsize=7, frameon=False)
    fig.savefig(OUT / "fig_atlas_policy.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ report
def fmt(x, n=3):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def classify(ms, mc):
    de = ms["edit_inv_block"]["intended"] - mc["edit_inv_block"]["intended"]
    dh = max(ms["edit_add1_block"]["intended"] - mc["edit_add1_block"]["intended"],
             ms["edit_dbl_block"]["intended"] - mc["edit_dbl_block"]["intended"])
    did, dood = ms["acc_id"] - mc["acc_id"], ms["acc_ood_final"] - mc["acc_ood_final"]
    diia = ms["iia_M_block"] - mc["iia_M_block"]
    better = de >= 0.05 and dh >= 0.03
    safe = did >= -0.005 and dood >= -0.02 and diia >= -0.01
    if better and safe:
        o = "improved"
    elif better:
        o = "traded off"
    elif abs(de) < 0.05 and abs(dh) < 0.03:
        o = "no effect"
    else:
        o = "harmed"
    return o, dict(d_edit_inv=de, d_best_heldout_edit=dh, d_id=did, d_ood_final=dood, d_iia=diia)


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    C = {n: load(n) for n in TRAJ}
    L = ["# Concept-formation atlas and continuous upstream edit (auto-generated by `run_atlas.py report`; registration docs/24)", ""]
    # integrity
    cc, cs = C["control"][0], C["supervised"][0]
    first_div = next((a["step"] for a, b in zip(cc, cs) if a["state_hash"] != b["state_hash"]), None)
    first_app = next((d["step"] for d in C["supervised"][1] if d.get("applied")), None)
    same_batches = all(a["batch_digest"] == b["batch_digest"] for a, b in zip(cc, cs))
    L += ["## Integrity", "",
          f"- checkpoints: control {len(cc)}, supervised {len(cs)} (registered: 61 each)",
          f"- batch digests identical at every checkpoint: **{same_batches}**",
          f"- first accepted edit (supervised): step {fmt(first_app)}; first checkpoint whose state hash differs: "
          f"step {fmt(first_div)} (must be the checkpoint after the first accepted edit, or none)",
          f"- before the first accepted edit the two networks are bitwise identical: "
          f"**{all(a['state_hash'] == b['state_hash'] for a, b in zip(cc, cs) if first_app is None or a['step'] <= first_app)}**", ""]
    # events
    ev = {n: AT.registered_events(C[n][0]) for n in TRAJ}
    (OUT / "events.json").write_text(json.dumps(ev, indent=1))
    L += ["## Registered timings (first checkpoint, sustained; docs/24 §24.7)", "", "| event | control | supervised |", "|---|---|---|"]
    for k in ev["control"]:
        L.append(f"| {k} | {fmt(ev['control'][k])} | {fmt(ev['supervised'][k])} |")
    # atlas
    atlas = {}
    for n in TRAJ:
        atlas[n] = {c["step"]: {k: AT.concept_state(c["observe"], k) for k in AT.CONCEPTS} for c in C[n][0]}
    (OUT / "atlas.json").write_text(json.dumps(atlas, indent=1))
    show = (0, 300, 600, 800, 900, 1000, 1100, 1200, 1300, 1500, 2000, 3000, 6000)
    L += ["", "## Concept-to-site atlas (control; supervised in atlas.json) — status, best decoding site (κ), best causal test (IIA)", "",
          "| step | " + " | ".join(AT.CONCEPTS) + " |", "|---|" + "---|" * len(AT.CONCEPTS)]
    for s in show:
        if s not in atlas["control"]:
            continue
        row = []
        for k in AT.CONCEPTS:
            a = atlas["control"][s][k]
            row.append(f"{a['status'].split(' (')[0]}; {a['best_decode_site']} {a['best_kappa']:.2f}"
                       + (f"; {a['best_causal_test'].replace('iia_', '')} {a['best_iia']:.2f}" if a["best_causal_test"] else ""))
        L.append(f"| {s} | " + " | ".join(row) + " |")
    # trajectory table
    L += ["", "## Matched trajectories (OBS panel; every 100 steps to 2,000, then every 500)", "",
          "| step | traj | ID | OOD_DEV | block IIA(M) | IIA(M) dims 16–63 | L1 tokens IIA(M) | heads IIA(M) 0/1/2/3 | edit (inv) | copy | interaction | route main | unit-transfer ratio (block) | M unit-CKA (block) | M mult-char (h) | decision |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, b in zip(cc, cs):
        s = a["step"]
        if s > 2000 and s % 500:
            continue
        for n, c in (("C", a), ("S", b)):
            o, z, g = c["observe"], c["observe"]["causal"], c["observe"]["geometry"]["block"]
            t = o["unit_transfer"]["M@h_block"]
            d = c["decision"]
            L.append(f"| {s} | {n} | {o['acc_obs']:.3f} | {o['acc_ood_dev']:.3f} | {z['iia_M_block']:.3f} | {z['iia_M_rest']:.3f} | "
                     f"{z['iia_M_L1_val+unit']:.3f} | {'/'.join(f'{z[f'iia_M_head{i}']:.2f}' for i in range(4))} | "
                     f"{z['edit_block_intended']:.3f} | {z['copy_block_intended']:.3f} | {g['M_interaction_share']:.3f} | "
                     f"{g['M_route_main_share']:.3f} | {t['transfer'] / max(t['within'], 1e-9):.3f} | {g['M_unit_cka']:.3f} | "
                     f"{o['geometry']['h']['M_char']['mult']:.3f} | {d['status']}{' **applied**' if d.get('applied') else ''} |")
    # decisions
    L += ["", "## Policy decisions (control = shadow, never applied)", "", "| trajectory | accepted | rejected | skipped | first non-skip step | failing checks among rejections |", "|---|---|---|---|---|---|"]
    allrec = []
    for n in TRAJ:
        decs = C[n][1]
        allrec += decs
        fails = {}
        for d in decs:
            if d["status"] == "rejected":
                for f in d["reason"].split(": ")[-1].split(","):
                    fails[f] = fails.get(f, 0) + 1
        fn = next((d["step"] for d in decs if d["status"] != "skipped"), None)
        L.append(f"| {n} | {sum(d['status'] == 'accepted' for d in decs)} | {sum(d['status'] == 'rejected' for d in decs)} | "
                 f"{sum(d['status'] == 'skipped' for d in decs)} | {fmt(fn)} | {fails} |")
    with open(OUT / "decisions.jsonl", "w") as f:
        for d in allrec:
            f.write(json.dumps(d) + "\n")
    ret = retention(C["supervised"][1])
    L += ["", "## Accepted edits (supervised): immediate effect and retention at the next checkpoint (VAL)", "",
          "| step | rel. norm | α equiv. | VAL edit before → after | gain | sham mean gain | IIT-grad comparator gain | VAL edit at next ckpt | retained fraction | interaction before → after → next |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ret:
        L.append(f"| {r['step']} | {r['rel_norm']:.3f} | {r['alpha_equiv']:.3f} | {r['val_edit_before']:.3f} → {r['val_edit_after']:.3f} | "
                 f"{r['immediate_gain']:+.3f} | {fmt(r['sham_mean_gain'])} | {fmt(r['iit_grad_gain'])} | {fmt(r.get('val_edit_next_checkpoint'))} | "
                 f"{fmt(r.get('retained_fraction'), 2)} | {r['interaction_before']:.3f} → {r['interaction_after']:.3f} → {fmt(r.get('interaction_next_checkpoint'))} |")
    (OUT / "accepted_edits.json").write_text(json.dumps(ret, indent=1))
    # shadow comparison
    L += ["", "## Same steps, both trajectories: proposal outcome (supervised applied / control shadow)", "",
          "| step | supervised | control (shadow) |", "|---|---|---|"]
    for a, b in zip(C["supervised"][1], C["control"][1]):
        if a["status"] != "skipped" or b["status"] != "skipped":
            if a["step"] <= 2000 or a["status"] == "accepted" or b["status"] == "accepted" or a["step"] % 1000 == 0:
                L.append(f"| {a['step']} | {a['status']}: {a.get('reason', '')} | {b['status']}: {b.get('reason', '')} |")
    # weights / attribution in the formation window
    L += ["", "## Which parameters moved and which reduced the IIT (mass-interchange) loss, steps 500–2,000", "",
          "| tensor | control Σ rel. change | supervised Σ rel. change | control Σ IIT attribution | supervised Σ IIT attribution | control Σ task attribution |",
          "|---|---|---|---|---|---|"]
    def sums(cards, key, sub=None):
        out = {}
        for c in cards:
            if not (FORMATION[0] < c["step"] <= FORMATION[1]) or c["attribution"] is None:
                continue
            if key == "w":
                for t, v in c["weights"].items():
                    out[t] = out.get(t, 0) + v["rel_from_prev"]
            else:
                for t, v in c["attribution"][sub][key].items():
                    out[t] = out.get(t, 0) + v
        return out
    wc, ws = sums(cc, "w"), sums(cs, "w")
    ac, as_, tc = sums(cc, "per_tensor", "iit"), sums(cs, "per_tensor", "iit"), sums(cc, "per_tensor", "task")
    for t in sorted(ac, key=lambda k: ac[k]):
        L.append(f"| {t} | {wc.get(t, 0):.3f} | {ws.get(t, 0):.3f} | {ac[t]:+.4f} | {as_[t]:+.4f} | {tc[t]:+.4f} |")
    hc, hs = sums(cc, "per_head", "iit"), sums(cs, "per_head", "iit")
    L += ["", "Per attention head (IIT attribution, Σ over 500–2,000; negative = reduced the mass-interchange loss): " +
          ", ".join(f"{k} {hc[k]:+.3f} / {hs.get(k, 0):+.3f}" for k in sorted(hc)) + " (control / supervised)."]
    # milestones
    mp = OUT / "milestones.json"
    if mp.exists():
        M = json.loads(mp.read_text())
        L += ["", "## Offline milestones (DAS, D1, family ablations; OBS)", "",
              "| step | traj | DAS IIA(M) | DAS∩block (M) | DAS edit M | DAS IIA(A) | D1 acc without route variance (OBS / OOD_DEV) | add-one edit (block) | ablate M-family neurons acc |",
              "|---|---|---|---|---|---|---|---|---|"]
        for n in TRAJ:
            for r in M[n]["rows"] + M[n]["post_edit_rows"]:
                ab = r.get("ablate_family_M", {})
                L.append(f"| {r['step']}{' (edited)' if r.get('edited') else ''} | {n[0].upper()} | {r['das_iia_M']:.3f} | {r['das_block_overlap_M']:.3f} | "
                         f"{r['edit_das_M_intended']:.3f} | {r['das_iia_A']:.3f} | {r['D1_acc_without_route_variance_obs']:.3f} / "
                         f"{r['D1_acc_without_route_variance_ood_dev']:.3f} | {r['edit_block_add1_intended']:.3f} | "
                         f"{fmt(ab.get('acc'))} (n={ab.get('n', 0)}) |")
    # sealed final
    fp = OUT / "final_sealed.json"
    if fp.exists():
        Fz = json.loads(fp.read_text())
        mc, ms = Fz["control"]["metrics"], Fz["supervised"]["metrics"]
        L += ["", "## Sealed final evaluation (step 6,000; FINAL / OOD_FINAL; run once)", "", "| measure | control | supervised | Δ |", "|---|---|---|---|"]
        for k in ("acc_id", "acc_ood_final", "acc_ood_legacy", "acc_ood_dev", "iia_M_block", "iia_M_rest", "iia_A_block", "das_iia_M",
                  "das_block_overlap_M", "copy_block_intended", "block_interaction_share", "block_route_main_share"):
            L.append(f"| {k} | {mc[k]:.3f} | {ms[k]:.3f} | {ms[k] - mc[k]:+.3f} |")
        for kind in ("inv", "add1", "dbl"):
            for loc in ("block", "das"):
                a, b = mc[f"edit_{kind}_{loc}"], ms[f"edit_{kind}_{loc}"]
                L.append(f"| persistent edit {kind} @ {loc}: intended / wrong / preserved | {a['intended']:.3f} / {a['wrong']:.3f} / "
                         f"{a['preserved_on_fixed_points']:.3f} | {b['intended']:.3f} / {b['wrong']:.3f} / {b['preserved_on_fixed_points']:.3f} | "
                         f"{b['intended'] - a['intended']:+.3f} |")
        o, dd = classify(ms, mc)
        L += ["", f"**Registered outcome (docs/24 §24.9): {o if first_app is not None else 'not tested (no edit accepted)'}** — " +
              ", ".join(f"{k} {v:+.3f}" for k, v in dd.items()), "",
              "### Post-training, evaluation-only benchmarks", "", "| benchmark | network | result |", "|---|---|---|"]
        for n in TRAJ:
            for k, v in Fz[n]["benchmarks"].items():
                vv = {kk: (round(x, 3) if isinstance(x, float) else (x.get("intended") if isinstance(x, dict) else x))
                      for kk, x in v.items()}
                L.append(f"| {k} | {n} | {vv} |")
    # ledger
    led = {n: C[n][2]["ledger"] for n in TRAJ if C[n][2]}
    stor = {n: C[n][2]["storage_bytes"] for n in TRAJ if C[n][2]}
    extra = {}
    if mp.exists():
        extra["offline_milestones_cpu_s"] = json.loads(mp.read_text())["cpu_s"]
    if fp.exists():
        extra["sealed_eval_cpu_s"] = json.loads(fp.read_text())["cpu_s"]
    (OUT / "ledger.json").write_text(json.dumps(dict(per_trajectory=led, storage_bytes=stor, **extra), indent=1))
    L += ["", "## Compute and storage ledger (CPU-seconds, single thread)", "", "| phase | control | supervised |", "|---|---|---|"]
    for k in led.get("control", {}):
        L.append(f"| {k} | {led['control'][k]} | {led['supervised'][k]} |")
    L.append(f"| storage (MB) | {stor.get('control', 0) / 1e6:.1f} | {stor.get('supervised', 0) / 1e6:.1f} |")
    for k, v in extra.items():
        L.append(f"| {k} (both) | {v} | |")
    tot = sum(sum(v.values()) for v in led.values()) + sum(extra.values())
    L.append(f"\nTotal: {tot / 3600:.2f} CPU-hours.")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    # audit cards and long history
    with open(OUT / "audit_cards.jsonl", "w") as f:
        for n in TRAJ:
            prev = None
            for c in C[n][0]:
                o = c["observe"]
                st = atlas[n][c["step"]]
                changes = {k: [prev[k]["status"], st[k]["status"]] for k in st if prev and prev[k]["status"] != st[k]["status"]}
                top_w = sorted(c["weights"].items(), key=lambda kv: -kv[1]["rel_from_prev"])[:3]
                attr = c["attribution"]["iit"]["per_head"] if c["attribution"] else {}
                f.write(json.dumps(dict(
                    trajectory=n, step=c["step"], state_hash=c["state_hash"][:16], acc_obs=round(o["acc_obs"], 4),
                    acc_ood_dev=round(o["acc_ood_dev"], 4), iia_M_block=round(o["causal"]["iia_M_block"], 4),
                    iia_M_rest=round(o["causal"]["iia_M_rest"], 4), edit_block=round(o["causal"]["edit_block_intended"], 4),
                    interaction_share=o["geometry"]["block"]["M_interaction_share"],
                    equivalence_regime=AT.equivalence_regime(o), concept_status_changes=changes,
                    mlp_families=o["features"]["mlp"]["counts"], mlp_family_changes=o["features"]["mlp"].get("family_changes"),
                    mlp_match_status=o["features"]["mlp"].get("match_status"),
                    top_weight_updates=[[k, v["rel_from_prev"]] for k, v in top_w],
                    iit_attribution_by_head={k: round(v, 5) for k, v in attr.items()},
                    decision=c["decision"]["status"], decision_reason=c["decision"].get("reason"),
                    applied=c["decision"].get("applied", False),
                    causally_verified=[k for k in st if st[k]["status"].startswith("causal (verified")])) + "\n")
                prev = st
    with open(OUT / "history_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trajectory", "step", "key", "value"])
        for n in TRAJ:
            for c in C[n][0]:
                for k, v in flatten({kk: vv for kk, vv in c.items() if kk not in ("decision",)}):
                    w.writerow([n, c["step"], k, v])
    for fn in (fig_decode, fig_causal, fig_geometry, fig_features, fig_weights, fig_policy):
        fn(C)
    print("\n".join(L))


if __name__ == "__main__":
    report()
