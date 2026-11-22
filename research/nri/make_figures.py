"""Figures for docs/06 (static PNG; the tables in results/*/summary.md are the table view).

Palette: validated categorical slots 1-5 of the reference instance (dataviz skill),
fixed order by condition, never re-colored by rank. Direct labels + legend.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
CONDS = ["none", "typed", "shuffled", "canonical", "typed_fact"]
COLOR = dict(zip(CONDS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
REGIMES = [r for r in ("wd0.01", "wd1") if (ROOT / "results" / r / "metrics.json").exists()]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def learning_curves():
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(5.2 * len(REGIMES), 3.4), sharey=True,
                             facecolor=SURF, squeeze=False)
    for ax, reg in zip(axes[0], REGIMES):
        style(ax)
        for c in CONDS:
            runs = sorted((ROOT / "runs" / f"zoo_{reg}").glob(f"{c}_s*"))
            curves = []
            for r in runs:
                recs = [json.loads(l) for l in (r / "log.jsonl").read_text().splitlines()]
                curves.append({x["step"]: x["eval_acc"] for x in recs if x["step"] <= 1000})
            steps = sorted(set.intersection(*(set(cv) for cv in curves)))
            mean = [st.mean(cv[s] for cv in curves) for s in steps]
            ax.plot(steps, mean, color=COLOR[c], linewidth=2, label=c)
        ax.set_title(f"held-out accuracy, first 1,000 steps ({reg})", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("training step", fontsize=8, color=INK2)
    axes[0][0].set_ylabel("accuracy (mean of 5 seeds)", fontsize=8, color=INK2)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right", title="condition",
                      title_fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "fig_learning.png", dpi=160)


METRICS = [("iia_op", "class-mean IIA, op"), ("das_op", "DAS IIA, op (k=16)"),
           ("das_a", "DAS IIA, a (k=16)"), ("head_op_best", "best head-patch IIA, op"),
           ("mean_head_purity", "mean head role-purity"), ("das_edit_success", "DAS-edit success")]


def causal_dots():
    fig, axes = plt.subplots(len(REGIMES), len(METRICS), figsize=(2.3 * len(METRICS), 2.6 * len(REGIMES)),
                             facecolor=SURF, squeeze=False)
    for i, reg in enumerate(REGIMES):
        models = json.loads((ROOT / "results" / reg / "metrics.json").read_text())["models"]
        for j, (k, title) in enumerate(METRICS):
            ax = axes[i][j]
            style(ax)
            for x, c in enumerate(CONDS):
                ys = [m[k] for m in models if m["condition"] == c and m[k] is not None]
                ax.scatter([x] * len(ys), ys, s=22, color=COLOR[c], edgecolor=SURF, linewidth=1, zorder=3)
                if ys:
                    ax.hlines(st.mean(ys), x - 0.3, x + 0.3, color=INK, linewidth=1.2, zorder=4)
            ax.set_xticks(range(len(CONDS)))
            ax.set_xticklabels(CONDS, rotation=45, ha="right", fontsize=7)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{title}\n({reg})", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "fig_causal.png", dpi=160)


UI_CONDS = ["none", "typed_always", "typed_dropout", "typed_dropout_aux", "aux_only",
            "aux_shuffled", "aux_lexical", "aux_contextual"]
UI_METRICS = [("acc_untyped", "accuracy, untyped inference"), ("das_op", "DAS IIA, op (k=16)"),
              ("iia_op", "class-mean IIA, op"), ("edit_success", "closed-form edit success"),
              ("das_edit_success", "DAS-guided edit success")]


def untyped_dots():
    p = ROOT / "results" / "untyped_inf" / "metrics.json"
    if not p.exists():
        return
    models = json.loads(p.read_text())
    conds = [c for c in UI_CONDS if any(m["condition"] == c for m in models)]
    role_sup = {"typed_dropout_aux", "aux_only"}  # highlighted: role-supervised arms
    fig, axes = plt.subplots(1, len(UI_METRICS), figsize=(3.0 * len(UI_METRICS), 3.6), facecolor=SURF)
    for ax, (k, title) in zip(axes, UI_METRICS):
        style(ax)
        for x, c in enumerate(conds):
            ys = [m[k] for m in models if m["condition"] == c]
            col = "#eb6834" if c in role_sup else "#2a78d6"
            ax.scatter([x] * len(ys), ys, s=20, color=col, edgecolor=SURF, linewidth=1, zorder=3)
            ax.hlines(st.mean(ys), x - 0.3, x + 0.3, color=INK, linewidth=1.2, zorder=4)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=55, ha="right", fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
    fig.suptitle("Typed training, untyped inference: orange = role used as an auxiliary TARGET",
                 fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "fig_untyped_inference.png", dpi=160)


IIT_CONDS = ["none", "aux_only", "iit", "iit_aux"]
IIT_METRICS = [("das_op", "DAS IIA, op"), ("das_a", "DAS IIA, a (operand)"),
               ("edit_success", "op-swap edit success"),
               ("shift_edit_das", "held-out edit: a→a+1 (DAS-located)"),
               ("additive_r2", "additive R² at site")]


def iit_dots():
    p = ROOT / "results" / "iit" / "metrics.json"
    if not p.exists():
        return
    models = json.loads(p.read_text())
    fig, axes = plt.subplots(1, len(IIT_METRICS), figsize=(2.7 * len(IIT_METRICS), 3.2), facecolor=SURF)
    for ax, (k, title) in zip(axes, IIT_METRICS):
        style(ax)
        for x, c in enumerate(IIT_CONDS):
            ys = [m[k] for m in models if m["condition"] == c]
            col = "#eb6834" if c.startswith("iit") else "#2a78d6"
            ax.scatter([x] * len(ys), ys, s=20, color=col, edgecolor=SURF, linewidth=1, zorder=3)
            ax.hlines(st.mean(ys), x - 0.3, x + 0.3, color=INK, linewidth=1.2, zorder=4)
        ax.set_xticks(range(len(IIT_CONDS)))
        ax.set_xticklabels(IIT_CONDS, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
    fig.suptitle("R6: structure enforced inside the network (orange = interchange intervention training)",
                 fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "fig_iit.png", dpi=160)


if __name__ == "__main__":
    learning_curves()
    causal_dots()
    untyped_dots()
    iit_dots()
    print("ok", REGIMES)
