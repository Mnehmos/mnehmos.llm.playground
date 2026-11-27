"""Figures for the preprint. Reads only committed result files and reconstruction audits.
Palette: validated categorical slots 1-4 (blue, orange, aqua, yellow; CVD-separated); low-contrast slots are always
direct-labelled. Thin marks, recessive grid, one y-axis per panel.

    python paper/make_figures.py [--zoo /home/user/swap_recon/zoo]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).parent / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
SEEDS = range(2001, 2013)


def fig_primary():
    ft = json.loads((ROOT / "runs/concept_factorial/final_test.json").read_text())
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    cols = {"A": MUTED, "B": BLUE, "C": ORANGE, "D": AQUA}
    names = {"A": "A narrow", "B": "B diverse", "C": "C canonical aux", "D": "D scrambled aux"}
    x = 0
    ticks, labels = [], []
    for w in (0.1, 1.0):
        for p in "ABCD":
            v = [ft[f"wd{w}_{p}_s{s}"]["test"] for s in SEEDS]
            ax.scatter([x + (i - 5.5) * 0.025 for i in range(12)], v, s=10, color=cols[p], alpha=0.8, linewidths=0, zorder=3)
            ax.plot([x - 0.25, x + 0.25], [statistics.mean(v)] * 2, color=INK, lw=1.5, zorder=4)
            ticks.append(x)
            labels.append(p)
            x += 1
        ax.text(x - 2.5, -0.17, f"weight decay {w}", ha="center", color=INK, fontsize=7.5, transform=ax.transData)
        x += 0.6
    ax.axhline(1 / 6, color=MUTED, lw=1, ls="--")
    ax.text(x - 0.7, 1 / 6 + 0.02, "best constant", color=MUTED, ha="right", fontsize=7.5)
    ax.set_xticks(ticks, labels, fontsize=8)
    ax.set_ylabel("reserved-test accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "fig1_primary.pdf")
    plt.close(fig)


def fig_i1():
    d = ROOT / "runs/instruments/factorial"
    conds = [("wd1.0_C", "C, wd 1.0", ORANGE), ("wd0.1_C", "C, wd 0.1", BLUE), ("wd1.0_D", "D, wd 1.0", AQUA),
             ("wd1.0_B", "B, wd 1.0", YELLOW)]
    S = "base_trained|can_implied_heldout"
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    any_ = False
    for pre, lab, col in conds:
        files = [d / f"{pre}_s{s}.json" for s in SEEDS]
        if not all(f.exists() for f in files):
            continue
        any_ = True
        runs = [json.loads(f.read_text())["I1"] for f in files]
        steps = sorted(int(k) for k in runs[0])
        m = [statistics.mean(r[str(t)][S]["canonical_switch"] for r in runs) for t in steps]
        lo = [min(r[str(t)][S]["canonical_switch"] for r in runs) for t in steps]
        hi = [max(r[str(t)][S]["canonical_switch"] for r in runs) for t in steps]
        ax.fill_between(steps, lo, hi, color=col, alpha=0.12, linewidth=0)
        ax.plot(steps, m, color=col)
        ax.text(steps[-1] + 80, m[-1], lab, color=INK, va="center", fontsize=7.5)
    if not any_:
        plt.close(fig)
        return
    ax.set_xlabel("training step")
    ax.set_ylabel("canonical switch rate\n(implied combination never trained)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 6000)
    fig.savefig(FIG / "fig2_heldout_causal.pdf")
    plt.close(fig)


def fig_swap(zoo):
    zoo = Path(zoo)
    nets = [(307, "307 (generalises)", ORANGE), (311, "311 (generalises)", YELLOW), (305, "305 (partial)", AQUA), (304, "304 (fails)", BLUE), (310, "310 (fails)", MUTED)]
    data = {}
    for s, _, _ in nets:
        f = zoo / f"mass_only_s{s}" / "audits.jsonl"
        if f.exists():
            a = sorted((json.loads(l) for l in f.read_text().splitlines() if l.strip()), key=lambda r: r["step"])
            if a and a[-1]["step"] == 6000:
                data[s] = a
    if len(data) < 5:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharex=True)
    panels = [("acc_id", "training accuracy"), ("oodM", "declared-variable OOD accuracy"), ("rsi_M", "route-specificity (mass)")]
    for ax, (key, title) in zip(axes, panels):
        for s, lab, col in nets:
            a = [r for r in data[s] if r["step"] <= 3000]
            st = [r["step"] for r in a]
            if key == "oodM":
                y = [statistics.mean(v for k, v in r["ood_combo_acc"].items() if k[0] == "M") for r in a]
            else:
                y = [r[key] for r in a]
            ax.plot(st, y, color=col, lw=1.6, label=lab)
            if key == "rsi_M":
                ax.text(3040, y[-1], lab.split()[0], color=INK, fontsize=6.8, va="center")
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("step")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "fig3_swap_development.pdf")
    plt.close(fig)


def fig_swap_endpoint():
    h = json.loads((ROOT / "runs/swap_recon/historical_endpoints_audit.json").read_text())
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.text(0.98, 0.04, "overlapping points at (0.09, 0.09)\nand (1, 1); n = 24", transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)
    for g, v, col, lab in (("mass_only", "M", ORANGE, "mass declared"), ("acc_only", "A", BLUE, "acceleration declared")):
        xs, ys = [], []
        for n, r in h.items():
            if n.startswith(g):
                xs.append(r[f"iia_{v}_oodsrc"])
                ys.append(statistics.mean(x for k, x in r["ood_combo_acc"].items() if k[0] == v))
        ax.scatter(xs, ys, s=22, color=col, edgecolors="white", linewidths=1.2, label=lab, zorder=3)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=1)
    ax.set_xlabel("declared block carries never-trained source\n(interchange accuracy)")
    ax.set_ylabel("declared-variable OOD accuracy")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(FIG / "fig4_swap_endpoint.pdf")
    plt.close(fig)


def fig_decoder():
    b = ROOT / "runs/phase5/decoder_birth.json"
    t = ROOT / "runs/phase5/tolerance.json"
    if not (b.exists() and t.exists()):
        return
    B = json.loads(b.read_text())
    T_ = json.loads(t.read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    nets = [("307", "307 (generalises)", ORANGE), ("311", "311 (generalises)", YELLOW), ("305", "305 (partial)", AQUA),
            ("304", "304 (fails)", BLUE), ("310", "310 (fails)", MUTED)]
    for s, lab, col in nets:
        st_ = sorted(int(k) for k in B[s] if int(k) <= 2000)
        ax1.plot(st_, [B[s][str(k)]["syn_unseen"] for k in st_], color=col, lw=1.6, label=lab)
    ax1.set_xlabel("step")
    ax1.set_ylabel("decoder alone: accuracy on\nsynthetic unseen inputs")
    ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False, fontsize=6.5, loc="upper left")
    for s, lab, col in nets[:2]:
        st_ = sorted(int(k) for k in T_[s])
        ax2.plot(st_, [T_[s][str(k)]["clean"]["oodM"] for k in st_], color=col, lw=1.6, label=f"{s} clean")
        ax2.plot(st_, [T_[s][str(k)]["inS"]["oodM"] for k in st_], color=col, lw=1.6, ls="--",
                 label=f"{s} + in-subspace interaction")
        ax2.plot(st_, [T_[s][str(k)]["outS"]["oodM"] for k in st_], color=col, lw=0.9, ls=":",
                 label=f"{s} + out-of-subspace control")
    ax2.set_xlabel("checkpoint step")
    ax2.set_ylabel("unseen-mass accuracy")
    ax2.set_ylim(0, 1.02)
    ax2.legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_decoder.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", default="/home/user/swap_recon/zoo")
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    fig_primary()
    fig_i1()
    fig_swap(a.zoo)
    fig_swap_endpoint()
    print(sorted(p.name for p in FIG.iterdir()))


def fig_factorization():
    w = ROOT / "runs/swap_recon/window_factorization.json"
    p = ROOT / "runs/swap_recon/factorization_trajectories.json"
    a = ROOT / "runs/swap_recon/analysis.json"
    if not (w.exists() and p.exists() and a.exists()):
        return
    W_ = json.loads(w.read_text())
    F = json.loads(p.read_text())
    A = json.loads(a.read_text())["networks"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    nets = [("307", "307 (generalises)", ORANGE), ("311", "311 (generalises)", YELLOW), ("305", "305 (partial)", AQUA),
            ("304", "304 (fails)", BLUE), ("310", "310 (fails)", MUTED)]
    for s, lab, col in nets:
        st_ = sorted(int(k) for k in W_[s] if 0 <= int(k) <= 2500)
        ax1.plot(st_, [W_[s][str(k)]["r2_additive"] for k in st_], color=col, lw=1.6, label=lab)
    ax1.set_xlabel("step")
    ax1.set_ylabel("block factorisation (R² of value + unit)")
    ax1.set_ylim(0.6, 1.01)
    ax1.legend(frameon=False, fontsize=6.5, loc="lower left")
    for g, col, lab in (("mass_only", ORANGE, "mass declared"), ("acc_only", BLUE, "acceleration declared")):
        xs, ys, fit = [], [], []
        for n, tr in F.items():
            if n.startswith(g):
                xs.append(tr["6000"]); ys.append(A[n]["outcome"]); fit.append(A[n]["acc_id"] >= 0.95)
        ax2.scatter([x for x, f in zip(xs, fit) if f], [y for y, f in zip(ys, fit) if f], s=22, color=col,
                    edgecolors="white", linewidths=1.2, label=f"{lab}, fits training data", zorder=3)
        ax2.scatter([x for x, f in zip(xs, fit) if not f], [y for y, f in zip(ys, fit) if not f], s=18, facecolors="none",
                    edgecolors=col, linewidths=1.0, label=f"{lab}, does not fit", zorder=3)
    ax2.axhline(0.95, color=MUTED, lw=0.8, ls="--")
    ax2.axvline(0.98, color=MUTED, lw=0.8, ls=":")
    ax2.set_xlabel("final block factorisation")
    ax2.set_ylabel("declared-variable OOD accuracy")
    ax2.set_xlim(0.1, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(0.0, 0.7))
    fig.tight_layout()
    fig.savefig(FIG / "fig5_factorisation.pdf")
    plt.close(fig)


def fig_decoder():
    b = ROOT / "runs/phase5/decoder_birth.json"
    t = ROOT / "runs/phase5/tolerance.json"
    if not (b.exists() and t.exists()):
        return
    B = json.loads(b.read_text())
    T_ = json.loads(t.read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    nets = [("307", "307 (generalises)", ORANGE), ("311", "311 (generalises)", YELLOW), ("305", "305 (partial)", AQUA),
            ("304", "304 (fails)", BLUE), ("310", "310 (fails)", MUTED)]
    for s, lab, col in nets:
        st_ = sorted(int(k) for k in B[s] if int(k) <= 2000)
        ax1.plot(st_, [B[s][str(k)]["syn_unseen"] for k in st_], color=col, lw=1.6, label=lab)
    ax1.set_xlabel("step")
    ax1.set_ylabel("decoder alone: accuracy on\nsynthetic unseen inputs")
    ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False, fontsize=6.5, loc="upper left")
    for s, lab, col in nets[:2]:
        st_ = sorted(int(k) for k in T_[s])
        ax2.plot(st_, [T_[s][str(k)]["clean"]["oodM"] for k in st_], color=col, lw=1.6, label=f"{s} clean")
        ax2.plot(st_, [T_[s][str(k)]["inS"]["oodM"] for k in st_], color=col, lw=1.6, ls="--",
                 label=f"{s} + in-subspace interaction")
        ax2.plot(st_, [T_[s][str(k)]["outS"]["oodM"] for k in st_], color=col, lw=0.9, ls=":",
                 label=f"{s} + out-of-subspace control")
    ax2.set_xlabel("checkpoint step")
    ax2.set_ylabel("unseen-mass accuracy")
    ax2.set_ylim(0, 1.02)
    ax2.legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_decoder.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_factorization()
    fig_decoder()
