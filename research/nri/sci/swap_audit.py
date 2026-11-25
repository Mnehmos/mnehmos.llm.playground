"""Phase III per-checkpoint audit of the reconstructed role-swap networks (doc 41 §41.6 Q4, §41.7).
Runs offline on saved checkpoints (never inside training). All evaluation data come from fixed generators.

Measures (site h = final-token residual before the MLP, where the declared block lives; x1 = layer-1 residual):
  acc_id, acc_ood                      frozen-seed sets (as run_dimensional.acc, n=4096)
  ood_combo_acc                        accuracy per never-trained (variable, value in 9..12) combination
  iia_M, iia_A                         designed-block interchange accuracy (dims 0-15), ID base and source
  iia_M_oodsrc, iia_A_oodsrc           same, but the SOURCE is a never-trained combination for that variable:
                                       does the block carry the canonical value of an unseen input?
  probe_{M,A}_{block,rest,x1val}_{id,ood}
                                       linear (ridge) probe for canonical M / A fitted on ID examples only, tested
                                       on ID (held-out sample) and on OOD examples (the unseen variable only)
  rsi_M, rsi_A                         route-specificity index at h: 1 - R^2 of (unit, value) class means on one-hot
                                       canonical value (ID classes)

    python -m sci.swap_audit <zoo> [--workers N]      # writes <run>/audits.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_dimensional as D  # noqa: E402
import run_swap as RS  # noqa: E402
from nri import dimtask as T  # noqa: E402
from nri.model import TinyTyped  # noqa: E402

N = 4096
SEED_ID, SEED_OOD, SEED_FIT, SEED_SRC = 990_001, 990_002, 990_003, 990_004


def lat(n, seed, split="train"):
    return T.sample_latents(n, torch.Generator().manual_seed(seed), split)


def ood_for(var, n, seed):
    """ID latents where ONLY `var`'s (unit, value) is a never-trained combination (unit 2, value 9..12)."""
    g = torch.Generator().manual_seed(seed)
    L = T.sample_latents(n, g)
    vk, uk = ("m", "mu") if var == "M" else ("a", "au")
    L[uk] = torch.full((n,), T.OOD_UNIT)
    L[vk] = torch.randint(9, 13, (n,), generator=g)
    return L


@torch.no_grad()
def states(m, L):
    b = T.build(L)
    zero = torch.zeros_like(b.tokens)
    x1 = m.resid_layer1(b.tokens, zero)
    h = m.resid_pre_mlp(b.tokens, zero)
    pos = torch.where(L["order"] == 0, 0, 3)                         # position of the mass value token
    return b, x1, h, pos


def ridge_acc(Xtr, ytr, tests, k=T.P):
    """Nearest-class-centroid probe (fitted on ID). NOTE (doc 41R): probes fitted on ID inputs FAIL the known-positive
    check on OOD inputs (0.00 on mass_only_s307 whose block causally carries the OOD canonical value at 0.994), so
    probe_*_ood values are reported as an instrument-validity finding, not as evidence about OOD representation.
    The validated OOD measure is the causal iia_*_oodsrc."""
    Cn = torch.stack([Xtr[ytr == c].mean(0) if (ytr == c).any() else torch.full_like(Xtr[0], float("inf")) for c in range(k)])
    return [float((torch.cdist(Xt, Cn).argmin(1) == yt).float().mean()) for Xt, yt in tests]


def rsi(h, unit, val, canon):
    keys = sorted(set(zip(unit.tolist(), val.tolist())))
    means, q = [], []
    for u, v in keys:
        sel = (unit == u) & (val == v)
        if sel.sum() >= 5:
            means.append(h[sel].mean(0))
            q.append(int(canon[sel][0]))
    M = torch.stack(means).double()
    Mc = M - M.mean(0)
    Q = torch.nn.functional.one_hot(torch.tensor(q), T.P).double()
    beta = torch.linalg.lstsq(Q, Mc).solution
    return 1 - float(((Mc - Q @ beta) ** 2).sum() / (Mc ** 2).sum())


@torch.no_grad()
def audit(m):
    m.eval()
    out = dict(acc_id=D.acc(m, n=N, seed=SEED_ID), acc_ood=D.acc(m, "ood", n=N, seed=SEED_OOD))
    # per never-trained combination
    combo = {}
    for var in ("M", "A"):
        L = ood_for(var, N, SEED_OOD + (1 if var == "M" else 2))
        b = T.build(L)
        pred = m(b.tokens, torch.zeros_like(b.tokens)).argmax(-1)
        vk = "m" if var == "M" else "a"
        for v in range(9, 13):
            sel = L[vk] == v
            combo[f"{var}_u2_v{v}"] = float((pred[sel] == b.y[sel]).float().mean())
    out["ood_combo_acc"] = combo
    # designed-block interchange (ID) and with never-trained sources
    mk = D.mask(*RS.BLOCK)
    base = lat(N, SEED_FIT + 7)
    _, _, hb, _ = states(m, base)
    for var in ("M", "A"):
        for tag, src in (("", lat(N, SEED_SRC)), ("_oodsrc", ood_for(var, N, SEED_SRC + 1))):
            _, _, hs, _ = states(m, src)
            pred = m.head_from_resid(hb * (1 - mk) + hs * mk).argmax(-1)
            out[f"iia_{var}{tag}"] = float((pred == D.counterfactual(base, src, var)).float().mean())
    # probes for canonical M / A: fit on ID, test on held-out ID and on OOD-for-that-variable
    fit, tid = lat(N, SEED_FIT), lat(N, SEED_FIT + 1)
    _, x1f, hf, pf = states(m, fit)
    _, x1t, ht, pt = states(m, tid)
    Mf, Af, _ = T.canon(fit)
    Mt, At, _ = T.canon(tid)
    ar = torch.arange(N)
    for var, yf, yt in (("M", Mf, Mt), ("A", Af, At)):
        Lo = ood_for(var, N, SEED_OOD + 10)
        _, x1o, ho, po = states(m, Lo)
        Mo, Ao, _ = T.canon(Lo)
        yo = Mo if var == "M" else Ao
        for site, F_, T_, O_ in (("block", hf[:, :16], ht[:, :16], ho[:, :16]),
                                 ("rest", hf[:, 16:], ht[:, 16:], ho[:, 16:])):
            a_id, a_ood = ridge_acc(F_, yf, [(T_, yt), (O_, yo)])
            out[f"probe_{var}_{site}_id"], out[f"probe_{var}_{site}_ood"] = a_id, a_ood
        # x1 at the variable's own value token (mass token pos for M; the other value slot for A)
        pf_v = pf if var == "M" else 3 - pf
        pt_v = pt if var == "M" else 3 - pt
        po_v = po if var == "M" else 3 - po
        a_id, a_ood = ridge_acc(x1f[ar, pf_v], yf, [(x1t[ar, pt_v], yt), (x1o[ar, po_v], yo)])
        out[f"probe_{var}_x1val_id"], out[f"probe_{var}_x1val_ood"] = a_id, a_ood
        # x1 at the value token after the unit token has been attended: use the unit position (value pos + 1)
        a_id, a_ood = ridge_acc(x1f[ar, pf_v + 1], yf, [(x1t[ar, pt_v + 1], yt), (x1o[ar, po_v + 1], yo)])
        out[f"probe_{var}_x1unit_id"], out[f"probe_{var}_x1unit_ood"] = a_id, a_ood
    big = lat(16384, SEED_FIT + 2)
    _, _, hbig, _ = states(m, big)
    Mb, Ab, _ = T.canon(big)
    out["rsi_M"] = rsi(hbig, big["mu"], big["m"], Mb)
    out["rsi_A"] = rsi(hbig, big["au"], big["a"], Ab)
    return out


def audit_run(run):
    torch.set_num_threads(1)
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else None
    group = run.name.rsplit("_s", 1)[0]
    m = TinyTyped(group, **RS.MODEL_KW)
    done = set()
    path = run / "audits.jsonl"
    if path.exists():
        done = {json.loads(l)["step"] for l in path.read_text().splitlines() if l.strip()}
    with open(path, "a") as f:
        for d in sorted((run / "ckpt").iterdir()):
            step = int(d.name.split("_")[1])
            if step in done or not (d / "COMPLETE").exists():
                continue
            m.load_state_dict(load_file(str(d / "model.safetensors")))
            r = audit(m)
            r.update(step=step, run=run.name)
            f.write(json.dumps(r) + "\n")
            f.flush()
    return run.name


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("zoo")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    runs = sorted(p for p in Path(a.zoo).iterdir() if (p / "ckpt").exists() and (not a.only or p.name in a.only))
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(audit_run, runs):
            print("[audited]", r, flush=True)
