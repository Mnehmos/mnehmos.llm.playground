"""Doc 57 experiment R: why constructed networks are fragile, and constructed (training-free) remedies.

Noise models: (a) additive, sigma * RMS(tensor); (b) multiplicative w(1 + sigma eps); (c) additive on h, relative to |h|.
Remedies: R1 balanced q/k scales + minimal attention gain (argmax-exact); R2 flat-top (trapezoid) ladder bumps;
R3 K redundant ladder copies; R4 readout margin scaling; combinations.

    python -m sci.robust      # -> runs/phase7/robust.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import construct as C  # noqa: E402
from sci import instructions as INS  # noqa: E402
from sci import rescue as R  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase7"
SIGMAS = (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3)
COMPONENTS = {"embed": ("E_c", "E_p", "E_t"), "attn1_qkv": ("attn1.qkv",), "attn1_o": ("attn1.o",), "attn2_qkv": ("attn2.qkv",),
              "attn2_o": ("attn2.o",), "W_in": ("W_in",), "W_out": ("W_out",), "W_U": ("W_U",)}
L_ = C.domain()
B_ = T.build(L_)
Y_ = B_.y


@torch.no_grad()
def acc(m):
    return float((m(B_.tokens, torch.zeros_like(B_.tokens)).argmax(-1) == Y_).float().mean())


@torch.no_grad()
def noisy(m, sigma, kind, seed, only=None):
    e = copy.deepcopy(m)
    g = torch.Generator().manual_seed(seed)
    for name, p in e.named_parameters():
        if only and not any(name.startswith(o) for o in only):
            continue
        z = torch.randn(p.shape, generator=g)
        if kind == "a":
            p.add_(z * sigma * p.pow(2).mean().sqrt())
        elif kind == "b":
            p.mul_(1 + sigma * z)
    return e


@torch.no_grad()
def h_noise_acc(m, sigma, seed):
    g = torch.Generator().manual_seed(seed)
    h = m.resid_pre_mlp(B_.tokens, torch.zeros_like(B_.tokens))
    z = torch.randn(h.shape, generator=g)
    z = z / z.norm(dim=1, keepdim=True) * h.norm(dim=1, keepdim=True) * sigma
    return float((m.head_from_resid(h + z).argmax(-1) == Y_).float().mean())


def curve(m, kind, only=None, seeds=5):
    out = {}
    for s in SIGMAS:
        vals = [h_noise_acc(m, s, k) if kind == "c" else acc(noisy(m, s, kind, k, only)) for k in range(seeds)]
        out[s] = sum(vals) / len(vals)
    return out


@torch.no_grad()
def single_weight_dependence(m, max_n=4000):
    base = m(B_.tokens, torch.zeros_like(B_.tokens)).argmax(-1)
    nz = [(n, i) for n, p in m.named_parameters() for i in torch.nonzero(p.flatten()).flatten().tolist()]
    nz = nz[:max_n]
    crit = 0
    for n, i in nz:
        e = copy.deepcopy(m)
        dict(e.named_parameters())[n].data.view(-1)[i] = 0
        crit += int((e(B_.tokens, torch.zeros_like(B_.tokens)).argmax(-1) != base).any())
    return dict(n_nonzero_tested=len(nz), frac_critical=crit / max(len(nz), 1))


# ------------------------------------------------------------------ remedies
@torch.no_grad()
def rebalance(m):
    """R1a: split each head's score gain equally between its query and key rows (scores unchanged exactly)."""
    e = copy.deepcopy(m)
    for attn in (e.attn1, e.attn2):
        W, b = attn.qkv.weight.data, attn.qkv.bias.data
        d = e.d
        for h in range(attn.h):
            q = slice(16 * h, 16 * h + 16); k = slice(d + 16 * h, d + 16 * h + 16)
            nq = torch.cat([W[q].abs().flatten(), b[q].abs()]).max()
            nk = torch.cat([W[k].abs().flatten(), b[k].abs()]).max()
            if nq > 0 and nk > 0:
                c = (nk / nq).sqrt()
                W[q] *= c; b[q] *= c; W[k] /= c; b[k] /= c
    return e


@torch.no_grad()
def scale_gain(m, gamma):
    e = copy.deepcopy(m)
    for attn in (e.attn1, e.attn2):
        attn.qkv.weight.data[:e.d] *= gamma
        attn.qkv.bias.data[:e.d] *= gamma
    return e


def minimal_gain(m):
    """smallest gamma (on a log grid) keeping argmax exact on all 9,126 prompts."""
    best = 1.0
    for gamma in [10 ** (-k / 8) for k in range(1, 40)]:
        if acc(scale_gain(m, gamma)) == 1.0:
            best = gamma
        else:
            break
    return best


def trapezoid_mlp(m, op, copies=1, width=0.25):
    """R2/R3: flat-top bumps b_k(t) = (1/w')[relu(t-k+a) - relu(t-k+c) - relu(t-k-c) + relu(t-k-a)], a = 0.5 + w, c = 0.5 - w
    -> 1 on [k - c, k + c], 0 outside (k - a, k + a); exact at integers; `copies` redundant ladders, logits averaged."""
    e = copy.deepcopy(m)
    D = C.DIM
    vals = [a + b + c for a in op.phiM for b in op.phiA for c in op.phiS]
    tmin, tmax = min(vals), max(vals)
    a_, c_ = 0.5 + width, 0.5 - width
    B = float(tmax - tmin + 10)
    Win, bin_, Wout = e.W_in.weight.data, e.W_in.bias.data, e.W_out.weight.data
    Win.zero_(); bin_.zero_(); Wout.zero_()
    per = 4 * (tmax - tmin + 1) + (2 if op.gate else 0)
    assert per * copies <= e.d_mlp, (per, copies)
    u = 0
    for _ in range(copies):
        for k in range(tmin, tmax + 1):
            cls = D["OUT"][op.g(k)]
            for off, sgn in ((-a_, 1), (-c_, -1), (c_, -1), (a_, 1)):
                for i in range(P):
                    Win[u, D["Q"][i]] = op.phiM[i]; Win[u, D["A"][i]] = op.phiA[i]
                for f in range(3):
                    Win[u, D["FU"][f]] = op.phiS[f]
                if "M" in op.gate: Win[u, D["Q"][0]] -= B
                if "A" in op.gate: Win[u, D["A"][0]] -= B
                bin_[u] = -(k + off)
                Wout[cls, u] += sgn / (a_ - c_) / copies
                u += 1
        if op.gate:
            for off in (0.0, -1.0):
                if "M" in op.gate: Win[u, D["Q"][0]] = 1
                if "A" in op.gate: Win[u, D["A"][0]] = 1
                bin_[u] = off
                Wout[D["OUT"][op.zero_out], u] += (1 if off == 0 else -1) / copies
                u += 1
    return e


def scale_readout(m, kappa):
    e = copy.deepcopy(m)
    e.W_out.weight.data *= kappa
    return e


def exact_check(m):
    """exhaustive accuracy + fraction of logits exactly one-hot (activation-exactness indicator)."""
    with torch.no_grad():
        lg = m(B_.tokens, torch.zeros_like(B_.tokens))
    oh = torch.nn.functional.one_hot(Y_, P).float()
    return dict(acc=float((lg.argmax(-1) == Y_).float().mean()), logits_exactly_onehot=float((lg == oh).all(1).float().mean()),
                min_margin=float((lg.gather(1, Y_[:, None]).squeeze(1) - lg.scatter(1, Y_[:, None], -1e9).max(1).values).min()))


def instruction_check(m):
    """re-run the Phase VI instruction-system edits that touch the encoder and decoding on the robust variant."""
    ok = {}
    for (role, i) in ((("MU", 1)), ("AU", 2)):
        for s_new in (3, 11):
            e = INS.unit(m, role, i, s_new)
            sp = INS.Spec(); sp.scales[role][i] = s_new
            ok[f"unit {role}{i}->{s_new}"] = INS.check(e, sp)["acc"]
    for k in ("F+1", "F^-1"):
        sp = INS.Spec(); sp.out = INS.OUTOPS[k]
        ok[f"out {k}"] = INS.check(INS.out(m, INS.OUTOPS[k]), sp)["acc"]
    return ok


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    base, _ = C.build_C()
    op = C.OPS["MAS"]
    # reference: learned generalisers
    for n in ("mass_only_s307", "mass_only_s311"):
        m = R.load(n)
        res[n] = dict(kind="learned", a=curve(m, "a"), b=curve(m, "b"), c=curve(m, "c"))
        print(n, {k: {s: round(v, 3) for s, v in res[n][k].items()} for k in "abc"}, flush=True)
    # R0 diagnosis on Design C
    r0 = dict(kind="constructed", a=curve(base, "a"), b=curve(base, "b"), c=curve(base, "c"),
              components_a={c: curve(base, "a", only=v) for c, v in COMPONENTS.items()},
              components_b={c: curve(base, "b", only=v) for c, v in COMPONENTS.items()},
              single_weight=single_weight_dependence(base), exact=exact_check(base))
    res["designC"] = r0
    print("designC a", {s: round(v, 3) for s, v in r0["a"].items()}, "| b", {s: round(v, 3) for s, v in r0["b"].items()}, "| c", {s: round(v, 3) for s, v in r0["c"].items()}, flush=True)
    for c in COMPONENTS:
        print("   comp", c, "a", {s: round(v, 2) for s, v in r0["components_a"][c].items()}, "| b", {s: round(v, 2) for s, v in r0["components_b"][c].items()}, flush=True)
    print("   single-weight", r0["single_weight"], flush=True)
    # remedies
    bal = rebalance(base)
    gam = minimal_gain(bal)
    r1 = scale_gain(bal, gam)
    variants = {"R1a_balanced": bal, "R1_balanced_min_gain": r1, "R2_trapezoid": trapezoid_mlp(base, op),
                "R3_redundant_K": trapezoid_mlp(base, op, copies=256 // (4 * 30 + 2)),
                "R4_margin_x10": scale_readout(base, 10.0)}
    variants["R1+R2+R3"] = trapezoid_mlp(r1, op, copies=256 // (4 * 30 + 2))
    res["R1_gamma"] = gam
    for k, m in variants.items():
        res[k] = dict(kind="constructed", exact=exact_check(m), a=curve(m, "a"), b=curve(m, "b"), c=curve(m, "c"),
                      single_weight=single_weight_dependence(m), instructions=instruction_check(m))
        print(k, "exact", res[k]["exact"], "| a", {s: round(v, 3) for s, v in res[k]["a"].items()}, "| b", {s: round(v, 3) for s, v in res[k]["b"].items()},
              "| c", {s: round(v, 3) for s, v in res[k]["c"].items()}, "| sw", res[k]["single_weight"], "| instr", res[k]["instructions"], flush=True)
        (OUT / "robust.json").write_text(json.dumps(res, indent=1))
    (OUT / "robust.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
