"""Doc 57 experiment D: the direct residual pathway.

D1 exact logit decomposition: W_U b_out + W_U h (split by embedding / layer-1 / layer-2 heads / layer-2 bias) + sum of
   neuron writes a_i q_i.
D2 coordinated relabelling F -> F^5: refit the MLP writes (W_out, b_out) by least squares on familiar prompts to the
   sigma-relabelled full logits, direct path left as is; vs W_U row permutation.
D3 internal operation F(M^-1, A, S): same refit, targets = the network's own logits with the original answer and the new
   answer swapped per example (an internal change, not sigma o F).

    python -m sci.direct_path     # -> runs/phase7/direct_path.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import construct as C  # noqa: E402
from sci import neuron_lens as N  # noqa: E402
from sci import rescue as R  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase7"
SC = dict(MU=T.MASS_S, AU=T.ACC_S)


def margin_share(part, y, ystar):
    return (part.gather(1, y[:, None]) - part.gather(1, ystar[:, None])).squeeze(1)


@torch.no_grad()
def analyse(name, m=None):
    L = C.domain()
    un = C.unseen_mask(L)
    m = m or R.load(name)
    P_ = N.parts(m, L)
    y = P_["y"]
    WU = m.W_U.weight.double()
    full = P_["logits"].double()
    ystar = N.competitor(P_["logits"], y)
    mlp = P_["a"].double() @ P_["q"].double().T
    parts = {"bias_b_out": (WU @ m.W_out.bias.double()).expand(len(y), -1)}
    for k, v in P_["comp"].items():
        parts[f"direct_{k}"] = v.double() @ WU.T
    parts["mlp_neurons"] = mlp
    recon = sum(parts.values())
    out = dict(kind="learned" if m is None or name.startswith("mass") else "constructed",
               reconstruction_max_abs_err=float((recon - full).abs().max()))
    total_margin = margin_share(full, y, ystar).mean()
    D1 = {}
    for k, v in parts.items():
        D1[k] = dict(class_spread=float((v.max(1).values - v.min(1).values).mean()),
                     mean_margin=float(margin_share(v, y, ystar).mean()), margin_share=float(margin_share(v, y, ystar).mean() / total_margin),
                     corr_with_mlp=float(torch.corrcoef(torch.stack([v.flatten(), mlp.flatten()]))[0, 1]) if k != "mlp_neurons" and v.std() > 0 else None)
    direct = sum(v for k, v in parts.items() if k.startswith("direct_"))
    D1["direct_total"] = dict(class_spread=float((direct.max(1).values - direct.min(1).values).mean()),
                              mean_margin=float(margin_share(direct, y, ystar).mean()),
                              corr_with_mlp=float(torch.corrcoef(torch.stack([direct.flatten(), mlp.flatten()]))[0, 1]),
                              direct_only_acc=float((direct.argmax(-1) == y).double().mean()))
    out["D1"] = D1
    # D2 relabel
    sigma = torch.tensor([pow(c, 5, P) for c in range(P)])
    inv = torch.empty(P, dtype=torch.long); inv[sigma] = torch.arange(P)
    tgt = sigma[y]
    A1 = torch.cat([P_["a"].double(), torch.ones(len(y), 1, dtype=torch.float64)], 1)
    def refit(Tlog, target):
        Z = torch.linalg.lstsq(A1[~un], (Tlog - direct)[~un], driver="gelsd").solution   # 257 x 13 = [W_U W_out' ; W_U b_out']
        pr = (direct + A1 @ Z).argmax(-1)
        return dict(familiar=float((pr[~un] == target[~un]).double().mean()), unseen=float((pr[un] == target[un]).double().mean()),
                    fit_r2_familiar=float(1 - (((direct + A1 @ Z) - Tlog)[~un] ** 2).sum() / ((Tlog[~un] - Tlog[~un].mean(0)) ** 2).sum()))
    out["D2_relabel_F5"] = dict(W_U_permutation=dict(familiar=float((full[:, inv].argmax(-1)[~un] == tgt[~un]).double().mean()),
                                                     unseen=float((full[:, inv].argmax(-1)[un] == tgt[un]).double().mean())),
                                W_out_only_relabel_writes=dict(familiar=float(((direct + mlp[:, inv]).argmax(-1)[~un] == tgt[~un]).double().mean()),
                                                               unseen=float(((direct + mlp[:, inv]).argmax(-1)[un] == tgt[un]).double().mean())),
                                W_out_refit_coordinated=refit(full[:, inv], tgt))
    # D3 internal operation
    M, A, S = C.canon_scales(L, SC)
    Minv = torch.tensor([pow(int(x), -1, P) if x else 0 for x in M])
    t3 = T.target(Minv, A, S)
    T3 = full.clone()
    idx = torch.arange(len(y))
    T3[idx, t3], T3[idx, y] = full[idx, y], full[idx, t3]
    out["D3_internal_Minv"] = dict(W_out_refit_coordinated=refit(T3, t3),
                                   is_relabelling=False)
    out["acc"] = dict(familiar=float((full.argmax(-1)[~un] == y[~un]).double().mean()), unseen=float((full.argmax(-1)[un] == y[un]).double().mean()))
    return out


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for s in (307, 311, 300, 305, 302, 304, 310):
        n = f"mass_only_s{s}"
        res[n] = r = analyse(n)
        d = r["D1"]
        print(n, "recon err", f"{r['reconstruction_max_abs_err']:.1e}",
              "| margin share:", {k: round(v["margin_share"], 2) for k, v in d.items() if "margin_share" in v},
              "| spread:", {k: round(v["class_spread"], 1) for k, v in d.items()},
              "| corr:", {k: round(v["corr_with_mlp"], 2) for k, v in d.items() if v.get("corr_with_mlp") is not None},
              "| D2", {k: {a: round(b, 3) for a, b in v.items()} for k, v in r["D2_relabel_F5"].items()},
              "| D3", {a: round(b, 3) for a, b in r["D3_internal_Minv"]["W_out_refit_coordinated"].items()}, flush=True)
    m, _ = C.build_C()
    res["designC"] = r = analyse("designC", m)
    print("designC", {k: {a: round(b, 3) for a, b in v.items()} for k, v in r["D2_relabel_F5"].items()}, "D3", r["D3_internal_Minv"])
    (OUT / "direct_path.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
