"""Doc 53 §53.3: exhaustive qualification of constructed networks (Q-acc, Q-zero, Q-inv, Q-iit, Q-data, Q-exact).

    python -m sci.construct_qualify            # -> runs/phase6/qualification.json
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import construct as C  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase6"
TASK_SCALES = dict(MU=T.MASS_S, AU=T.ACC_S)


@torch.no_grad()
def trace(m, L, dtype):
    m = m.to(dtype)
    b = T.build(L)
    x0 = m.embed(b.tokens, torch.zeros_like(b.tokens))
    a1, p1 = m.attn1(x0, return_pattern=True)
    x1 = x0 + a1
    a2, p2 = m.attn2(x1, return_pattern=True)
    h = (x1 + a2)[:, -1]
    pre = h @ m.W_in.weight.T + m.W_in.bias
    hid = torch.relu(pre)
    logits = m.W_U(h + m.W_out(hid))
    return dict(x1=x1, p1=p1, p2=p2, h=h, pre=pre, hid=hid, logits=logits)


def _dev_from(x, allowed):
    """max distance of each entry from the nearest allowed value."""
    d = torch.stack([(x - a).abs() for a in allowed]).min(0).values
    return float(d.max())


def exactness_C(m, L, op, dtype):
    tr = trace(m, L, dtype)
    D = C.DIM
    M, A, S = C.canon_scales(L, TASK_SCALES)
    y = C.target(op, M, A, S)
    n = len(y)
    # attention rows that are read downstream: layer 1 at unit positions (1, 4), x (2), force (6), '=' (7);
    # layer 2 at '=' (7)
    rows1 = tr["p1"][:, :, [1, 2, 4, 6, 7], :]
    rows2 = tr["p2"][:, :3, 7, :]
    sym_h = torch.zeros(n, 64, dtype=torch.float64)
    sym_h[torch.arange(n), torch.tensor(D["Q"])[M]] = 1
    sym_h[torch.arange(n), torch.tensor(D["A"])[A]] = 1
    sym_h[torch.arange(n), torch.tensor(D["FU"])[L["fu"]]] = 1
    used = D["Q"] + D["A"] + D["FU"] + D["MU"] + D["AU"] + [D["NULL"]] + D["OUT"]
    h_used_equal = bool((tr["h"][:, used].double() == sym_h[:, used]).all())
    # symbolic pre-activations in exact integer arithmetic
    Wi = m.W_in.weight.double()
    assert (Wi == Wi.round()).all() and (m.W_in.bias.double() == m.W_in.bias.double().round()).all()
    pre_sym = sym_h[:, used].round().long() @ Wi[:, used].round().long().T + m.W_in.bias.double().round().long()
    pre_equal = bool((tr["pre"].double() == pre_sym.double()).all())
    onehot = torch.nn.functional.one_hot(y, P).to(tr["logits"].dtype)
    logits_equal = bool((tr["logits"] == onehot).all())
    return dict(dtype=str(dtype), attention_layer1_max_dev_from_0_1=_dev_from(rows1, (0.0, 1.0)),
                attention_layer2_max_dev_from_0_half_1=_dev_from(rows2, (0.0, 0.5, 1.0)),
                h_regions_equal_symbolic=h_used_equal, preactivations_equal_integer_symbolic=pre_equal,
                logits_exactly_onehot_target=logits_equal,
                logits_max_abs_error=float((tr["logits"].double() - onehot.double()).abs().max()),
                min_margin=float((tr["logits"].gather(1, y[:, None]).squeeze(1)
                                  - tr["logits"].masked_fill(onehot.bool(), -1e9).max(1).values).min()))


@torch.no_grad()
def head_from_h(m, h):
    return m.W_U(h + m.W_out(torch.relu(h @ m.W_in.weight.T + m.W_in.bias)))


@torch.no_grad()
def iit_C(m, L, op):
    D = C.DIM
    tr = trace(m, L, torch.float32)
    M, A, S = C.canon_scales(L, TASK_SCALES)
    base_y = C.target(op, M, A, S)
    res = {}
    for var, dims, k in (("M", D["Q"], P), ("A", D["A"], P), ("S", D["FU"], 3)):
        ok = disc_ok = disc_n = 0
        for src in range(k):
            h = tr["h"].clone()
            h[:, dims] = 0
            h[:, dims[src]] = 1
            pred = head_from_h(m, h).argmax(-1)
            if var == "M":
                want = C.target(op, torch.full_like(M, src), A, S)
            elif var == "A":
                want = C.target(op, M, torch.full_like(A, src), S)
            else:
                want = C.target(op, M, A, torch.full_like(S, C.SINV[src]))
            ok += int((pred == want).sum())
            d = want != base_y
            disc_n += int(d.sum())
            disc_ok += int((pred[d] == want[d]).sum())
        res[var] = dict(n=len(M) * k, acc=ok / (len(M) * k), discriminating_n=disc_n,
                        discriminating_acc=disc_ok / max(disc_n, 1))
    return res


def invariance_C(m, L):
    D = C.DIM
    tr = trace(m, L, torch.float32)
    M, A, S = C.canon_scales(L, TASK_SCALES)
    res = {}
    for var, dims, key in (("M", D["Q"], M), ("A", D["A"], A), ("S", D["FU"], L["fu"])):
        blk = tr["h"][:, dims]
        same = all(bool((blk[key == v] == blk[key == v][0]).all()) for v in key.unique())
        distinct = len({tuple(blk[key == v][0].tolist()) for v in key.unique()}) == len(key.unique())
        res[var] = dict(bitwise_identical_within_class=same, distinct_between_classes=distinct,
                        n_classes=int(len(key.unique())))
    return res


def q_data():
    """Construct with the OOD split removed from the task module; weights must be bitwise identical."""
    m1, _ = C.build_C()
    keep = (T.OOD_VALUES, T.OOD_UNIT)
    T.OOD_VALUES, T.OOD_UNIT = (), None
    try:
        m2, _ = C.build_C()
    finally:
        T.OOD_VALUES, T.OOD_UNIT = keep
    same = all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    src = (ROOT / "sci/construct.py").read_text()
    reads = [w for w in ("load_file", "zoo", "R.load", "sample_latents", "safetensors", "Generator") if w in src]
    return dict(weights_identical_without_ood_split=same, data_access_tokens_in_constructor=reads)


def qualify_C(op_name="MAS"):
    op = C.OPS[op_name]
    m, info = C.build_C(op)
    L = C.domain()
    un = C.unseen_mask(L)
    M, A, S = C.canon_scales(L, TASK_SCALES)
    y = C.target(op, M, A, S)
    pred = C.run(m, L)["pred"]
    zero_in = (L["m"] == 0) | (L["a"] == 0)
    out = dict(kind="constructed", design="C", op=op.name, compile=info,
               nonzero_params=int(sum((p != 0).sum() for p in m.parameters())), total_params=m.n_params(),
               acc_all=float((pred == y).float().mean()), acc_familiar=float((pred[~un] == y[~un]).float().mean()),
               acc_unseen=float((pred[un] == y[un]).float().mean()), n_familiar=int((~un).sum()), n_unseen=int(un.sum()))
    if op.gate:
        out["zero"] = dict(zero_inputs_to_zero=float((pred[zero_in] == 0).float().mean()),
                           nonzero_inputs_never_zero=bool((pred[~zero_in] != 0).all()))
    out["exact_float32"] = exactness_C(m, L, op, torch.float32)
    out["exact_float64"] = exactness_C(m, L, op, torch.float64)
    out["iit"] = iit_C(m, L, op)
    out["invariance"] = invariance_C(m, L)
    return out


# ------------------------------------------------------------------ Design L
@torch.no_grad()
def qualify_L(op_name="MAS"):
    op = C.OPS[op_name]
    m, info = C.build_L(op)
    L = C.domain()
    un = C.unseen_mask(L)
    M, A, S = C.canon_scales(L, TASK_SCALES)
    y = C.target(op, M, A, S)
    tr = trace(m, L, torch.float32)
    pred = tr["logits"].argmax(-1)
    D = C.DL
    lift = tr["h"][:, D["HM"]]
    nz = M != 0
    lm = torch.tensor([C.DLOG[int(x)] for x in M[nz]])
    inv_mod12 = bool(((lift[nz].round().long() % 12) == lm).all())
    inv_bitwise = all(len({float(v) for v in lift[(M == c)]}) == 1 for c in M.unique())
    # interchange with every distinct (lift, zero) state that occurs for the source
    states = {}
    for i in range(len(M)):
        states.setdefault(int(M[i]), set()).add((float(tr["h"][i, D["HM"]]), float(tr["h"][i, D["HZM"]])))
    ok = n = 0
    for src, sts in states.items():
        for (lv, zv) in sts:
            h = tr["h"].clone()
            h[:, D["HM"]], h[:, D["HZM"]] = lv, zv
            p = head_from_h(m, h).argmax(-1)
            want = C.target(op, torch.full_like(M, src), A, S)
            ok += int((p == want).sum()); n += len(M)
    onehot = torch.nn.functional.one_hot(y, P).float()
    return dict(kind="constructed", design="L", op=op.name, compile=info,
                nonzero_params=int(sum((p != 0).sum() for p in m.parameters())),
                acc_all=float((pred == y).float().mean()), acc_unseen=float((pred[un] == y[un]).float().mean()),
                acc_familiar=float((pred[~un] == y[~un]).float().mean()),
                M_state_equal_mod12=inv_mod12, M_state_bitwise_invariant=inv_bitwise,
                n_distinct_M_states=sum(len(s) for s in states.values()),
                iit_M=dict(n=n, acc=ok / n), logits_exactly_onehot=bool((tr["logits"] == onehot).all()),
                attention_layer2_eq_row_exact_1_8=bool((tr["p2"][:, 0, 7, :] == 0.125).all()),
                attention_layer1_max_dev=_dev_from(tr["p1"][:, :2, [1, 2, 4, 6, 7], :], (0.0, 1.0)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(C={k: qualify_C(k) for k in C.OPS}, L={k: qualify_L(k) for k in ("MAS", "M2AS", "MA2S", "M/A")},
               q_data=q_data())
    (OUT / "qualification.json").write_text(json.dumps(res, indent=1))
    for k, v in res["C"].items():
        print("C", k, v["compile"], "acc", v["acc_familiar"], v["acc_unseen"], "| f32", {a: b for a, b in v["exact_float32"].items() if a != "dtype"},
              "| iit", {a: (round(b["acc"], 4), round(b["discriminating_acc"], 4)) for a, b in v["iit"].items()},
              "| inv", {a: b["bitwise_identical_within_class"] for a, b in v["invariance"].items()}, "| zero", v.get("zero"))
        print("   f64", {a: b for a, b in v["exact_float64"].items() if a != "dtype"})
    for k, v in res["L"].items():
        print("L", k, {a: b for a, b in v.items() if a not in ("kind", "design")})
    print("Q-data", res["q_data"])


if __name__ == "__main__":
    main()
