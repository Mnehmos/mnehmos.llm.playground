"""Examiner-side sampling, reparameterization and controls for the symbolic pilot."""
import itertools

import torch

import run_dimensional as D
import run_swap as S
from nri import dimtask as T


def universe():
    """Enumerate every distinct token input, including the original OOD combinations."""
    keys = ("m", "mu", "a", "au", "fu", "order")
    sizes = (T.P, 3, T.P, 3, 3, 2)
    cols = torch.cartesian_prod(*(torch.arange(size) for size in sizes)).T
    return dict(zip(keys, cols))


def choose_split(rng):
    # Randomize *semantic* questions, grouping all units and word orders for each (M,A).
    # Leave a third pool unused so the test is not simply the complement of calibration.
    pairs = list(itertools.product(range(T.P), repeat=2))
    while True:
        rng.shuffle(pairs)
        fit, test = pairs[:84], pairs[84:126]
        if all(len({p[i] for p in part}) == T.P for part in (fit, test) for i in (0, 1)):
            return sorted(fit), sorted(test)


def split_rows(pairs):
    lat = universe()
    M, A, _ = T.canon(lat)
    allowed = torch.zeros(T.P, T.P, dtype=torch.bool)
    ms, accs = zip(*pairs)
    allowed[list(ms), list(accs)] = True
    mask = allowed[M, A]
    for value, unit in (("m", "mu"), ("a", "au")):
        mask &= ~((lat[unit] == T.OOD_UNIT) & torch.isin(lat[value], torch.tensor(T.OOD_VALUES)))
    return {name: col[mask] for name, col in lat.items()}


def calibration(pairs):
    lat = split_rows(pairs)
    batch = T.build(lat)
    M, A, S_out = T.canon(lat)
    return {"tokens": batch.tokens, "M": M, "A": A, "S": S_out, "force": batch.y}


def permute_residual(state, order):
    """For row activations h, the new coordinate vector is h[:, order]. QKV head order is fixed."""
    axes = {"mask_c": 0, "mask_t": 0, "E_c.weight": 1, "E_t.weight": 1, "E_p.weight": 1,
            "aux.weight": 1, "W_in.weight": 1, "W_out.weight": 0, "W_out.bias": 0,
            "W_U.weight": 1}
    unchanged = {"W_in.bias", "aux.bias"}
    for layer in ("attn1", "attn2"):
        axes.update({f"{layer}.qkv.weight": 1, f"{layer}.o.weight": 0, f"{layer}.o.bias": 0})
        unchanged.add(f"{layer}.qkv.bias")
    if set(state) != axes.keys() | unchanged or sorted(order) != list(range(64)):
        raise ValueError("unexpected architecture or residual permutation")
    idx = torch.tensor(order)
    return {k: v.index_select(axes[k], idx) if k in axes else v.clone() for k, v in state.items()}


def logits(model, tokens):
    return torch.cat([model(t, torch.zeros_like(t)) for t in tokens.split(256)])


def resid(model, tokens):
    return torch.cat([model.resid_pre_mlp(t, torch.zeros_like(t)) for t in tokens.split(256)])


def edit_scores(pred, original, lat, var):
    result = S.score(pred, lat, var, "inv")
    f0, fm, fa = S.targets(lat, "inv")
    unchanged = f0 == (fm if var == "M" else fa)
    result["n_math_unchanged"] = int(unchanged.sum())
    result["agreement_where_math_unchanged"] = (pred[unchanged] == original[unchanged]).float().mean().item()
    return result


def swap_pairs(lat, h, pair_seed):
    # A donor changes both operands. Holding its output unit fixed lets us reject simply
    # copying the entire donor answer. Keep only four-way distinguishable trials.
    rng = torch.Generator().manual_seed(pair_seed)
    donors = torch.arange(len(h))
    for fu, order in itertools.product(range(3), range(2)):
        group = torch.where((lat["fu"] == fu) & (lat["order"] == order))[0]
        group = group[torch.randperm(len(group), generator=rng)]
        donors[group] = group.roll(1)
    src = {k: v[donors] for k, v in lat.items()}
    answers = torch.stack([T.build(lat).y, D.counterfactual(lat, src, "M"),
                           D.counterfactual(lat, src, "A"), T.build(src).y], 1)
    ok = (answers.sort(1).values.diff(dim=1) != 0).all(1)
    if not ok.any():
        raise ValueError("no discriminating donor pairs")
    return h[ok], h[donors][ok], answers[ok]


def swap_rate(model, pairs, coordinates, var):
    base, donor, answers = pairs
    patched = base.clone()
    patched[:, coordinates] = donor[:, coordinates]
    pred = model.head_from_resid(patched).argmax(-1)
    intended = answers[:, 1 if var == "M" else 2]
    return (pred == intended).float().mean().item()


def python_baseline(model, cal_h, values, coordinates, test_h):
    """Fit doc 14's subspace affine inversion using only participant calibration rows."""
    z = cal_h[:, coordinates]
    means = torch.stack([z[values == i].mean(0) for i in range(T.P)])
    moves = means[S.INV[values]] - means[values]
    center = z.mean(0)
    X = torch.cat([z - center, torch.ones(len(z), 1)], 1)
    coeff = torch.linalg.lstsq(X, moves, driver="gelsd").solution
    # Same write path as folding the affine adjustment into W_in; leave residual skip intact.
    delta = (test_h[:, coordinates] - center) @ coeff[:-1] + coeff[-1]
    pre = model.W_in(test_h) + delta @ model.W_in.weight[:, coordinates].T
    return model.W_U(test_h + model.W_out(pre.relu())).argmax(-1)


def baseline_coordinates(h, values, width):
    """Rank axes by fraction of variation explained by the named calibration value."""
    means = torch.stack([h[values == i].mean(0) for i in range(T.P)])
    return (means[values].var(0) / h.var(0).clamp_min(1e-12)).topk(width).indices.tolist()
