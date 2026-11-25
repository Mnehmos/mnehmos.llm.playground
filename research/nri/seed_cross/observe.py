"""Read-only historical role-swap measurements. All examples and generator seeds are fixed."""

import hashlib
import itertools
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

NRI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NRI))
import run_dimensional as D
import run_swap as R
from nri import dimtask as T


def oracle(lat):
    # Independent scalar specification, including the inverse output scale modulo thirteen.
    return torch.tensor(
        [
            (m * (1, 5, 8)[mu] * a * (1, 2, 7)[au] * pow((1, 6, 11)[fu], -1, 13)) % 13
            for m, mu, a, au, fu in zip(
                *(lat[k].tolist() for k in ("m", "mu", "a", "au", "fu"))
            )
        ]
    )


def enumerate_domain():
    rows = list(
        itertools.product(range(13), range(3), range(13), range(3), range(3), range(2))
    )
    lat = {
        k: torch.tensor(v)
        for k, v in zip(("m", "mu", "a", "au", "fu", "order"), zip(*rows))
    }
    held_m = (lat["mu"] == 2) & (lat["m"] >= 9)
    held_a = (lat["au"] == 2) & (lat["a"] >= 9)
    masks = dict(
        familiar=~(held_m | held_a),
        unfamiliar=held_m | held_a,
        mass_only=held_m & ~held_a,
        acc_only=held_a & ~held_m,
        both=held_m & held_a,
    )
    assert T.build(lat).y.equal(oracle(lat))
    assert [int(m.sum()) for m in masks.values()] == [7350, 1776, 840, 840, 96]
    return lat, masks


def subset(lat, sel):
    return {k: v[sel] for k, v in lat.items()}


def sample(n, seed, split="train"):
    return T.sample_latents(n, torch.Generator().manual_seed(seed), split)


def forced(n, seed, key):
    g = torch.Generator().manual_seed(seed)
    lat = T.sample_latents(n, g)
    v, u = ("m", "mu") if key == "M" else ("a", "au")
    lat[u] = torch.full((n,), 2)
    lat[v] = torch.randint(9, 13, (n,), generator=g)
    return lat


def raw_sha(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def interval_change(original, source):
    # This exact computation order is also the intervention used during training.
    mk = torch.zeros(original.shape[-1])
    mk[:16] = 1
    return original * (1 - mk) + source * mk


def correct(pred, target):
    return int((pred == target).sum())


@torch.no_grad()
def hidden(m, lat):
    tok = T.build(lat).tokens
    return torch.cat([D.encode(m, chunk)[1] for chunk in tok.split(256)])


@torch.no_grad()
def predictions(m, lat):
    return m.head_from_resid(hidden(m, lat)).argmax(-1)


@torch.no_grad()
def causal(m, n, detailed=True):
    b = sample(n, 990010)
    hb = hidden(m, b)
    base = m.head_from_resid(hb).argmax(-1)
    z0 = T.build(b).y
    out = {}
    for name, src in [
        ("familiar", sample(n, 990004)),
        ("unfamiliar", forced(n, 990005, "M")),
    ]:
        hs = hidden(m, src)
        hm = interval_change(hb, hs)
        zm = D.counterfactual(b, src, "M")
        za = D.counterfactual(b, src, "A")
        pred = m.head_from_resid(hm).argmax(-1)
        sel = (zm != za) & (zm != z0) & (za != z0)
        out[name] = dict(
            n=n,
            hit=correct(pred, zm),
            eligible=int(sel.sum()),
            intended=correct(pred[sel], zm[sel]),
            unchanged=correct(pred[sel], z0[sel]),
            wrong=correct(pred[sel], za[sel]),
            unpatched_target=correct(base[sel], zm[sel]),
        )
        if detailed:
            not_block = torch.cat((hb[:, :16], hs[:, 16:]), 1)
            out[name]["complement_target"] = correct(
                m.head_from_resid(not_block).argmax(-1)[sel], zm[sel]
            )
            out[name]["residual_only_target"] = correct(
                m.W_U(hm).argmax(-1)[sel], zm[sel]
            )
            out[name]["mlp_only_target"] = correct(
                m.W_U(m.W_out(F.relu(m.W_in(hm)))).argmax(-1)[sel], zm[sel]
            )
            # Independent intervention from an alternative written form with the very same mass.
            alias = {k: v.clone() for k, v in src.items()}
            alias["mu"] = (src["mu"] + 1) % 3
            alias["m"] = T.canon(src)[0] * torch.tensor([1, 8, 5])[alias["mu"]] % 13
            assert T.canon(src)[0].equal(T.canon(alias)[0])
            ha = hidden(m, alias)
            pa = m.head_from_resid(interval_change(hb, ha)).argmax(-1)
            out[name]["alias_both_correct"] = int(
                ((pa == zm) & (pred == zm) & sel).sum()
            )
            out[name]["alias_prediction_agreement"] = correct(pa[sel], pred[sel])
            out[name]["block_alias_rms"] = float(
                (hs[:, :16] - ha[:, :16]).square().mean().sqrt()
            )
    return out


@torch.no_grad()
def audit(m, step, full=True):
    # Separate frozen evaluators for intermediate cards and the exact domain at the endpoint.
    n = 512
    out = {"step": step, "panels": {}}
    for key, lat in [
        ("familiar", sample(n, 74101)),
        ("unfamiliar", sample(n, 74102, "ood")),
        ("unfamiliar_mass", forced(n, 74103, "M")),
        ("unfamiliar_acc", forced(n, 74104, "A")),
    ]:
        pred = predictions(m, lat)
        out["panels"][key] = {"n": n, "hit": correct(pred, oracle(lat))}
    out["swap"] = causal(m, n, full)
    if full:
        basis = torch.eye(64)[:, :16]
        dw, db = R.fit_edit(m, basis, "M", "inv", n=512)
        # A weight edit in an independent inference copy, expressed as exact added W_in and b_in.
        lat = sample(1024, 74105)
        h = hidden(m, lat)
        p = m.head_from_resid(h, m.W_in.weight + dw, m.W_in.bias + db).argmax(-1)
        out["permanent_inversion"] = R.score(p, lat, "M", "inv") | {
            "weight_l2": float(dw.norm()),
            "bias_l2": float(db.norm()),
        }
    return out


@torch.no_grad()
def endpoint(m):
    lat, masks = enumerate_domain()
    pred = predictions(m, lat)
    y = oracle(lat)
    panels = {
        k: {"hit": correct(pred[v], y[v]), "n": int(v.sum())} for k, v in masks.items()
    }
    frozen = {}
    for name, split in [("familiar", "train"), ("unfamiliar", "ood")]:
        L = sample(8192, 999983, split)
        p = predictions(m, L)
        frozen[name] = {
            "hit": correct(p, oracle(L)),
            "n": 8192,
            "pred_sha256": raw_sha(p),
        }
    return {
        "universe": panels,
        "pred_sha256": raw_sha(pred),
        "frozen": frozen,
        "swap": causal(m, 4096),
    }


def controls():
    class Planted:
        def __init__(self, use_block):
            self.use_block = use_block

        def read(self, h):
            mass = h[:, :13].argmax(-1) if self.use_block else h[:, 16]
            return (mass * h[:, 17] * h[:, 18]).long() % 13

    b = sample(1024, 7)
    s = forced(1024, 8, "M")

    def enc(l):
        m, a, u = T.canon(l)
        h = torch.zeros(len(m), 64)
        h[:, :13] = F.one_hot(m, 13).float()
        h[:, 16] = m
        h[:, 17] = a
        h[:, 18] = u
        return h

    h, v = enc(b), enc(s)
    target = D.counterfactual(b, s, "M")
    wrong = D.counterfactual(b, s, "A")
    base = oracle(b)
    good = (base != target) & (base != wrong) & (wrong != target)
    p = interval_change(h, v)
    n = int(good.sum())
    hits = [
        correct(Planted(mode).read(p)[good], target[good]) for mode in (True, False)
    ]
    assert hits == [n, 0] and n > 0
    assert torch.equal(h, interval_change(h, h))
    return {
        "discriminating_pairs": n,
        "canonical_planted_hits": hits[0],
        "bypassed_planted_hits": hits[1],
        "identity_exact": True,
    }


if __name__ == "__main__":
    R.set_determinism()
    out = {"controls": controls(), "archive": {}}
    for seed in (307, 311, 305, 304, 310):
        path = NRI / "runs/zoo_swap" / f"mass_only_s{seed}"
        model, cfg = D.load(path)
        rec = endpoint(model)
        rec["model_file_sha256"] = hashlib.sha256(
            (path / "model.pt").read_bytes()
        ).hexdigest()
        rec["historical_metrics_matched"] = all(
            rec["frozen"][s]["hit"] / 8192 == cfg[k]
            for s, k in [("familiar", "acc_id"), ("unfamiliar", "acc_ood")]
        )
        assert rec["historical_metrics_matched"]
        out["archive"][str(seed)] = rec
    Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
