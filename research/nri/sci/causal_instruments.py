"""Validated held-out causal instruments (doc 41 §41.2). Retrospective, exploratory.

A generic pair-and-stratum engine over ADAPTERS, so the same code scores real networks and planted mechanisms
with known ground truth.

Adapter protocol:
    enc = adapter.encode(combos)            -> dict(states={site: [N,T,d]}, pos={"src": [N], "unit": [N]}, ctx=...)
    adapter.readout(states, enc, site)       -> LongTensor[N] of integer answers 0..12
    adapter.sites                            -> tuple of patchable sites

Seven conditions (I1..I7) and the pair families:
  * "cross" pairs (doc 39 enumeration, reserved test excluded everywhere), stratified by base pool x
    canonical-implied pool; rates are reported as EXCESS over the no-intervention baseline;
  * "same_canonical" pairs (I6): donor has the base's canonical quantity, a different source unit and number.

    python -m sci.causal_instruments validate      # planted-mechanism + control validation (gate)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import concept as C  # noqa: E402
from sci import heldout_causal as HC  # noqa: E402
from sem import data as D  # noqa: E402
from sem import lang as L  # noqa: E402

P = C.P
CANON_STYLE = dict(num="num", unit="full")


# ------------------------------------------------------------------ pair families
def cross_pairs(split):
    return HC.enumerate_pairs(split)


def same_canonical_pairs(split):
    """Donor (trained) shares dim and canonical quantity with the base, has a different source unit, and the
    number prediction differs from the base answer. Canonical code predicts UNCHANGED."""
    Pl = HC.pools(split)
    out = []
    for rb, vb in sorted(Pl["trained"] | Pl["heldout"]):
        dim, ub, tb = rb
        qb = C.canonical(dim, vb, ub)
        a_base = C.answer(dim, vb, ub, tb)
        for rd, vd in sorted(Pl["trained"]):
            if rd[0] != dim or rd[1] == ub or C.canonical(dim, vd, rd[1]) != qb:
                continue
            a_num = C.answer(dim, vd, ub, tb)
            if a_num == a_base:
                continue
            num_c = HC.classify((rb, vd), Pl)
            if num_c == "test":
                continue
            out.append(dict(base=(rb, vb), donor=(rd, vd), a_base=a_base, a_num=a_num, a_can=a_base,
                            base_pool=HC.classify((rb, vb), Pl), can_implied=HC.classify((rb, vb), Pl)))
    return out


# ------------------------------------------------------------------ subspaces
def class_mean_basis(adapter, split, site="x1", rank=None):
    """Orthonormal basis of centred canonical-class means at the source-number token, fitted on TRAINED bases only."""
    combos = list(split["train"])
    enc = adapter.encode(combos)
    x = enc["states"][site][torch.arange(len(combos)), enc["pos"]["src"]]
    q = torch.tensor([C.canonical(r[0], v, r[1]) for r, v in combos])
    means = torch.stack([x[q == k].mean(0) for k in range(P) if (q == k).any()])
    means = means - means.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(means, full_matrices=False)
    k = int((S > S[0] * 1e-6).sum()) if rank is None else rank
    return Vh[:k].T.contiguous()                                    # [d, k]


def random_basis(d, k, seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
    return Q


# ------------------------------------------------------------------ engine
@torch.no_grad()
def run_condition(adapter, pairs, site, positions=("src",), basis=None, mode="full"):
    """Patch donor -> base at `site` and `positions`. mode: full | sub (project onto basis) | comp (complement)."""
    eb = adapter.encode([p["base"] for p in pairs])
    ed = adapter.encode([p["donor"] for p in pairs])
    xs = {k: v.clone() for k, v in eb["states"].items()}
    x = xs[site]
    a = torch.arange(len(pairs))
    for pos in positions:
        pb, pd = eb["pos"][pos], ed["pos"][pos]
        hb, hd = x[a, pb], ed["states"][site][a, pd]
        delta = hd - hb
        if mode == "sub":
            delta = delta @ basis @ basis.T
        elif mode == "comp":
            delta = delta - delta @ basis @ basis.T
        x[a, pb] = hb + delta
    pred = adapter.readout(xs, eb, site)
    base_pred = adapter.readout(eb["states"], eb, site)
    t = lambda k: torch.tensor([p[k] for p in pairs])
    return dict(canonical=pred == t("a_can"), number=pred == t("a_num"), unchanged=pred == t("a_base"),
                b_canonical=base_pred == t("a_can"), b_number=base_pred == t("a_num"), b_unchanged=base_pred == t("a_base"))


def summarise(r):
    m = {k: round(v.float().mean().item(), 4) for k, v in r.items()}
    # switch rate (doc 41R deviation 1): among pairs whose UNPATCHED prediction is not the target, the fraction the
    # patch moves onto the target. The excess is bounded by 1 - baseline; the switch rate is not.
    for k in ("canonical", "number"):
        nb = ~r["b_" + k]
        m[k + "_switch"] = round(r[k][nb].float().mean().item(), 4) if nb.any() else None
    m["canonical_excess"] = round(m["canonical"] - m["b_canonical"], 4)
    m["number_excess"] = round(m["number"] - m["b_number"], 4)
    return m


def strata(adapter, pairs, n=300, seed=0, **kw):
    rng = random.Random(seed)
    groups = {}
    for p in pairs:
        groups.setdefault(f"base_{p['base_pool']}|can_implied_{p['can_implied']}", []).append(p)
    out = {}
    for k, g in sorted(groups.items()):
        g = rng.sample(g, min(n, len(g)))
        out[k] = dict(n=len(g), **summarise(run_condition(adapter, g, **kw)))
    return out


def conditions(adapter, split, cp, sp, n=300, n_random=10, which=("I1", "I2", "I3", "I4", "I5", "I6", "I7")):
    """All seven doc-41 conditions on one adapter."""
    res = {}
    if "I1" in which:
        res["I1"] = strata(adapter, cp, n, site="x1")
    if "I2" in which and "unit" in adapter.positions:
        res["I2"] = strata(adapter, cp, n, site="x1", positions=("src", "unit"))
    if "I3" in which and "x2" in adapter.sites:
        res["I3"] = strata(adapter, cp, n, site="x2")
    if {"I4", "I5", "I7"} & set(which):
        B = class_mean_basis(adapter, split, "x1")
        res["I4_rank"] = B.shape[1]
        if "I4" in which:
            res["I4"] = strata(adapter, cp, n, site="x1", basis=B, mode="sub")
        if "I5" in which:
            res["I5"] = strata(adapter, cp, n, site="x1", basis=B, mode="comp")
        if "I7" in which:
            draws = [strata(adapter, cp, n, site="x1", basis=random_basis(B.shape[0], B.shape[1], 1000 + i), mode="sub")
                     for i in range(n_random)]
            res["I7"] = {k: {m: round(sum(d[k][m] for d in draws) / n_random, 4) for m in draws[0][k]} for k in draws[0]}
    if "I6" in which:
        res["I6"] = strata(adapter, sp, n, site="x1")
    return res


# ------------------------------------------------------------------ adapters
class StudentAdapter:
    """SemStudent: states are the residual streams x0 / x1 (after block 1) / x2 (after block 2)."""
    sites = ("x0", "x1", "x2")
    positions = ("src", "unit")

    def __init__(self, model, vocab):
        self.m, self.vocab = model.eval(), vocab

    _cache = {}

    def batch(self, combos):
        key = tuple(combos)
        if key not in self._cache:                                   # tokenisation is model-independent: cache it
            rng = random.Random(0)
            items = [C.example(r, v, "C01", CANON_STYLE, rng) for r, v in combos]
            self._cache[key] = D.encode_batch(items, self.vocab)
        return self._cache[key]

    @torch.no_grad()
    def encode(self, combos):
        b = self.batch(combos)
        st = self.m.trunk(b["tok"], ret=True)
        return dict(states={"x0": st["x0"], "x1": st["x1"], "x2": st["x2"]},
                    pos={"src": C.src_pos(b), "unit": (b["role"] == L.ROLE_ID["src_unit"]).long().argmax(1)},
                    ctx=b)

    @torch.no_grad()
    def readout(self, states, enc, site):
        b = enc["ctx"]
        km = b["tok"] != D.PAD
        m = self.m
        if site == "x0":
            x = states["x0"]
            x = x + m.attn1(m.ln1a(x), km)
            x = x + m.mlp1(m.ln1m(x))
            logits = C.finish_from(m, x, km, 1)
        elif site == "x1":
            logits = C.finish_from(m, states["x1"], km, 1)
        else:
            logits = C.finish_from(m, states["x2"], km, 2)
        return logits[:, :P].argmax(-1)                              # answer ids 0..12 == integer answers


class PlantedAdapter:
    """Known mechanisms. One state vector at the source-number token (T=1); the route is read from context."""
    sites = ("x1",)
    positions = ("src",)

    def __init__(self, kind, split, d=64, noise=20, seed=0):
        self.kind, self.d = kind, d
        self.trained = set(split["train"])
        g = torch.Generator().manual_seed(seed)
        self.R, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        self.noise, self.g = noise, g

    def code(self, r, v):
        dim, u, t = r
        q = C.canonical(dim, v, u)
        z = torch.zeros(self.d)
        if self.kind in ("P1", "P3", "P5"):
            z[q] = 1.0
        elif self.kind == "P2":
            z[v] = 1.0
        elif self.kind == "P4":
            z[q] = 1.0
            z[P + u] = 1.0
        z[P + 3:P + 3 + self.noise] = torch.randn(self.noise, generator=self.g) * 0.3
        return self.R @ z if self.kind == "P4" else z

    def encode(self, combos):
        x = torch.stack([self.code(r, v) for r, v in combos])[:, None, :]
        zero = torch.zeros(len(combos), dtype=torch.long)
        return dict(states={"x1": x}, pos={"src": zero}, ctx=list(combos))

    def readout(self, states, enc, site):
        x = states["x1"][:, 0]
        if self.kind == "P4":
            x = x @ self.R                                            # R^T applied row-wise
        out = []
        for i, (r, v) in enumerate(enc["ctx"]):
            dim, u, t = r
            if self.kind == "P5":
                out.append(C.answer(dim, v, u, t))
                continue
            k = int(x[i, :P].argmax())
            if self.kind == "P2":
                out.append(C.answer(dim, k, u, t))
                continue
            a = k * L.INV[L.SCALES[dim][t]] % P                        # convert canonical quantity to target unit
            if self.kind == "P3":
                v_imp = k * L.INV[L.SCALES[dim][u]] % P
                if (r, v_imp) not in self.trained:
                    a = (a + 5) % P                                    # route-specific reader: unseen input -> wrong
            out.append(a)
        return torch.tensor(out)


# ------------------------------------------------------------------ validation (gate)
CRITERIA = {
    "P1": "I1 canonical_excess >= 0.9 in all strata; I6 unchanged >= 0.9",
    "P2": "I1 number_excess >= 0.9 and canonical_excess <= 0.05 in all strata",
    "P3": "I1 canonical_switch >= 0.9 where implied trained and <= 0.10 where implied held-out (deviation 1: switch, not excess)",
    "P4": "I4 canonical_excess >= 0.9; I5 and I7 canonical_excess <= 0.10 (all strata)",
    "P5": "every |excess| <= 0.05",
}


def check(kind, r):
    I1 = r["I1"]
    ce = {k: v["canonical_excess"] for k, v in I1.items()}
    if kind == "P1":
        return all(x >= 0.9 for x in ce.values()) and all(v["unchanged"] >= 0.9 for v in r["I6"].values())
    if kind == "P2":
        return all(v["number_excess"] >= 0.9 and v["canonical_excess"] <= 0.05 for v in I1.values())
    if kind == "P3":
        sw = {k: v["canonical_switch"] for k, v in I1.items()}
        return all((x >= 0.9) if k.endswith("implied_trained") else (x <= 0.10) for k, x in sw.items())
    if kind == "P4":
        return (all(v["canonical_excess"] >= 0.9 for v in r["I4"].values())
                and all(v["canonical_excess"] <= 0.10 for v in r["I5"].values())
                and all(v["canonical_excess"] <= 0.10 for v in r["I7"].values()))
    if kind == "P5":
        return all(abs(v[m]) <= 0.05 for c in ("I1", "I4", "I5", "I6", "I7") for v in r[c].values()
                   for m in ("canonical_excess", "number_excess"))


def validate():
    split = C.make_split()
    cp, sp = cross_pairs(split), same_canonical_pairs(split)
    out = dict(n_cross=len(cp), n_same_canonical=len(sp), planted={}, criteria=CRITERIA)
    for kind in ("P1", "P2", "P3", "P4", "P5"):
        r = conditions(PlantedAdapter(kind, split), split, cp, sp, n=300, n_random=5)
        r["pass"] = bool(check(kind, r))
        out["planted"][kind] = r
    out["all_pass"] = all(v["pass"] for v in out["planted"].values())
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["validate"]:
        res = validate()
        path = ROOT / "runs" / "instruments" / "planted_validation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1))
        for k, v in res["planted"].items():
            print(k, "PASS" if v["pass"] else "FAIL", "| rank", v.get("I4_rank"))
            for c in ("I1", "I4", "I5", "I6", "I7"):
                print("   ", c, {s.replace("base_", "").replace("can_implied_", ""): (x["canonical_excess"], x["canonical_switch"], x["number_excess"], x["unchanged"]) for s, x in v[c].items()})
        print("ALL PASS:", res["all_pass"], "| pairs", res["n_cross"], res["n_same_canonical"])


# ------------------------------------------------------------------ application to the archived factorial
WEIGHTS = Path("/home/user/nri_weights")
OUT_DIR = ROOT / "runs" / "instruments"


def load_student(run, step):
    from safetensors.torch import load_file
    from sem import model as SM
    vocab = D.Vocab.from_json(json.loads((ROOT / "runs/zoo_sem/eval/vocab.json").read_text()))
    m = SM.SemStudent(len(vocab.itos))
    m.load_state_dict(load_file(WEIGHTS / run / "ckpt" / f"step_{step:05d}" / "model.safetensors"))
    return StudentAdapter(m, vocab)


COMPACT = ("canonical_excess", "canonical_switch", "number_excess", "number_switch", "unchanged", "b_unchanged")


def compact(res):
    if isinstance(res, int):
        return res
    return {k: {m: v[m] for m in ("n",) + COMPACT} for k, v in res.items()}


def apply_run(run):
    torch.set_num_threads(1)
    split = C.make_split()
    cp, sp = cross_pairs(split), same_canonical_pairs(split)
    out = dict(run=run, I1={}, full={})
    for step in range(0, 6001, 100):
        ad = load_student(run, step)
        if step % 500 == 0:
            r = conditions(ad, split, cp, sp, n=300, n_random=3)
            out["full"][step] = {k: compact(v) for k, v in r.items()}
            out["I1"][step] = out["full"][step]["I1"]
        else:
            out["I1"][step] = compact(strata(ad, cp, 300, site="x1"))
    path = OUT_DIR / "factorial" / f"{run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return run


def real_controls(seeds=(2001, 2002, 2003, 2004)):
    """Known-answer controls on real networks (doc 41 §41.2): input-level swaps at x0 (final weights) and every
    condition on untrained (step 0) networks."""
    split = C.make_split()
    cp, sp = cross_pairs(split), same_canonical_pairs(split)
    out = {}
    for s in seeds:
        run = f"wd1.0_C_s{s}"
        ad = load_student(run, 6000)
        out[f"{run}_x0_number_token"] = compact(strata(ad, cp, 300, site="x0"))
        out[f"{run}_x0_number+unit"] = compact(strata(ad, cp, 300, site="x0", positions=("src", "unit")))
        out[f"{run}_step0_all"] = {k: compact(v) for k, v in conditions(load_student(run, 0), split, cp, sp, 300, 3).items()}
    step0_max = max(abs(v[m]) for k, r in out.items() if k.endswith("step0_all") for c, cc in r.items() if c != "I4_rank"
                    for v in cc.values() for m in ("canonical_excess", "number_excess"))
    out["step0_max_abs_excess"] = step0_max
    out["step0_pass"] = step0_max <= 0.05
    return out


def apply_all(workers=3):
    from concurrent.futures import ProcessPoolExecutor
    from sci import factorial as Fx
    runs = [Fx.run_name(p, w, s) for p, w, s in Fx.all_runs()]
    todo = [r for r in runs if not (OUT_DIR / "factorial" / f"{r}.json").exists()]
    # C wd1.0 and C wd0.1 first (the mechanism contrasts), then the rest
    todo.sort(key=lambda r: (not r.startswith("wd1.0_C"), not r.startswith("wd0.1_C"), r))
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(apply_run, todo):
            print("[done]", r, flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["apply"]:
    apply_all(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
