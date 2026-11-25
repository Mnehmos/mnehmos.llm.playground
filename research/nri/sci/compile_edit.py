"""Doc 48 E5/E6: permanent weight-level edits (no hooks at run time, no gradient training) and composition.

Compilation categories:
  (a) EXACT   a closed-form parameter change whose effect on the network's own computation is exactly the intended
              change of the internal input (e.g. replacing a role-specific token's embedding row by another's makes
              the network compute exactly as if the other token had been written);
  (b) APPROX  a closed-form least-squares parameter change fitted on calibration data;
  (c) gradient retraining: not used.

Semantic edits on the declared mass M = v * s_u (mod 13):
  unit conversion  "read unit u as unit u'":  E[MU_u] <- E[MU_u']        (EXACT; mass tokens are role-specific)
                   effect on M: M -> M * s_u' / s_u   for inputs written in unit u
  scaling by c in the unit group: compose a unit-row permutation pi with s_pi(u) = c * s_u for every u. Exists only if
                   c * {1,5,8} = {1,5,8} (mod 13) -- checked; none exists except c = 1, so global scaling is NOT
                   compilable by unit-row permutation.
  value edits (increment, decrement, doubling, inversion, constant): the value tokens 0..12 are SHARED by mass and
                   acceleration, so a mass-only value edit is input-dependent (depends on the role of the token). It
                   cannot be an embedding-row edit. Approximate construction (b): a low-rank update to the attention-2
                   value/output path, fitted by least squares on calibration data so that the block shift equals the
                   per-part shift a_{f(v)} - a_v (doc 46 P4) for mass-value contributions and zero otherwise.

    python -m sci.compile_edit        # writes runs/phase4/compile_edit.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_dimensional as D  # noqa: E402
from nri import dimtask as T  # noqa: E402
from sci import window_analysis as W  # noqa: E402

P = T.P
OUT = ROOT / "runs" / "phase4"
N = 8192


def load(name):
    m, _ = D.load(ROOT / "runs/zoo_swap" / name)
    return m


@torch.no_grad()
def evaluate(m, edit_M, edit_A=None, n=N):
    """Intended rate of an edited network: answers must equal target(edit_M(M,u,v), edit_A(A), S). Reported on trained
    and unseen (unit 2, value 9-12) mass renderings; eligible = intended answer differs from the unedited answer;
    preservation = accuracy on inputs where the edit is the identity."""
    out = {}
    for name, L in (("id", W.lat(n, 660_001)), ("oodM", W.ood_mass(n, 660_002))):
        b = T.build(L)
        pred = m(b.tokens, torch.zeros_like(b.tokens)).argmax(-1)
        M, A, S = T.canon(L)
        M2 = edit_M(M, L["mu"], L["m"])
        A2 = A if edit_A is None else edit_A(A, L["au"], L["a"])
        want, base = T.target(M2, A2, S), T.target(M, A, S)
        ch = want != base
        out[f"{name}_intended"] = float((pred[ch] == want[ch]).float().mean()) if ch.any() else None
        out[f"{name}_preserved"] = float((pred[~ch] == base[~ch]).float().mean()) if (~ch).any() else None
        out[f"{name}_n_changed"] = int(ch.sum())
    return out


def rel_disp(m0, m1):
    num = sum(float((a - b).norm() ** 2) for a, b in zip(m0.state_dict().values(), m1.state_dict().values()))
    den = sum(float(a.norm() ** 2) for a in m0.state_dict().values())
    return (num / den) ** 0.5


def unit_conversion(m, u_from, u_to):
    """EXACT: mass unit token u_from is read as u_to."""
    e = copy.deepcopy(m)
    e.E_c.weight.data[T.MU[u_from]] = m.E_c.weight.data[T.MU[u_to]]
    s = torch.tensor(T.MASS_S)

    def edit_M(M, mu, mv):
        new = (mv * s[mu]) % P
        sel = mu == u_from
        new[sel] = (mv[sel] * T.MASS_S[u_to]) % P
        return new
    return e, edit_M


def unit_permutation(m, perm):
    """EXACT: all three mass unit rows permuted (u read as perm[u]). Reversible by the inverse permutation."""
    e = copy.deepcopy(m)
    rows = [m.E_c.weight.data[T.MU[perm[u]]].clone() for u in range(3)]
    for u in range(3):
        e.E_c.weight.data[T.MU[u]] = rows[u]
    s = torch.tensor(T.MASS_S)
    ps = torch.tensor([T.MASS_S[perm[u]] for u in range(3)])
    return e, (lambda M, mu, mv: (mv * ps[mu]) % P)


def exact_suite(name):
    m = load(name)
    res = {}
    base = evaluate(m, lambda M, mu, mv: M)
    res["identity"] = dict(category="exact (no change)", **base)
    for uf, ut in ((0, 1), (0, 2), (1, 2), (2, 0)):
        e, f = unit_conversion(m, uf, ut)
        r = evaluate(e, f)
        r["param_disp"] = rel_disp(m, e)
        r["acc_A_preserved"] = evaluate(e, lambda M, mu, mv: f(M, mu, mv))["id_preserved"]
        res[f"unit_{uf}_as_{ut}"] = dict(category="exact (embedding row)", **r)
    for perm in ((1, 2, 0), (2, 0, 1), (0, 2, 1)):
        e, f = unit_permutation(m, perm)
        r = evaluate(e, f)
        inv = [perm.index(u) for u in range(3)]
        e2, _ = unit_permutation(e, tuple(inv))
        r["reversible_bitwise"] = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), e2.state_dict().values()))
        r["param_disp"] = rel_disp(m, e)
        res[f"unit_perm_{''.join(map(str, perm))}"] = dict(category="exact (embedding rows)", **r)
    return res


def compiled_value_edit(m, f, rank=8, n=32768):
    """APPROX (b): a rank-r additive update to the attention-2 value projection so that, for every position, the
    change of the last-token block approximates the per-part value shift a_{f(v)} - a_v when the position holds a
    MASS value, and zero otherwise. Fitted by least squares on calibration activations; then compiled into
    attn2.qkv (value rows) and checked without hooks."""
    from sci.part_edit import additive
    a, b = additive(m)
    L = W.lat(n, 550_001)
    bt = T.build(L)
    z = torch.zeros_like(bt.tokens)
    x0 = m.embed(bt.tokens, z)
    x1 = x0 + m.attn1(x0)
    _, pat = m.attn2(x1, return_pattern=True)                       # B, H, L, L
    Hn, dh = m.attn2.h, m.attn2.dh
    p_last = pat[:, :, -1, :]                                         # B, H, L
    mpos = torch.where(L["order"] == 0, 0, 3)
    target = torch.zeros(n, 64)
    vnew = torch.tensor([f(int(v)) for v in L["m"]])
    target[:, :16] = (a[vnew] - a[L["m"]]).float()
    # The change of the last-token output is o(concat_h dV_h feat_h), feat_h = sum_pos p_last[h,pos] * x1[pos].
    # Pull the target back through the (invertible) output projection: G = Wo^-1 target, then solve per head the
    # small least-squares problem dV_h feat_h ~= G_h (dh x 64 per head), and truncate each dV_h to rank r.
    Wo = m.attn2.o.weight
    G = torch.linalg.solve(Wo.double(), target.double().T).T.float()      # B x 64 (head-major: h*dh + j)
    dV = torch.zeros(Hn, dh, 64)
    for hh in range(Hn):
        feat = (p_last[:, hh, :, None] * x1).sum(1)                        # B x 64
        Gh = G[:, hh * dh:(hh + 1) * dh]                                    # B x dh
        Xf = feat.double()
        sol = torch.linalg.solve(Xf.T @ Xf + 1e-3 * torch.eye(64, dtype=Xf.dtype), Xf.T @ Gh.double()).float()  # 64 x dh
        U, S, Vh = torch.linalg.svd(sol.T, full_matrices=False)
        dV[hh] = (U[:, :rank] * S[:rank]) @ Vh[:rank]
    e = copy.deepcopy(m)
    with torch.no_grad():
        vrows = slice(2 * 64, 3 * 64)
        e.attn2.qkv.weight.data[vrows] += dV.reshape(64, 64)
    return e


def value_suite(name, edits):
    m = load(name)
    res = {}
    s = torch.tensor(T.MASS_S)
    for ename, f in edits.items():
        e = compiled_value_edit(m, f)
        r = evaluate(e, lambda M, mu, mv, f=f: (torch.tensor([f(int(x)) for x in mv]) * s[mu]) % P)
        r["param_disp"] = rel_disp(m, e)
        res[ename] = dict(category="approximate (least-squares, attention-2 value path)", **r)
    return res


VALUE_EDITS = {
    "increment_value": lambda v: (v + 1) % P,
    "decrement_value": lambda v: (v - 1) % P,
    "double_value": lambda v: 2 * v % P,
    "invert_value": lambda v: pow(v, -1, P) if v else 0,
    "constant_value_5": lambda v: 5,
}


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for s in (307, 311, 300, 304):
        n = f"mass_only_s{s}"
        res[n] = {"exact": exact_suite(n)}
        print(n, "EXACT", {k: (v.get("id_intended"), v.get("oodM_intended"), v.get("id_preserved")) for k, v in res[n]["exact"].items()}, flush=True)
    (OUT / "compile_edit.json").write_text(json.dumps(res, indent=1, default=str))
    for s in (307, 311, 300, 304):
        n = f"mass_only_s{s}"
        res[n]["approx_value"] = value_suite(n, VALUE_EDITS)
        print(n, "APPROX", {k: (round(v["id_intended"], 3), round(v["oodM_intended"], 3), round(v["id_preserved"] or 0, 3)) for k, v in res[n]["approx_value"].items()}, flush=True)
        (OUT / "compile_edit.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ exact token-level instruction set + composition
class Program:
    """An exact edit: a map on token identities (value tokens, mass units, acceleration units, force units).
    Compiled by replacing embedding rows: row[t] <- original_row[map(t)]. Semantics: the network computes exactly as
    if token map(t) had been written in place of t. Composition: (g after f) has map f∘g on tokens, i.e. the token t is
    read as f(g(t))... we keep maps as dicts token -> token and compose explicitly."""

    def __init__(self, val=None, mu=None, au=None, fu=None):
        self.val = val or {v: v for v in range(P)}
        self.mu = mu or {u: u for u in range(3)}
        self.au = au or {u: u for u in range(3)}
        self.fu = fu or {u: u for u in range(3)}

    def then(self, other):
        """Apply self, then other: token t is first read as self(t), and that reading is then edited by other.
        Since an embedding-row edit makes t behave as map(t), sequential compilation gives t -> self(other(t))."""
        return Program({v: self.val[other.val[v]] for v in range(P)}, {u: self.mu[other.mu[u]] for u in range(3)},
                       {u: self.au[other.au[u]] for u in range(3)}, {u: self.fu[other.fu[u]] for u in range(3)})

    def compile(self, m):
        e = copy.deepcopy(m)
        W0 = m.E_c.weight.data
        for v in range(P):
            e.E_c.weight.data[T.VAL[v]] = W0[T.VAL[self.val[v]]]
        for u in range(3):
            e.E_c.weight.data[T.MU[u]] = W0[T.MU[self.mu[u]]]
            e.E_c.weight.data[T.AU[u]] = W0[T.AU[self.au[u]]]
            e.E_c.weight.data[T.FU[u]] = W0[T.FU[self.fu[u]]]
        return e

    def target(self, L):
        v = torch.tensor([self.val[int(x)] for x in L["m"]])
        a = torch.tensor([self.val[int(x)] for x in L["a"]])
        mu = torch.tensor([self.mu[int(x)] for x in L["mu"]])
        au = torch.tensor([self.au[int(x)] for x in L["au"]])
        fu = torch.tensor([self.fu[int(x)] for x in L["fu"]])
        M2, A2, S2 = T.canon(dict(m=v, mu=mu, a=a, au=au, fu=fu, order=L["order"]))
        return T.target(M2, A2, S2), (mu == 2) & (v >= 9) | (au == 2) & (a >= 9)


def compiled_seq(m, progs):
    e = m
    for p_ in progs:
        e = p_.compile(e)
    return e


INSTR = {
    "inc_value": Program(val={v: (v + 1) % P for v in range(P)}),
    "dec_value": Program(val={v: (v - 1) % P for v in range(P)}),
    "double_value": Program(val={v: 2 * v % P for v in range(P)}),
    "invert_value": Program(val={v: (pow(v, -1, P) if v else 0) for v in range(P)}),
    "const_value_5": Program(val={v: 5 for v in range(P)}),
    "mass_unit_0_as_2": Program(mu={0: 2, 1: 1, 2: 2}),
    "mass_units_rotate": Program(mu={0: 1, 1: 2, 2: 0}),
    "acc_units_rotate": Program(au={0: 1, 1: 2, 2: 0}),
    "force_units_swap01": Program(fu={0: 1, 1: 0, 2: 2}),
    "identity": Program(),
}


@torch.no_grad()
def eval_program(m, progs, n=8192):
    """Compile the sequence into the weights (no hooks) and score it on trained-rendering inputs and on unseen-mass
    inputs, against the mathematically composed target. Also splits by whether the edited reading is itself a
    never-trained rendering (i.e. the program routes the input into an unseen combination)."""
    e = compiled_seq(m, progs)
    comp = Program()
    for p_ in progs:
        comp = p_.then(comp) if False else comp.then(p_)
    out = {}
    for name, L in (("id", W.lat(n, 660_011)), ("oodM", W.ood_mass(n, 660_012))):
        b = T.build(L)
        pred = e(b.tokens, torch.zeros_like(b.tokens)).argmax(-1)
        want, novel = comp.target(L)
        ok = pred == want
        out[f"{name}_correct"] = float(ok.float().mean())
        out[f"{name}_correct_on_trained_readings"] = float(ok[~novel].float().mean()) if (~novel).any() else None
        out[f"{name}_correct_on_unseen_readings"] = float(ok[novel].float().mean()) if novel.any() else None
        out[f"{name}_frac_unseen_readings"] = float(novel.float().mean())
    out["param_disp"] = rel_disp(m, e)
    return out


def instruction_suite(name):
    m = load(name)
    return {k: eval_program(m, [p_]) for k, p_ in INSTR.items()}


def composition_benchmark(name, n_seq=40, seed=0):
    import random as _r
    rng = _r.Random(seed)
    m = load(name)
    keys = [k for k in INSTR if k != "identity"]
    res = []
    for _ in range(n_seq):
        seq = [rng.choice(keys) for _ in range(rng.choice((2, 3)))]
        r = eval_program(m, [INSTR[k] for k in seq])
        res.append(dict(seq=seq, **r))
    # reversibility: inverse programs (permutations only) restore the network bitwise
    inv_ok = {}
    for k in ("inc_value", "double_value", "invert_value", "mass_units_rotate", "acc_units_rotate", "force_units_swap01"):
        p_ = INSTR[k]
        inv = Program({v: k2 for k2, v in p_.val.items()} if len(set(p_.val.values())) == P else None,
                      {v: k2 for k2, v in p_.mu.items()}, {v: k2 for k2, v in p_.au.items()}, {v: k2 for k2, v in p_.fu.items()})
        e2 = inv.compile(p_.compile(m))
        inv_ok[k] = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), e2.state_dict().values()))
    return dict(sequences=res, reversible=inv_ok)


def run_programs():
    torch.set_num_threads(3)
    out = {}
    for s in (307, 311, 300, 305, 304, 310):
        n = f"mass_only_s{s}"
        out[n] = dict(instructions=instruction_suite(n), composition=composition_benchmark(n))
        c = out[n]["composition"]["sequences"]
        import statistics as st
        print(n, "single:", {k: (round(v["id_correct"], 3), round(v["oodM_correct"], 3)) for k, v in out[n]["instructions"].items()},
              "| composed mean id/ood:", round(st.mean(x["id_correct"] for x in c), 3), round(st.mean(x["oodM_correct"] for x in c), 3),
              "| reversible:", all(out[n]["composition"]["reversible"].values()), flush=True)
    (OUT / "programs.json").write_text(json.dumps(out, indent=1))
