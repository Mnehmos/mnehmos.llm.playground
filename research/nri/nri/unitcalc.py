"""Unit-calculus task v2 (design: docs/20; registration: docs/21).

Every question is 8 tokens:   x1 u1  OP  x2 u2  ->  out =      (arithmetic mod 13)

Four modes. The operator token and the labels' dimensions determine the mode:
  mul   mass x acceleration -> force          y = M * A * s_out^-1
  div   force / mass -> acceleration          y = F * M^-1 * s_out^-1      (mass value != 0)
  msum  mass + mass -> mass                   y = (M1 + M2) * s_out^-1
  tsum  temperature + temperature -> temp     y = T1 + T2 - c_out          (phase B: offset labels)

Label relationships (what a label does to its value):
  scale    Q = x * s_u                         (mass, force, and 4 of the 6 acceleration labels)
  inverse  Q = s_u * x^-1, with 0^-1 = 0       (2 "per" acceleration labels)
  offset   T = x + c_u                         (temperature labels)

Label parameters are given as discrete logs base 2 (scales) or offsets. Each dimension contains
label pairs with EQUAL ratios / differences, so "same relationship => same internal change" can
be tested. Held-out label combinations (never seen together in training) test composition.

meaningless=True removes every label's meaning (scale 1, offset 0, no inversion): labels then
carry no information about the answer (negative control).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

P = 13
G = 2
LOG = {pow(G, k, P): k for k in range(12)}
INV = torch.tensor([0] + [pow(x, -1, P) for x in range(1, P)])

# label parameters: discrete logs for scale labels, offsets for temperature labels
MASS_LOG = (0, 1, 3, 4, 7, 9)
ACC_LOG = (0, 2, 5, 8, 0, 3)            # labels 4 and 5 are inverse ("per") labels
ACC_INVERSE = (False, False, False, False, True, True)
FORCE_LOG = (0, 1, 4, 6, 9, 10)
TEMP_OFF = (0, 1, 3, 4, 8, 11)
ACC_OUT = (0, 1, 2, 3)                  # quotient outputs use the direct acceleration labels

# token ids
VAL = list(range(P))
MASS_U = [P + i for i in range(6)]           # 13..18
ACC_U = [P + 6 + i for i in range(6)]        # 19..24
FORCE_U = [P + 12 + i for i in range(6)]     # 25..30
TEMP_U = [P + 18 + i for i in range(6)]      # 31..36
TIMES, DIVIDE, PLUS, ARROW, EQ = P + 24, P + 25, P + 26, P + 27, P + 28
VOCAB = EQ + 1                               # 42
SEQ_LEN = 8
MODES = ("mul", "div", "msum", "tsum")

# held-out label combinations (composition test): never in training
HELD = {"mul": {(i, i) for i in range(6)},                      # (mass label, acc label)
        "div": {(i, i) for i in range(6)},                      # (force label, mass label)
        "msum": {(i, (i + 1) % 6) for i in range(6)} | {((i + 1) % 6, i) for i in range(6)},
        "tsum": {(i, (i + 1) % 6) for i in range(6)} | {((i + 1) % 6, i) for i in range(6)}}


def scale(log):
    return torch.tensor([pow(G, int(k), P) for k in log])


@dataclass
class Batch:
    tokens: torch.Tensor
    y: torch.Tensor
    lat: dict


def _params(meaningless):
    if meaningless:
        z6 = (0,) * 6
        return dict(mass=scale(z6), acc=scale(z6), acc_inv=torch.zeros(6, dtype=torch.bool),
                    force=scale(z6), temp=torch.zeros(6, dtype=torch.long))
    return dict(mass=scale(MASS_LOG), acc=scale(ACC_LOG), acc_inv=torch.tensor(ACC_INVERSE),
                force=scale(FORCE_LOG), temp=torch.tensor(TEMP_OFF))


HELD_T = torch.zeros(len(MODES), 6, 6, dtype=torch.bool)
for _k, _m in enumerate(MODES):
    for _a, _b in HELD[_m]:
        HELD_T[_k, _a, _b] = True


def _draw(n, g, modes):
    mode = torch.tensor([MODES.index(m) for m in modes])[torch.randint(len(modes), (n,), generator=g)]
    x1 = torch.randint(0, P, (n,), generator=g)
    x2 = torch.randint(0, P, (n,), generator=g)
    u1 = torch.randint(0, 6, (n,), generator=g)
    u2 = torch.randint(0, 6, (n,), generator=g)
    uo = torch.randint(0, 6, (n,), generator=g)
    order = torch.randint(0, 2, (n,), generator=g)          # mul only: 0 = mass written first
    div = mode == MODES.index("div")
    uo = torch.where(div, uo % 4, uo)                       # quotient outputs: 4 direct labels
    x2 = torch.where(div & (x2 == 0), torch.randint(1, P, (n,), generator=g), x2)  # mass != 0
    return dict(mode=mode, x1=x1, u1=u1, x2=x2, u2=u2, uo=uo, order=order)


def is_held(lat):
    return HELD_T[lat["mode"], lat["u1"], lat["u2"]]


def sample(n, g, split="train", modes=MODES):
    """split: 'train' excludes held-out label combinations; 'held' uses only them; 'all' any."""
    if split == "all":
        return _draw(n, g, modes)
    parts, have = [], 0
    while have < n:
        lat = _draw(max(n, 1024), g, modes)
        keep = ~is_held(lat) if split == "train" else is_held(lat)
        parts.append({k: v[keep] for k, v in lat.items()})
        have += int(keep.sum())
    return {k: torch.cat([p_[k] for p_ in parts])[:n] for k in parts[0]}


def quantities(lat, meaningless=False):
    """Canonical quantities per operand (None where not applicable) and the answer."""
    p = _params(meaningless)
    mode = lat["mode"]
    x1, x2, u1, u2, uo = lat["x1"], lat["x2"], lat["u1"], lat["u2"], lat["uo"]
    # operand roles
    is_mul, is_div = mode == 0, mode == 1
    is_msum, is_tsum = mode == 2, mode == 3
    # mul convention: (x1, u1) is the mass and (x2, u2) the acceleration; `order` only moves
    # their token positions (see build)
    m_val, m_lab, a_val, a_lab = x1, u1, x2, u2
    # canonical values
    M_mul = m_val * p["mass"][m_lab] % P
    a_dir = a_val * p["acc"][a_lab] % P
    a_inv = p["acc"][a_lab] * INV[a_val] % P
    A_mul = torch.where(p["acc_inv"][a_lab], a_inv, a_dir)
    F_div = x1 * p["force"][u1] % P
    M_div = x2 * p["mass"][u2] % P
    M1 = x1 * p["mass"][u1] % P
    M2 = x2 * p["mass"][u2] % P
    T1 = (x1 + p["temp"][u1]) % P
    T2 = (x2 + p["temp"][u2]) % P
    s_out_force = p["force"][uo]
    s_out_acc = p["acc"][uo]
    s_out_mass = p["mass"][uo]
    y = torch.zeros_like(mode)
    y = torch.where(is_mul, M_mul * A_mul * INV[s_out_force] % P, y)
    y = torch.where(is_div, F_div * INV[M_div] * INV[s_out_acc] % P, y)
    y = torch.where(is_msum, (M1 + M2) * INV[s_out_mass] % P, y)
    y = torch.where(is_tsum, (T1 + T2 - p["temp"][uo]) % P, y)
    return dict(y=y, M_mul=M_mul, A_mul=A_mul, F_div=F_div, M_div=M_div, M1=M1, M2=M2, T1=T1, T2=T2)


def build(lat, meaningless=False) -> Batch:
    n = lat["mode"].shape[0]
    mode = lat["mode"]
    lab1 = torch.stack([torch.tensor(MASS_U)[lat["u1"]], torch.tensor(FORCE_U)[lat["u1"]],
                        torch.tensor(MASS_U)[lat["u1"]], torch.tensor(TEMP_U)[lat["u1"]]], 1)
    lab2 = torch.stack([torch.tensor(ACC_U)[lat["u2"]], torch.tensor(MASS_U)[lat["u2"]],
                        torch.tensor(MASS_U)[lat["u2"]], torch.tensor(TEMP_U)[lat["u2"]]], 1)
    labo = torch.stack([torch.tensor(FORCE_U)[lat["uo"]], torch.tensor(ACC_U)[lat["uo"]],
                        torch.tensor(MASS_U)[lat["uo"]], torch.tensor(TEMP_U)[lat["uo"]]], 1)
    ar = torch.arange(n)
    t1, t2, to = lab1[ar, mode], lab2[ar, mode], labo[ar, mode]
    # mul: u1 is the MASS label and u2 the ACC label by convention; order swaps their positions
    swap = (mode == 0) & (lat["order"] == 1)
    x1, x2 = lat["x1"], lat["x2"]
    first_v = torch.where(swap, x2, x1)
    first_u = torch.where(swap, t2, t1)
    second_v = torch.where(swap, x1, x2)
    second_u = torch.where(swap, t1, t2)
    op = torch.tensor([TIMES, DIVIDE, PLUS, PLUS])[mode]
    tok = torch.stack([first_v, first_u, op, second_v, second_u, torch.full((n,), ARROW), to,
                       torch.full((n,), EQ)], 1)
    return Batch(tok, quantities(lat, meaningless)["y"], lat)
