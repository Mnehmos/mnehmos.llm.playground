"""Dimensional-reasoning task with a genuine unit-rescaling symmetry (docs/11, docs/12).

Input (8 tokens):   x1 u1  ×  x2 u2  →  f  =
  One (value, unit) pair is a MASS, the other an ACCELERATION, in random order. Which is
  which is determined only by the unit token's DIMENSION (contextual functional typing).
Output:             F = (m · s_mu) · (a · s_au) · s_f^-1   (mod 13)

Unit scale factors are elements of the multiplicative group mod 13. The canonical,
unit-invariant causal variables are
    M = m · s_mu  (mass in base units),  A = a · s_au,  S = s_f^-1  (output-unit factor),
and F = M · A · S. Rescaling a unit while rescaling its value inversely leaves F unchanged:
a real group action on inputs under which the target is invariant.

Out-of-distribution split (never trained): mass unit 2 with m in {9..12}, and acceleration
unit 2 with a in {9..12}. Their canonical M / A values all occur in training via other units,
so solving them requires the unit-scaling structure, not memorization.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

P = 13
MASS_S = (1, 5, 8)
ACC_S = (1, 2, 7)
FORCE_S = (1, 6, 11)
INV = {x: pow(x, -1, P) for x in range(1, P)}

VAL = list(range(P))                       # 0..12
MU = [P + i for i in range(3)]             # 13..15 mass units
AU = [P + 3 + i for i in range(3)]         # 16..18 acceleration units
FU = [P + 6 + i for i in range(3)]         # 19..21 force units
TIMES, ARROW, EQ = P + 9, P + 10, P + 11   # 22, 23, 24
VOCAB = EQ + 1
SEQ_LEN = 8

# contextual roles (auxiliary targets for the ANLU-style functional-supervision arm)
R_MASS_VAL, R_MASS_UNIT, R_ACC_VAL, R_ACC_UNIT, R_OP, R_OUT_UNIT, R_QUERY = range(7)
N_ROLES = 7

OOD_VALUES = (9, 10, 11, 12)
OOD_UNIT = 2


@dataclass
class DBatch:
    tokens: torch.Tensor
    roles: torch.Tensor
    lat: dict
    y: torch.Tensor


def canon(lat):
    ms = torch.tensor(MASS_S)[lat["mu"]]
    as_ = torch.tensor(ACC_S)[lat["au"]]
    S = torch.tensor([INV[s] for s in FORCE_S])[lat["fu"]]
    M = lat["m"] * ms % P
    A = lat["a"] * as_ % P
    return M, A, S


def target(M, A, S):
    return M * A * S % P


def _sample_val(unit, g, split):
    n = unit.shape[0]
    v = torch.randint(0, P, (n,), generator=g)
    ood = (unit == OOD_UNIT) & (v >= OOD_VALUES[0])
    if split == "train":  # resample forbidden combos until none remain
        while ood.any():
            v[ood] = torch.randint(0, P, (int(ood.sum()),), generator=g)
            ood = (unit == OOD_UNIT) & (v >= OOD_VALUES[0])
    return v


def sample_latents(n, g, split="train"):
    """split: 'train' (no OOD combos), 'ood' (every sample has >= 1 held-out combo)."""
    mu = torch.randint(0, 3, (n,), generator=g)
    au = torch.randint(0, 3, (n,), generator=g)
    fu = torch.randint(0, 3, (n,), generator=g)
    order = torch.randint(0, 2, (n,), generator=g)  # 0: mass first
    if split == "ood":
        which = torch.randint(0, 2, (n,), generator=g)  # which pair is forced OOD
        mu = torch.where(which == 0, torch.full_like(mu, OOD_UNIT), mu)
        au = torch.where(which == 1, torch.full_like(au, OOD_UNIT), au)
        m = _sample_val(mu, g, "any")
        a = _sample_val(au, g, "any")
        m = torch.where(which == 0, torch.randint(9, 13, (n,), generator=g), m)
        a = torch.where(which == 1, torch.randint(9, 13, (n,), generator=g), a)
    else:
        m = _sample_val(mu, g, "train")
        a = _sample_val(au, g, "train")
    return dict(m=m, mu=mu, a=a, au=au, fu=fu, order=order)


def build(lat) -> DBatch:
    n = lat["m"].shape[0]
    mass_v, mass_u = lat["m"], torch.tensor(MU)[lat["mu"]]
    acc_v, acc_u = lat["a"], torch.tensor(AU)[lat["au"]]
    first = lat["order"] == 0
    x1 = torch.where(first, mass_v, acc_v)
    u1 = torch.where(first, mass_u, acc_u)
    x2 = torch.where(first, acc_v, mass_v)
    u2 = torch.where(first, acc_u, mass_u)
    tok = torch.stack([x1, u1, torch.full((n,), TIMES), x2, u2, torch.full((n,), ARROW),
                       torch.tensor(FU)[lat["fu"]], torch.full((n,), EQ)], 1)
    rv1 = torch.where(first, R_MASS_VAL, R_ACC_VAL)
    ru1 = torch.where(first, R_MASS_UNIT, R_ACC_UNIT)
    rv2 = torch.where(first, R_ACC_VAL, R_MASS_VAL)
    ru2 = torch.where(first, R_ACC_UNIT, R_MASS_UNIT)
    roles = torch.stack([rv1, ru1, torch.full((n,), R_OP), rv2, ru2, torch.full((n,), R_OP),
                         torch.full((n,), R_OUT_UNIT), torch.full((n,), R_QUERY)], 1)
    M, A, S = canon(lat)
    return DBatch(tok, roles, lat, target(M, A, S))


def with_latent(lat, **kw):
    out = {k: v.clone() for k, v in lat.items()}
    out.update(kw)
    return out
