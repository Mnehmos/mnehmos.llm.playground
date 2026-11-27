"""Doc 59: independently generated rules. Imported ONLY by the label oracle (training targets) and by the final
verification step — never by the extractor (sci/p8_extract.py)."""
from __future__ import annotations

import random

import torch

P = 13
_rng = random.Random(880001)
RULE4 = tuple(_rng.randrange(1, P) for _ in range(3)) + (_rng.randrange(P),)        # (a, b, c, d), a,b,c != 0
_g = torch.Generator().manual_seed(880002)
TABLE_R = torch.randint(0, P, (P, P, P), generator=_g)                               # random lookup (negative control)


def op4(M, A, S):
    a, b, c, d = RULE4
    return (a * M + b * A + c * S + d) % P


def opr(M, A, S):
    return TABLE_R[M, A, S]
