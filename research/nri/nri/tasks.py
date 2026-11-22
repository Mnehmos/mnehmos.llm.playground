"""Typed modular-arithmetic task with *contextual* functional types.

A sequence holds six content tokens and a final query token "=":

    d  d  a  OP  b  d  =        (OP position varies, here 3)

The operands are the numbers immediately LEFT and RIGHT of the single operator
token; every other number is a DISTRACTOR. The same number token can therefore
carry any of three roles depending on context, which mirrors ANLU's
*contextual* (not lexical) functional types. The target is OP(a, b) mod P.

Conditions (all share architecture and parameter count):

  none        e = c + p                (type id is a constant)
  typed       e = c + t_tau + p        (true contextual role ids)
  shuffled    e = c + t_tau' + p       (role ids permuted among content positions
                                        per sequence: same marginals, no role info)
  canonical   e = c + p, sequence re-ordered to [a, OP, b, d, d, d, =]
              (same role information as `typed`, delivered through position
               instead of an additive type geometry)
  typed_fact  typed, but content lives in dims [0, d-k) and type in [d-k, d)
              (a hard "orthogonal subspace" geometric prior)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

P = 11  # numbers 0..P-1, arithmetic mod P
OPS = ("ADD", "SUB", "MUL")
N_CONTENT = 6
SEQ_LEN = N_CONTENT + 1  # + query token

# vocabulary
NUM_TOKENS = list(range(P))
OP_TOKENS = [P + i for i in range(len(OPS))]
EQ_TOKEN = P + len(OPS)
VOCAB = EQ_TOKEN + 1

# contextual types
T_DISTRACTOR, T_LEFT, T_RIGHT, T_OP, T_QUERY = range(5)
N_TYPES = 5
TYPE_NAMES = ("DISTRACTOR", "LEFT", "RIGHT", "OP", "QUERY")

CONDITIONS = ("none", "typed", "shuffled", "canonical", "typed_fact")


def apply_op(op: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.where(op == 0, a + b, torch.where(op == 1, a - b, a * b))
    return out % P


@dataclass
class Batch:
    tokens: torch.Tensor  # [B, L] long
    types: torch.Tensor  # [B, L] long (already transformed for the condition)
    a: torch.Tensor  # [B]
    b: torch.Tensor  # [B]
    op: torch.Tensor  # [B]
    y: torch.Tensor  # [B]


def sample_latents(n: int, g: torch.Generator):
    a = torch.randint(0, P, (n,), generator=g)
    b = torch.randint(0, P, (n,), generator=g)
    op = torch.randint(0, len(OPS), (n,), generator=g)
    op_pos = torch.randint(1, N_CONTENT - 1, (n,), generator=g)  # 1..4
    distract = torch.randint(0, P, (n, N_CONTENT), generator=g)
    perm_noise = torch.rand(n, N_CONTENT, generator=g)
    return dict(a=a, b=b, op=op, op_pos=op_pos, distract=distract, perm_noise=perm_noise)


def build(lat: dict, condition: str) -> Batch:
    """Deterministically render latents into a batch for one condition."""
    a, b, op, op_pos, distract = lat["a"], lat["b"], lat["op"], lat["op_pos"], lat["distract"]
    n = a.shape[0]
    idx = torch.arange(n)
    if condition == "canonical":
        content = distract.clone()
        content[:, 0] = a
        content[:, 1] = OP_TOKENS[0] + op
        content[:, 2] = b
        types = torch.full((n, N_CONTENT), T_DISTRACTOR)
        types[:, 0], types[:, 1], types[:, 2] = T_LEFT, T_OP, T_RIGHT
    else:
        content = distract.clone()
        content[idx, op_pos - 1] = a
        content[idx, op_pos] = OP_TOKENS[0] + op
        content[idx, op_pos + 1] = b
        types = torch.full((n, N_CONTENT), T_DISTRACTOR)
        types[idx, op_pos - 1] = T_LEFT
        types[idx, op_pos] = T_OP
        types[idx, op_pos + 1] = T_RIGHT
    if condition == "shuffled":
        perm = lat["perm_noise"].argsort(dim=1)
        types = torch.gather(types, 1, perm)
    if condition in ("none", "canonical"):
        types = torch.zeros_like(types)  # constant type id == learned bias
    tokens = torch.cat([content, torch.full((n, 1), EQ_TOKEN)], dim=1)
    qtype = T_QUERY if condition not in ("none", "canonical") else 0
    types = torch.cat([types, torch.full((n, 1), qtype)], dim=1)
    return Batch(tokens, types, a, b, op, apply_op(op, a, b))


def make_batch(n: int, condition: str, g: torch.Generator) -> Batch:
    return build(sample_latents(n, g), condition)


def with_latent(lat: dict, **overrides) -> dict:
    out = {k: v.clone() for k, v in lat.items()}
    for k, v in overrides.items():
        out[k] = v
    return out
