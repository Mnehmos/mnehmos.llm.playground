"""Inspect H01's actual frozen phase adder and method selector.

This is not the full H01 arithmetic runtime or a new training result.
The exact family solvers and evaluation below are conventional Python.
"""
import hashlib
import json
import math
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import torch
from torch import nn

R = Path(__file__).resolve().parent
NAMES = ['linear', 'quadratic', 'cubic', 'exponential', 'reciprocal', 'hinge', 'periodic']


def load_weights(name):
    record = json.loads((R / 'weights.json').read_text())[name]
    raw = bytes.fromhex(record['hex'])
    if hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('weight hash mismatch')
    return torch.from_numpy(np.frombuffer(raw, dtype='<f4').copy())


@torch.no_grad()
def digit_addition():
    theta = load_weights('add.f32')
    x = torch.cartesian_prod(torch.arange(13), torch.arange(13))
    scores = 30 * torch.cos(theta[x[:, 0], None] + theta[x[:, 1], None] - theta[None, :])
    return scores.argmax(1).reshape(13, 13)


def features(y):
    d = np.diff(np.array([float(x) for x in y], dtype=np.float64))
    s = max(abs(d).max(), 1e-30)
    z, p = list(d / s), d
    for _ in range(3):
        p = np.diff(p)
        z.append(math.log1p(float(np.max(abs(p))) / s))
    return z


def differences(y):
    out = [list(y)]
    while len(out[-1]) > 1:
        out.append([b - a for a, b in zip(out[-1], out[-1][1:])])
    return out


def fittedval(t, n):
    k, *a = t
    if k < 3:
        return sum(c * math.comb(n, i) for i, c in enumerate(a[0]))
    if k == 3:
        return Q(a[0] * a[1] ** n + a[2])
    if k == 4:
        return Q(a[0], n + a[1]) + a[2]
    if k == 5:
        return Q(a[0] * abs(n - a[1]) + a[2])
    if k == 6:
        return Q(a[0] * ((n + a[2]) % a[1]) + a[3])
    raise ValueError(k)


def fit(k, y):
    if k < 3:
        d = differences(y)
        cands = [(k, [d[i][0] for i in range(k + 2)])]
    elif k == 3:
        cands = [(k, (y[1]-y[0])/(r-1), r, y[0]-(y[1]-y[0])/(r-1)) for r in range(2, 7)]
    elif k == 4:
        cands = []
        for c in range(1, 10):
            a = (y[0] - y[1]) * c * (c + 1)
            cands.append((k, a, c, y[0] - a/c))
    elif k == 5:
        a = y[0] - y[1]
        cands = [(k, a, c, y[0] - a*c) for c in range(2, 10)]
    elif k == 6:
        cands = []
        for p in range(2, 8):
            for c in range(p):
                a = (y[1]-y[0])/(((1+c) % p)-c)
                cands.append((k, a, p, c, y[0]-a*c))
    else:
        raise ValueError(k)
    for t in cands:
        if all(fittedval(t, n) == v for n, v in enumerate(y)):
            return t
    return None


@torch.no_grad()
def load_router():
    m = nn.Sequential(nn.Linear(14, 8), nn.Tanh(), nn.Linear(8, 7))
    v = load_weights('router.f32')
    pos = 0
    for p in m.parameters():
        n = p.numel()
        p.copy_(v[pos:pos+n].reshape(p.shape))
        pos += n
    assert pos == len(v) == 183
    return m.eval()


@torch.no_grad()
def infer(observations, router, budget=7):
    if len(observations) != 16:
        raise ValueError('16 exact observations required, n=0..15')
    y = [Q(x) for x in observations]
    order = router(torch.tensor([features(y[:12])], dtype=torch.float32))[0].argsort(descending=True).tolist()
    for count, k in enumerate(order[:budget], 1):
        t = fit(k, y[:12])
        if t is not None and all(fittedval(t, n) == y[n] for n in range(12, 16)):
            return t, count
    return None, min(len(order), budget)


def main():
    torch.set_num_threads(1)
    a = digit_addition()
    expected = torch.tensor([[(i+j) % 13 for j in range(13)] for i in range(13)])
    correct = int((a == expected).sum())
    assert correct == 169
    print(f'Frozen 13-parameter digit adder: {correct}/169')
    m = load_router()
    y = [Q(3, i+8)+2 for i in range(16)]
    t, attempts = infer(y, m)
    assert t is not None and fittedval(t, 31) == Q(27, 13)
    print(f'Frozen 183-parameter router: {NAMES[t[0]]}; {attempts} solver attempt(s)')
    print(f'Conventional exact evaluation at n=31: {fittedval(t, 31)}')
    print('PASS: narrow inspection demo, not full H01 qualification.')


if __name__ == '__main__':
    main()
