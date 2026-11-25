"""Doc 57 experiments H and K (addendum A): hybrid learning and protection of programmed knowledge.

Dual-operation task: '->' (token 23) = OP1, F = M*A*S (programmed); token 25 = OP2, U = (M*A + S) mod 13 (to be learned).
Hybrid model: TinyTyped d=128, 8x16 heads, MLP 512, vocab 26, with the Design-C mechanism installed in dims 0-63,
heads 0-3 and MLP units 0-35 (OP1 path gated off for OP2 by an OP-flag head).
Clarification fixed before any run: the constructed heads' entire q/k/v rows (all 128 input columns, zero beyond 63),
the o columns of heads 0-3 (all rows) and the constructed MLP units' entire W_in rows / W_out columns are constructed.

    python -m sci.hybrid A                 # condition A (entirely constructed), exhaustive check
    python -m sci.hybrid train [--workers 4]
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri.model import TinyTyped  # noqa: E402
from sci import construct as C  # noqa: E402
from sci import graph_compiler as G  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase7/hybrid"
OP2_TOK = 25
FLAG_DIM, KEY_DIM = 51, 52
SC = dict(MU=T.MASS_S, AU=T.ACC_S)


# ------------------------------------------------------------------ task
def domain2():
    L = C.domain()
    n = len(L["m"])
    L2 = {k: torch.cat([v, v]) for k, v in L.items()}
    L2["op"] = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    return L2


def build2(L):
    b = T.build({k: v for k, v in L.items() if k != "op"})
    tok = b.tokens.clone()
    tok[:, 5] = torch.where(L["op"] == 1, OP2_TOK, T.ARROW)
    M, A, S = C.canon_scales(L, SC)
    y = torch.where(L["op"] == 0, M * A * S % P, (M * A + S) % P)
    return tok, y


def unseen2(L):
    return C.unseen_mask(L)


# ------------------------------------------------------------------ constructed install
def make_model(kind, seed):
    torch.manual_seed(seed)
    if kind == "Bprime":
        return TinyTyped("mass_only", d=120, n_heads=8, d_mlp=480, n_types=1, aux_classes=0, vocab=26, seq_len=8, n_out=P)
    return TinyTyped("mass_only", d=128, n_heads=8, d_mlp=512, n_types=1, aux_classes=0, vocab=26, seq_len=8, n_out=P)


@torch.no_grad()
def install(m):
    """Install Design C (+ OP-flag head and gate) into dims 0-63 / heads 0-3 / units 0-35. Returns {param: bool mask}."""
    src, _ = C.build_C()
    d, d0 = m.d, 64
    mask = {n: torch.zeros_like(p, dtype=torch.bool) for n, p in m.named_parameters()}
    def put(name, idx, val):
        p = dict(m.named_parameters())[name]
        p.data[idx] = val
        mask[name][idx] = True
    E = torch.zeros(26, d0); E[:25] = src.E_c.weight
    E[OP2_TOK, KEY_DIM] = 1
    put("E_c.weight", (slice(0, 26), slice(0, d0)), E)
    put("E_p.weight", (slice(None), slice(0, d0)), src.E_p.weight)
    put("E_t.weight", (slice(None), slice(0, d0)), src.E_t.weight)
    for layer, sa in (("attn1", src.attn1), ("attn2", src.attn2)):
        Wb = torch.zeros(3 * d, d); bb = torch.zeros(3 * d)
        for part in range(3):                        # q, k, v
            for h in range(4):
                rs, rb = part * 64 + 16 * h, part * d + 16 * h
                Wb[rb:rb + 16, :d0] = sa.qkv.weight[rs:rs + 16]
                bb[rb:rb + 16] = sa.qkv.bias[rs:rs + 16]
        if layer == "attn2":                          # OP-flag head (head 3): q const, keys on OP2 flag / null
            bb[16 * 3] = C.G2; bb[16 * 3 + 1] = C.G2 / 2
            Wb[d + 16 * 3, KEY_DIM] = 1; Wb[d + 16 * 3 + 1, C.DIM["NULL"]] = 1
            Wb[2 * d + 16 * 3, KEY_DIM] = 1
        rows = torch.cat([torch.arange(part * d + 16 * h, part * d + 16 * h + 16) for part in range(3) for h in range(4)])
        p = dict(m.named_parameters())[f"{layer}.qkv.weight"]
        p.data[rows] = Wb[rows]; mask[f"{layer}.qkv.weight"][rows] = True
        p = dict(m.named_parameters())[f"{layer}.qkv.bias"]
        p.data[rows] = bb[rows]; mask[f"{layer}.qkv.bias"][rows] = True
        O = torch.zeros(d, 64); O[:d0, :] = sa.o.weight
        if layer == "attn2":
            O[FLAG_DIM, 48] = 1
        put(f"{layer}.o.weight", (slice(None), slice(0, 64)), O)
        put(f"{layer}.o.bias", (slice(0, d0),), sa.o.bias[:d0])
    # MLP: Design C MAS ladder (34 units) + 2 zero units, all gated by the OP flag
    Win = torch.zeros(36, d); bin_ = torch.zeros(36)
    Win[:, :d0] = src.W_in.weight[:36]; bin_[:] = src.W_in.bias[:36]
    B = 39.0
    Win[:, FLAG_DIM] -= B
    put("W_in.weight", (slice(0, 36), slice(None)), Win)
    put("W_in.bias", (slice(0, 36),), bin_)
    Wo = torch.zeros(d, 36); Wo[:d0] = src.W_out.weight[:, :36]
    put("W_out.weight", (slice(None), slice(0, 36)), Wo)
    put("W_out.bias", (slice(0, d0),), torch.zeros(d0))
    put("W_U.weight", (slice(None), slice(0, d0)), src.W_U.weight)
    return mask


def k2_mask(m):
    """learned writes into constructed dims 0-63 (held at zero under K2/K5; frozen at init under K3)."""
    mk = {n: torch.zeros_like(p, dtype=torch.bool) for n, p in m.named_parameters()}
    for layer in ("attn1", "attn2"):
        mk[f"{layer}.o.weight"][0:64, 64:] = True
    mk["W_out.weight"][0:64, 36:] = True
    return mk


# ------------------------------------------------------------------ condition A (entirely constructed, two stages)
def build_A():
    """Design C encoder + OP flag (layer-2 head 3), StagedNet: block 1 = gated OP1 ladder + node X = M*A;
    block 2 = node U = X + S gated by (1 - flag). Both roots write the shared OUT dims."""
    enc, _ = C.build_C()
    with torch.no_grad():
        E = torch.zeros(26, 64); E[:25] = enc.E_c.weight; E[OP2_TOK, KEY_DIM] = 1
        enc.E_c = torch.nn.Embedding.from_pretrained(E, freeze=False)
        W, b = enc.attn2.qkv.weight.data, enc.attn2.qkv.bias.data
        b[48], b[49] = C.G2, C.G2 / 2
        W[64 + 48, KEY_DIM] = 1; W[64 + 49, C.DIM["NULL"]] = 1; W[128 + 48, KEY_DIM] = 1
        enc.attn2.o.weight.data[FLAG_DIM, 48] = 1
    D = 64 + P
    X = list(range(64, 64 + P))
    OUTD = C.DIM["OUT"]
    opm = C.OPS["MAS"]
    w1, w2 = 36 + 27, 26                                                # MAS ladder+zero + X ladder (25+2); U ladder (t in 1..24)
    net = G.StagedNet(enc, D, [w1, 0])
    with torch.no_grad():
        for p in net.parameters():
            if p is not None and p.dim() > 0 and p not in set(enc.parameters()):
                p.zero_()
    # block 1a: MAS ladder (reuse Design C's MLP rows) gated by the flag
    blk = net.blocks[0]
    Win, bi, Wo = blk["W_in"].weight.data, blk["W_in"].bias.data, blk["W_out"].weight.data
    Win.zero_(); bi.zero_(); Wo.zero_(); blk["W_out"].bias.data.zero_()
    Win[:36, :64] = enc.W_in.weight.data[:36]; bi[:36] = enc.W_in.bias.data[:36]; Win[:36, FLAG_DIM] -= 39.0
    Wo[:64, :36] = enc.W_out.weight.data[:, :36]
    # block 1b: X = M*A (log statistic, gated on zeros) written one-hot into X
    def ladder(Win, bi, Wo, u0, terms, gfun, gate_dims, dest, B, flag_coef=0.0, flag_bias=0.0):
        vals = {0}
        for _, tab in terms:
            vals = {a + b for a in vals for b in tab}
        tmin, tmax = min(vals), max(vals)
        js = list(range(tmin - 1, tmax + 2))
        for u, j in enumerate(js):
            for dims, tab in terms:
                for d_, v in zip(dims, tab):
                    Win[u0 + u, d_] += v
            for gd in gate_dims:
                Win[u0 + u, gd] -= B
            Win[u0 + u, FLAG_DIM] += flag_coef
            bi[u0 + u] = -j + flag_bias
        for k in range(tmin, tmax + 1):
            c = dest[gfun(k)]
            Wo[c, u0 + (k - tmin)] += 1; Wo[c, u0 + (k - tmin + 1)] -= 2; Wo[c, u0 + (k - tmin + 2)] += 1
        n = len(js)
        if gate_dims:
            for uu, off in ((u0 + n, 0.0), (u0 + n + 1, -1.0)):
                for gd in gate_dims:
                    Win[uu, gd] = 1
                Win[uu, FLAG_DIM] += flag_coef
                bi[uu] = off + flag_bias
            Wo[dest[0], u0 + n] += 1
            Wo[dest[0], u0 + n + 1] -= 1
            n += 2
        return n
    lg = [C._lg(x) for x in range(P)]
    n = ladder(Win, bi, Wo, 36, [(C.DIM["Q"], lg), (C.DIM["A"], lg)], lambda t: pow(2, t % 12, P),
               [C.DIM["Q"][0], C.DIM["A"][0]], X, 40.0)
    # block 2: U = X + S, gated by (1 - flag)
    net.blocks[1] = torch.nn.ModuleDict(dict(W_in=torch.nn.Linear(D, w2), W_out=torch.nn.Linear(w2, D)))
    blk2 = net.blocks[1]
    with torch.no_grad():
        for p in blk2.parameters():
            p.zero_()
        Sv = C.SINV
        ladder(blk2["W_in"].weight.data, blk2["W_in"].bias.data, blk2["W_out"].weight.data, 0,
               [(X, list(range(P))), (C.DIM["FU"], Sv)], lambda t: t % P, [], OUTD, 40.0, flag_coef=40.0, flag_bias=-40.0)
        net.W_U.weight.data.zero_()
        for c in range(P):
            net.W_U.weight.data[c, OUTD[c]] = 1
    return net


@torch.no_grad()
def run_A():
    net = build_A()
    L = domain2()
    tok, y = build2(L)
    lg = net(tok)
    un = unseen2(L)
    oh = F.one_hot(y, P).float()
    r = dict(kind="constructed", acc=float((lg.argmax(-1) == y).float().mean()),
             acc_OP1=float((lg.argmax(-1) == y)[L["op"] == 0].float().mean()), acc_OP2=float((lg.argmax(-1) == y)[L["op"] == 1].float().mean()),
             acc_unseen=float((lg.argmax(-1) == y)[un].float().mean()), logits_exact_onehot=bool((lg == oh).all()),
             nonzero_params=int(sum((p != 0).sum() for p in net.parameters())), n_params=sum(p.numel() for p in net.parameters()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "A.json").write_text(json.dumps(r, indent=1))
    print(r)


# ------------------------------------------------------------------ training (conditions B, B', C=K1, K2, K3, K0, D=K4, K5)
CONDS = ("B", "Bprime", "C", "K2", "K3", "K0", "D", "K5")


@torch.no_grad()
def evaluate(m, L, tok, y, un, constructed):
    lg_all, sub_exact = [], []
    for i in range(0, len(y), 4096):
        t = tok[i:i + 4096]
        h = m.resid_pre_mlp(t, torch.zeros_like(t))
        r = h + m.W_out(torch.relu(h @ m.W_in.weight.T + m.W_in.bias))
        lg_all.append(m.W_U(r))
        if constructed:
            sub_exact.append(r[:, :64] @ m.W_U.weight[:, :64].T)
    lg = torch.cat(lg_all)
    pr = lg.argmax(-1)
    op = L["op"]
    res = {}
    for o, nm in ((0, "OP1"), (1, "OP2")):
        sel = op == o
        res[f"{nm}_familiar"] = float((pr[sel & ~un] == y[sel & ~un]).float().mean())
        res[f"{nm}_unseen"] = float((pr[sel & un] == y[sel & un]).float().mean())
    if constructed:
        sc = torch.cat(sub_exact)
        want = torch.where((op == 0)[:, None], F.one_hot(y, P).float(), torch.zeros(len(y), P))
        res["OP1_subcircuit_exact"] = float((sc[op == 0] == want[op == 0]).all(1).float().mean())
        res["OP2_subcircuit_silent"] = float((sc[op == 1] == 0).all(1).float().mean())
    # OP1 M-region interchange (discriminating pairs)
    g = torch.Generator().manual_seed(7)
    idx = torch.nonzero(op == 0).flatten()
    bi = idx[torch.randint(0, len(idx), (2048,), generator=g)]
    si = idx[torch.randint(0, len(idx), (2048,), generator=g)]
    M, A, S = C.canon_scales({k: v for k, v in L.items()}, SC)
    want = M[si] * A[bi] * S[bi] % P
    disc = want != y[bi]
    hb = m.resid_pre_mlp(tok[bi], torch.zeros_like(tok[bi]))
    hs = m.resid_pre_mlp(tok[si], torch.zeros_like(tok[si]))
    hb[:, :13] = hs[:, :13]
    pi = m.head_from_resid(hb).argmax(-1)
    res["OP1_iit_M_disc"] = float((pi[disc] == want[disc]).float().mean())
    return res


def train_one(args):
    cond, regime, seed = args
    torch.set_num_threads(1)
    m = make_model("Bprime" if cond == "Bprime" else "full", seed)
    constructed = cond not in ("B", "Bprime")
    mask = install(m) if constructed else None
    frozen = cond in ("C", "K2", "K3", "K5")
    kz = k2_mask(m) if cond in ("K2", "K5") else None
    k3 = k2_mask(m) if cond == "K3" else None
    with torch.no_grad():
        if kz is not None:
            for n, p in m.named_parameters():
                p.data[kz[n]] = 0
    snap = {n: p.detach().clone() for n, p in m.named_parameters()}
    L = domain2()
    tok, y = build2(L)
    un = unseen2(L)
    pool = torch.nonzero(~un).flatten()
    if regime == "q25":
        gq = torch.Generator().manual_seed(25)
        pool = pool[torch.randperm(len(pool), generator=gq)[: len(pool) // 4]]
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.01, betas=(0.9, 0.999))
    gb = torch.Generator().manual_seed(1000 + seed)
    M, A, S = C.canon_scales(L, SC)
    hist = []
    crit = None
    for step in range(6001):
        if step % 100 == 0:
            m.eval()
            e = evaluate(m, L, tok, y, un, constructed)
            e["step"] = step
            hist.append(e)
            if crit is None and e["OP2_familiar"] >= 0.99:
                crit = step
            m.train()
        if step == 6000:
            break
        bidx = pool[torch.randint(0, len(pool), (256,), generator=gb)]
        t = tok[bidx]
        loss = F.cross_entropy(m(t, torch.zeros_like(t)), y[bidx])
        if cond in ("D", "K5"):
            o1 = bidx[L["op"][bidx] == 0]
            if len(o1) > 1:
                src = o1[torch.randperm(len(o1), generator=gb)]
                hb = m.resid_pre_mlp(tok[o1], torch.zeros_like(tok[o1]))
                hs = m.resid_pre_mlp(tok[src], torch.zeros_like(tok[src]))
                hmix = torch.cat([hs[:, :13], hb[:, 13:]], 1)
                loss = loss + F.cross_entropy(m.head_from_resid(hmix), M[src] * A[o1] * S[o1] % P)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for n, p in m.named_parameters():
                if frozen and mask is not None:
                    p.data[mask[n]] = snap[n][mask[n]]
                if kz is not None:
                    p.data[kz[n]] = 0
                if k3 is not None:
                    p.data[k3[n]] = snap[n][k3[n]]
                    p.data[mask[n]] = snap[n][mask[n]]
    n_tr = sum(p.numel() for p in m.parameters()) - (int(sum(v.sum() for v in mask.values())) if (frozen and mask is not None) else 0)
    r = dict(cond=cond, regime=regime, seed=seed, kind="hybrid-trained" if constructed else "learned", trainable_params=n_tr,
             steps_to_OP2_criterion=crit, final=hist[-1], history=hist)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{cond}_{regime}_s{seed}.json").write_text(json.dumps(r, indent=1))
    return cond, regime, seed, crit, hist[-1]


def run_train(workers=4):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(c, r, s) for r in ("full", "q25") for c in CONDS for s in (0, 1, 2)]
    with ProcessPoolExecutor(workers) as ex:
        for cond, regime, seed, crit, fin in ex.map(train_one, jobs):
            print(cond, regime, seed, "crit", crit, {k: round(v, 3) for k, v in fin.items() if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "A":
        run_A()
    else:
        run_train(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
