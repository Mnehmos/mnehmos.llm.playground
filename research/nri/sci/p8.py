"""Doc 59: Phase VIII core — masked/protected TinyTyped networks and a resumable, cost-accounted trainer.

Permissions are STRUCTURAL: every parameter W is used through its effective value
    W_eff = Zbar * (F * W0 + Fbar * W)
F = frozen (installed, value W0), Z = write-blocked (forced zero). Gradients reach only trainable entries. Optional
OUTPUT OWNERSHIP gate (ProtectedNet): logits = compiled(r) + flag(x) * learned(r), where flag is the OP2 indicator computed
exactly by the frozen OP-flag head (dim 51), compiled(r) = W_U[:, :64] r[:64] and learned(r) = W_U[:, 64:] r[64:].

Conditions (dual-operation task of doc 57 addendum A):
  B, Bp (d=96, parameter-matched to K2's effective trainable count), Bc (B, FLOP-matched to K5),
  S1 typed embeddings, S2 + binding, S3 + canonical quantities/conversion, S4 + arithmetic (=K2 structurally),
  Srand (S4 slices filled with random values at installed RMS), P (S4 + output ownership gate), K5s (S4 + interchange loss).

    python -m sci.p8 run GROUP [--workers 3]
"""
from __future__ import annotations

import copy
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nri.model import TinyTyped  # noqa: E402
from sci import construct as C  # noqa: E402
from sci import hybrid as H  # noqa: E402
from sci import p8_cost as PC  # noqa: E402

T, P = C.T, C.P
OUT = ROOT / "runs/phase8"
CKPT = Path("/home/user/p8_ckpt")          # checkpoints outside Git
FLAG = H.FLAG_DIM


# ------------------------------------------------------------------ masks for installation levels
def head_rows(d, h, parts=(0, 1, 2)):
    return torch.cat([torch.arange(part * d + 16 * h, part * d + 16 * h + 16) for part in parts])


def level_masks(m, level):
    """returns (W0 dict, F dict, Z dict) for the installation level on a d=128 hybrid model (after H.install)."""
    full = H.install(m)                                   # installs Design C + OP flag; returns its full mask
    W0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    Fm = {n: torch.zeros_like(p, dtype=torch.bool) for n, p in m.named_parameters()}
    Zm = {n: torch.zeros_like(p, dtype=torch.bool) for n, p in m.named_parameters()}
    d = m.d
    if level in ("S4", "Srand", "P", "K5s"):
        Fm = {n: v.clone() for n, v in full.items()}
    else:
        for n in ("E_c.weight", "E_p.weight", "E_t.weight"):
            Fm[n] = full[n].clone()
        if level in ("S2", "S3"):
            heads1 = [0] if level == "S2" else [0, 1, 2, 3]
            for h in heads1:
                r = head_rows(d, h)
                Fm["attn1.qkv.weight"][r] = True; Fm["attn1.qkv.bias"][r] = True
                Fm["attn1.o.weight"][:, 16 * h:16 * h + 16] = True
            Fm["attn1.o.bias"][:64] = True
        if level == "S3":
            for h in range(4):
                r = head_rows(d, h)
                Fm["attn2.qkv.weight"][r] = True; Fm["attn2.qkv.bias"][r] = True
                Fm["attn2.o.weight"][:, 16 * h:16 * h + 16] = True
            Fm["attn2.o.bias"][:64] = True
    if level != "none":
        # write protection: nothing learned may write into dims 0-63
        for layer in ("attn1", "attn2"):
            Zm[f"{layer}.o.weight"][:64, :] = ~Fm[f"{layer}.o.weight"][:64, :]
            Zm[f"{layer}.o.bias"][:64] = ~Fm[f"{layer}.o.bias"][:64]
        Zm["W_out.weight"][:64, :] = ~Fm["W_out.weight"][:64, :]
        Zm["W_out.bias"][:64] = ~Fm["W_out.bias"][:64]
    if level == "Srand":
        g = torch.Generator().manual_seed(4242)
        for n in W0:
            if Fm[n].any():
                rms = W0[n][Fm[n]].float().pow(2).mean().sqrt()
                W0[n] = W0[n].clone()
                W0[n][Fm[n]] = torch.randn(int(Fm[n].sum()), generator=g) * rms
    # for levels below S4, the un-installed parts of the slice must be randomly initialised (not the installed values)
    if level in ("S1", "S2", "S3"):
        ref = copy.deepcopy(m)
        torch.manual_seed(1000 + hash(level) % 1000)
        fresh = TinyTyped("mass_only", d=m.d, n_heads=8, d_mlp=m.d_mlp, n_types=1, aux_classes=0, vocab=26, seq_len=8, n_out=P)
        with torch.no_grad():
            for (n, p), (_, q) in zip(m.named_parameters(), fresh.named_parameters()):
                inst = full[n] & ~Fm[n]
                p.data[inst] = q.data[inst]
                W0[n] = p.detach().clone()
    return W0, Fm, Zm


class _ResidView(torch.nn.Module):
    """exposes TinyTyped.resid_pre_mlp as forward, so torch.func.functional_call can substitute effective weights."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, tokens):
        return self.m.resid_pre_mlp(tokens, torch.zeros_like(tokens))


class Masked(torch.nn.Module):
    def __init__(self, base, W0=None, Fm=None, Zm=None, gate=False):
        super().__init__()
        self.base, self.gate = base, gate
        self.names = [n for n, _ in base.named_parameters()]
        for n in self.names:
            k = n.replace(".", "__")
            self.register_buffer(f"W0__{k}", W0[n] if W0 else torch.zeros(0))
            self.register_buffer(f"F__{k}", Fm[n] if Fm else torch.zeros(0, dtype=torch.bool))
            self.register_buffer(f"Z__{k}", Zm[n] if Zm else torch.zeros(0, dtype=torch.bool))

    def eff(self):
        out = {}
        for n, p in self.base.named_parameters():
            k = n.replace(".", "__")
            W0, Fm, Zm = getattr(self, f"W0__{k}"), getattr(self, f"F__{k}"), getattr(self, f"Z__{k}")
            w = p
            if Fm.numel():
                w = torch.where(Fm, W0, w)
            if Zm.numel():
                w = w.masked_fill(Zm, 0.0)
            out[n] = w
        return out

    def forward(self, tokens):
        e = self.eff()
        return self.head_from_resid(self.residual(tokens, e), e)

    def residual(self, tokens, e=None):
        e = e or self.eff()
        return torch.func.functional_call(_ResidView(self.base), {f"m.{k}": v for k, v in e.items()}, (tokens,))

    def head_from_resid(self, h, e=None):
        e = e or self.eff()
        r = h + torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(h, e["W_in.weight"], e["W_in.bias"])),
                                           e["W_out.weight"], e["W_out.bias"])
        WU = e["W_U.weight"]
        if not self.gate:
            return r @ WU.T
        return r[:, :64] @ WU[:, :64].T + h[:, FLAG:FLAG + 1] * (r[:, 64:] @ WU[:, 64:].T)


# ------------------------------------------------------------------ conditions
def build(cond, seed):
    if cond == "Bp":
        torch.manual_seed(seed)
        base = TinyTyped("mass_only", d=96, n_heads=8, d_mlp=384, n_types=1, aux_classes=0, vocab=26, seq_len=8, n_out=P)
        return Masked(base), dict(d=96, heads=8, mlp=384)
    base = H.make_model("full", seed)
    if cond in ("B", "Bc"):
        return Masked(base), dict(d=128, heads=8, mlp=512)
    level = cond
    W0, Fm, Zm = level_masks(base, level)
    return Masked(base, W0, Fm, Zm, gate=(cond == "P")), dict(d=128, heads=8, mlp=512)


def counts(model):
    tot = trn = 0
    for n, p in model.base.named_parameters():
        k = n.replace(".", "__")
        Fm, Zm = getattr(model, f"F__{k}"), getattr(model, f"Z__{k}")
        tot += p.numel()
        blocked = torch.zeros_like(p, dtype=torch.bool)
        if Fm.numel():
            blocked |= Fm
        if Zm.numel():
            blocked |= Zm
        trn += int((~blocked).sum())
    return tot, trn


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, L, tok, y, un, train_idx=None):
    model.eval()
    e = model.eff()
    lg, sub = [], []
    for i in range(0, len(y), 4096):
        t = tok[i:i + 4096]
        h = model.residual(t, e)
        lg.append(model.head_from_resid(h, e))
        r = h + torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(h, e["W_in.weight"], e["W_in.bias"])),
                                           e["W_out.weight"], e["W_out.bias"])
        sub.append(r[:, :64] @ e["W_U.weight"][:, :64].T)
    lg, sc = torch.cat(lg), torch.cat(sub)
    pr = lg.argmax(-1)
    op = L["op"]
    res = {}
    for o, nm in ((0, "OP1"), (1, "OP2")):
        s = op == o
        res[f"{nm}_familiar"] = float((pr[s & ~un] == y[s & ~un]).float().mean())
        res[f"{nm}_unseen"] = float((pr[s & un] == y[s & un]).float().mean())
    want = torch.where((op == 0)[:, None], F.one_hot(y, P).float(), torch.zeros(len(y), P))
    res["OP1_subcircuit_exact"] = float((sc[op == 0] == want[op == 0]).all(1).float().mean())
    res["OP1_endtoend_correct"] = res["OP1_familiar"]
    if train_idx is not None:
        res["OP2_train_acc"] = float((pr[train_idx] == y[train_idx]).float().mean())
    model.train()
    return res


def competent(e):
    return e["OP2_familiar"] >= 0.99 and e["OP2_unseen"] >= 0.95


# ------------------------------------------------------------------ resumable trainer
def train(name, cond, seed, steps=6000, frac=1.0, sampling="random", eval_every=200, interchange=False):
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    fout = OUT / "runs" / f"{name}.json"
    fout.parent.mkdir(parents=True, exist_ok=True)
    if fout.exists():
        return json.loads(fout.read_text())
    model, arch = build(cond, seed)
    L = H.domain2()
    tok, y = H.build2(L)
    un = H.unseen2(L)
    pool = torch.nonzero(~un).flatten()
    if frac < 1.0:
        pool = select_pool(L, pool, frac, sampling, seed)
    opt = torch.optim.AdamW([p for p in model.base.parameters()], lr=2e-3, weight_decay=0.01, betas=(0.9, 0.999))
    gb = torch.Generator().manual_seed(1000 + seed)
    M, A, S = C.canon_scales(L, dict(MU=T.MASS_S, AU=T.ACC_S))
    hist, step0, wall0 = [], 0, 0.0
    ck = CKPT / f"{name}.pt"
    if ck.exists():
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); gb.set_state(st["gen"])
        hist, step0, wall0 = st["hist"], st["step"], st["wall"]
    tot, trn = counts(model)
    fstep = PC.step_flops(arch["d"], arch["heads"], arch["mlp"], interchange_frac=0.5 if interchange else 0.0)
    feval = PC.eval_flops(arch["d"], arch["heads"], arch["mlp"])
    t0 = time.time()
    for step in range(step0, steps + 1):
        if step % eval_every == 0 and (not hist or hist[-1]["step"] != step):
            e = evaluate(model, L, tok, y, un, train_idx=pool[L["op"][pool] == 1])
            e.update(step=step, train_flops=step * fstep, wall=wall0 + time.time() - t0)
            hist.append(e)
        if step == steps:
            break
        if step % 500 == 0 and step > step0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), gen=gb.get_state(), hist=hist, step=step,
                            wall=wall0 + time.time() - t0), ck)
        bidx = pool[torch.randint(0, len(pool), (256,), generator=gb)]
        t = tok[bidx]
        e = model.eff()
        h = model.residual(t, e)
        loss = F.cross_entropy(model.head_from_resid(h, e), y[bidx])
        if interchange:
            o1 = bidx[L["op"][bidx] == 0]
            if len(o1) > 1:
                src = o1[torch.randperm(len(o1), generator=gb)]
                hb = model.residual(tok[o1], e); hs = model.residual(tok[src], e)
                loss = loss + F.cross_entropy(model.head_from_resid(torch.cat([hs[:, :13], hb[:, 13:]], 1), e),
                                              M[src] * A[o1] * S[o1] % P)
        opt.zero_grad(); loss.backward(); opt.step()
    wall = wall0 + time.time() - t0
    comp = next((h_ for h_ in hist if competent(h_)), None)
    r = dict(name=name, cond=cond, seed=seed, frac=frac, sampling=sampling, steps=steps, arch=arch,
             params_total=tot, params_effective_trainable=trn, flops_per_step=fstep, eval_flops_each=feval,
             steps_to_competence=comp["step"] if comp else None, flops_to_competence=comp["train_flops"] if comp else None,
             wall_to_competence=comp["wall"] if comp else None, wall_total=wall,
             peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, final=hist[-1], history=hist)
    fout.write_text(json.dumps(r, indent=1))
    if ck.exists():
        ck.unlink()
    return r


def select_pool(L, pool, frac, sampling, seed):
    g = torch.Generator().manual_seed(25 if sampling == "random_phase7" else 7000 + seed + int(frac * 100))
    n = int(len(pool) * frac)
    if sampling.startswith("random"):
        return pool[torch.randperm(len(pool), generator=g)[:n]]
    # structured semantic coverage: round-robin over canonical (op, M, A, S) cells
    M, A, S = C.canon_scales(L, dict(MU=T.MASS_S, AU=T.ACC_S))
    key = (L["op"][pool] * 10000 + M[pool] * 1000 + A[pool] * 10 + L["fu"][pool]).tolist()
    cells = {}
    perm = torch.randperm(len(pool), generator=g).tolist()
    for i in perm:
        cells.setdefault(key[i], []).append(int(pool[i]))
    out, lists = [], list(cells.values())
    k = 0
    while len(out) < n:
        for lst in lists:
            if k < len(lst) and len(out) < n:
                out.append(lst[k])
        k += 1
    return torch.tensor(out)


GROUPS = {
    "S": [(f"S_{c}_s{s}", c, s, {}) for c in ("S1", "S2", "S3", "S4", "Srand") for s in (0, 1, 2)],
    "A3": [(f"A3_Bp_s{s}", "Bp", s, {}) for s in (0, 1, 2)] + [(f"A3_Bc_s{s}", "Bc", s, dict(steps=12000)) for s in (0, 1, 2)],
    "P": [(f"P_{c}_{fr}_s{s}", c, s, dict(frac=f_, sampling="random_phase7" if f_ < 1 else "random",
                                          interchange=(c == "K5s")))
          for c in ("P", "K5s") for (fr, f_) in (("full", 1.0), ("q25", 0.25)) for s in (0, 1, 2)],
    "E": [(f"E_{c}_{fr}_{smp}_s{s}", c, s, dict(frac=fr, sampling=smp, interchange=(c == "K5s")))
          for c in ("B", "K5s", "P") for fr in (0.25, 0.40, 0.55, 0.70, 0.85, 1.0)
          for smp in (("random", "structured") if fr in (0.25, 0.40) else ("random",)) for s in (0, 1)],
}


def _job(j):
    name, cond, seed, kw = j
    r = train(name, cond, seed, **kw)
    return name, r["steps_to_competence"], {k: round(v, 3) for k, v in r["final"].items() if isinstance(v, float)}


def run(group, workers=3):
    from concurrent.futures import ProcessPoolExecutor
    jobs = GROUPS[group]
    with ProcessPoolExecutor(workers) as ex:
        for out in ex.map(_job, jobs):
            print(*out, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run":
        run(a[1], int(a[a.index("--workers") + 1]) if "--workers" in a else 3)
