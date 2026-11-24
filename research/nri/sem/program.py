"""Post-training feature programming (Issue #25 §3; docs/30 §30.8). Exploratory.

Adds NEW capabilities to a trained SemStudent by three bounded methods and measures acquisition,
paraphrase/order generalisation, held-out values, preservation of every previous dev task, interference
between two new capabilities, parameters added, and exact rollback.

Capabilities (new question types over existing concepts; modular toy, not real-world arithmetic):
  hard pair:  gap = (M - A) mod 13 ("gap"), total = (M + A) mod 13 ("total")
  easy pair:  null = is the product zero? yes/no ("null"; balanced), kind = which mass unit, 0/1/2 ("kind")
  (the easy pair was added after a non-study development run showed no method learns the hard pair in 600
  steps on this student - it is the positive control that the methods can add anything at all)
M, A are the canonical mass and acceleration. Both capabilities share their frames and differ only in the cue
word, so coexistence is a hard test. New words get new embedding rows in every method (counted).

Methods (the original network is never modified in place; it is deep-copied, hashed and restored):
  head     frozen student + new token rows + gated residual head  logits = W_U r + G(h)
  adapter  frozen student + new token rows + rank-r residual adapters after mlp1 / mlp2 / attn3
  bounded  existing weights of a declared subset fine-tuned with a per-tensor Frobenius bound
           ||W - W0|| <= eps ||W0|| (projection after every step) + new token rows

    python -m sem.program <checkpoint_dir> <out.json> [--steps N] [--methods head,adapter,bounded]
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sem import data as D  # noqa: E402
from sem import lang as L  # noqa: E402
from sem import model as SM  # noqa: E402
from sem import observe as O  # noqa: E402

CAPS = {"gap": lambda a, M, A: str((M - A) % 13), "total": lambda a, M, A: str((M + A) % 13),
        "null": lambda a, M, A: "yes" if M * A % 13 == 0 else "no",        # easy pair: lexically grounded
        "kind": lambda a, M, A: str(a["mu"])}                            # which mass unit (0/1/2)
PAIRS = {"hard": ("gap", "total"), "easy": ("null", "kind")}
FRAMES = {   # {cue} is the capability word; train frames are used for training, the others only for evaluation
    "train": [T for T in (L.T("{m} {mu} and {a} {au} : what is the {cue} ?"), L.T("find the {cue} for {m} {mu} and {a} {au} ."))],
    "heldout_order": [L.T("for {a} {au} and {m} {mu} , what is the {cue} ?")],        # acceleration first
    "heldout_phrase": [L.T("the {cue} of {m} {mu} and {a} {au} is what ?")],
}
NEW_WORDS = ["gap", "total", "null", "kind"]
STEPS, LR, BATCH, NEW_FRAC = 1200, 3e-3, 128, 0.5
EPS_BOUND = 0.10
BOUNDED_SUBSET = ("attn3", "mlp2", "W_in", "W_out")


# ------------------------------------------------------------------ data + truth contract
def numeric_split(ast):
    k = set()
    for v, u in ((ast["m"], ast["mu"]), (ast["a"], ast["au"])):
        if u == L.OOD_UNIT and v >= 9:
            k.add("dev" if v in L.DEV_OOD_VALUES else "sealed")
    return "train" if not k else k.pop() if len(k) == 1 else "crossed"


def tuple_split(ast):
    return L.tuple_split(dict(task="prog", m=ast["m"], mu=ast["mu"], a=ast["a"], au=ast["au"]))   # cue-independent


def answer(ast):
    M = ast["m"] * L.MASS_S[ast["mu"]] % 13
    A = ast["a"] * L.ACC_S[ast["au"]] % 13
    return CAPS[ast["cap"]](ast, M, A)


def render(ast, frame, style):
    out = []
    for w in frame:
        if w == "{cue}":
            out.append(ast["cap"])
        elif w in ("{m}", "{a}"):
            out.append(L.num_tok(ast[w[1:-1]], style))
        elif w == "{mu}":
            out.append(L.unit_tok("mass", ast["mu"], ast["m"], style))
        elif w == "{au}":
            out.append(L.unit_tok("acc", ast["au"], ast["a"], style))
        else:
            out.append(w)
    return out


def validate(ast, toks):
    """Independent re-reading: exactly one cue word, exactly one mass and one acceleration value-unit pair
    (number immediately followed by a unit of that dimension); recomputed answer must match."""
    cues = [w for w in toks if w in CAPS]
    if len(cues) != 1:
        return False, "cue"
    got = {}
    for i in range(len(toks) - 1):
        if toks[i] in L._NUM and toks[i + 1] in L._UNIT:
            dim, u = L._UNIT[toks[i + 1]]
            if dim in got:
                return False, "duplicate " + dim
            got[dim] = (L._NUM[toks[i]], u)
    if set(got) != {"mass", "acc"}:
        return False, "pairs"
    re = dict(cap=cues[0], m=got["mass"][0], mu=got["mass"][1], a=got["acc"][0], au=got["acc"][1])
    return (answer(re) == answer(ast), "ok" if answer(re) == answer(ast) else "answer")


def sample(rng, cap, splits=("train",), numeric=("train",), frames="train", style_p=(0.3, 0.3)):
    want_yes = rng.random() < 0.5                                     # balance the yes/no capability
    while True:
        ast = dict(cap=cap, m=rng.randrange(13), mu=rng.randrange(3), a=rng.randrange(13), au=rng.randrange(3))
        if cap == "null" and (answer(ast) == "yes") != want_yes:
            continue
        if tuple_split(ast) in splits and numeric_split(ast) in numeric:
            break
    toks = render(ast, rng.choice(FRAMES[frames]), D.draw_style(rng, *style_p))
    ok, why = validate(ast, toks)
    assert ok, (ast, toks, why)                      # renders are deterministic; a failure is a bug
    return dict(ast=ast, tokens=toks, roles=["other"] * len(toks), answer=answer(ast))


def encode(exs, stoi):
    tok = torch.full((len(exs), D.MAX_LEN), D.PAD, dtype=torch.long)
    for i, ex in enumerate(exs):
        ids = [stoi.get(w, D.UNK) for w in ex["tokens"]][-(D.MAX_LEN - 1):] + [D.QRY]
        tok[i, D.MAX_LEN - len(ids):] = torch.tensor(ids)
    return tok, torch.tensor([L.ANS_ID[ex["answer"]] for ex in exs])


# ------------------------------------------------------------------ programmable wrapper
class ExtEmbed(nn.Module):
    def __init__(self, old: nn.Embedding, n_new):
        super().__init__()
        self.old, self.V = old, old.num_embeddings
        self.new = nn.Embedding(n_new, old.embedding_dim)
        nn.init.normal_(self.new.weight, std=float(old.weight.std()))

    def forward(self, tok):
        is_new = tok >= self.V
        e = self.old(tok.clamp(max=self.V - 1))
        return torch.where(is_new.unsqueeze(-1), self.new((tok - self.V).clamp(min=0)), e)


class Adapter(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.A, self.B = nn.Linear(d, r), nn.Linear(r, d)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def forward(self, x):
        return self.B(F.relu(self.A(x)))


class Programmed(nn.Module):
    def __init__(self, base: SM.SemStudent, method, rank=8, head_hidden=64):
        super().__init__()
        self.base, self.method = base, method
        self.base.E = ExtEmbed(base.E, len(NEW_WORDS))
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.E.new.weight.requires_grad_(True)
        self.added = nn.ModuleDict()
        self._hooks = []
        if method == "head":
            self.added["gate_head"] = nn.Sequential(nn.Linear(base.d, head_hidden), nn.ReLU(), nn.Linear(head_hidden, len(L.ANSWERS)))
            nn.init.zeros_(self.added["gate_head"][2].weight)
            nn.init.zeros_(self.added["gate_head"][2].bias)
        elif method == "adapter":
            for site in ("mlp1", "mlp2", "attn3"):
                self.added[site] = Adapter(base.d, rank)
                mod = getattr(self.base, site)
                self._hooks.append(mod.register_forward_hook(
                    lambda m, i, o, s=site: (o[0] + self.added[s](i[0]),) + tuple(o[1:]) if isinstance(o, tuple) else o + self.added[s](i[0])))
        elif method == "bounded":
            self.W0 = {}
            for n, p in self.base.named_parameters():
                if n.split(".")[0] in BOUNDED_SUBSET:
                    p.requires_grad_(True)
                    self.W0[n] = p.detach().clone()
        else:
            raise ValueError(method)

    def forward(self, tok):
        h, _ = self.base.trunk(tok)
        logits = self.base.head(h)
        if self.method == "head":
            logits = logits + self.added["gate_head"](h)
        return logits

    @torch.no_grad()
    def project(self):
        if self.method != "bounded":
            return
        for n, p in self.base.named_parameters():
            if n in self.W0:
                d = p - self.W0[n]
                lim = EPS_BOUND * self.W0[n].norm()
                if d.norm() > lim:
                    p.copy_(self.W0[n] + d * (lim / d.norm()))

    def added_params(self):
        n = self.base.E.new.weight.numel() + sum(p.numel() for p in self.added.parameters())
        mod = sum(p.numel() for k, p in self.base.named_parameters() if k in getattr(self, "W0", {}))
        return dict(added=n, existing_modified=mod)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def sd_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ evaluation
def eval_sets(caps, seed=5, n=384):
    rng = random.Random(seed)
    out = {}
    for cap in caps:
        out[f"{cap}/id"] = [sample(rng, cap, ("dev",), ("train",), "train") for _ in range(n)]
        out[f"{cap}/order"] = [sample(rng, cap, ("dev",), ("train",), "heldout_order") for _ in range(n)]
        out[f"{cap}/phrase"] = [sample(rng, cap, ("dev",), ("train",), "heldout_phrase") for _ in range(n)]
        out[f"{cap}/ood_value"] = [sample(rng, cap, ("train", "dev"), ("dev",), "train") for _ in range(n)]
    return out


@torch.no_grad()
def acc_new(net, exs, stoi):
    tok, ans = encode(exs, stoi)
    return round((net(tok).argmax(-1) == ans).float().mean().item(), 4)


@torch.no_grad()
def acc_old(net, devsets, vocab):
    return {k: round((net(D.encode_batch(v, vocab)["tok"]).argmax(-1) == D.encode_batch(v, vocab)["ans"]).float().mean().item(), 4)
            for k, v in devsets.items()}


def replay_lesson():
    import run_sem as RS
    return RS.FIXED_CURRICULUM[-1]                   # all five old tasks, training pools only


def train_cap(net, caps, stoi, vocab, steps, seed):
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    rng = random.Random(seed)
    rep = replay_lesson()
    net.train()
    for s in range(steps):
        nn_new = int(BATCH * NEW_FRAC)
        new = [sample(rng, rng.choice(caps)) for _ in range(nn_new)]
        old, _, _ = D.lesson_batch(rep, BATCH - nn_new, rng)
        tn, an = encode(new, stoi)
        bo = D.encode_batch(old, vocab)
        tok, ans = torch.cat([tn, bo["tok"]]), torch.cat([an, bo["ans"]])
        loss = F.cross_entropy(net(tok), ans)
        opt.zero_grad()
        loss.backward()
        opt.step()
        net.project()
    net.eval()


def run(ckpt, methods=("head", "adapter", "bounded"), steps=STEPS, seed=0, pair="hard"):
    c1, c2 = PAIRS[pair]
    torch.manual_seed(seed)
    vocab = D.Vocab.from_json(json.loads((ROOT / "runs/zoo_sem/eval/vocab.json").read_text()))
    stoi = dict(vocab.stoi, **{w: len(vocab.itos) + i for i, w in enumerate(NEW_WORDS)})
    base = SM.SemStudent(len(vocab.itos))
    base.load_state_dict(load_file(Path(ckpt) / "model.safetensors"))
    base.eval()
    h0 = sd_hash(base)
    backup = copy.deepcopy(base.state_dict())
    devsets = O.dev_sets(ROOT / "runs/zoo_sem/eval")
    ev = eval_sets((c1, c2))
    res = dict(checkpoint=str(ckpt), pair=pair, capabilities=[c1, c2], base_hash=h0, steps_per_capability=steps, before=dict(old=acc_old(base, devsets, vocab)))
    for meth in methods:
        t0 = time.process_time()
        net = Programmed(copy.deepcopy(base), meth)
        r = dict(params=net.added_params())
        r["new_before"] = {k: acc_new(net, v, stoi) for k, v in ev.items()}
        train_cap(net, [c1], stoi, vocab, steps, seed + 1)                         # capability 1
        r["after_cap1"] = dict(new={k: acc_new(net, v, stoi) for k, v in ev.items()}, old=acc_old(net, devsets, vocab))
        train_cap(net, [c2, c1], stoi, vocab, steps, seed + 2)                     # capability 2 (+ cap-1 replay)
        r["after_cap2"] = dict(new={k: acc_new(net, v, stoi) for k, v in ev.items()}, old=acc_old(net, devsets, vocab))
        r["cpu_s"] = round(time.process_time() - t0, 1)
        net.remove_hooks()
        res[meth] = r
    # rollback: the original object is untouched and a restore from the backup reproduces the hash
    assert sd_hash(base) == h0, "original weights changed"
    fresh = SM.SemStudent(len(vocab.itos))
    fresh.load_state_dict(backup)
    res["rollback"] = dict(original_untouched=sd_hash(base) == h0, restore_hash_equal=sd_hash(fresh) == h0)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--methods", default="head,adapter,bounded")
    ap.add_argument("--pair", default="hard", choices=sorted(PAIRS))
    a = ap.parse_args()
    torch.set_num_threads(2)
    r = run(a.ckpt, tuple(a.methods.split(",")), a.steps, pair=a.pair)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "before"}, indent=1)[:6000])
