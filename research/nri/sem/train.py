"""Training state, objectives, safe persistence and the per-100-step observation (Issue #25; docs/30).

Persistence never uses pickle for anything that is loaded back: model and RNG tensors go to
safetensors, optimizer/scheduler state is loaded with torch.load(weights_only=True) (tensor-only
unpickler), Python RNG state and metadata go to JSON.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from sem import data as D
from sem import lang as L
from sem.model import BLOCK, SemStudent

THREADS = 4
BATCH = 128
PRE_LR, PRE_WARMUP, PRE_CAP = 2e-3, 200, 5000
CONT_STEPS, CONT_LR, CONT_WARMUP, CONT_MIN_LR = 1500, 1e-3, 100, 1e-4


def set_det():
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


# ------------------------------------------------------------------ state
def new_state(vocab, seed, phase="pretrain", lr=PRE_LR):
    set_det()
    torch.manual_seed(seed)
    m = SemStudent(len(vocab.itos))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    return dict(model=m, opt=opt, step=0, phase=phase, rng=random.Random(seed), seed=seed, vocab=vocab,
                lr_fn=phase, history=[])


def lr_at(phase, step):
    if phase == "pretrain":
        return PRE_LR * min(1.0, step / PRE_WARMUP)
    if step <= CONT_WARMUP:
        return CONT_LR * step / CONT_WARMUP
    p = (step - CONT_WARMUP) / (CONT_STEPS - CONT_WARMUP)
    return CONT_MIN_LR + 0.5 * (CONT_LR - CONT_MIN_LR) * (1 + math.cos(math.pi * min(p, 1.0)))


def restart_for_continuation(st, seed):
    """Continuation policy (identical for A and B): same weights, FRESH AdamW, cosine schedule, step 0,
    a continuation-specific data RNG (identical seed for both arms)."""
    st["opt"] = torch.optim.AdamW(st["model"].parameters(), lr=CONT_LR, weight_decay=0.01)
    st["step"] = 0
    st["phase"] = "continuation"
    st["rng"] = random.Random(seed)
    st["history"] = []
    return st


def save_state(st, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in st["model"].state_dict().items()}, folder / "model.safetensors")
    torch.save(dict(opt=st["opt"].state_dict()), folder / "optim.pt")
    save_file(dict(torch_rng=torch.get_rng_state()), folder / "rng.safetensors")
    meta = dict(step=st["step"], phase=st["phase"], seed=st["seed"], py_rng=_rng_to_json(st["rng"].getstate()),
                vocab=st["vocab"].to_json(), history=st["history"])
    (folder / "state.json").write_text(json.dumps(meta))
    return state_hash(st)


def load_state(folder):
    folder = Path(folder)
    meta = json.loads((folder / "state.json").read_text())
    vocab = D.Vocab.from_json(meta["vocab"])
    st = new_state(vocab, meta["seed"], meta["phase"])
    st["model"].load_state_dict(load_file(folder / "model.safetensors"))
    st["opt"] = torch.optim.AdamW(st["model"].parameters(), lr=CONT_LR if meta["phase"] == "continuation" else PRE_LR,
                                  weight_decay=0.01)
    st["opt"].load_state_dict(torch.load(folder / "optim.pt", weights_only=True)["opt"])
    torch.set_rng_state(load_file(folder / "rng.safetensors")["torch_rng"])
    st["step"], st["phase"], st["history"] = meta["step"], meta["phase"], meta["history"]
    st["rng"].setstate(_rng_from_json(meta["py_rng"]))
    return st


def _rng_to_json(s):
    return [s[0], list(s[1]), s[2]]


def _rng_from_json(s):
    return (s[0], tuple(s[1]), s[2])


def state_hash(st):
    h = hashlib.sha256()
    for k, v in st["model"].state_dict().items():
        h.update(k.encode())
        h.update(v.detach().numpy().tobytes())
    for k, sd in st["opt"].state_dict()["state"].items():
        for kk, vv in sd.items():
            h.update(f"{k}{kk}".encode())
            h.update(vv.numpy().tobytes() if torch.is_tensor(vv) else str(vv).encode())
    h.update(json.dumps(_rng_to_json(st["rng"].getstate())).encode())
    h.update(str(st["step"]).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ objectives
def loss_on(model, b, twins, obj, *, retain_parts=False):
    """Answer CE + aux role CE (training only) + IIT on force examples (block swap within the batch;
    no extra forward) + paraphrase consistency (symmetric KL between twins; no extra forward)."""
    h, x1 = model.trunk(b["tok"])
    logits = model.head(h)
    parts = dict(task=F.cross_entropy(logits, b["ans"]))
    loss = parts["task"]
    if obj.get("aux_role", 0) > 0:
        mask = b["role"] != L.ROLE_ID["pad"]
        parts["aux"] = F.cross_entropy(model.aux(x1)[mask], b["role"][mask])
        loss = loss + obj["aux_role"] * parts["aux"]
    fi = (b["MAS"][:, 0] >= 0).nonzero().squeeze(1)
    if obj.get("iit", 0) > 0 and len(fi) >= 2:
        src = fi.roll(1)
        hp = h[fi].clone()
        hp[:, BLOCK[0]:BLOCK[1]] = h[src][:, BLOCK[0]:BLOCK[1]]
        M, A, S = b["MAS"][src, 0], b["MAS"][fi, 1], b["MAS"][fi, 2]
        cf = M * A * S % 13
        parts["iit"] = F.cross_entropy(model.head(hp), cf)
        loss = loss + obj["iit"] * parts["iit"]
    if obj.get("consistency", 0) > 0 and twins:
        i = torch.tensor([t[0] for t in twins])
        j = torch.tensor([t[1] for t in twins])
        lp, lq = F.log_softmax(logits[i], -1), F.log_softmax(logits[j], -1)
        parts["cons"] = 0.5 * (F.kl_div(lp, lq, log_target=True, reduction="batchmean") +
                               F.kl_div(lq, lp, log_target=True, reduction="batchmean"))
        loss = loss + obj["consistency"] * parts["cons"]
    return loss, parts if retain_parts else {k: v.item() for k, v in parts.items()}


def train_steps(st, lesson, n, log=None, *, recorder=None):
    """n optimizer steps on batches drawn from `lesson` (validated). Returns per-step digests."""
    D.validate_lesson(lesson)
    m, opt = st["model"], st["opt"]
    obj = dict(dict(aux_role=0.5, iit=3.0, consistency=0.0), **lesson.get("objectives", {}))
    digests = []
    tokens = 0
    for _ in range(n):
        st["step"] += 1
        for g in opt.param_groups:
            g["lr"] = lr_at(st["phase"], st["step"])
        exs, twins, quar = D.lesson_batch(lesson, BATCH, st["rng"])
        b = D.encode_batch(exs, st["vocab"])
        tokens += int((b["tok"] != D.PAD).sum())
        o = dict(obj)
        if st["phase"] == "pretrain":
            o["iit"] = obj["iit"] * min(1.0, st["step"] / 1000)      # IIT warm-up (doc 14 style)
        loss, parts = loss_on(m, b, twins, o)
        opt.zero_grad()
        loss.backward()
        if recorder is None:
            opt.step()
        else:
            recorder(st, b, exs, twins, o, loss, parts)
        digests.append(hashlib.sha256(b["tok"].numpy().tobytes()).hexdigest()[:12])
        if log is not None:
            log.append(dict(step=st["step"], loss=round(loss.item(), 5), quarantined=quar, **{k: round(v, 5) for k, v in parts.items()}))
    return digests, tokens
