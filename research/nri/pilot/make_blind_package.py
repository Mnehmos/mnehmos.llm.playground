"""Build a blinded package for an LLM interpreter (Experiment A pilot).

    python pilot/make_blind_package.py --run runs/zoo/typed_s0 --blind-id net_A --out <dir>

The package holds ONLY: the architecture code (task-agnostic), the weights, a pool of
valid inputs (token ids + the second "channel" ids, unlabelled), and instructions.
It holds no task description, no type names, no training code and no labels.
The answer key (which run, which condition) is written to <out>/../answer_keys/, which
the interpreter is told not to read.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nri import tasks  # noqa: E402

ARCH = '''"""Architecture of the network you are inspecting. Load with load_net()."""
import math, torch, torch.nn as nn, torch.nn.functional as F

VOCAB, N_CHANNEL2, SEQ_LEN, N_OUT = {vocab}, {ntypes}, {seqlen}, {nout}

class Attn(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__(); self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
    def forward(self, x, return_pattern=False):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        pat = ((q @ k.transpose(-1, -2)) / math.sqrt(self.dh)).softmax(-1)
        out = self.o((pat @ v).transpose(1, 2).reshape(B, L, d))
        return (out, pat) if return_pattern else out

class Net(nn.Module):
    """x0 = E_c[tokens]*mask_c + E_t[channel2]*mask_t + E_p[pos]; two attention layers
    (bidirectional, no LayerNorm); h = x[:, -1]; MLP with residual; logits = W_U(r)."""
    def __init__(self, d=64, n_heads=4, d_mlp=256):
        super().__init__(); self.d = d
        self.E_c = nn.Embedding(VOCAB, d); self.E_t = nn.Embedding(N_CHANNEL2, d)
        self.E_p = nn.Embedding(SEQ_LEN, d)
        self.attn1 = Attn(d, n_heads); self.attn2 = Attn(d, n_heads)
        self.W_in = nn.Linear(d, d_mlp); self.W_out = nn.Linear(d_mlp, d)
        self.W_U = nn.Linear(d, N_OUT, bias=False)
        self.register_buffer("mask_c", torch.ones(d)); self.register_buffer("mask_t", torch.ones(d))
    def embed(self, tokens, ch2):
        pos = torch.arange(tokens.shape[1])
        return self.E_c(tokens) * self.mask_c + self.E_t(ch2) * self.mask_t + self.E_p(pos)
    def resid_pre_mlp(self, tokens, ch2):
        x = self.embed(tokens, ch2); x = x + self.attn1(x); x = x + self.attn2(x); return x[:, -1]
    def head_from_resid(self, h, W_in=None, b_in=None):
        W_in = self.W_in.weight if W_in is None else W_in
        b_in = self.W_in.bias if b_in is None else b_in
        return self.W_U(h + self.W_out(F.relu(h @ W_in.T + b_in)))
    def forward(self, tokens, ch2):
        return self.head_from_resid(self.resid_pre_mlp(tokens, ch2))

def load_net(path="weights.pt"):
    net = Net(); net.load_state_dict(torch.load(path, weights_only=True)); net.eval(); return net
'''

INSTRUCTIONS = """# Task: reverse-engineer an unknown neural network

You are given a small trained network and nothing about what it was trained to do.

Files here:
- `arch.py`      architecture; `from arch import load_net; net = load_net()`
- `weights.pt`   the trained weights
- `inputs.pt`    a dict with `tokens` [N, {seqlen}] and `ch2` [N, {seqlen}] (long tensors):
                 {npool} inputs drawn from the network's training distribution. `ch2` is a second
                 input channel; its meaning is not given (it may be constant).
- `nri-0.1.schema.json` and `nri_example_shape.json`: the output format.

Rules:
1. Work only inside this directory and your scratch subdirectory `work/`. Do NOT read any other
   files on this machine: no repositories, no parent directories, no answer keys. This is a
   blinded evaluation, and reading outside this directory invalidates it.
2. You may write and run Python (torch is installed) to query the network, read its weights and
   activations, and run interventions.
3. Deliverables, written into this directory:
   a. `nri.json`: an NRI v0.1 document (see the schema and the example shape). It must contain:
      - `variables`: every intermediate quantity you believe the network computes, each with a
        `binding` (a Python lambda over the token list `t` (list of ints, length {seqlen}) that
        returns an int) and an `ftype`.
      - `mechanisms`: at least one with an `output_fn` (lambda over dict `v` of variable values ->
        predicted output class).
      - `claims`: `behavior`, `linear_rep` (site `resid_pre_mlp`, give a `rank`) and `necessity`
        claims you believe are true. NOTE: `linear_rep` is a CAUSAL claim, tested by interchange
        intervention: the executor swaps the rank-r class-mean subspace of the variable between two
        inputs and checks that the output follows output_fn with that variable taken from the
        source input. Linear decodability alone does not pass it. Each will be executed and verified or refuted automatically,
        so only claim what you are willing to have tested.
   b. `edit_program.py` with `def program(host, doc, **args)` returning `(dW_in, db_in)`, the
      additive changes to `W_in.weight` [256, 64] and `W_in.bias` [256], so that:
        **wherever token {tx} occurs in the input, the network outputs what it would output if
        that token were {ty} instead; all other inputs keep their outputs.**
      `host.sampler(n, generator)` returns `(tokens, ch2)` from the input distribution,
      `host.site("resid_pre_mlp", tokens, ch2)` returns the [n, 64] activations, and `host.model`
      is the network. Compute the edit with code, not by gradient descent. Also add a claim of kind
      `edit` to `nri.json` that references `edit_program.py` (entry `program`), with a `scope`
      lambda and an `edit_fn` lambda written over your variables.
   c. `report.md`: what you think the network does, how you found out, and your confidence.
4. Budget: try to finish within about 60 tool calls.
"""

EXAMPLE = {
    "nri_version": "0.1",
    "model": {"id": "<blind id>", "weights_sha256": "<sha256 of weights.pt>",
              "sites": [{"id": "resid_pre_mlp", "dim": 64}]},
    "author": {"kind": "llm", "blinded": True, "model_name": "<you>"},
    "variables": [{"id": "V_example", "name": "...", "ftype": "CATEGORY",
                   "binding": "lambda t: t[0]"}],
    "mechanisms": [{"id": "M_main", "site_in": "resid_pre_mlp", "reads": ["V_example"],
                    "output_fn": "lambda v: v['V_example'] % 2"}],
    "claims": [
        {"id": "C1", "kind": "behavior", "mechanism": "M_main", "predicate": [">=", 0.95]},
        {"id": "C2", "kind": "linear_rep", "site": "resid_pre_mlp", "variable": "V_example",
         "rank": 3, "mechanism": "M_main", "predicate": [">=", 0.9]},
        {"id": "C3", "kind": "necessity", "site": "resid_pre_mlp", "variable": "V_example",
         "rank": 3, "mechanism": "M_main", "predicate": ["<=", 0.5]},
        {"id": "C4", "kind": "edit", "mechanism": "M_main",
         "program": {"path": "edit_program.py", "entry": "program", "args": {}},
         "scope": "lambda v: v['V_example'] == 3",
         "edit_fn": "lambda v: 0", "predicate": {"success": [">=", 0.9], "preserve": [">=", 0.97]}},
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--blind-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--npool", type=int, default=20000)
    a = ap.parse_args()
    cfg = json.loads((a.run / "config.json").read_text())
    cond = cfg["condition"]
    pkg = a.out / a.blind_id
    pkg.mkdir(parents=True, exist_ok=False)
    (pkg / "work").mkdir()
    sd = torch.load(a.run / "model.pt", weights_only=True)
    torch.save(sd, pkg / "weights.pt")
    g = torch.Generator().manual_seed(2026)
    b = tasks.make_batch(a.npool, cond, g)
    torch.save({"tokens": b.tokens, "ch2": b.types}, pkg / "inputs.pt")
    (pkg / "arch.py").write_text(ARCH.format(vocab=tasks.VOCAB, ntypes=tasks.N_TYPES,
                                             seqlen=tasks.SEQ_LEN, nout=tasks.P))
    tx, ty = tasks.OP_TOKENS[0], tasks.OP_TOKENS[1]  # ADD -> SUB, stated only behaviorally
    (pkg / "INSTRUCTIONS.md").write_text(INSTRUCTIONS.format(seqlen=tasks.SEQ_LEN, npool=a.npool,
                                                             tx=tx, ty=ty))
    shutil.copy(ROOT / "schema" / "nri-0.1.schema.json", pkg)
    (pkg / "nri_example_shape.json").write_text(json.dumps(EXAMPLE, indent=2))
    keys = a.out / "answer_keys"
    keys.mkdir(exist_ok=True)
    (keys / f"{a.blind_id}.json").write_text(json.dumps(
        {"run": str(a.run), "condition": cond, "edit_goal": {"token_from": tx, "token_to": ty}}))
    print(pkg)


if __name__ == "__main__":
    main()
