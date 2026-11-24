"""student_architecture_map: a versioned, machine-readable description of the student, generated from
the ACTUAL source files and a runtime trace (Issue #25 §1). Code and runtime shapes are authoritative.

    python -m toolsmith.archmap [checkpoint_dir]      -> sci/student_architecture_map.json

The version is the SHA-256 of (source file hashes + runtime site shapes); `revalidate()` recomputes it
and reports any drift, so tools can refuse to run against a map that no longer matches the build.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sem import data as D  # noqa: E402
from sem import lang as L  # noqa: E402
from sem import model as SM  # noqa: E402
from sem import observe as O  # noqa: E402
from sem import train as TR  # noqa: E402

SOURCES = ["sem/lang.py", "sem/data.py", "sem/model.py", "sem/train.py", "sem/observe.py", "run_sem.py",
           "toolsmith/sandbox_child.py", "toolsmith/runner.py"]
OUT = ROOT / "sci" / "student_architecture_map.json"


def _sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def runtime_trace(model, vocab):
    """Run one real batch and record every registered site's shape, plus hook shapes of all modules."""
    import random
    rng = random.Random(0)
    exs = []
    while len(exs) < 4:
        ex = D.make_example(D.sample_ast("force", rng), D.PRETRAIN_FAMILIES[0], D.draw_style(rng), rng)
        if ex["valid"]:
            exs.append(ex)
    b = D.encode_batch(exs, vocab)
    shapes = {}
    hooks = []
    for name, mod in model.named_modules():
        if name and not list(mod.children()):
            hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: shapes.__setitem__(
                n, list(o[0].shape) if isinstance(o, tuple) else list(o.shape))))
    with torch.no_grad():
        c = model.trunk(b["tok"], ret=True)
        logits = model.head(c["h"])
        aux = model.aux(c["x1"])
    for h in hooks:
        h.remove()
    sites = {k: list(v.shape) for k, v in c.items()}
    sites["logits"] = list(logits.shape)
    sites["aux_logits"] = list(aux.shape)
    return dict(batch_example=4, sites=sites, module_outputs=shapes)


def build(model=None, vocab=None, checkpoint=None):
    vocab = vocab or D.Vocab.from_json(json.loads((ROOT / "runs/zoo_sem/eval/vocab.json").read_text()))
    if model is None:
        model = SM.SemStudent(len(vocab.itos))
        if checkpoint:
            from safetensors.torch import load_file
            model.load_state_dict(load_file(Path(checkpoint) / "model.safetensors"))
    params = {n: dict(shape=list(p.shape), numel=p.numel(), group=n.split(".")[0]) for n, p in model.named_parameters()}
    groups = {}
    for n, p in params.items():
        groups[p["group"]] = groups.get(p["group"], 0) + p["numel"]
    rt = runtime_trace(model, vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    m = dict(
        schema="student_architecture_map/1",
        source_sha256={p: _sha(p) for p in SOURCES if (ROOT / p).exists()},
        tokenization=dict(kind="whole-word lexicon (lower-case, space-separated; punctuation as tokens)",
                          vocab_size=len(vocab.itos), specials=D.SPECIALS, max_len=D.MAX_LEN, padding="left; <q> marker last",
                          unknown="<unk> (words never seen in training renders)", note="NOT BPE; no BPE-vs-ANLU comparison is made"),
        semantics=dict(modulus=L.P, mass_scales=L.MASS_S, acc_scales=L.ACC_S, force_output_factor=L.OUT_FACTOR,
                       units={k: [u["sing"] for u in v] for k, v in L.UNITS.items()}, answers=L.ANSWERS,
                       tasks=list(D.TASKS), roles=L.ROLES, polysemous_words=sorted({c["word"] for c in L.CONTEXTS})),
        data_generator=dict(families={f.fid: dict(task=f.task, register=f.register, syntax=f.syntax, split=f.split) for f in L.FAMILIES},
                            contexts={c["cid"]: dict(word=c["word"], sense=c["sense"], split=c["split"]) for c in L.CONTEXTS},
                            split_axes=["family", "context", "tuple(hash 60/15/25)", "numeric(held-out values: dev 9-10, sealed 11-12 on unit 2)"],
                            truth_contract="sem.lang.validate: independent re-reading must recover the AST and answer, else quarantine"),
        model=dict(class_name="SemStudent", source=inspect.getsource(SM.SemStudent.trunk).splitlines()[0],
                   n_params=model.n_params(), n_params_without_aux=model.n_params(False), param_groups=groups, params=params,
                   heads_per_layer=model.attn1.h, head_dim=model.attn1.dh, layers=["attn1+mlp1", "attn2+mlp2", "attn3", "readout MLP"],
                   declared_block=list(SM.BLOCK)),
        runtime=rt,
        patch_points={
            "h": "final-position residual after attn3 (input of the readout); swap/patch exact, no normalisation",
            "h[0:16]": "declared mass block (IIT-supervised on force examples)",
            "W_in": "readout input weights: affine edits of h fold exactly into W_in/b_in",
            "x1[pos], x2[pos]": "token residuals after block 1/2 (role head reads x1); positional patching needs aligned positions",
            "pre3[head]": "attn3 per-head output at the query position (before the o projection)",
            "pat{1,2,3}": "attention patterns (keys masked at <pad>)"},
        objectives=dict(task="CE on the answer at <q>", aux_role="CE on per-token semantic role from x1 (training only)",
                        iit="block swap between force examples within the batch; target = source mass x base acceleration x base output factor",
                        consistency="symmetric KL between paraphrase twins in the batch",
                        defaults=dict(aux_role=0.5, iit=3.0, consistency=0.0)),
        optimizer=dict(kind="AdamW", weight_decay=0.01, state_per_param=["step", "exp_avg", "exp_avg_sq"],
                       pretrain_lr=dict(peak=TR.PRE_LR, warmup=TR.PRE_WARMUP, cap_steps=TR.PRE_CAP),
                       continuation_lr=dict(peak=TR.CONT_LR, warmup=TR.CONT_WARMUP, min=TR.CONT_MIN_LR, steps=TR.CONT_STEPS, schedule="cosine"),
                       continuation_restart="fresh AdamW from the common checkpoint for both arms", batch=TR.BATCH, threads=TR.THREADS),
        checkpoints=dict(model="model.safetensors", optimizer="optim.pt (loaded with torch.load(weights_only=True))",
                         rng="rng.safetensors (torch) + state.json (python random state, JSON)", metadata="state.json",
                         pickle="never loaded from artifacts"),
        evaluation_permissions=dict(training=["families pretrain/train", "contexts train", "tuples train", "numeric train"],
                                    supervisor_visible=["dev sets (runs/zoo_sem/eval/dev)", "FIT pool", "all checkpoints of A and B"],
                                    sealed=["generated only in memory by run_sem.py final; content hashes frozen in eval/manifest.json",
                                            "sem.data.SEALED_LOCK blocks every sealed request"]),
        tool_api=dict(context="toolsmith.sandbox_child.Ctx", methods=["load_model(name)", "vocab", "data (guarded sem.data facade)",
                                                                       "lang", "encode(examples)", "dev_sets()", "observe (sem.observe)", "out(name, obj)"]),
        instruments_prebuilt=[n for n, f in inspect.getmembers(O, inspect.isfunction) if f.__module__ == "sem.observe"])
    m["version"] = hashlib.sha256(json.dumps(dict(src=m["source_sha256"], rt=m["runtime"]["sites"]), sort_keys=True).encode()).hexdigest()[:16]
    return m


def revalidate(path=OUT):
    old = json.loads(Path(path).read_text())
    new = build()
    drift = dict(source=[k for k in set(old["source_sha256"]) | set(new["source_sha256"])
                         if old["source_sha256"].get(k) != new["source_sha256"].get(k)],
                 runtime_sites=old["runtime"]["sites"] != new["runtime"]["sites"])
    return dict(valid=old["version"] == new["version"], old=old["version"], new=new["version"], drift=drift)


if __name__ == "__main__":
    mp = build(checkpoint=sys.argv[1] if len(sys.argv) > 1 else None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(mp, indent=1))
    print("architecture map", mp["version"], "params", mp["model"]["n_params"], "sites", mp["runtime"]["sites"])
