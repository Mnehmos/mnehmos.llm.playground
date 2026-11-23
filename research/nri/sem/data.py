"""Data pools, split permissions, lesson specifications, tokenizer and frozen evaluation sets (Issue #25).

Split axes (frozen; docs/30):
  1. phrasing family     pretrain / train / dev / sealed            (sem.lang.FAMILIES[*].split)
  2. polysemy context    train / dev / sealed                        (sem.lang.CONTEXTS[*].split)
  3. canonical tuple     train 60% / dev 15% / sealed 25%            (sem.lang.tuple_split, hash)
  4. value-unit numeric  train / dev (held-out values 9-10) / sealed (11-12)   (sem.lang.numeric_split)
Training may only touch: families in {pretrain, train}, contexts 'train', tuples 'train', numeric
'train'. Development sets use 'dev' on at least one axis and never 'sealed'. Sealed sets use 'sealed'
axes and are written once to disk, hashed, and only read by the final evaluation.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch

from sem import lang as L

TASKS = ("force", "convert", "equiv", "inverse", "sense")
MAX_LEN = 42
SPECIALS = ["<pad>", "<unk>", "<q>"]
PAD, UNK, QRY = 0, 1, 2


# ------------------------------------------------------------------ permissions
SEALED_LOCK = [True]       # unlocked only by run_sem.py freeze / final (and tests); list so it is shared


def _guard(*split_lists):
    if SEALED_LOCK[0] and any(s in ("sealed", "crossed") for sl in split_lists if sl for s in sl):
        raise PermissionError("sealed split requested while the sealed lock is on")


def families(task, splits):
    _guard(splits)
    return [f.fid for f in L.FAMILIES if f.task == task and f.split in splits]


def contexts(splits, sense=None, numbered=False):
    _guard(splits)
    return [c["cid"] for c in L.CONTEXTS if c["split"] in splits and (sense is None or c["sense"] == sense)
            and (not numbered or "{n}" in c["tokens"])]


TRAIN_FAMILY_SPLITS = ("pretrain", "train")
PRETRAIN_FAMILIES = families("force", ("pretrain",))


# ------------------------------------------------------------------ AST sampling
def sample_ast(task, rng, tuple_splits=("train",), numeric=("train",), sense_ctx=None, hard_negative=False,
               focus=None, tries=500):
    """Rejection-sample a canonical AST whose tuple split and numeric split are allowed.
    focus: optional dict restricting e.g. {'mass_unit': [1, 2]} (lessons may focus on units/values)."""
    _guard(tuple_splits, numeric)
    if task == "sense" and sense_ctx:
        _guard([L.CONTEXT[c]["split"] for c in sense_ctx])
    for _ in range(tries):
        if task == "force":
            ast = dict(task="force", m=rng.randrange(13), mu=rng.randrange(3), a=rng.randrange(13), au=rng.randrange(3),
                       fu=rng.randrange(3))
        elif task == "convert":
            dim = rng.choice(["mass", "acc"])
            ast = dict(task="convert", dim=dim, v=rng.randrange(13), u=rng.randrange(3), t=rng.randrange(3))
        elif task == "equiv":
            m1, u1, u2 = rng.randrange(13), rng.randrange(3), rng.randrange(3)
            M = m1 * L.MASS_S[u1] % 13
            if rng.random() < 0.5:                                         # true equivalence
                m2 = M * L.INV[L.MASS_S[u2]] % 13
            elif hard_negative and rng.random() < 0.7:                     # near miss / same digits
                m2 = rng.choice([(M * L.INV[L.MASS_S[u2]] + d) % 13 for d in (1, 12)] + [m1])
            else:
                m2 = rng.randrange(13)
            ast = dict(task="equiv", m1=m1, u1=u1, m2=m2, u2=u2)
        elif task == "inverse":
            ast = dict(task="inverse", F=rng.randrange(13), fu=rng.randrange(3), a=rng.randrange(1, 13), au=rng.randrange(3),
                       mu=rng.randrange(3))
            if ast["a"] * L.ACC_S[ast["au"]] % 13 == 0:
                continue
        elif task == "sense":
            cid = rng.choice(sense_ctx)
            c = L.CONTEXT[cid]
            ast = dict(task="sense", word=c["word"], sense=c["sense"], ctx=cid)
            if "{qv}" in c["tokens"]:
                ast["qv"] = rng.randrange(13)
                ast["qu"] = rng.randrange(3)
            return ast
        else:
            raise ValueError(task)
        if L.tuple_split(ast) not in tuple_splits or L.numeric_split(ast) not in numeric:
            continue
        if focus and not _focus_ok(ast, focus):
            continue
        return ast
    raise RuntimeError(f"could not sample {task} with {tuple_splits} {numeric} {focus}")


def _focus_ok(ast, focus):
    for k, allowed in focus.items():
        if k == "mass_unit":
            us = [ast[x] for x in ("mu", "u1", "u2") if x in ast] + ([ast["u"], ast["t"]] if ast.get("dim") == "mass" else [])
            if us and not any(u in allowed for u in us):
                return False
        elif k == "acc_unit":
            us = [ast["au"]] if "au" in ast else ([ast["u"], ast["t"]] if ast.get("dim") == "acc" else [])
            if us and not any(u in allowed for u in us):
                return False
    return True


# ------------------------------------------------------------------ examples
def make_example(ast, fid, style, rng, distractor=None):
    _guard([L.FAMILY[fid].split] + ([L.CONTEXT[distractor]["split"]] if distractor else [])
           + ([L.CONTEXT[ast["ctx"]]["split"]] if ast.get("task") == "sense" else []))
    toks, roles = L.render(ast, fid, style, rng, distractor)
    ok, why = L.validate(ast, toks)
    return dict(ast=ast, fid=fid, style=style, distractor=distractor, tokens=toks, roles=roles,
                answer=L.answer(ast), valid=ok, reason=why)


def draw_style(rng, p_word=0.0, p_abbr=0.0):
    return dict(num="word" if rng.random() < p_word else "num", unit="abbr" if rng.random() < p_abbr else "full")


# ------------------------------------------------------------------ lesson specifications
LESSON_BOUNDS = dict(min_replay=0.2, max_paraphrase_pair_rate=0.5, objectives=dict(aux_role=(0.0, 1.0), iit=(0.0, 3.0),
                     consistency=(0.0, 1.0)))


class LessonError(ValueError):
    pass


def validate_lesson(spec):
    """Reject any lesson that touches non-training pools or exceeds bounds. Returns the normalized spec."""
    allowed_keys = {"task_weights", "replay_weight", "families", "style", "distractor_rate", "distractor_contexts",
                    "sense_contexts", "hard_negative_rate", "paraphrase_pair_rate", "objectives", "focus", "note"}
    extra = set(spec) - allowed_keys
    if extra:
        raise LessonError(f"unknown lesson fields {extra}")
    tw = spec.get("task_weights", {})
    if not tw or any(t not in TASKS for t in tw) or any(w < 0 for w in tw.values()) or sum(tw.values()) <= 0:
        raise LessonError("task_weights must be non-negative over known tasks with a positive sum")
    if spec.get("replay_weight", 0) < LESSON_BOUNDS["min_replay"] or spec.get("replay_weight", 0) > 0.8:
        raise LessonError("replay_weight outside [0.2, 0.8]")
    for task, fids in spec.get("families", {}).items():
        allowed = families(task, TRAIN_FAMILY_SPLITS)
        bad = [f for f in fids if f not in allowed]
        if bad or not fids:
            raise LessonError(f"families {bad} not in the training pool for {task}")
    for key in ("distractor_contexts", "sense_contexts"):
        bad = [c for c in spec.get(key, []) if c not in contexts(("train",))]
        if bad:
            raise LessonError(f"{key} {bad} not in the training pool")
    if spec.get("distractor_rate", 0) > 0 and not spec.get("distractor_contexts"):
        raise LessonError("distractor_rate > 0 needs training distractor contexts")
    for c in spec.get("distractor_contexts", []):
        if L.CONTEXT[c]["sense"] != "other" or "{n}" not in L.CONTEXT[c]["tokens"]:
            raise LessonError(f"{c} cannot serve as a distractor")
    if tw.get("sense", 0) > 0 and not spec.get("sense_contexts"):
        raise LessonError("sense lessons need training sense contexts")
    for k in ("distractor_rate", "hard_negative_rate"):
        if not 0 <= spec.get(k, 0) <= 1:
            raise LessonError(f"{k} outside [0, 1]")
    if not 0 <= spec.get("paraphrase_pair_rate", 0) <= LESSON_BOUNDS["max_paraphrase_pair_rate"]:
        raise LessonError("paraphrase_pair_rate out of bounds")
    st = spec.get("style", {})
    if any(k not in ("p_word", "p_abbr") or not 0 <= v <= 1 for k, v in st.items()):
        raise LessonError("style must be {p_word, p_abbr} in [0, 1]")
    for k, v in spec.get("objectives", {}).items():
        lo, hi = LESSON_BOUNDS["objectives"].get(k, (None, None))
        if lo is None or not lo <= v <= hi:
            raise LessonError(f"objective {k}={v} not allowed")
    for k, v in spec.get("focus", {}).items():
        if k not in ("mass_unit", "acc_unit") or not v or any(u not in (0, 1, 2) for u in v):
            raise LessonError("focus must restrict mass_unit / acc_unit to unit ids")
    return spec


def lesson_batch(spec, n, rng):
    """Sample n training examples from a validated lesson (training pools only). A fraction of slots are
    paraphrase twins (same AST, different family/style) for the consistency objective; twins are
    returned as index pairs. Quarantined examples are resampled and counted."""
    validate_lesson(spec)
    tw = dict(spec["task_weights"])
    tasks, weights = list(tw), [tw[t] for t in tw]
    st = spec.get("style", {})
    out, quarantined, twins = [], 0, []
    n_replay = int(round(spec["replay_weight"] * n))
    n_pair = int(spec.get("paraphrase_pair_rate", 0) * (n - n_replay) / 2)
    while len(out) < n_replay:                                    # original-task replay
        ast = sample_ast("force", rng)
        ex = make_example(ast, rng.choice(PRETRAIN_FAMILIES), draw_style(rng), rng)
        if ex["valid"]:
            ex["replay"] = True
            out.append(ex)
        else:
            quarantined += 1
    while len(out) < n:
        task = rng.choices(tasks, weights)[0]
        fams = spec.get("families", {}).get(task) or families(task, TRAIN_FAMILY_SPLITS)
        ast = sample_ast(task, rng, sense_ctx=spec.get("sense_contexts"), hard_negative=rng.random() < spec.get("hard_negative_rate", 0),
                         focus=spec.get("focus"))
        dis = None
        if task == "force" and rng.random() < spec.get("distractor_rate", 0):
            dis = rng.choice(spec["distractor_contexts"])
        ex = make_example(ast, rng.choice(fams), draw_style(rng, st.get("p_word", 0), st.get("p_abbr", 0)), rng, dis)
        if not ex["valid"]:
            quarantined += 1
            continue
        ex["replay"] = False
        out.append(ex)
        if len(twins) < n_pair and len(out) < n and len(fams) > 1:
            other = rng.choice([f for f in fams if f != ex["fid"]])
            tw_ex = make_example(ast, other, draw_style(rng, st.get("p_word", 0), st.get("p_abbr", 0)), rng, dis)
            if tw_ex["valid"]:
                tw_ex["replay"] = False
                out.append(tw_ex)
                twins.append((len(out) - 2, len(out) - 1))
            else:
                quarantined += 1
    return out[:n], [t for t in twins if t[1] < n], quarantined


# ------------------------------------------------------------------ tokenizer
class Vocab:
    """Whole-word lexicon built from TRAINING renders only (unseen words map to <unk>)."""

    def __init__(self, words):
        self.itos = SPECIALS + sorted(set(words) - set(SPECIALS))
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, toks):
        return [self.stoi.get(w, UNK) for w in toks]

    def to_json(self):
        return dict(itos=self.itos)

    @classmethod
    def from_json(cls, d):
        v = cls([])
        v.itos = d["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v


def build_vocab(seed=0, n=40_000):
    rng = random.Random(seed)
    words = set()
    for task in TASKS:
        for fid in families(task, TRAIN_FAMILY_SPLITS):
            for _ in range(n // 30):
                ast = sample_ast(task, rng, sense_ctx=contexts(("train",)))
                dis = rng.choice(contexts(("train",), "other", numbered=True)) if task == "force" and rng.random() < 0.5 else None
                st = draw_style(rng, 0.5, 0.5)
                toks, _ = L.render(ast, fid, st, rng, dis)
                words.update(toks)
    for v in range(13):                                    # every number in both styles
        words.update([str(v), L.NUM_WORDS[v]])
    for dim in L.UNITS.values():
        for e in dim:
            words.update(e.values())
    return Vocab(words)


def encode_batch(examples, vocab):
    """Left-padded token ids with the query marker last; roles aligned; answer ids; canonical fields."""
    n = len(examples)
    tok = torch.full((n, MAX_LEN), PAD, dtype=torch.long)
    role = torch.zeros((n, MAX_LEN), dtype=torch.long)
    for i, ex in enumerate(examples):
        ids = vocab.encode(ex["tokens"])[-(MAX_LEN - 1):] + [QRY]
        rs = [L.ROLE_ID[r] for r in ex["roles"]][-(MAX_LEN - 1):] + [L.ROLE_ID["query"]]
        tok[i, MAX_LEN - len(ids):] = torch.tensor(ids)
        role[i, MAX_LEN - len(rs):] = torch.tensor(rs)
    ans = torch.tensor([L.ANS_ID[ex["answer"]] for ex in examples])
    task = torch.tensor([TASKS.index(ex["ast"]["task"]) for ex in examples])
    # canonical quantities for force examples (interchange supervision); -1 elsewhere
    MAS = torch.full((n, 3), -1, dtype=torch.long)
    for i, ex in enumerate(examples):
        a = ex["ast"]
        if a["task"] == "force":
            MAS[i] = torch.tensor([a["m"] * L.MASS_S[a["mu"]] % 13, a["a"] * L.ACC_S[a["au"]] % 13, L.OUT_FACTOR[a["fu"]]])
    return dict(tok=tok, role=role, ans=ans, task=task, MAS=MAS)


# ------------------------------------------------------------------ frozen evaluation sets
EVAL_SPECS = {
    # name: (split kind, task, family splits, tuple splits, numeric splits, context splits, distractor)
    # ---- development (supervisor-visible)
    "dev/force_id_pretrainfam": ("dev", "force", ("pretrain",), ("dev",), ("train",), None, False),
    "dev/force_id_trainfam": ("dev", "force", ("pretrain", "train"), ("dev",), ("train",), None, False),
    "dev/force_newfam": ("dev", "force", ("dev",), ("dev",), ("train",), None, False),
    "dev/force_ood_value": ("dev", "force", ("pretrain", "train"), ("train", "dev"), ("dev",), None, False),
    "dev/force_distractor": ("dev", "force", ("pretrain", "train"), ("dev",), ("train",), ("dev",), True),
    "dev/force_crossed": ("dev", "force", ("dev",), ("train", "dev"), ("dev",), ("dev",), True),
    "dev/convert_trainfam": ("dev", "convert", ("train",), ("dev",), ("train",), None, False),
    "dev/convert_newfam": ("dev", "convert", ("dev",), ("dev",), ("train",), None, False),
    "dev/convert_ood_value": ("dev", "convert", ("train",), ("train", "dev"), ("dev",), None, False),
    "dev/equiv_trainfam": ("dev", "equiv", ("train",), ("dev",), ("train",), None, False),
    "dev/equiv_newfam": ("dev", "equiv", ("dev",), ("dev",), ("train",), None, False),
    "dev/inverse_trainfam": ("dev", "inverse", ("train",), ("dev",), ("train",), None, False),
    "dev/inverse_newfam": ("dev", "inverse", ("dev",), ("dev",), ("train",), None, False),
    "dev/inverse_ood_value": ("dev", "inverse", ("train",), ("train", "dev"), ("dev",), None, False),
    "dev/sense_trainframe_newctx": ("dev", "sense", ("train",), None, None, ("dev",), False),
    "dev/sense_newframe_trainctx": ("dev", "sense", ("dev",), None, None, ("train",), False),
    # ---- sealed (final evaluation only)
    "sealed/force_retention_pretrainfam": ("sealed", "force", ("pretrain",), ("sealed",), ("train",), None, False),
    "sealed/force_id_trainfam": ("sealed", "force", ("pretrain", "train"), ("sealed",), ("train",), None, False),
    "sealed/force_newfam": ("sealed", "force", ("sealed",), ("sealed",), ("train",), None, False),
    "sealed/force_ood_value": ("sealed", "force", ("pretrain", "train"), ("train", "dev", "sealed"), ("sealed",), None, False),
    "sealed/force_distractor": ("sealed", "force", ("pretrain", "train"), ("sealed",), ("train",), ("sealed",), True),
    "sealed/force_crossed": ("sealed", "force", ("sealed",), ("train", "dev", "sealed"), ("sealed",), ("sealed",), True),
    "sealed/convert_newfam": ("sealed", "convert", ("sealed",), ("sealed",), ("train",), None, False),
    "sealed/convert_ood_value": ("sealed", "convert", ("train",), ("train", "dev", "sealed"), ("sealed",), None, False),
    "sealed/equiv_trainfam": ("sealed", "equiv", ("train",), ("sealed",), ("train",), None, False),
    "sealed/equiv_newfam": ("sealed", "equiv", ("sealed",), ("sealed",), ("train",), None, False),
    "sealed/inverse_newfam": ("sealed", "inverse", ("sealed",), ("sealed",), ("train",), None, False),
    "sealed/inverse_ood_value": ("sealed", "inverse", ("train",), ("train", "dev", "sealed"), ("sealed",), None, False),
    "sealed/sense_newframe_newctx": ("sealed", "sense", ("sealed",), None, None, ("sealed",), False),
    "sealed/sense_trainframe_newctx": ("sealed", "sense", ("train",), None, None, ("sealed",), False),
}
EVAL_N = 384


def make_eval_set(name, seed):
    kind, task, fsplits, tsplits, nsplits, csplits, dis = EVAL_SPECS[name]
    _guard([kind])
    rng = random.Random(seed)
    fams = families(task, fsplits)
    out, quarantined = [], 0
    while len(out) < EVAL_N:
        if task == "sense":
            ast = sample_ast("sense", rng, sense_ctx=contexts(csplits))
        else:
            ast = sample_ast(task, rng, tuple_splits=tsplits, numeric=nsplits, hard_negative=(task == "equiv"))
        d = rng.choice(contexts(csplits, "other", numbered=True)) if dis else None
        ex = make_example(ast, rng.choice(fams), draw_style(rng, 0.3, 0.3), rng, d)
        if ex["valid"]:
            ex["set"] = name
            out.append(ex)
        else:
            quarantined += 1
    return out, quarantined


def paraphrase_panel(splits, n_ast=64, seed=7):
    """Force ASTs rendered in EVERY family of the given family splits (paraphrase invariance)."""
    _guard(splits)
    rng = random.Random(seed)
    tsplit = ("dev",) if "sealed" not in splits else ("sealed",)
    out = []
    for k in range(n_ast):
        ast = sample_ast("force", rng, tuple_splits=tsplit)
        for fid in families("force", splits):
            ex = make_example(ast, fid, dict(num="num", unit="full"), rng)
            if ex["valid"]:
                ex["group"] = k
                out.append(ex)
    return out


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items()}, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(l) for l in open(path)]
