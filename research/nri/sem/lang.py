"""Semantic language for Issue #25: canonical ASTs, a surface realizer with phrasing families, and an
INDEPENDENT semantic validator (truth contract).

Semantics (declared, NOT real-world units): the dimensional toy of docs 11-29. Values are residues
mod 13. Units are FICTIONAL words with the doc-11 scale factors:
  mass units          brel (1), klot (5), vosk (8)         canonical M = m * s (mod 13)
  acceleration units  zep (1),  quill (2), torv (7)          canonical A = a * s (mod 13)
  force units         nub, farl, gant with output factor S = inverse of (1, 6, 11)
  force F = M * A * S (mod 13).  Conversion of value v from unit u to unit t: v * s_u * s_t^-1.

Task families (AST 'task' field):
  force     (m, mu, a, au, fu)                 -> F
  convert   (dim, v, u, t)                      -> value in unit t
  equiv     (m1, u1, m2, u2)                    -> yes / no   (same canonical mass)
  inverse   (F, fu, a, au, mu)  A != 0          -> mass value in unit mu
  sense     (word, sense, context id)           -> phys / other   (contextual polysemy)
  force tasks may carry a DISTRACTOR sentence that uses a polysemous word in its non-physics sense
  together with an irrelevant number (hard negative).

The realizer knows every slot's role and emits per-token role labels (auxiliary targets). The
validator does NOT use templates: it re-reads the token list with its own lexicons and cue rules and
must recover the same AST and answer, or the example is quarantined.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

P = 13
INV = {x: pow(x, -1, P) for x in range(1, P)}
MASS_S = (1, 5, 8)
ACC_S = (1, 2, 7)
FORCE_S = (1, 6, 11)
OUT_FACTOR = tuple(INV[s] for s in FORCE_S)
OOD_UNIT = 2
DEV_OOD_VALUES, SEALED_OOD_VALUES = (9, 10), (11, 12)

NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve"]
UNITS = {
    "mass": [dict(sing="brel", plur="brels", abbr="br"), dict(sing="klot", plur="klots", abbr="kl"),
             dict(sing="vosk", plur="vosks", abbr="vk")],
    "acc": [dict(sing="zep", plur="zeps", abbr="zp"), dict(sing="quill", plur="quills", abbr="ql"),
            dict(sing="torv", plur="torvs", abbr="tv")],
    "force": [dict(sing="nub", plur="nubs", abbr="nb"), dict(sing="farl", plur="farls", abbr="fr"),
              dict(sing="gant", plur="gants", abbr="gt")],
}
SCALES = {"mass": MASS_S, "acc": ACC_S}
ANSWERS = [str(v) for v in range(P)] + ["yes", "no", "phys", "other"]
ANS_ID = {a: i for i, a in enumerate(ANSWERS)}

ROLES = ["pad", "other", "mass_val", "mass_unit", "acc_val", "acc_unit", "out_unit", "force_val",
         "target_unit", "src_val", "src_unit", "poly_phys", "poly_other", "distract_num", "query"]
ROLE_ID = {r: i for i, r in enumerate(ROLES)}


def answer(ast):
    t = ast["task"]
    if t == "force":
        M = ast["m"] * MASS_S[ast["mu"]] % P
        A = ast["a"] * ACC_S[ast["au"]] % P
        return str(M * A * OUT_FACTOR[ast["fu"]] % P)
    if t == "convert":
        s = SCALES[ast["dim"]]
        return str(ast["v"] * s[ast["u"]] * INV[s[ast["t"]]] % P)
    if t == "equiv":
        return "yes" if ast["m1"] * MASS_S[ast["u1"]] % P == ast["m2"] * MASS_S[ast["u2"]] % P else "no"
    if t == "inverse":
        A = ast["a"] * ACC_S[ast["au"]] % P
        M = ast["F"] * INV[OUT_FACTOR[ast["fu"]]] * INV[A] % P
        return str(M * INV[MASS_S[ast["mu"]]] % P)
    if t == "sense":
        return ast["sense"]
    raise ValueError(t)


def value_unit_pairs(ast):
    """Every (dimension, value, unit) pair an example exposes, as input OR as answer (for the
    held-out-combination guard)."""
    t = ast["task"]
    out = []
    if t == "force":
        out += [("mass", ast["m"], ast["mu"]), ("acc", ast["a"], ast["au"])]
    elif t == "convert":
        out += [(ast["dim"], ast["v"], ast["u"]), (ast["dim"], int(answer(ast)), ast["t"])]
    elif t == "equiv":
        out += [("mass", ast["m1"], ast["u1"]), ("mass", ast["m2"], ast["u2"])]
    elif t == "inverse":
        out += [("acc", ast["a"], ast["au"]), ("mass", int(answer(ast)), ast["mu"])]
    return out


def numeric_split(ast):
    """'train' unless a held-out unit-value combination is exposed: 'dev' (values 9-10) or 'sealed'
    (11-12); 'crossed' if both kinds appear."""
    kinds = set()
    for dim, v, u in value_unit_pairs(ast):
        if dim in ("mass", "acc") and u == OOD_UNIT and v >= 9:
            kinds.add("dev" if v in DEV_OOD_VALUES else "sealed")
    if not kinds:
        return "train"
    return kinds.pop() if len(kinds) == 1 else "crossed"


def tuple_split(ast):
    """Hash split of the canonical tuple (independent of surface): train 60% / dev 15% / sealed 25%."""
    key = ast["task"] + ":" + ",".join(f"{k}={ast[k]}" for k in sorted(ast) if k not in ("task", "ctx"))
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 20
    return "train" if h < 12 else "dev" if h < 15 else "sealed"


# ------------------------------------------------------------------ surface realizer
# A template is a list of tokens; '{slot}' tokens are filled from the AST. Families carry metadata.
@dataclass
class Family:
    fid: str
    task: str
    register: str
    syntax: str
    split: str           # pretrain / train / dev / sealed
    tokens: list
    order: str = "ma"    # which quantity is mentioned first (for force)
    meta: dict = field(default_factory=dict)


def T(s):
    return s.split()


FAMILIES = [
    # ---- force
    Family("F01", "force", "textbook", "decl+imperative", "pretrain", T("a body of mass {m} {mu} accelerates at {a} {au} . find the force in {fu} .")),
    Family("F04", "force", "engineering", "key-value", "pretrain", T("compute force [ {fu} ] : m = {m} {mu} , a = {a} {au}")),
    Family("F11", "force", "concise", "fragment", "pretrain", T("force in {fu} for {m} {mu} at {a} {au} .")),
    Family("F02", "force", "textbook", "reordered-question", "train", T("an object accelerating at {a} {au} has a mass of {m} {mu} . what is the force in {fu} ?"), order="am"),
    Family("F03", "force", "casual", "fragment-question", "train", T("quick question : {m} {mu} cart , speeding up at {a} {au} , how much force in {fu} ?")),
    Family("F05", "force", "symbolic", "equation", "train", T("f = m * a with m = {m} {mu} and a = {a} {au} ; f in {fu} = ?")),
    Family("F06", "force", "textbook", "passive", "train", T("a sled of {m} {mu} is accelerated at {a} {au} by a force . give that force in {fu} .")),
    Family("F08", "force", "conversational", "request", "train", T("could you tell me the force , in {fu} , needed to push {m} {mu} at {a} {au} ?")),
    Family("F07", "force", "textbook", "coreference", "dev", T("a block has a mass of {m} {mu} . it accelerates at {a} {au} . what force in {fu} acts on it ?")),
    Family("F10", "force", "verbose", "relative-clause", "dev", T("when a cart whose mass is {m} {mu} accelerates at a rate of {a} {au} , the force it feels , measured in {fu} , is what ?")),
    Family("F09", "force", "textbook", "declarative-completion", "sealed", T("the force in {fu} on a mass of {m} {mu} that accelerates at {a} {au} is")),
    Family("F12", "force", "casual", "unit-fronted-question", "sealed", T("in {fu} , what force moves {m} {mu} at {a} {au} ?")),
    # ---- conversion
    Family("C01", "convert", "concise", "imperative", "train", T("convert {v} {u} to {t} .")),
    Family("C02", "convert", "casual", "how-many", "train", T("how many {t} is {v} {u} ?")),
    Family("C04", "convert", "symbolic", "equation", "train", T("{v} {u} = ? {t}")),
    Family("C05", "convert", "casual", "what-is", "train", T("what is {v} {u} in {t} ?")),
    Family("C03", "convert", "textbook", "express", "dev", T("express {qty} of {v} {u} in {t} .")),
    Family("C06", "convert", "verbose", "equals-how-many", "sealed", T("{qty} of {v} {u} equals how many {t} ?")),
    # ---- equivalence
    Family("E01", "equiv", "textbook", "same-as", "train", T("is {m1} {u1} the same mass as {m2} {u2} ?")),
    Family("E03", "equiv", "symbolic", "equation", "train", T("{m1} {u1} = {m2} {u2} ? yes or no")),
    Family("E04", "equiv", "casual", "equivalent", "train", T("are {m1} {u1} and {m2} {u2} equivalent ?")),
    Family("E02", "equiv", "textbook", "describe-equal", "dev", T("do {m1} {u1} and {m2} {u2} describe equal masses ?")),
    Family("E05", "equiv", "casual", "colon-question", "sealed", T("a block of {m1} {u1} and a sled of {m2} {u2} : same mass ?")),
    # ---- inverse
    Family("I01", "inverse", "textbook", "decl+question", "train", T("a force of {F} {fu} accelerates a mass at {a} {au} . what is the mass in {mu} ?")),
    Family("I03", "inverse", "symbolic", "equation", "train", T("f = {F} {fu} , a = {a} {au} , m in {mu} = ?")),
    Family("I04", "inverse", "engineering", "key-value", "train", T("find m [ {mu} ] given f = {F} {fu} and a = {a} {au} .")),
    Family("I02", "inverse", "textbook", "fronted-question", "dev", T("what mass , in {mu} , accelerates at {a} {au} under {F} {fu} ?")),
    Family("I05", "inverse", "casual", "conditional", "sealed", T("if pushing with {F} {fu} makes it accelerate at {a} {au} , how many {mu} is the cart ?")),
    # ---- sense question frames (the sentence comes from a context family)
    Family("S01", "sense", "textbook", "question", "train", T("{sentence} . is {word} a physics quantity here ?")),
    Family("S02", "sense", "casual", "question", "train", T("{sentence} . physics sense of {word} ?")),
    Family("S03", "sense", "textbook", "question", "dev", T("{sentence} . does {word} mean a measured quantity ?")),
    Family("S04", "sense", "casual", "question", "sealed", T("{sentence} . here , is {word} the physics term ?")),
]
FAMILY = {f.fid: f for f in FAMILIES}

# contextual-polysemy context families. '{n}' is an irrelevant number (unitless), '{qv}'/'{qu}' a
# physical quantity value and unit (physical sense only).
CONTEXTS = [
    # physical senses
    dict(cid="P-force-acts", word="force", sense="phys", split="train", tokens=T("a force of {qv} {qu} acts on the cart"), qdim="force"),
    dict(cid="P-force-net", word="force", sense="phys", split="dev", tokens=T("the net force on the block is {qv} {qu}"), qdim="force"),
    dict(cid="P-force-applied", word="force", sense="phys", split="sealed", tokens=T("the applied force grew to {qv} {qu}"), qdim="force"),
    dict(cid="P-mass-has", word="mass", sense="phys", split="train", tokens=T("the cart has a mass of {qv} {qu}"), qdim="mass"),
    dict(cid="P-mass-of", word="mass", sense="phys", split="dev", tokens=T("the mass of the sled is {qv} {qu}"), qdim="mass"),
    dict(cid="P-mass-measured", word="mass", sense="phys", split="sealed", tokens=T("we measured its mass as {qv} {qu}"), qdim="mass"),
    dict(cid="P-work-done", word="work", sense="phys", split="train", tokens=T("the work done by the force on the cart is large")),
    dict(cid="P-work-net", word="work", sense="phys", split="dev", tokens=T("net work on the block equals its energy change")),
    dict(cid="P-work-done2", word="work", sense="phys", split="sealed", tokens=T("the push did work on the sled over a distance")),
    # other senses
    dict(cid="O-force-compel", word="force", sense="other", split="train", tokens=T("they will force him to resign at {n}")),
    dict(cid="O-force-habit", word="force", sense="other", split="dev", tokens=T("by force of habit she woke at {n}")),
    dict(cid="O-force-org", word="force", sense="other", split="sealed", tokens=T("the air force flew {n} jets over the city")),
    dict(cid="O-mass-church", word="mass", sense="other", split="train", tokens=T("we attended mass at {n} on sunday")),
    dict(cid="O-mass-crowd", word="mass", sense="other", split="dev", tokens=T("a mass of {n} people gathered in the square")),
    dict(cid="O-mass-priest", word="mass", sense="other", split="sealed", tokens=T("the priest said mass for {n} families")),
    dict(cid="O-work-job", word="work", sense="other", split="train", tokens=T("she goes to work at {n} each day")),
    dict(cid="O-work-art", word="work", sense="other", split="dev", tokens=T("the museum shows a work of art from {n} painters")),
    dict(cid="O-work-shift", word="work", sense="other", split="sealed", tokens=T("his work shift lasts {n} hours")),
]
CONTEXT = {c["cid"]: c for c in CONTEXTS}

SLOT_ROLE = {"m": "mass_val", "mu": "mass_unit", "a": "acc_val", "au": "acc_unit", "fu": "out_unit",
             "F": "force_val", "t": "target_unit", "v": "src_val", "u": "src_unit", "m1": "mass_val",
             "u1": "mass_unit", "m2": "mass_val", "u2": "mass_unit"}


def num_tok(v, style):
    return NUM_WORDS[v] if style["num"] == "word" else str(v)


def unit_tok(dim, u, v, style):
    e = UNITS[dim][u]
    if style["unit"] == "abbr":
        return e["abbr"]
    return e["sing"] if v == 1 else e["plur"]


def render(ast, fid, style, rng, distractor=None):
    """Return (tokens, roles). `distractor`: a context id of an OTHER-sense context with a number,
    prepended as a separate sentence (force tasks only)."""
    fam = FAMILY[fid]
    assert fam.task == ast["task"]
    toks, roles = [], []
    if distractor:
        c = CONTEXT[distractor]
        assert c["sense"] == "other" and "{n}" in c["tokens"]
        n = rng.randrange(P)
        for w in c["tokens"]:
            if w == "{n}":
                toks.append(num_tok(n, style))
                roles.append("distract_num")
            else:
                toks.append(w)
                roles.append("poly_other" if w == c["word"] else "other")
        toks.append(".")
        roles.append("other")
    for w in fam.tokens:
        if w.startswith("{") and w.endswith("}"):
            k = w[1:-1]
            if k == "sentence":
                c = CONTEXT[ast["ctx"]]
                for cw in c["tokens"]:
                    if cw == "{n}":
                        toks.append(num_tok(rng.randrange(P), style))
                        roles.append("distract_num")
                    elif cw == "{qv}":
                        toks.append(num_tok(ast["qv"], style))
                        roles.append("other")
                    elif cw == "{qu}":
                        toks.append(unit_tok(c["qdim"], ast["qu"], ast["qv"], style))
                        roles.append("other")
                    else:
                        toks.append(cw)
                        roles.append(("poly_phys" if c["sense"] == "phys" else "poly_other") if cw == c["word"] else "other")
                continue
            if k == "word":
                toks.append(ast["word"])
                roles.append("poly_phys" if ast["sense"] == "phys" else "poly_other")
                continue
            if k == "qty":
                toks += (["an", "acceleration"] if ast["dim"] == "acc" else ["a", "mass"])
                roles += ["other", "other"]
                continue
            role = SLOT_ROLE[k]
            if k in ("m", "a", "F", "v", "m1", "m2"):
                toks.append(num_tok(ast[k], style))
            else:
                dim = {"mu": "mass", "au": "acc", "fu": "force", "u1": "mass", "u2": "mass"}.get(k) or ast["dim"]
                # the unit agrees in number with the value it qualifies; a bare target unit is plural
                paired = {"mu": "m", "au": "a", "fu": "F", "u1": "m1", "u2": "m2", "u": "v"}.get(k)
                val = ast.get(paired, 2) if paired else 2
                toks.append(unit_tok(dim, ast[k], val, style))
            roles.append(role)
        else:
            toks.append(w)
            roles.append("other")
    return toks, roles


# ------------------------------------------------------------------ independent validator
# Built from its own lexicons; it never looks at templates, slots or roles.
_NUM = {**{str(i): i for i in range(P)}, **{w: i for i, w in enumerate(NUM_WORDS)}}
_UNIT = {}
for _dim, _lst in UNITS.items():
    for _i, _e in enumerate(_lst):
        for _form in _e.values():
            _UNIT[_form] = (_dim, _i)
# sense cue lexicon (validator's own): words that mark a NON-physics sentence context
_OTHER_CUES = {"resign", "habit", "air", "jets", "attended", "sunday", "people", "gathered", "priest", "families",
               "goes", "museum", "art", "painters", "shift", "hours", "day", "city", "square", "woke", "said", "each"}
_PHYS_CUES = {"acts", "net", "applied", "measured", "done", "energy", "distance", "push", "did", "grew", "has", "is"}
_POLY = {"force", "mass", "work"}


class Ambiguous(Exception):
    pass


def _sentences(toks):
    out, cur = [], []
    for w in toks:
        if w in (".", ";") and cur:
            out.append(cur)
            cur = []
        else:
            cur.append(w)
    if cur:
        out.append(cur)
    return out


def _sense_of_sentence(sent):
    s = set(sent)
    o, p = bool(s & _OTHER_CUES), bool(s & {"acts", "net", "applied", "measured", "done", "energy", "distance", "push", "did", "grew"})
    if any(w in _UNIT for w in sent):
        p = True
    if o and not p:
        return "other"
    if p and not o:
        return "phys"
    raise Ambiguous("sense cues conflict or are absent: " + " ".join(sent))


def extract(toks):
    """Recover the AST from surface tokens alone. Raises Ambiguous when the reading is not unique."""
    sents = _sentences(toks)
    q = sents[-1]
    allw = [w for s in sents for w in s]
    # sense questions: the last sentence asks about one polysemous word
    if ("physics" in q or "quantity" in q) and any(w in _POLY for w in q) and not any(w in _UNIT for w in q):
        words = [w for w in q if w in _POLY]
        if len(words) != 1 or len(sents) < 2:
            raise Ambiguous("sense question target")
        body = [w for s in sents[:-1] for w in s]
        if words[0] not in body:
            raise Ambiguous("target word absent from the sentence")
        return dict(task="sense", word=words[0], sense=_sense_of_sentence(body))
    # drop OTHER-sense sentences (distractors) when they contain no units
    kept = []
    for s in sents:
        if any(w in _UNIT for w in s):
            kept.append(s)
        elif set(s) & _OTHER_CUES and not (set(s) & {"what", "find", "compute", "give", "?"}):
            continue                     # distractor: other-sense sentence, numbers ignored
        else:
            kept.append(s)
    seq = [w for s in kept for w in s]
    # pair each number with the unit that follows it (number-unit adjacency is the binding rule)
    pairs, lone_units = [], []
    i = 0
    while i < len(seq):
        w = seq[i]
        if w in _NUM and i + 1 < len(seq) and seq[i + 1] in _UNIT:
            pairs.append((_NUM[w],) + _UNIT[seq[i + 1]])
            i += 2
            continue
        if w in _NUM:
            raise Ambiguous(f"unbound number {w}")
        if w in _UNIT:
            lone_units.append(_UNIT[w])
        i += 1
    dims = [p[1] for p in pairs]
    if "yes" in seq or "same" in seq or "equivalent" in seq or "equal" in seq:
        if dims == ["mass", "mass"] and not lone_units:
            return dict(task="equiv", m1=pairs[0][0], u1=pairs[0][2], m2=pairs[1][0], u2=pairs[1][2])
        raise Ambiguous("equivalence reading")
    if dims.count("force") == 1 and dims.count("acc") == 1 and len(pairs) == 2:
        tgt = [u for u in lone_units if u[0] == "mass"]
        if len(tgt) != 1:
            raise Ambiguous("inverse target unit")
        f = [p for p in pairs if p[1] == "force"][0]
        a = [p for p in pairs if p[1] == "acc"][0]
        return dict(task="inverse", F=f[0], fu=f[2], a=a[0], au=a[2], mu=tgt[0][1])
    if sorted(dims) == ["acc", "mass"] and len(pairs) == 2:
        tgt = [u for u in lone_units if u[0] == "force"]
        if len(tgt) != 1:
            raise Ambiguous("force target unit")
        m = [p for p in pairs if p[1] == "mass"][0]
        a = [p for p in pairs if p[1] == "acc"][0]
        return dict(task="force", m=m[0], mu=m[2], a=a[0], au=a[2], fu=tgt[0][1])
    if len(pairs) == 1 and len(lone_units) == 1 and lone_units[0][0] == pairs[0][1]:
        return dict(task="convert", dim=pairs[0][1], v=pairs[0][0], u=pairs[0][2], t=lone_units[0][1])
    raise Ambiguous(f"no unique reading: pairs={pairs} lone={lone_units}")


def validate(ast, toks):
    """Truth contract: the independent reading must give the same task, the same canonical content and
    the same deterministic answer. Returns (ok, reason)."""
    try:
        got = extract(toks)
    except Ambiguous as e:
        return False, f"ambiguous: {e}"
    keys = [k for k in got if k != "task"]
    if got["task"] != ast["task"]:
        return False, f"task mismatch {got['task']} != {ast['task']}"
    for k in keys:
        if got[k] != ast[k]:
            return False, f"field {k}: {got[k]} != {ast[k]}"
    if answer(got) != answer(ast):
        return False, "answer mismatch"
    return True, "ok"
