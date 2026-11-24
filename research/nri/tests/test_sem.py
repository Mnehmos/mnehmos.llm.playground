"""Issue #25 pre-experiment checks: truth contract (validator negative controls), sealed lock, lesson
bounds, deterministic save/restore, identical A/B continuation starts, sandbox isolation (static check,
audit hook, nonmutation), tool validation gate, architecture-map revalidation."""
import json
import random
import shutil
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sem import data as D  # noqa: E402
from sem import lang as L  # noqa: E402
from sem import train as TR  # noqa: E402
from toolsmith import runner as R  # noqa: E402

EVAL = ROOT / "runs" / "zoo_sem" / "eval"


def _vocab():
    return D.Vocab.from_json(json.loads((EVAL / "vocab.json").read_text()))


# ------------------------------------------------------------------ truth contract
def _example(task="force", seed=0):
    rng = random.Random(seed)
    fams = D.families(task, ("pretrain", "train"))
    ctx = D.contexts(("train",)) if task == "sense" else None
    ast = D.sample_ast(task, rng, sense_ctx=ctx)
    return ast, D.make_example(ast, fams[0], D.draw_style(rng), rng)


@pytest.mark.parametrize("task", D.TASKS)
def test_validator_accepts_renders(task):
    for s in range(30):
        _, ex = _example(task, s)
        assert ex["valid"], ex["reason"]


def test_validator_rejects_corrupted_ast():
    ast, ex = _example("force", 1)
    bad = dict(ast, m=(ast["m"] + 1) % 13)
    ok, _ = L.validate(bad, ex["tokens"])
    assert not ok


def test_validator_rejects_unbound_number():
    ast, ex = _example("force", 2)
    toks = [w for w in ex["tokens"] if w not in {u["sing"] for us in L.UNITS.values() for u in us}
            and w not in {u["plur"] for us in L.UNITS.values() for u in us}]
    ok, _ = L.validate(ast, toks)
    assert not ok


def test_validator_rejects_sense_without_cues():
    ast, ex = _example("sense", 3)
    cues = L._OTHER_CUES | L._PHYS_CUES | set(L._UNIT)                  # unit words are physical cues too
    toks = [w if w not in cues else "the" for w in ex["tokens"]]
    ok, _ = L.validate(ast, toks)
    assert not ok


# ------------------------------------------------------------------ sealed lock
def test_sealed_lock_blocks_every_entry_point():
    assert D.SEALED_LOCK[0]
    rng = random.Random(0)
    sealed_fam = next(f.fid for f in L.FAMILIES if f.split == "sealed" and f.task == "force")
    with pytest.raises(PermissionError):
        D.families("force", ("sealed",))
    with pytest.raises(PermissionError):
        D.sample_ast("force", rng, numeric=("sealed",))
    with pytest.raises(PermissionError):
        D.make_example(D.sample_ast("force", rng), sealed_fam, D.draw_style(rng), rng)
    with pytest.raises(PermissionError):
        D.make_eval_set(next(k for k in D.EVAL_SPECS if k.startswith("sealed/")), 0)
    assert not list((ROOT / "runs" / "zoo_sem").glob("**/sealed*")), "sealed data must never be on disk"


def test_frozen_eval_manifest_has_hashes_only_for_sealed():
    man = json.loads((EVAL / "manifest.json").read_text())
    for name, info in man["sealed"].items():
        assert set(info) >= {"seed", "content_sha256"}
        assert "rows" not in info and "examples" not in info


# ------------------------------------------------------------------ lesson bounds
GOOD = dict(task_weights=dict(force=1.0), replay_weight=0.3)


@pytest.mark.parametrize("patch", [
    dict(task_weights=dict(teleport=1.0)),
    dict(replay_weight=0.1),
    dict(families=dict(force=["F07"])),                                   # dev family
    dict(families=dict(force=["F09"])),                                   # sealed family
    dict(sense_contexts=["P-force-net"], task_weights=dict(sense=1.0)),   # dev context
    dict(distractor_rate=0.3),                                            # no contexts given
    dict(distractor_rate=0.3, distractor_contexts=["P-force-acts"]),      # physical sense cannot distract
    dict(objectives=dict(iit=9.0)),
    dict(objectives=dict(gold_types=1.0)),                                # privileged inference signal
    dict(paraphrase_pair_rate=0.9),
    dict(focus=dict(mass_unit=[5])),
    dict(eval_peek=True),
])
def test_lesson_validator_rejects(patch):
    with pytest.raises(D.LessonError):
        D.validate_lesson(dict(GOOD, **patch))


def test_lesson_batches_stay_in_training_pools():
    import run_sem as RS
    for spec in RS.FIXED_CURRICULUM[::5]:
        exs, _, _ = D.lesson_batch(spec, 256, random.Random(0))
        for ex in exs:
            assert L.FAMILY[ex["fid"]].split in ("pretrain", "train")
            assert L.tuple_split(ex["ast"]) == "train" or ex["ast"]["task"] == "sense"
            assert L.numeric_split(ex["ast"]) == "train"
            assert ex["valid"]


# ------------------------------------------------------------------ determinism, restore, matched arms
LESSON = dict(task_weights=dict(force=0.5, convert=0.5), replay_weight=0.3)


def test_save_restore_is_bit_exact(tmp_path):
    V = _vocab()
    st = TR.new_state(V, 7)
    TR.train_steps(st, LESSON, 3)
    TR.save_state(st, tmp_path / "s")
    h = TR.state_hash(st)
    d1, _ = TR.train_steps(st, LESSON, 3)
    h1 = TR.state_hash(st)
    st2 = TR.load_state(tmp_path / "s")
    assert TR.state_hash(st2) == h
    d2, _ = TR.train_steps(st2, LESSON, 3)
    assert d1 == d2 and TR.state_hash(st2) == h1


def test_continuations_start_identical_and_rollback(tmp_path):
    V = _vocab()
    st = TR.new_state(V, 7)
    TR.train_steps(st, LESSON, 2)
    TR.save_state(st, tmp_path / "common")
    common_hash = TR.state_hash(TR.load_state(tmp_path / "common"))
    arms = {}
    for arm in "AB":
        s = TR.load_state(tmp_path / "common")
        TR.restart_for_continuation(s, 11)
        arms[arm] = s
    assert TR.state_hash(arms["A"]) == TR.state_hash(arms["B"])
    # identical lessons from identical states produce identical streams and weights (no-op supervision = control)
    da, _ = TR.train_steps(arms["A"], LESSON, 2)
    db, _ = TR.train_steps(arms["B"], LESSON, 2)
    assert da == db and TR.state_hash(arms["A"]) == TR.state_hash(arms["B"])
    # the frozen reference is untouched by continuation training (rollback point)
    assert TR.state_hash(TR.load_state(tmp_path / "common")) == common_hash


def test_rejected_lesson_does_not_touch_state():
    V = _vocab()
    st = TR.new_state(V, 7)
    h = TR.state_hash(st)
    with pytest.raises(D.LessonError):
        TR.train_steps(st, dict(LESSON, families=dict(force=["F09"])), 2)
    assert TR.state_hash(st) == h


def test_no_pickle_load_paths():
    for p in ["sem/train.py", "run_sem.py", "toolsmith/runner.py", "toolsmith/sandbox_child.py", "sem/observe.py"]:
        src = (ROOT / p).read_text()
        assert "pickle.load" not in src
        for line in src.splitlines():
            if "torch.load(" in line:
                assert "weights_only=True" in line, f"{p}: {line.strip()}"


# ------------------------------------------------------------------ toolsmith isolation
@pytest.mark.parametrize("src,why", [
    ("import os\ndef controls(c): pass\ndef run(c): pass", "import os"),
    ("import subprocess\ndef controls(c): pass\ndef run(c): pass", "import subprocess"),
    ("def controls(c): open('/etc/passwd')\ndef run(c): pass", "open"),
    ("def controls(c): eval('1')\ndef run(c): pass", "eval"),
    ("def controls(c): c.__class__\ndef run(c): pass", "dunder"),
    ("def controls(c): c.data.SEALED_LOCK[0] = False\ndef run(c): pass", "lock"),
    ("def controls(c): c.data.families('force', ('sea' 'led',))\ndef run(c): pass", "sealed string"),
    ("def controls(c): getattr(c, 'x')\ndef run(c): pass", "getattr"),
    ("def run(c): pass", "missing controls"),
])
def test_static_check_rejects(src, why):
    with pytest.raises(R.ToolRejected):
        R.static_check(src)


def _sandbox_tool(tmp_path, body):
    """Run an UNCHECKED tool body in the sandbox child (bypasses the static check to test the audit hook)."""
    import subprocess
    tdir = tmp_path / "tool"
    tdir.mkdir()
    (tdir / "tool.py").write_text(body)
    ck = tmp_path / "ck"
    out = tmp_path / "out"
    ck.mkdir()
    out.mkdir()
    shutil.copy(EVAL / "vocab.json", ck / "vocab.json")
    (tmp_path / "a.json").write_text("{}")
    p = subprocess.run([sys.executable, "-I", str(ROOT / "toolsmith" / "sandbox_child.py"), str(ROOT), str(tdir), str(ck),
                        str(out), "run", str(tmp_path / "a.json")], capture_output=True, text=True, timeout=120,
                       env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    return p


@pytest.mark.parametrize("body", [
    "import socket\ndef run(c):\n    socket.socket()\ndef controls(c): pass",
    "import subprocess\ndef run(c):\n    subprocess.run(['true'])\ndef controls(c): pass",
    "def run(c):\n    open('/home/escape.txt', 'w').write('x')\ndef controls(c): pass",
    "def run(c):\n    open('/etc/hostname').read()\ndef controls(c): pass",
    "import pickle, io\ndef run(c):\n    pickle.loads(pickle.dumps(c.__class__))\ndef controls(c): pass",
    "def run(c):\n    c.data.families('force', ('sealed',))\ndef controls(c): pass",
    "def run(c):\n    c.data.SEALED_LOCK[0] = False\n    c.lang.FAMILY['F09']\ndef controls(c): pass",
    "def run(c):\n    c.data.EVAL_SPECS['sealed/force_newfam']\ndef controls(c): pass",
])
def test_audit_hook_and_child_guards_block(tmp_path, body):
    p = _sandbox_tool(tmp_path, body)
    assert p.returncode != 0, p.stdout + p.stderr
    last = p.stderr.strip().splitlines()[-1]
    assert last.startswith(("PermissionError", "KeyError")), last              # blocked by policy, not by accident
    assert not (tmp_path / "out" / "result.json").exists()


def test_sandbox_allows_benign_tool(tmp_path):
    p = _sandbox_tool(tmp_path, "def run(c):\n    return dict(v=len(c.vocab.itos))\ndef controls(c): pass")
    assert p.returncode == 0, p.stderr
    assert json.loads((tmp_path / "out" / "result.json").read_text())["v"] == len(_vocab().itos)


def test_tool_run_requires_validation_of_current_source(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "REGISTRY", tmp_path / "reg.jsonl")
    monkeypatch.setattr(sys, "argv", ["runner", "run", "T001_edit_decomposer", f"x={tmp_path}"])
    with pytest.raises(SystemExit, match="no passing validation"):
        R.main()


def test_nonmutation_and_provenance(tmp_path):
    V = _vocab()
    st = TR.new_state(V, 3)
    TR.save_state(st, tmp_path / "ck")
    res, rec = R.execute("T001_edit_decomposer", {"ck": tmp_path / "ck"}, mode="controls")
    assert rec["nonmutation"]["ok"] and rec["returncode"] == 0, rec["error"]
    assert len(rec["source_sha256"]) == 64 and rec["manifest"]["tool_id"] == "T001_edit_decomposer"
    assert all(c["pass"] for c in res["cases"])


# ------------------------------------------------------------------ architecture map
def test_archmap_revalidates():
    from toolsmith import archmap
    if not archmap.OUT.exists():
        pytest.skip("map not generated")
    r = archmap.revalidate()
    assert r["valid"], r


def test_primary_endpoint_bootstrap():
    import run_sem as RS
    g = torch.Generator().manual_seed(1)
    mk = lambda p: {k: (torch.rand(384, generator=g) < p).int().tolist() for k in RS.PRIMARY_SETS + (RS.RETENTION_SET,)}
    correct = {"A_fixed": mk(0.5), "B_adaptive": mk(0.7), "common (pre-curriculum reference)": mk(0.9)}
    res = {n: {k: sum(v) / len(v) for k, v in c.items()} for n, c in correct.items()}
    r = RS.primary(correct, res)
    assert r["ci95"][0] > 0 and not r["B_better_claimed"]              # B better but retention guard fails
    correct["common (pre-curriculum reference)"][RS.RETENTION_SET] = correct["B_adaptive"][RS.RETENTION_SET]
    res = {n: {k: sum(v) / len(v) for k, v in c.items()} for n, c in correct.items()}
    assert RS.primary(correct, res)["B_better_claimed"]
    same = {"A_fixed": correct["A_fixed"], "B_adaptive": correct["A_fixed"], "common (pre-curriculum reference)": correct["A_fixed"]}
    r0 = RS.primary(same, {n: res["A_fixed"] for n in same})
    assert r0["diff_B_minus_A"] == 0 and not r0["B_better_claimed"] and not r0["A_better"]


# ------------------------------------------------------------------ post-training feature programming
def test_program_validator_and_rollback():
    import copy
    from sem import model as SM
    from sem import program as PG
    rng = random.Random(0)
    for cap in PG.CAPS:
        ex = PG.sample(rng, cap)
        assert PG.validate(ex["ast"], ex["tokens"])[0]
        bad = dict(ex["ast"], m=(ex["ast"]["m"] + 1) % 13)
        if PG.answer(bad) != PG.answer(ex["ast"]):
            assert not PG.validate(bad, ex["tokens"])[0]
    V = _vocab()
    stoi = dict(V.stoi, **{w: len(V.itos) + i for i, w in enumerate(PG.NEW_WORDS)})
    base = SM.SemStudent(len(V.itos))
    h0 = PG.sd_hash(base)
    for meth in ("head", "adapter", "bounded"):
        net = PG.Programmed(copy.deepcopy(base), meth)
        before = {n: p.detach().clone() for n, p in net.base.named_parameters() if not n.startswith("E.new")}
        PG.train_cap(net, ["null"], stoi, V, 3, 0)
        for n, p in net.base.named_parameters():
            if n in before and meth != "bounded":
                assert torch.equal(p, before[n]), f"{meth} changed frozen {n}"
            if meth == "bounded" and n in net.W0:
                assert (p - net.W0[n]).norm() <= PG.EPS_BOUND * net.W0[n].norm() * (1 + 1e-5)
        net.remove_hooks()
    assert PG.sd_hash(base) == h0                                         # the original object is never touched
