"""Birth of a Concept: split coverage, population definitions, control equivalence, patch identity."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import concept as C  # noqa: E402
from sem import data as D  # noqa: E402
from sem import model as SM  # noqa: E402


def test_split_components_trained_and_disjoint():
    s = C.make_split()
    tr, dev, te = set(s["train"]), set(s["dev"]), set(s["test"])
    assert not (tr & dev) and not (tr & te) and not (dev & te)
    assert len(tr | dev | te) == len(C.ROUTES) * C.P
    for r, v in dev | te:                                   # every withheld combo: value and route trained elsewhere
        assert any(vv == v for _, vv in tr) and any(rr == r for rr, _ in tr) and v != 0
    assert s == C.make_split()                              # deterministic


def test_narrow_population_covers_components():
    a = C.narrow_subset(C.make_split()["train"])
    assert {v for _, v in a} == set(range(C.P)) and {r for r, _ in a} == set(C.ROUTES)
    assert len(a) < len(C.make_split()["train"]) / 2


def test_control_matches_entropy_but_breaks_equivalence():
    # C and D targets: both 13-way and deterministic; equal canonical quantities share a C label across source
    # units, while D labels disagree for some cross-unit equivalent pairs
    for d in C.DIMS:
        for u in range(3):
            assert sorted(C.aux_target("D", d, v, u) for v in range(C.P)) == list(range(C.P))
    same_c = diff_d = 0
    for d in C.DIMS:
        for v1 in range(C.P):
            for v2 in range(C.P):
                for u1, u2 in ((0, 1), (0, 2), (1, 2)):
                    if C.canonical(d, v1, u1) == C.canonical(d, v2, u2):
                        same_c += C.aux_target("C", d, v1, u1) == C.aux_target("C", d, v2, u2)
                        diff_d += C.aux_target("D", d, v1, u1) != C.aux_target("D", d, v2, u2)
    assert same_c > 0 and diff_d > 0.7 * same_c


def test_examples_validate_and_answers():
    rng = random.Random(0)
    for r, v in C.make_split()["train"][:40]:
        ex = C.example(r, v, rng.choice(C.TRAIN_FRAMES), D.draw_style(rng, .5, .5), rng)
        assert ex["answer"] == str(C.answer(r[0], v, r[1], r[2]))


@torch.no_grad()
def test_finish_from_reproduces_forward_and_self_patch_is_identity():
    vocab = D.Vocab.from_json(json.loads((ROOT / "runs/zoo_sem/eval/vocab.json").read_text()))
    torch.manual_seed(0)
    m = SM.SemStudent(len(vocab.itos)).eval()
    sets = C.eval_sets(C.make_split())
    b = D.encode_batch(sets["practiced"][:30], vocab)
    c = m.trunk(b["tok"], ret=True)
    for site, key in ((1, "x1"), (2, "x2")):
        assert torch.allclose(C.finish_from(m, c[key], b["tok"] != D.PAD, site), m(b["tok"]), atol=1e-5)


def test_circle_score_detects_planted_dlog_circle():
    th = torch.tensor([2 * 3.141592653589793 * C.DLOG[v] / 12 for v in range(1, C.P)])
    codes = torch.zeros(12, 8)
    codes[:, 0], codes[:, 1] = th.cos(), th.sin()
    codes += 0.01 * torch.randn(12, 8, generator=torch.Generator().manual_seed(0))
    s = C.circle_scores(codes)
    assert s["dlog"] > 0.95 and s["dlog_k"] == 1 and s["additive"] < 0.6 and s["shuffled_mean"] < 0.6


def test_complete_checkpoints_every_100_and_bit_exact_resume(tmp_path):
    from safetensors.torch import load_file
    _, full = C.run("C", 7, tmp_path / "full", steps=300, audit=False)
    _, _ = C.run("C", 7, tmp_path / "cut", steps=200, audit=False)            # stop at 200
    _, resumed = C.run("C", 7, tmp_path / "cut", steps=300, resume=True, audit=False)
    assert resumed == full                                                     # bit-exact continuation
    cks = sorted(p.name for p in (tmp_path / "full" / "ckpt").iterdir())
    assert cks == ["step_00000", "step_00100", "step_00200", "step_00300"]
    d = tmp_path / "full" / "ckpt" / "step_00100"
    for f in ("model.safetensors", "aux.safetensors", "optim.safetensors", "rng.safetensors", "state.json", "traces.safetensors"):
        assert (d / f).exists(), f
    tr = load_file(d / "traces.safetensors")
    assert tr["grad_norm"].shape == (48, 42) and tr["grad_norm"].abs().sum() > 0
    assert len((tmp_path / "full" / "interval.jsonl").read_text().splitlines()) == 300
    # checkpointed weights equal the interval endpoint weights used for update analysis
    a, b = load_file(d / "model.safetensors"), load_file(tmp_path / "cut" / "ckpt" / "step_00100" / "model.safetensors")
    assert all(torch.equal(a[k], b[k]) for k in a)
