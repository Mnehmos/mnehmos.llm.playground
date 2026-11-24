"""Registered factorial infrastructure: interleaved == standalone, bit-exact restart, exact AdamW decomposition,
complete checkpoints, audits never touch the reserved test combinations, verified archives."""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sci import concept as C  # noqa: E402
from sci import concept_audit as A  # noqa: E402
from sci import factorial as Fx  # noqa: E402
from sem import data as D  # noqa: E402

RUNS = [("C", 1.0, 7), ("B", 0.1, 7)]


def _w(live, name, step):
    return load_file(live / name / "ckpt" / f"step_{step:05d}" / "model.safetensors")


def _eq(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_interleaved_equals_standalone_and_restart_is_bit_exact(tmp_path):
    Fx.train_worker(0, 1, live=tmp_path / "inter", runs=RUNS, steps=200)
    for r in RUNS:
        Fx.train_worker(0, 1, live=tmp_path / "solo", runs=[r], steps=200)
    Fx.train_worker(0, 1, live=tmp_path / "restart", runs=RUNS, steps=100)        # "crash" after 100
    Fx.train_worker(0, 1, live=tmp_path / "restart", runs=RUNS, steps=200)        # resume
    for p, w, s in RUNS:
        n = Fx.run_name(p, w, s)
        assert _eq(_w(tmp_path / "inter", n, 200), _w(tmp_path / "solo", n, 200))
        assert _eq(_w(tmp_path / "inter", n, 200), _w(tmp_path / "restart", n, 200))
        lines = (tmp_path / "restart" / n / "interval.jsonl").read_text().splitlines()
        assert [json.loads(l)["step"] for l in lines] == list(range(1, 201))
    # matched initialisation across conditions with the same seed
    a, b = _w(tmp_path / "inter", "wd1.0_C_s7", 0), _w(tmp_path / "inter", "wd0.1_B_s7", 0)
    assert _eq(a, b)


def test_update_decomposition_is_exact(tmp_path):
    Fx.train_worker(0, 1, live=tmp_path, runs=[("C", 1.0, 3)], steps=100)
    d0, d1 = tmp_path / "wd1.0_C_s3" / "ckpt" / "step_00000", tmp_path / "wd1.0_C_s3" / "ckpt" / "step_00100"
    w0, w1, up = load_file(d0 / "model.safetensors"), load_file(d1 / "model.safetensors"), load_file(d1 / "updates.safetensors")
    for k in w0:
        total = w1[k] - w0[k]
        assert torch.allclose(up[k + "::adam"] + up[k + "::wd"], total, atol=1e-5), k
    assert up["aux.weight::wd"].abs().sum() == 0                      # the unused role head is never decayed
    for f in ("model.safetensors", "aux.safetensors", "optim.safetensors", "rng.safetensors", "state.json",
              "updates.safetensors", "updates.json", "COMPLETE"):
        assert (d1 / f).exists(), f
    s = json.loads((d1 / "updates.json").read_text())
    assert s["W_in"]["wd_net"] > 0 and s["W_in"]["adam_net"] > 0


def test_audit_runs_and_never_uses_reserved_test(tmp_path):
    test = set(C.make_split()["test"])
    vocab = D.Vocab.from_json(json.loads((ROOT / "runs/zoo_sem/eval/vocab.json").read_text()))
    for pop in ("A", "C"):
        ctx = A.AuditContext(vocab, pop)
        for exs in list(ctx.sets.values()) + [ctx.panel_ex, ctx.own]:
            for e in exs:
                a = e["ast"]
                assert ((a["dim"], a["u"], a["t"]), a["v"]) not in test
    Fx.train_worker(0, 1, live=tmp_path, runs=[("D", 0.1, 5)], steps=100)
    assert Fx.audit_pending(live=tmp_path) == 2
    s = json.loads((tmp_path / "wd0.1_D_s5" / "ckpt" / "step_00100" / "audit.json").read_text())
    for k in ("behaviour", "traces", "representation", "causal", "updates", "weight_norm"):
        assert k in s
    t = load_file(tmp_path / "wd0.1_D_s5" / "ckpt" / "step_00100" / "audit.safetensors")
    assert t["loss_task_per_item"].shape == (48,) and t["loss_task_per_item"].std() > 0     # per item, not repeated
    assert "x1m::task_aux_cos" in t and "logits::task_gnorm" in t and "basis_canonical_x1m" in t
    assert s["traces"]["param_groups"]["E"]["task_norm"] > 0


def test_archive_roundtrip(tmp_path):
    live = tmp_path / "live"
    Fx.train_worker(0, 1, live=live, runs=[("A", 1.0, 9)], steps=100)
    Fx.audit_pending(live=live)
    tgt = Fx.archive("wd1.0_A_s9", live=live, archive_dir=tmp_path / "arch", repo=tmp_path / "repo", steps=100)
    man = json.loads((tmp_path / "repo" / "wd1.0_A_s9" / "manifest.json").read_text())
    assert man["archive_sha256"] == Fx.sha256_file(tgt) and len(man["files"]) > 20
    import subprocess
    subprocess.run(["tar", "-C", str(tmp_path / "x"), "-xJf", str(tgt)] if (tmp_path / "x").mkdir() is None else [], check=True)
    for rel, h in man["files"].items():
        assert Fx.sha256_file(tmp_path / "x" / "wd1.0_A_s9" / rel) == h


def test_episode_counter_on_planted_logs():
    from sci import episodes as E
    calm = [dict(step=i, task=0.01, grad_norm=0.1) for i in range(1, 601)]
    planted = [dict(r) for r in calm]
    for i in range(450, 456):                                    # a planted episode in block 401-500
        planted[i - 1].update(task=0.5, grad_norm=5.0)
    assert not any(b["episode"] for b in E.blocks(calm, 600))
    eps = [b["step"] for b in E.blocks(planted, 600) if b["episode"]]
    assert eps == [500]


def test_registered_statistics_known_answers():
    from sci import factorial_analysis as FA
    assert FA.sign_flip_p([1.0] * 12) == 1 / 4096                      # all-positive: only the identity pattern
    assert abs(FA.sign_flip_p([1, -1] * 6) - 1.0) < 1e-9 or FA.sign_flip_p([1, -1] * 6) > 0.5
    assert FA.sign_flip_p([0.5] * 12, "two-sided") == 2 / 4096
    assert FA.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    lo, hi = FA.bootstrap_ci([0.2] * 12)
    assert lo == hi == 0.2
    assert abs(FA.binom_greater(10, 10) - 1 / 1024) < 1e-12
