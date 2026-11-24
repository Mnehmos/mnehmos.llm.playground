"""Regression checks on a completed two-interval feasibility run."""
import argparse
from copy import deepcopy

import torch
from safetensors.torch import load_file

from run_sem import FIXED_CURRICULUM
from sci import loop
from sci.measure import record_step
from sem import train


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def rejected(checkpoint, config, record):
    try:
        loop.validate_review(checkpoint, config, record)
    except (KeyError, ValueError):
        return
    raise AssertionError("accepted a disallowed review")


def main(run):
    zero, first = loop.at(run, 0), loop.at(run, 100)
    config = loop.read(run / "run.json")
    accepted = loop.read(zero / "review.json")
    faulty = deepcopy(accepted)
    del faulty["assessment"]["tokens_and_objectives"]
    rejected(zero, config, faulty)
    faulty = deepcopy(accepted)
    faulty["checkpoint_sha256"] = "0"*64
    rejected(zero, config, faulty)
    faulty = deepcopy(accepted)
    faulty["action"] = "curriculum"
    faulty["lesson"] = FIXED_CURRICULUM[0]
    rejected(zero, {**config, "arm": "fixed"}, faulty)
    faulty = {**accepted, "action": "apply_weights"}
    rejected(zero, config, faulty)

    replay = train.load_state(zero)
    replay["interval"] = []
    train.train_steps(replay, FIXED_CURRICULUM[0], 100, recorder=record_step)
    rng_after_replay = torch.get_rng_state()
    restored = train.load_state(first)
    assert torch.equal(rng_after_replay, torch.get_rng_state())
    assert_equal(replay["model"].state_dict(), restored["model"].state_dict())
    assert_equal(replay["opt"].state_dict(), restored["opt"].state_dict())
    assert_equal(replay["rng"].getstate(), restored["rng"].getstate())
    assert replay["step"] == restored["step"] == 100
    import json
    assert json.loads(json.dumps(replay["interval"])) == [json.loads(s) for s in (first/"interval.jsonl").read_text().splitlines()]
    fields = dict(restored["model"].named_parameters())
    original = load_file(zero/"model.safetensors")
    evidence = loop.verify(first)
    assert evidence["weight_changes"].keys() == fields.keys()
    for name, tensor in fields.items():
        expected = float((tensor.detach()-original[name]).norm())
        assert expected == evidence["weight_changes"][name]["previous"]["l2"]
    results = load_file(first/"traces.safetensors")
    for example in evidence["token_trace"]["examples"]:
        i = example["index"]
        total = torch.zeros_like(fields["E.weight"])
        total.index_add_(0,torch.tensor(example["token_ids"]),results[f"{i}.gradient.E"][0])
        torch.testing.assert_close(total, results[f"{i}.parameter_gradient.E.weight"])
    print("Verified initial-to-100 exact replay, Adam moments, data RNG, 100 step records, full tensor accounting, occurrence traces, and audit/permission rejection.")


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run.resolve())
