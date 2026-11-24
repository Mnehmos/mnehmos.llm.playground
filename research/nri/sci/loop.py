"""Disk-backed checkpoint gate for prospective semantic-student development."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time

import torch
from safetensors.torch import load_file, save_file
from sem import data as D, observe as O, train as T
from sci import measure as M, conversion as C
from run_sem import FIXED_CURRICULUM

ROOT = Path(__file__).resolve().parents[1]
INTERVAL = 100
QUESTIONS = (
    "weights", "effective_updates", "tokens_and_objectives", "representations", "causal",
    "capabilities", "failures", "alternatives", "prediction", "next_experiment", "instruments",
    "decision", "prior_predictions",
)
ACTIONS = {"continue", "diagnose", "curriculum", "propose_weight_edit"}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def at(run, step):
    return run / f"step_{step:06d}"


def verify(checkpoint):
    manifest = read(checkpoint / "manifest.json")
    required = {"model.safetensors", "optim.pt", "rng.safetensors", "state.json", "procedure.json",
                "interval.jsonl", "evidence.json", "traces.safetensors"}
    if set(manifest) != required or any(sha(checkpoint/name) != digest for name,digest in manifest.items()):
        raise ValueError(f"checkpoint integrity failure: {checkpoint}")
    return read(checkpoint / "evidence.json")


def source_hashes():
    files = [*sorted((ROOT/"sem").glob("*.py")), *sorted((ROOT/"sci").glob("*.py")),
             ROOT/"run_sem.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def panel(vocab):
    rng = random.Random(3905)
    examples = []
    for task in D.TASKS:
        fams = D.families(task, ("train",))
        if task == "sense":
            for sense in ("phys", "other"):
                contexts = [cid for cid in D.contexts(("train",), sense)
                            if D.L.CONTEXT[cid]["word"] == "mass"]
                for _ in range(2):
                    while True:
                        ast = D.sample_ast(task, rng, sense_ctx=contexts)
                        ex = D.make_example(ast, fams[0], D.draw_style(rng), rng)
                        if ex["valid"]:
                            examples.append(ex)
                            break
            continue
        while True:
            ast = D.sample_ast(task, rng)
            variants = [D.make_example(ast, fams[0], dict(num=num, unit=unit), rng)
                        for num,unit in (("num","abbr"),("word","full"))]
            if all(ex["valid"] for ex in variants):
                examples.extend(variants)
                break
    return examples


def snapshot(run, state, config, interval, lesson, training_seconds):
    start = time.perf_counter()
    step = state["step"]
    target = at(run, step)
    current = state["model"].state_dict()
    initial = load_file(at(run, 0)/"model.safetensors") if step else current
    previous_path = at(run, step-INTERVAL)
    previous = load_file(previous_path/"model.safetensors") if step else current
    previous_card = verify(previous_path) if step else None
    with tempfile.TemporaryDirectory(dir=run) as directory:
        pending = Path(directory)/"snapshot"
        T.save_state(state, pending)
        model = T.load_state(pending)["model"].eval()
        vocab = state["vocab"]
        dev = run/"development"
        card, _ = O.observe(model, vocab, dev)
        model.eval()
        force_panel = O.dev_sets(dev)["dev/force_id_trainfam"][:128]
        card["causal_controls"] = M.causal(model, vocab, force_panel)
        traces, tensors = M.trace(model, vocab, panel(vocab) + (C.panel(vocab) if config.get("conversion") else []))
        if config.get("conversion"):
            card["conversion"] = C.audit(model,vocab)
        gradients, signed = M.objective_gradients(model, vocab, lesson)
        tensors.update({f"objective.{k}": v for k,v in signed.items()})
        inactive = {}
        for site in ("mlp1.fc","mlp2.fc","readout.pre_relu"):
            points = torch.cat([tensors[f"{i}.activation.{site}"].reshape(-1,
                              tensors[f"{i}.activation.{site}"].shape[-1])
                                for i in range(len(traces["examples"]))])
            inactive[site] = dict(panel_inactive=int((points.abs().amax(0)<1e-4).sum())
                                  if site != "readout.pre_relu" else int((points.amax(0)<=0).sum()),
                                  units=points.shape[-1])
        lineage = M.head_lineage(card["heads"]["signature"],
                                 previous_card["concepts"]["heads"]["signature"] if step else None,
                                 previous_card["head_lineage"] if step else None)
        concept = "h[0:16]/canonical_mass"
        causal = card["causal_controls"]
        supported = (causal["mass_block"]["mass"] >= .6
                     and causal["mass_block"]["mass"]-max(causal["mass_block"]["acceleration"],
                          causal["comparison_block"]["mass"],causal["unchanged_on_target"]) >= .2)
        prior_lineage = previous_card["concept_lineage"][concept] if step else {}
        streak = prior_lineage.get("support_streak", 0) + 1 if supported else 0
        weak = prior_lineage.get("weak_streak", 0) + 1 if not supported else 0
        had_support = prior_lineage.get("status") == "sustained causal support"
        concept_lineage = {concept: dict(support_streak=streak, weak_streak=weak,
            status="sustained causal support" if streak >= 2 or (had_support and weak < 2) else "candidate",
            binding=card["binding_probe"], swap=causal["mass_block"]["mass"])}
        with (pending/"interval.jsonl").open("x") as stream:
            for row in interval:
                stream.write(json.dumps(row,allow_nan=False)+"\n")
        save_file({k: v.clone() for k, v in tensors.items()}, pending/"traces.safetensors")
        write(pending/"procedure.json", dict(config=config, active_lesson=lesson,
              scheduler=dict(phase=state["phase"], step=step, learning_rates=[g["lr"] for g in state["opt"].param_groups]),
              tokenizer="sem.data.Vocab", schema="sem.lang", source_hashes=source_hashes(),
              previous_manifest=sha(previous_path/"manifest.json") if step else None,
              rng="CPU Torch and dedicated Python data-stream RNG; no NumPy or CUDA training RNG",
              objective_gradients="unweighted, local, controlled batch", batch_fingerprint="canonical JSON AST, render and twin indices"))
        by_tensor = {}
        if interval:
            for name in interval[0]["tensors"]:
                seq = [r["tensors"][name] for r in interval]
                by_tensor[name] = {key: sum(v[key] or 0 for v in seq)/len(seq) for key in
                                   ("raw_grad","actual_update","update_over_weight","adam_m","adam_v")}
        write(pending/"evidence.json", dict(step=step, training_wall_s=training_seconds,
              instrument_wall_s=time.perf_counter()-start, torch_version=torch.__version__,
              weight_changes=M.weight_changes(current, previous, initial), token_trace=traces,
              objective_dynamics=gradients, interval_overview=by_tensor, concepts=card, inactive=inactive,
              head_lineage=lineage, concept_lineage=concept_lineage))
        write(pending/"manifest.json", {f.name:sha(f) for f in pending.iterdir()})
        # Completed snapshot directories are published atomically and never replaced.
        if target.exists():
            raise FileExistsError(target)
        for file in pending.iterdir():
            file.chmod(0o444)
        pending.rename(target)


def init(run, seed, budget, arm, conversion=False):
    if budget <= 0 or budget % INTERVAL:
        raise ValueError("budget must be a positive multiple of 100")
    run.mkdir()
    config = dict(conversion=conversion, seed=seed, budget=budget, arm=arm, checkpoint_interval=INTERVAL,
                  source=source_hashes(), purpose="non-study feasibility",
                  base_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    write(run/"run.json", config)
    dev = ROOT/"runs/zoo_sem/eval"
    for file in (dev/"dev").glob("*.jsonl"):
        dest = run/"development/dev"/file.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(file.read_bytes())
    vocab = D.Vocab.from_json(read(dev/"vocab.json"))
    lesson = C.lesson() if conversion else FIXED_CURRICULUM[0]
    snapshot(run, T.new_state(vocab, seed), config, [], lesson, 0)


def pointer(document, reference):
    for key in reference.split("/")[1:]:
        key = key.replace("~1", "/").replace("~0", "~")
        document = document[int(key)] if isinstance(document, list) else document[key]
    return document


def validate_review(checkpoint, config, review, *, previous=None):
    card = verify(checkpoint)
    if review["checkpoint_sha256"] != sha(checkpoint/"manifest.json") or review["step"] != card["step"]:
        raise ValueError("review belongs to another checkpoint")
    if not isinstance(review.get("supervisor"), str) or not review["supervisor"].strip():
        raise ValueError("review must identify the supervisor")
    for key in QUESTIONS:
        answer = review["assessment"][key]
        if not answer["analysis"].strip() or not answer["evidence"]:
            raise ValueError(f"empty assessment: {key}")
        for ref in answer["evidence"]:
            if not ref.startswith("/") or pointer(card,ref) is None:
                raise ValueError(f"invalid evidence reference: {ref}")
    decision = review["action"]
    if decision not in ACTIONS:
        raise ValueError("unrecognized or unauthorized action")
    if decision == "curriculum":
        if config["arm"] == "fixed":
            raise ValueError("fixed curriculum is independent of reviews")
        D.validate_lesson(review["lesson"])
    elif "lesson" in review:
        raise ValueError("only curriculum decisions may supply lessons")
    prediction = review["prediction"]
    if (not prediction["statement"].strip() or not prediction["metric"].startswith("/")
        or type(pointer(card,prediction["metric"])) not in (int,float)
        or prediction["direction"] not in ("at_least","at_most")
        or not isinstance(prediction["threshold"], (int,float))
        or prediction["due"] != card["step"] + INTERVAL):
        raise ValueError("prediction must identify a measured outcome at the next checkpoint")
    if previous:
        old = previous["prediction"]
        value = pointer(card,old["metric"])
        hit = value >= old["threshold"] if old["direction"] == "at_least" else value <= old["threshold"]
        resolution = review["previous_resolution"]
        if resolution["status"] != ("confirmed" if hit else "falsified") or resolution["observed"] != value:
            raise ValueError("prior prediction resolution does not match checkpoint")
    return card


def advance(run, checkpoint, config, previous):
    step = int(checkpoint.name.split("_")[-1])
    review = read(checkpoint/"review.json")
    validate_review(checkpoint, config, review, previous=previous)
    if step >= config["budget"]:
        print("Budget completed. Final review verified.")
        return
    if review["action"] in ("diagnose","propose_weight_edit"):
        raise ValueError("diagnosis or edit proposal is pending. Review resolution required before continuing.")
    state = T.load_state(checkpoint)
    baseline = FIXED_CURRICULUM[min(step//INTERVAL, len(FIXED_CURRICULUM)-1)]
    if config.get("conversion"):
        baseline = read(checkpoint/"procedure.json")["active_lesson"]
    lesson = review.get("lesson", baseline)
    state["interval"] = []
    started = time.perf_counter()
    T.train_steps(state, lesson, INTERVAL, recorder=M.record_step)
    snapshot(run, state, config, state["interval"], lesson, time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init","review","advance","verify"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--seed", type=int, default=409327)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--arm", choices=("fixed","adaptive","toolsmith"), default="fixed")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--conversion", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.action == "init":
        init(run,args.seed,args.budget,args.arm,args.conversion)
        return
    with (run/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        config = read(run/"run.json")
        if config["source"] != source_hashes():
            raise ValueError("frozen code mismatch")
        checkpoints = sorted(run.glob("step_*"))
        previous = None
        for ck in checkpoints[:-1]:
            entry = read(ck/"review.json")
            validate_review(ck,config,entry,previous=previous)
            previous = entry
        checkpoint = checkpoints[-1]
        if args.action == "review":
            review = read(args.review)
            validate_review(checkpoint,config,review,previous=previous)
            review_path = checkpoint/"review.json"
            if review_path.exists():
                if read(review_path)["action"] not in ("diagnose","propose_weight_edit"):
                    raise ValueError("terminal decision already submitted")
                archive = checkpoint / f"review.pending.{sha(review_path)}.json"
                review_path.rename(archive)
            write(review_path,review)
        elif args.action == "advance":
            advance(run,checkpoint,config,previous)
        else:
            card = verify(checkpoint)
            validate_review(checkpoint,config,read(checkpoint/"review.json"),previous=previous)
            print(f"Verified {len(checkpoints)} complete audits through step {card['step']}")


if __name__ == "__main__":
    main()
