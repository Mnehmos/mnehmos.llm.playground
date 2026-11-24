"""Bind an authored qualification review to each measured checkpoint hash."""
import argparse
import json
from pathlib import Path
import sys

from pilot import ARMS, SEEDS, hash_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("step",type=int,choices=range(0,601,100))
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    notes=sys.stdin.read().strip().splitlines()
    runs=[f"{seed}-{arm}" for seed in SEEDS for arm in ARMS]
    if len(notes)!=len(runs)+1 or not all(note.strip() for note in notes):
        parser.error("Provide a cohort assessment followed by one nonempty note per run.")
    hashes={}
    for run in runs:
        ck=args.root/run/f"{args.step:04}.pt"
        card=json.loads(ck.with_suffix(".json").read_text())
        hashes[run]=hash_file(ck)
        assert card["hash"]==hashes[run]
    with (args.root/f"review-{args.step:04}.json").open("x") as f:
        json.dump(dict(step=args.step,assessment=notes[0],notes=dict(zip(runs,notes[1:])),hashes=hashes),f,indent=2)


if __name__=="__main__":
    main()
