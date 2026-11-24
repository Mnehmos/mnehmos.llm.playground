"""Validate the complete pilot and replay its final interval on a disposable copy."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch

from pilot import ARMS, SEEDS, hash_file, job


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for v,w in zip(a,b):
            equal(v,w)
    else:
        assert a==b


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    root=p.parse_args().root
    for step in range(0,601,100):
        gate=json.loads((root/f"review-{step:04}.json").read_text())
        assert gate["step"]==step and gate["assessment"].strip()
        for seed in SEEDS:
            batches=[]
            start=None
            for arm in ARMS:
                tag=f"{seed}-{arm}"
                file=root/tag/f"{step:04}.pt"
                card=json.loads(file.with_suffix(".json").read_text())
                assert hash_file(file)==card["hash"]==gate["hashes"][tag]
                assert gate["notes"][tag].strip()
                ck=torch.load(file,weights_only=True)
                assert ck["step"]==step
                assert [r["step"] for r in ck["telemetry"]]==(list(range(step-99,step+1)) if step else [])
                batches.append([r["fingerprint"] for r in ck["telemetry"]])
                if step==0:
                    if start is None:
                        start=ck["model"]
                    equal(start,ck["model"])
            assert all(trace==batches[0] for trace in batches)
    seed,arm=SEEDS[0],"D"
    folder=f"{seed}-{arm}"
    with tempfile.TemporaryDirectory() as tmp:
        copy=Path(tmp)
        (copy/folder).mkdir()
        for rel in ("review-0500.json",f"{folder}/0500.pt"):
            shutil.copy2(root/rel,copy/rel)
        job((copy,seed,arm,600))
        relative=f"{folder}/0600.pt"
        equal(torch.load(root/relative,weights_only=True),torch.load(copy/relative,weights_only=True))
    print("84 checkpoint hashes, paired initializations and streams, and exact D replay verified")


if __name__=="__main__":
    main()
