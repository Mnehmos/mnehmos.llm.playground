"""Prepare a public exclusion file and private assignment for a custodian-held run."""
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path

from pilot import CV, conv_ids, d, held_c, split_swaps, u, v


def prepare(secret):
    if len(secret) < 32:
        raise ValueError("Use a private salt of at least 32 bytes.")
    public = {"train_conversion_ids": conv_ids.tolist(), "visible": {}, "reserved": {}}
    private = {"hidden_dev": {}, "final": {}}
    counts = {}
    groups = {
        "numeric": (v>=9)&(d!=1)&held_c,
        "direction": (v<9)&(d==1)&held_c,
        "crossed": (v>=9)&(d==1)&held_c,
    }
    for label, mask in groups.items():
        ids = [idx for idx in range(len(CV)) if mask[idx]]
        ids.sort(key=lambda i:hmac.digest(secret,f"{label}:{i}".encode(),"sha256"))
        n = len(ids)//3
        public["visible"][label] = sorted(ids[:n])
        public["reserved"][label] = sorted(ids[n:])
        private["hidden_dev"][label] = sorted(ids[n:2*n])
        private["final"][label] = sorted(ids[2*n:])
    import torch
    for name, partition in [("visible",public["visible"]),*private.items()]:
        counts[name]={}
        for label, ids in partition.items():
            (base, *rest), _ = split_swaps(torch.tensor(ids),conv_ids)
            if not len(base) or len(base.unique()) != len(ids):
                raise ValueError(f"Undiscriminating conversion base in {name}/{label}")
            counts[name][label]={"base":len(ids),"discriminating":len(base)}
    public["visible_counts"] = counts["visible"]
    # Publishing hidden counts can disclose membership in this small universe.
    return public,private,counts


def main():
    p=argparse.ArgumentParser()
    p.add_argument("salt",type=Path)
    p.add_argument("public_output",type=Path)
    p.add_argument("private_output",type=Path)
    args=p.parse_args()
    public,private,counts=prepare(args.salt.read_bytes())
    # Do not put a digest of a tiny manifest without a secret nonce in a public register.
    payload=json.dumps({"nonce":hashlib.sha256(args.salt.read_bytes()).hexdigest(),
                        "splits":private,"counts":counts},sort_keys=True).encode()
    with os.fdopen(os.open(args.private_output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"wb") as f:
        f.write(payload)
    public["private_commitment"]=hashlib.sha256(payload).hexdigest()
    with args.public_output.open("x") as f:
        json.dump(public,f,indent=2)


if __name__=="__main__":
    main()
