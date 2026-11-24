"""Endpoint lesions on cloned qualification networks. No training or primary scoring."""
import argparse
import json
from pathlib import Path

import torch
from pilot import SEEDS, Student, assess, hash_file


@torch.no_grad()
def repairs(root):
    output={}
    torch.set_num_threads(1)
    for seed in SEEDS:
        for arm in "ABDF":
            path=root/f"{seed}-{arm}"/"0600.pt"
            original=hash_file(path)
            ck=torch.load(path,weights_only=True)
            model=Student(seed,arm)
            model.load_state_dict(ck["model"])
            output[path.parent.name]={}
            for mode,scale in (("intact",1.),("restore_maps",1.),("attenuate_half",.5),
                               ("attenuate_eighth",.125),("zero_conversion",0.),
                               ("unrelated_norm_matched",0.)):
                model.load_state_dict(ck["model"])
                weights=model.residual.W_U.weight
                if mode=="restore_maps":
                    fresh=Student(seed,arm)
                    model.encoder.copy_(fresh.encoder)
                    model.decoder.copy_(fresh.decoder)
                elif mode=="unrelated_norm_matched":
                    size=weights[:13].norm()
                    unrelated=weights[26:]
                    direction=unrelated/unrelated.norm()
                    unrelated.sub_(direction*size)
                else:
                    weights[:13].mul_(scale)
                change=(weights-ck["model"]["residual.W_U.weight"]).norm()
                output[path.parent.name][mode]=dict(readout_change=float(change),metrics=assess(model))
            assert hash_file(path)==original
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    print(json.dumps(repairs(args.root),indent=2))
