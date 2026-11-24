"""Frozen inference: scores are compared at the independent seed level."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from confirm import SEEDS

CONTRASTS=(('E','B'),('C','B'),('D','F'),('B','A'))
BOOT_SEED=831309


def infer(scores):
    result={}
    rng=np.random.default_rng(BOOT_SEED)
    index=rng.integers(0,len(SEEDS),(10000,len(SEEDS)))
    signs=np.array(list(itertools.product((-1,1),repeat=len(SEEDS))))
    for a,b in CONTRASTS:
        x=np.array([scores[f'{s}-{a}']['splits']['final']['authority']-
                    scores[f'{s}-{b}']['splits']['final']['authority'] for s in SEEDS])
        avg=x.mean()
        p=(np.abs((signs*x).mean(1))>=abs(avg)-1e-12).mean()
        result[f'{a}-{b}']=dict(mean=float(avg),ci=np.quantile(x[index].mean(1),[.025,.975]).tolist(),p=float(p),paired=x.tolist())
    previous=0.
    for rank,k in enumerate(sorted(result,key=lambda k:result[k]['p'])):
        previous=max(previous,result[k]['p']*(len(result)-rank))
        result[k]['holm']=min(1,previous)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    root=p.parse_args().root
    scores=json.loads((root/'sealed_scores.json').read_text())
    print(json.dumps(infer(scores),indent=2))
