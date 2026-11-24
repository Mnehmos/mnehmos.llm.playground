"""Primary confirmation runner. This module receives only public split identifiers."""
import argparse
import itertools
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
import study_model as model

SEEDS=(830701,830703,830709,830713,830719,830729,830731,830737,830741,830743,830747,830753)
STEPS=tuple(range(0,601,100))


def public_config(path):
    spec=json.loads(path.read_text())
    assert spec['train_conversion_ids']==model.conv_ids.tolist()
    visible=sum(spec['visible'].values(),[])
    reserved=sum(spec['reserved'].values(),[])
    arrays=(spec['train_conversion_ids'],visible,reserved)
    assert sorted(itertools.chain.from_iterable(arrays))==list(range(len(model.CV)))
    model.dev_c=torch.tensor(sorted(visible))
    return spec


def jobset(root,seeds,step,workers):
    params=((root,s,a,step) for s,a in itertools.product(seeds,model.ARMS))
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('fork')) as pool:
        print(step,len(list(pool.map(model.job,params))),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('step',type=int,choices=STEPS)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    public_config(a.root/'public.json')
    if not a.smoke:
        lock=json.loads((a.root/'lock.json').read_text())
        assert lock['public_sha256']==model.hash_file(a.root/'public.json')
        for name,sha in lock['sources'].items():
            assert model.hash_file(Path(__file__).parent/name)==sha
    jobset(a.root,(830617,) if a.smoke else SEEDS,a.step,8)


if __name__=='__main__':
    main()
