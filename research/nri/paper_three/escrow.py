"""Separate evaluator. Private allocation stays in process memory until the final audit."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time
import torch

import study_model as model
from confirm import SEEDS, STEPS, public_config
from splits import prepare


def write(path,data):
    with path.open('x') as f:
        json.dump(data,f,indent=2)


def verify(root):
    for step in STEPS:
        receipt=json.loads((root/f'review-{step:04}.json').read_text())
        assert receipt['step']==step and len(receipt['assessment'].split())>=60
        for seed in SEEDS:
            draws=[]
            for arm in model.ARMS:
                name=f'{seed}-{arm}'
                p=root/name/f'{step:04}.pt'
                assert receipt['hashes'][name]==model.hash_file(p)
                card=json.loads(p.with_suffix('.json').read_text())
                assert card['hash']==receipt['hashes'][name]
                ck=torch.load(p,weights_only=True)
                assert ck['step']==step
                assert [r['step'] for r in ck['telemetry']]==(list(range(step-99,step+1)) if step else [])
                draws.append([r['fingerprint'] for r in ck['telemetry']])
            assert all(r==draws[0] for r in draws)


def serve(root):
    torch.set_num_threads(1)
    # Prevent same-user process-memory inspection via standard procfs/ptrace.
    assert ctypes.CDLL(None).prctl(4,0,0,0,0)==0
    salt=os.urandom(32)
    public,private,counts=prepare(salt)
    release=dict(nonce=salt.hex(),splits=private,counts=counts)
    payload=json.dumps(release,sort_keys=True,separators=(',',':')).encode()
    public['private_commitment']=hashlib.sha256(payload).hexdigest()
    assert all(c['discriminating']>=20 and c['base']>=1 for p in counts.values() for c in p.values())
    root.mkdir(parents=True,exist_ok=True)
    write(root/'public.json',public)
    public_config(root/'public.json')
    print('Evaluator committed; hidden allocation withheld',flush=True)
    while not (root/'evaluate.request').exists():
        time.sleep(1)
    assert (root/'review-0600.json').is_file()
    lock=json.loads((root/'lock.json').read_text())
    assert lock['private_commitment']==public['private_commitment']
    assert lock['public_sha256']==model.hash_file(root/'public.json')
    for name,sha in lock['sources'].items():
        assert model.hash_file(Path(__file__).parent/name)==sha
    verify(root)
    outputs={}
    for seed in SEEDS:
        for arm in model.ARMS:
            name=f'{seed}-{arm}'
            s=model.Student(seed,arm)
            ck=torch.load(root/name/'0600.pt',weights_only=True)
            s.load_state_dict(ck['model'])
            panels={split:torch.tensor(sorted(sum(c.values(),[]))) for split,c in private.items()}
            panels.update({'final_'+k:torch.tensor(v) for k,v in private['final'].items()})
            scores=model.assess(s,panels=panels,include_secondary=False)
            with torch.no_grad():
                for k in panels:
                    if arm in 'CE':
                        assert scores[k]['authority']==scores[k]['integrity']==scores[k]['equivalence']==1
                original=s.residual.W_U.weight.clone()
                repairs={}
                if arm in 'ABDF':
                    for factor in (.5,.125,0.):
                        s.residual.W_U.weight[:13].copy_(factor*original[:13])
                        repairs[str(factor)]=model.assess(s,panels={'final':panels['final']},include_secondary=False)['final']['authority']
                    s.residual.W_U.weight.copy_(original)
            outputs[name]={'splits':scores,'repair_authority':repairs}
    (root/'released_manifest.json').write_bytes(payload)
    write(root/'sealed_scores.json',outputs)
    print('504 checkpoints and matched streams verified. Sealed endpoint released.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    serve(p.parse_args().root)
