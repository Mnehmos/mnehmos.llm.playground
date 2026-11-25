"""Same-machine process-separated custody for the prospective final split."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import torch
from experiment import World,Student,put,sha,evaluate,names

root=Path(sys.argv[1])
torch.set_num_threads(1)
assert ctypes.CDLL(None).prctl(4,0,0,0,0)==0
salt=os.urandom(32)
private={}
for key in ('obj','auto'):
    w=World(key)
    seed=int.from_bytes(hashlib.sha256(salt+key.encode()).digest()[:8],'big')%(2**63)
    gen=torch.Generator().manual_seed(seed)
    if key=='obj':
        # Partition semantic triples as a group so no surface aliases cross splits.
        held=((w.u==1)&((w.v>=6)|(w.d==2)))|((w.u==2)&(w.d==0))
        groups=torch.where(held.reshape(-1,4)[:,0])[0]
        groups=groups[torch.randperm(len(groups),generator=gen)]
        partitions=[(partition[:,None]*4+torch.arange(4)).flatten()
                    for partition in groups.tensor_split(3)]
        train=torch.where(~held)[0]
    else:
        order=torch.randperm(w.n,generator=gen)
        partitions=[order[i*384:(i+1)*384] for i in range(3)]
        train=order[1152:]
    dev,hid,end=partitions
    all_ids=torch.cat((train,dev,hid,end))
    assert len(all_ids)==len(all_ids.unique())==w.n
    a=w.panels(end,train)[0]
    assert len(a[0].unique())>=len(end)*(.75 if key=='obj' else .30)
    private[key]={'hidden':hid.tolist(),'final':end.tolist(),'nonce':salt.hex(),
                  'final_distinct_eligible_bases':len(a[0].unique()),'final_pairs':len(a[0])}
    payload=json.dumps(private[key],sort_keys=True,separators=(',',':')).encode()
    d=root/key;d.mkdir(exist_ok=True)
    put(d/'public.json',dict(train=train.tolist(),dev=dev.tolist(),commitment=hashlib.sha256(payload).hexdigest()))
print('Opaque splits committed. Train and visible inputs released only.',flush=True)
while not (root/'score.request').exists():time.sleep(1)
lock=json.loads((root/'lock.json').read_text())
for local,expected in lock.items():assert sha(Path(__file__).parent/local)==expected
for key in private:
    d=root/key
    for step in (0,100,200,300):
        audit=json.loads((d/f'review-{step:04}.json').read_text())
        assert audit['step']==step and len(audit['assessment'].split())>=55
        draws={}
        for seed,arm in names('study',key):
            tag=f'{seed}-{arm}'
            p=d/tag/f'{step:04}.pt'
            assert sha(p)==audit['hashes'][tag]
            ck=torch.load(p,weights_only=True)
            assert ck['step']==step
            assert [t['step'] for t in ck['telemetry']]==(list(range(step-99,step+1)) if step else [])
            sig=[t['batch'] for t in ck['telemetry']]
            if seed in draws:assert sig==draws[seed]
            else:draws[seed]=sig
    manifest=private[key]
    pay=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
    spec=json.loads((d/'public.json').read_text())
    assert hashlib.sha256(pay).hexdigest()==spec['commitment']
    assert sha(d/'public.json')==json.loads((root/'registered_manifests.json').read_text())[key]
    train=torch.tensor(spec['train']);w=World(key)
    result={}
    for seed,arm in names('study',key):
        tag=f'{seed}-{arm}';s=Student(w,seed,arm)
        ck=torch.load(d/tag/'0300.pt',weights_only=True);s.load_state_dict(ck['model'])
        result[tag]={p:evaluate(s,w,torch.tensor(manifest[p]),train) for p in ('hidden','final')}
        if arm in ('bounded','isolated'):assert result[tag]['final']['authority']==1
    (d/'released.json').write_bytes(pay)
    put(d/'scores.json',result)
print('Both registered studies verified and final partitions scored.',flush=True)
