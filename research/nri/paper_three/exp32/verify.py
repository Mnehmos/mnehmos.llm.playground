"""Independent endpoint and archive-integrity cross-check."""
import argparse
import itertools
import json
from pathlib import Path
import torch
from experiment import Student,World,T,S,I,names,sha

def read(p):return json.loads(p.read_text())

def check(root):
    torch.set_num_threads(1)
    total=0
    for task in ('obj','auto'):
        r=root/task;w=World(task)
        pub=read(r/'public.json');release=read(r/'released.json')
        assert sha(r/'released.json')==pub['commitment']
        held=torch.tensor(release['final']);train=torch.tensor(pub['train'])
        all_ids=pub['train']+pub['dev']+release['hidden']+release['final']
        assert sorted(all_ids)==list(range(w.n))
        if task=='obj':
            memberships=[{x//4 for x in z} for z in (pub['train'],pub['dev'],release['hidden'],release['final'])]
            assert len(set.union(*memberships))==sum(len(x) for x in memberships)
        (i,j,y,alt),_=w.panels(held,train)
        assert all(torch.all(a!=b) for a,b in itertools.combinations((w.y[i],w.y[j],y,alt),2))
        assert len(i)==release['final_pairs'] and len(i.unique())==release['final_distinct_eligible_bases']
        scores=read(r/'scores.json')
        cohorts=names('study',task)
        first=cohorts[0][0]
        for t in range(0,301,100):
            rev=read(r/f'review-{t:04}.json')
            matching={}
            for seed,arm in cohorts:
                tag=f'{seed}-{arm}';p=r/tag/f'{t:04}.pt'
                ck=torch.load(p,weights_only=True)
                assert all(torch.isfinite(v).all() for v in ck['model'].values())
                assert all(torch.isfinite(v).all() for s in ck['optimizer']['state'].values()
                           for v in s.values() if isinstance(v,torch.Tensor))
                assert sha(p)==rev['hashes'][tag]==read(p.with_suffix('.json'))['hash']
                batches=[m['batch'] for m in ck['telemetry']]
                if seed in matching:assert matching[seed]==batches
                else:matching[seed]=batches
                if t==300 and seed==first:
                    s=Student(w,seed,arm);s.load_state_dict(ck['model'])
                    with torch.no_grad():
                        # Compute the intact and switched program using a separate matrix layout.
                        prog=ck['model']['program'].numpy()
                        import numpy as np
                        scores0=[]
                        raw=s.net(w.bits[held],torch.zeros_like(w.bits[held]))[:,:w.size].numpy()
                        lookup={int(n):v for n,v in zip(held,raw)}
                        for k,l in zip(i.tolist(),j.tolist()):
                            if task=='obj':
                                code=prog[0,int(w.u[l]),int(w.v[l])]
                                pred=code @ prog[1,int(w.d[k])]
                                if arm=='scale':pred=pred*5
                            else:
                                pred=np.eye(5,dtype='float32')[0]
                                for symbol in w.bits[l,:3].tolist()+w.bits[k,3:].tolist():
                                    pred=pred @ prog[symbol]
                            res=lookup[k]
                            if arm=='bounded':res=.4*np.tanh(res)
                            if arm=='isolated':res=res*0
                            scores0.append((pred+res).argmax())
                        acc=np.mean(np.asarray(scores0)==y.numpy())
                        # Batched GEMMs can round near a tie differently. Use the exact observed tolerance here.
                        assert abs(float(acc)-scores[tag]['final']['authority'])<1e-8
                total+=1
    assert total==528
    print('528 full checkpoint hashes, matched draws and evaluation exclusions; independently rescored 11 endpoints.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    check(p.parse_args().root)
