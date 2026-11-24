"""Experiment 31 isolated continuations and prospective replications."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
import study_model as core

REPAIR_SEEDS=(830701,830703,830709)
HOST_SEEDS=(831211,831223,831227)
PARITY_SEEDS=(831301,831307,831311)
REPAIRS=("untouched","zero","eighth","budget","causal","protected","untouched_reset","zero_reset")
EARLY=(1,5,10,20,50)
OLD=Path('/mnt/data/paper-three-confirm')
SOURCE_ARMS="ABDF"


def sha(p):
    return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()


def jobs(study,smoke=False):
    if study=='repair':
        seeds=(REPAIR_SEEDS[0],) if smoke else REPAIR_SEEDS
        arms="B" if smoke else SOURCE_ARMS
        return sorted(f'{s}-{a}-{k}' for s,a,k in itertools.product(seeds,arms,REPAIRS))
    if study=='prehost':
        return [str(s) for s in ((831199,) if smoke else HOST_SEEDS)]
    if study=='host':
        return [f'{s}-{a}' for s,a in itertools.product(HOST_SEEDS,"ABCDEF")]
    return [f'{s}-{a}' for s,a in itertools.product((831299,) if smoke else PARITY_SEEDS,"ABCE")]


def from_ck(model,opt,rng,ck,reset=False):
    model.load_state_dict(ck['model'])
    if not reset:
        opt.load_state_dict(ck['optimizer'])
    rng.set_state(ck['generator'])
    torch.set_rng_state(ck['torch_rng'])


def new_optimizer(s):
    return torch.optim.AdamW([p for p in s.parameters() if p.requires_grad],lr=.002)


def partition(study):
    if study=='repair':
        private=json.loads((OLD/'released_manifest.json').read_text())
        core.dev_c=torch.tensor(sorted(sum(private['splits']['final'].values(),[])))
    elif study in ('host','prehost'):
        mask=(core.u==1)&((core.v>=8)|(core.d==2))
        core.conv_ids=torch.where(~mask)[0]
        held=torch.where(mask)[0]
        shuff=held[torch.randperm(len(held),generator=torch.Generator().manual_seed(830079))]
        core.dev_c=shuff[:len(held)//2].sort().values


def conversion_metrics(s,study,final=False):
    panels={'dev':core.dev_c}
    if final and study=='host':
        held=torch.where((core.u==1)&((core.v>=8)|(core.d==2)))[0]
        panels={'test':held[~torch.isin(held,core.dev_c)]}
    return core.assess(s,panels=panels)


def repair(s,kind):
    if kind.startswith('untouched'): return
    unit=s.residual.W_U.weight
    with torch.no_grad():
        reference=unit[:13].clone()
        if kind.startswith('zero'):
            unit[:13].zero_()
            return
        if kind=='eighth':
            unit[:13].mul_(.125)
            return
        ids=core.conv_ids
        i,j,target,_=core.split_swaps(ids,ids)[0]
        feature=s.residual.resid_pre_mlp(core.cv_tok,torch.zeros_like(core.cv_tok))
        h=feature+s.residual.W_out(s.residual.W_in(feature).relu())
        base=s.conversion(i,j,raw=torch.zeros((len(i),13)))[0]
        ordinary=s.conversion(ids,raw=torch.zeros((len(ids),13)))[0]
    edited=nn.Parameter(reference.clone())
    opt=torch.optim.Adam([edited],lr=.003)
    gen=torch.Generator().manual_seed(831500)
    cap=.10*reference.norm()
    for _ in range(50):
        ix=torch.randint(len(i),(96,),generator=gen)
        loss=F.cross_entropy(base[ix]+h[i[ix]]@edited.T,target[ix])
        loss=loss+F.cross_entropy(ordinary+h[ids]@edited.T,core.Y[ids])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if kind=='budget':
            with torch.no_grad():
                delta=edited-reference
                edited.copy_(reference+delta*min(1.,float(cap/delta.norm().clamp_min(1e-12))))
    with torch.no_grad():
        unit[:13].copy_(edited)
        if kind=='budget':
            assert (unit[:13]-reference).norm() <= cap+1e-6


BITS=torch.cartesian_prod(*([torch.arange(2)]*8))
PARITY=BITS.sum(1)%2
ORDER=torch.randperm(len(BITS),generator=torch.Generator().manual_seed(38577))
PT,PD,PF=ORDER[:160],ORDER[160:208],ORDER[208:]
assert len(set(PT.tolist())|set(PD.tolist())|set(PF.tolist()))==256


class Parity(nn.Module):
    def __init__(self,seed,arm):
        super().__init__()
        torch.manual_seed(seed)
        self.arm=arm
        self.transition=nn.Parameter(torch.tensor([[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]),requires_grad=arm=='A')
        self.residual=core.TinyTyped('untyped',d=32,n_heads=4,d_mlp=64,n_types=1,vocab=2,seq_len=8,n_out=2)
        nn.init.zeros_(self.residual.W_U.weight)

    def sweep(self,bits,state):
        for bit in bits.T:
            state=torch.einsum('ni,nij->nj',state,self.transition[bit])
        return state

    def forward(self,idx,donor=None,return_parts=False):
        tokens=BITS[idx]
        prefix=tokens if donor is None else BITS[donor]
        zero=F.one_hot(torch.zeros_like(idx),2).float()
        state=self.sweep(prefix[:,:4],zero)
        exact=self.sweep(tokens[:,4:],state)
        raw=self.residual(tokens,torch.zeros_like(tokens))
        residual=raw*.0 if self.arm=='E' else .4*raw.tanh() if self.arm=='C' else raw
        return (exact,residual,state) if return_parts else exact+residual


def parity_pairs(base):
    i,j=torch.cartesian_prod(base,PT).T
    target=(BITS[j,:4].sum(1)+BITS[i,4:].sum(1))%2
    keep=target!=PARITY[i]
    return i[keep],j[keep],target[keep]


@torch.no_grad()
def parity_metrics(s,final=False):
    base=PF if final else PD
    i,j,target=parity_pairs(base)
    groups=dict()
    for name,ids in [('dev',base),('practice',PT)]:
        exact,r,_=s(ids,return_parts=True)
        groups[name]=dict(integrity=core.rate(exact,PARITY[ids]),accuracy=core.rate(exact+r,PARITY[ids]))
    exact,r,_=s(i,j,return_parts=True)
    d=groups['dev']
    d.update(authority=core.rate(exact+r,target),isolated=core.rate(exact,target),n=len(i),
             unchanged_rate=1-core.rate(exact+r,target),unpatched=core.rate(s(i),target),
             spread_max=float((r[:,0]-r[:,1]).abs().max()), program_margin=float((exact.gather(1,target[:,None]).flatten()-exact.gather(1,(1-target)[:,None]).flatten()).min()),
             identity=core.rate(s(base,base),PARITY[base]))
    x,y=torch.cartesian_prod(base,PT).T
    same=(BITS[x,:4].sum(1)%2==BITS[y,:4].sum(1)%2)&BITS[x,:4].ne(BITS[y,:4]).any(1)
    d['equal_state']=core.rate(s(x[same],y[same]),PARITY[x[same]])
    return groups


def p_trace(s):
    net=s.residual
    trace={}
    i,j,y=parity_pairs(PD)
    for label,x,donor,target in [('practice',PT[:3],PT[:3],PARITY[PT[:3]]),
                                  ('heldout',PD[:3],PD[:3],PARITY[PD[:3]]),
                                  ('intervention',i[:3],j[:3],y[:3])]:
        def evaluate(z):
            zero=F.one_hot(torch.zeros_like(x),2).float()
            state=s.sweep(BITS[donor,:4],zero).detach().requires_grad_()
            exact=s.sweep(BITS[x,4:],state)
            res=z*.0 if s.arm=='E' else .4*z.tanh() if s.arm=='C' else z
            return F.cross_entropy(exact+res,target),dict(program_state=state)
        trace[label]=core.get_trace(s,BITS[x],evaluate)
    return trace


def host_score(s):
    with torch.no_grad():
        ids=core.force_groups['held_form']
        r=s.trunk(core.force_tok[ids])[:,26:]
        return r,core.rate(r,core.routing_y[ids])


def host_init(seed,arm,root):
    chk=root.parent/'prehost'/str(seed)/'0200.pt'
    receipt=json.loads((root.parent/'prehost'/'review-0200.json').read_text())
    assert receipt['hashes'][str(seed)]==sha(chk)
    ck=torch.load(chk,weights_only=True)
    s=core.Student(seed,arm)
    installed={k:s.state_dict()[k].clone() for k in ('encoder','decoder')}
    s.load_state_dict(ck['model'])
    before,acc=host_score(s)
    assert acc>=.95
    with torch.no_grad():
        for k,v in installed.items():
            getattr(s,k).copy_(v)
        assert s.residual.W_U.weight[:13].count_nonzero()==0
    after,_=host_score(s)
    assert torch.equal(before,after)
    return s,dict(source_hash=sha(chk),preinstall_accuracy=acc,retained_exact_logits=True)


def objective(s,study,gen,prot):
    if study=='parity':
        t=core.draw(gen,PT,64)
        return (F.cross_entropy(s(t),PARITY[t]),),t,BITS[t]
    if study=='prehost':
        f=core.draw(gen,core.force_ids,96)
        z=s.trunk(core.force_tok[f])[:,26:]
        return (F.cross_entropy(z,core.routing_y[f]),),f,core.force_tok[f]
    c,f=core.draw(gen,core.conv_ids,64),core.draw(gen,core.force_ids,96)
    extra,donor=core.draw(gen,core.conv_ids,64),core.draw(gen,core.conv_ids,64)
    pc,rc=s.conversion(c)
    fl,rl=s.additional(f)
    z=sum(s.conversion(extra,donor if s.arm=='D' else extra))
    y=core.q[donor]*core.INV[core.d[extra]]%13 if s.arm=='D' else core.Y[extra]
    losses=(F.cross_entropy(pc+rc,core.Y[c]),F.cross_entropy(fl,core.FY[f]),
            F.cross_entropy(rl,core.routing_y[f]),
            float(s.arm in 'DF')*F.cross_entropy(z,y),
            float(prot)*((rc-rc.mean(1,keepdim=True))**2).mean())
    return losses,torch.cat((c,f,extra,donor)),torch.cat((core.cv_tok[c],core.force_tok[f],core.cv_tok[extra],core.cv_tok[donor]))


def summarize_weights(s,initial,previous):
    out={}
    for k,p in s.named_parameters():
        p=p.detach()
        old=previous[k]
        step=p-old
        out[k]=dict(delta=float(step.norm()),since_start=float((p-initial[k]).norm()),
                    relative=float(step.norm()/old.norm().clamp_min(1e-12)),
                    cosine=float(F.cosine_similarity(p.flatten(),old.flatten(),dim=0)),
                    norm=float(p.norm()))
        if p.ndim==2 and (p.shape[0]>1 and p.shape[1]>1):
            spectrum=torch.linalg.svdvals(step)
            out[k]['update_rank1_energy']=float(spectrum[0]**2/spectrum.square().sum().clamp_min(1e-25))
    return out


def save_json(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def measure(s,study,early=False):
    if study=='parity':
        return parity_metrics(s)
    if study=='prehost':
        return dict(host=host_score(s)[1])
    if early:
        return core.assess(s,early=True)
    return conversion_metrics(s,study)


def job(args):
    root,study,tag,step=args
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.process_time()
    partition(study)
    seed,*rest=tag.split('-')
    seed=int(seed)
    arm=rest[0] if rest else 'B'
    kind=rest[1] if len(rest)>1 else ''
    meta={}
    if study=='host':
        s,meta=host_init(seed,arm,root)
    elif study=='parity':
        s=Parity(seed,arm)
    else:
        s=core.Student(seed,arm)
    if study=='prehost':
        with torch.no_grad():
            s.encoder.zero_()
            s.decoder.zero_()
    opt=new_optimizer(s)
    gen=torch.Generator().manual_seed(seed+273)
    folder=root/tag
    folder.mkdir(parents=True,exist_ok=True)
    if step:
        prev=folder/f'{step-100:04}.pt'
        receipt=json.loads((root/f'review-{step-100:04}.json').read_text())
        assert receipt['hashes'][tag]==sha(prev)
        assert len(receipt['assessment'].split())>=55
        ck=torch.load(prev,weights_only=True)
        from_ck(s,opt,gen,ck)
        meta=ck['metadata']
    elif study=='repair':
        path=OLD/f'{seed}-{arm}'/'0600.pt'
        origin=json.loads((OLD/'review-0600.json').read_text())['hashes'][f'{seed}-{arm}']
        assert sha(path)==origin
        from_ck(s,opt,gen,torch.load(path,weights_only=True),kind.endswith('_reset'))
        weights={k:p.detach().clone() for k,p in s.named_parameters()}
        # Preserve original trajectory RNG across each matched repair copy.
        repair(s,kind)
        edits=summarize_weights(s,weights,weights)
        meta=dict(source_hash=origin,optimizer_reset=kind.endswith('_reset'),
                  changes={k:v for k,v in edits.items() if v['delta']},
                  original_conversion_readout_norm=float(weights['residual.W_U.weight'][:13].norm()),
                  absolute_global_delta=sum(v['delta']**2 for v in edits.values())**.5,
                  relative_global_delta=(sum(v['delta']**2 for v in edits.values())/
                                         sum(float(p.norm())**2 for p in weights.values()))**.5)
        meta['relative_conversion_readout_delta']=meta['absolute_global_delta']/meta['original_conversion_readout_norm']
        assert sha(path)==origin
    initial={k:p.detach().clone() for k,p in s.named_parameters()}
    if step:
        zero=torch.load(folder/'0000.pt',weights_only=True)['model']
    else:
        zero=initial
    records,early=[],{}
    for t in range(step-99,step+1) if step else ():
        losses,ids,tok=objective(s,study,gen,kind=='protected')
        opt.zero_grad()
        before={k:p.detach().clone() for k,p in s.named_parameters()}
        sum(losses).backward()
        opt.step()
        records.append(dict(step=t,losses=[float(x.detach()) for x in losses],lr=.002,
            batch=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),token_exposure=tok.flatten().bincount().tolist(),
            tensors={k:[None if p.grad is None else float(p.grad.norm()),float((p.detach()-before[k]).norm())]
                     for k,p in s.named_parameters()}))
        if t in EARLY:
            early[str(t)]=measure(s,study,early=True)
    metrics=measure(s,study)
    if study in ('host','parity') and (step==0 or arm in 'CE'):
        assert metrics['dev']['integrity']==metrics['dev']['authority']==metrics['dev']['isolated']==1
        assert metrics['dev']['accuracy']==metrics['dev']['equal_state' if study=='parity' else 'equivalence']==1
    if study=='prehost' and step==200:
        assert metrics['host']>=.95
    if study!='parity' and arm!='A':
        assert all(torch.equal(s.state_dict()[k],zero[k]) for k in ('encoder','decoder'))
    if study=='parity' and arm!='A':
        assert torch.equal(s.transition,zero['transition'])
    stats=summarize_weights(s,zero,initial)
    if study=='parity':
        trace=p_trace(s)
    elif study=='prehost':
        ids=core.force_groups['held_form'][:4]
        trace={'routing':core.get_trace(s,core.force_tok[ids],
                               lambda z:(F.cross_entropy(z[:,26:],core.routing_y[ids]),{}))}
    else:
        trace=core.trace_panel(s)
    audit_gen=torch.Generator()
    audit_gen.set_state(gen.get_state())
    objectives,_,_=objective(s,study,audit_gen,kind=='protected')
    # Disaggregate local gradients on the same controlled batch; no live optimizer update.
    named={k:p for k,p in s.named_parameters() if p.requires_grad}
    local={str(i):{k:x.detach() for k,x in zip(named,torch.autograd.grad(loss,list(named.values()),allow_unused=True,retain_graph=True))
                   if x is not None} for i,loss in enumerate(objectives)}
    alignment={}
    for a,b in itertools.combinations(local,2):
        alignment[a+'-'+b]={k:float(F.cosine_similarity(local[a][k].flatten(),local[b][k].flatten(),dim=0))
                            for k in local[a].keys()&local[b].keys()}
    if step:
        previous_trace=ck['trace']
        for k,r in trace.items():
            before=previous_trace[k]['activations']
            trace_site='program_state' if study=='parity' else 'canonical'
            r['longitudinal_dot']={a:float(F.cosine_similarity(v.flatten(),before[a].flatten(),dim=0))
                                   for a,v in r['activations'].items()
                                   if a in (trace_site,'layer1','layer2','mlp') and a in before}
    path=folder/f'{step:04}.pt'
    assert not path.exists()
    tmp=path.with_suffix('.tmp')
    torch.save(dict(model=s.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
                    generator=gen.get_state(),step=step,scheduler={'type':'constant','lr':.002},trace=trace,
                    objective_gradients=local,telemetry=records,metadata=meta),tmp)
    tmp.replace(path)
    summaries={k:{x:float(v.norm()) for x,v in w['gradients'].items()
                   if x in ('embedding','canonical','program_state','mass','accel')} for k,w in trace.items()}
    trace_geometry={k:{'mlp_active':float((r['activations']['mlp']>0).float().mean()),
                      'attention2_last_to_first':float(r['activations']['attention2'][:,:,-1,0].mean())}
                      for k,r in trace.items()}
    updates={k:sum(x['tensors'][k][1] for x in records)/max(len(records),1) for k in stats}
    save_json(path.with_suffix('.json'),dict(hash=sha(path),metrics=metrics,early=early,weights=stats,
                trace=summaries,geometry=trace_geometry,
                gradient_alignment=alignment,lineage={k:r.get("longitudinal_dot",{}) for k,r in trace.items()},
                objective_readout={i:float(g['residual.W_U.weight'][:(2 if study=='parity' else 13)].norm())
                                   for i,g in local.items() if 'residual.W_U.weight' in g},
                updates=updates,metadata=meta,cpu=time.process_time()-start))
    return tag


def finish(root,study):
    torch.set_num_threads(1)
    partition(study)
    assert (root/'review-0600.json').exists()
    result={}
    for tag in jobs(study):
        seed,arm=tag.split('-')
        s=Parity(int(seed),arm) if study=='parity' else core.Student(int(seed),arm)
        path=root/tag/'0600.pt'
        receipt=json.loads((root/'review-0600.json').read_text())
        assert sha(path)==receipt['hashes'][tag]
        s.load_state_dict(torch.load(path,weights_only=True)['model'])
        result[tag]=parity_metrics(s,final=True) if study=='parity' else conversion_metrics(s,study,final=True)
    save_json(root/'final.json',result)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('study',choices=('repair','prehost','host','parity'))
    p.add_argument('step')
    p.add_argument('root',type=Path)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    if not a.smoke:
        lock=json.loads((a.root.parent/'lock.json').read_text())
        assert all(sha(Path(__file__).parent/name)==h for name,h in lock.items())
    if a.step=='final':
        finish(a.root,a.study)
    else:
        step=int(a.step)
        assert step in range(0,601,100)
        entries=jobs(a.study,a.smoke)
        # Re-entry after interruption only fills missing checkpoints.
        missing=[t for t in entries if not (a.root/t/f'{step:04}.pt').exists()]
        with ProcessPoolExecutor(8,mp_context=mp.get_context('fork')) as pool:
            done=list(pool.map(job,[(a.root,a.study,t,step) for t in missing]))
        print(a.study,step,len(done),flush=True)


if __name__=='__main__':
    main()
