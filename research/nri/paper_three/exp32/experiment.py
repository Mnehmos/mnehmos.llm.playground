"""Prospective confidence and finite-state experiments. All training reads a public manifest."""
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
from nri.model import TinyTyped

PILOT=('ce','satisfied','scale','iit','soft','routed','ordinary')
OBJECT=('ce','satisfied','scale','soft','routed','ordinary')
MACHINE=('frozen','trainable','bounded','isolated','soft')
SEEDS=tuple(range(832301,832313))
PSEEDS=(832201,832202)
T=torch.tensor([[1,0,3,2,4],[2,1,4,0,3],[0,0,1,3,2],[4,2,0,1,3]])
S=torch.tensor([1,5,8])
I=torch.tensor([1,8,5])
V=13
early=(1,5,10,20,50)

def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()

def put(p,data):
    with p.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)

def oracle(st,sequence):
    for x in sequence.T:
        st=T[x,st]
    return st

class World:
    def __init__(self,task):
        self.task=task
        if task=='obj':
            self.v,self.u,self.d,self.style=torch.cartesian_prod(torch.arange(13),torch.arange(3),torch.arange(3),torch.arange(4)).T
            self.bits=torch.stack([self.v,13+self.u,16+self.d,19+self.style,23+self.style*0],1)
            self.q=self.v*S[self.u]%13
            self.y=self.q*I[self.d]%13
            self.size=13
        else:
            self.bits=torch.cartesian_prod(*([torch.arange(4)]*6))
            self.q=oracle(torch.zeros(len(self.bits),dtype=torch.long),self.bits[:,:3])
            self.y=oracle(self.q,self.bits[:,3:])
            self.size=5
        self.n=len(self.y)

    def target(self,i,j):
        if self.task=='obj':return self.q[j]*I[self.d[i]]%13
        return oracle(self.q[j],self.bits[i,3:])

    def panels(self,base,train):
        g=torch.Generator().manual_seed(32171)
        i=base.repeat_interleave(64 if self.task=='auto' else 128)
        j=train[torch.randint(len(train),(len(i),),generator=g)]
        want=self.target(i,j)
        alt=(self.v[j]*S[self.u[i]]*I[self.d[i]]%13 if self.task=='obj'
             else oracle(self.q[j],self.bits[i,3:].flip(1)))
        valid=torch.stack([a!=b for a,b in itertools.combinations((want,self.y[i],self.y[j],alt),2)]).all(0)
        same=(self.q[i]==self.q[j])&(i!=j)
        return (i[valid],j[valid],want[valid],alt[valid]),(i[same],j[same])

class Student(nn.Module):
    def __init__(self,w,seed,arm):
        super().__init__()
        torch.manual_seed(seed)
        self.w,self.arm=w,arm
        maps=torch.stack((S,I))[:,:,None]*torch.arange(13)[None,None,:]%13
        programmed=F.one_hot(maps,13) if w.task=='obj' else F.one_hot(T,5)
        self.program=nn.Parameter(programmed.float(),requires_grad=arm=='trainable')
        self.net=TinyTyped('untyped',d=16,n_heads=2,d_mlp=32,vocab=26,seq_len=w.bits.shape[-1],n_types=1,n_out=w.size+2)
        nn.init.zeros_(self.net.W_U.weight)

    def tokens(self,ids,extra=False):
        x=self.w.bits[ids].clone()
        if extra:x[:,-1]=24
        return x

    def sweep(self,q,ids):
        for sym in ids.T:q=torch.einsum('bi,bij->bj',q,self.program[sym])
        return q

    def state(self,id):
        if self.w.task=='obj':return self.program[0,self.w.u[id],self.w.v[id]]
        return self.sweep(F.one_hot(torch.zeros_like(id),5).float(),self.w.bits[id,:3])

    def prog(self,ids,q):
        if self.w.task=='obj':
            return torch.einsum('bi,bij->bj',q,self.program[1,self.w.d[ids]])*(5 if self.arm=='scale' else 1)
        return self.sweep(q,self.w.bits[ids,3:])

    def branch(self,r):
        if self.arm=='bounded':return .4*r.tanh()
        if self.arm=='isolated':return r*0
        return r

    def raw(self,ids,extra=False):
        x=self.tokens(ids,extra)
        return self.net(x,torch.zeros_like(x))

    def forward(self,i,j=None,extra=False):
        return self.prog(i,self.state(i if j is None else j))+self.branch(self.raw(i,extra)[:,:self.w.size])

def margin(z,y):
    other=z.masked_fill(F.one_hot(y,z.shape[1]).bool(),-torch.inf).amax(1)
    return z.gather(1,y[:,None]).squeeze(1)-other

def hinge(z,y):return (.5-margin(z,y)).relu().mean()

def draw(gen,p,n):return p[torch.randint(len(p),(n,),generator=gen)]

def lossbatch(s,gen,train):
    w,arm=s.w,s.arm
    t=draw(gen,train,40);u=draw(gen,train,40);j=draw(gen,train,40)
    if w.task=='obj':
        x=s.tokens(t);z=s.tokens(t,True);k=s.tokens(u)
        tok=torch.cat((x,z,k))
    else:tok=torch.cat((s.tokens(t),s.tokens(u)))
    logits=s.net(tok,torch.zeros_like(tok))
    main=logits[:len(t)]
    base=s.prog(t,s.state(t))
    r=s.branch(main[:,:w.size])
    clean=base+(r.detach() if arm=='routed' else r)
    l=hinge(clean,w.y[t]) if arm=='satisfied' else F.cross_entropy(clean,w.y[t])
    cross=s.prog(u,s.state(j if arm in ('iit','soft') else u))+s.branch(logits[-len(u):,:w.size])
    target=w.target(u,j) if arm in ('soft','iit') else w.y[u]
    auxiliary=hinge(cross,target) if arm in ('soft','ordinary') else F.cross_entropy(cross,target)
    weight=4. if arm in ('soft','ordinary') or not (w.task=='obj') and arm=='soft' else 1.
    if arm not in ('soft','ordinary','iit'):weight=0.
    losses=[l,weight*auxiliary]
    if w.task=='obj':
        shifted=base+s.branch(logits[len(t):2*len(t),:w.size])
        # On this flag, the required label is the written value, not conversion.
        loss_extra=F.cross_entropy(shifted,w.v[t])
        route=F.cross_entropy(main[:,w.size:],(w.u[t]>0).long())
        losses.extend([loss_extra,route])
    return losses,torch.cat((t,u,j)),tok

@torch.no_grad()
def evaluate(s,w,base,train):
    (i,j,y,alt),(u,v)=w.panels(base,train)
    cache=torch.zeros(w.n,w.size)
    cache[base]=torch.cat([s.branch(s.raw(k)[:,:w.size]) for k in base.split(256)])
    def logits(p,d=None):
        return s.prog(p,s.state(p if d is None else d)),cache[p]
    def rate(pred,want):return float((pred==want).double().mean())
    p,r=logits(base);x=(p+r).argmax(1)
    a,b=logits(i,j);z=(a+b).argmax(1)
    g,h=logits(i);unpatched=(g+h).argmax(1);good=unpatched==w.y[i]
    qa,qb=logits(u,v)
    ident=sum(logits(base,base)).argmax(1)
    random=s.prog(i,s.state(train[torch.randint(len(train),(len(i),),generator=torch.Generator().manual_seed(991))]))
    out=dict(n=len(i),unique_pairs=len((i*w.n+j).unique()),bases=len(i.unique()),ordinary=rate(x,w.y[base]),
      practiced=rate(s(train[:128]).argmax(1),w.y[train[:128]]),
      integrity=rate(p.argmax(1),w.y[base]),isolated=rate(a.argmax(1),y),
      authority=rate(z,y),correct_n=int(good.sum()),conditional=rate(z[good],y[good]) if good.any() else None,
      unpatched=rate(unpatched,y),original=rate(z,w.y[i]),donor=rate(z,w.y[j]),alternative=rate(z,alt),
      equal=rate((qa+qb).argmax(1),w.y[u]),identity=rate(ident,w.y[base]),
      random=rate((random+b).argmax(1),y),
      unrelated_output_control=rate(torch.cat((g+h,s.raw(i[:1])[:,w.size:].expand(len(i),-1)),1)[:,:w.size].argmax(1),y),
      program_margin=float(margin(a,y).min()),effective_median=float(margin(a+b,y).median()),
      residual_max_spread=float((b.amax(1)-b.amin(1)).max()))
    if w.task=='obj':
        out['outside']=rate(s(base,extra=True).argmax(1),w.v[base])
        out['route']=rate(s.raw(base)[:,w.size:].argmax(1),(w.u[base]>0).long())
    return out

def traces(s,w,base,train):
    panels=w.panels(base,train)[0]
    i,j,y,_=panels
    data={}
    for label,ids,ds,ys,external in (
        ('ordinary',train[:2],train[:2],w.y[train[:2]],False),
        ('held',base[:2],base[:2],w.y[base[:2]],False),
        ('counterfactual',i[:2],j[:2],y[:2],False),
        ('external',base[2:4],base[2:4],w.v[base[2:4]] if w.task=='obj' else w.y[base[2:4]],w.task=='obj')):
        net=s.net
        tok=s.tokens(ids,external)
        emb=net.E_c(tok)
        h=emb+net.E_t(torch.zeros_like(tok))+net.E_p(torch.arange(tok.shape[1]))
        z,a=net.attn1(h,return_pattern=True);l1=h+z
        z,b=net.attn2(l1,return_pattern=True);l2=l1+z
        v=l2[:,-1];m=net.W_in(v).relu()
        raw=net.W_U(v+net.W_out(m))
        q=s.state(ds).detach().requires_grad_(True)
        logit=s.prog(ids,q)+s.branch(raw[:,:w.size])
        loss=F.cross_entropy(logit,ys)
        sites=dict(emb=emb,context=h,att1=a,att2=b,l1=l1,l2=l2,mlp=m,raw=raw,program_state=q,logits=logit)
        grad=torch.autograd.grad(loss,list(sites.values()),allow_unused=True)
        data[label]=dict(tokens=tok,sites={k:x.detach() for k,x in sites.items()},
          gradients={k:x.detach() for k,x in zip(sites,grad) if x is not None})
    return data

def names(kind,task):
    return [(s,a) for s,a in itertools.product(PSEEDS if kind=='pilot' else SEEDS if task=='obj' else tuple(range(832401,832413)),
             PILOT if kind=='pilot' else OBJECT if task=='obj' else MACHINE)]

def job(args):
    root,kind,task,step,seed,arm=args
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.process_time()
    public=json.loads((root/'public.json').read_text())
    train=torch.tensor(public['train']);dev=torch.tensor(public['dev'])
    w=World(task);s=Student(w,seed,arm)
    opt=torch.optim.AdamW([p for p in s.parameters() if p.requires_grad],lr=.002)
    gen=torch.Generator().manual_seed(seed+321)
    tag=f'{seed}-{arm}';d=root/tag;d.mkdir(exist_ok=True)
    if step:
        old=d/f'{step-100:04}.pt'
        receipt=json.loads((root/f'review-{step-100:04}.json').read_text())
        assert receipt['hashes'][tag]==sha(old)
        state=torch.load(old,weights_only=True)
        s.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng']);gen.set_state(state['gen'])
    before={k:p.detach().clone() for k,p in s.named_parameters()}
    init=torch.load(d/'0000.pt',weights_only=True)['model'] if step else before
    records=[];hi={}
    for t in range(step-99,step+1) if step else ():
        losses,ids,tokens=lossbatch(s,gen,train)
        old={k:p.detach().clone() for k,p in s.named_parameters()}
        opt.zero_grad();sum(losses).backward();opt.step()
        records.append(dict(step=t,loss=[float(z.detach()) for z in losses],
            batch=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),exposures=tokens.flatten().bincount().tolist(),
            tensors={k:[None if p.grad is None else float(p.grad.norm()),float((p.detach()-old[k]).norm())] for k,p in s.named_parameters()}))
        if t in early:hi[t]=evaluate(s,w,dev,train)
    card=evaluate(s,w,dev,train)
    if not step or arm in ('bounded','isolated'):
        assert card['authority']==card['ordinary']==card['equal']==1
    if arm!='trainable':assert torch.equal(s.program,init['program'])
    trace=traces(s,w,dev,train)
    g=torch.Generator();g.set_state(gen.get_state())
    losses,*_=lossbatch(s,g,train)
    par={k:p for k,p in s.named_parameters() if p.requires_grad}
    grads={str(i):{k:v.detach() for k,v in zip(par,torch.autograd.grad(l,list(par.values()),retain_graph=True,allow_unused=True)) if v is not None} for i,l in enumerate(losses) if l.requires_grad}
    tensor_data={}
    for k,p in s.named_parameters():
        delta=p.detach()-before[k]
        if p.ndim==2:
            spec=torch.linalg.svdvals(delta)
            energy=float(spec[0].square()/spec.square().sum().clamp_min(1e-20))
        else:energy=None
        tensor_data[k]=dict(delta=float(delta.norm()),since0=float((p.detach()-init[k]).norm()),
            norm=float(p.detach().norm()),cos=float(F.cosine_similarity(p.detach().flatten(),before[k].flatten(),dim=0)),
            leading_singular_energy=energy,
            mean_update=sum(r['tensors'][k][1] for r in records)/max(len(records),1))
    path=d/f'{step:04}.pt';assert not path.exists()
    tmp=path.with_suffix('.tmp')
    torch.save(dict(model=s.state_dict(),optimizer=opt.state_dict(),scheduler={'lr':.002,'kind':'constant'},
          step=step,rng=torch.get_rng_state(),gen=gen.get_state(),telemetry=records,trace=trace,objective_grads=grads),tmp)
    tmp.replace(path)
    put(path.with_suffix('.json'),dict(hash=sha(path),metrics=card,early=hi,weights=tensor_data,
        max_update=max(tensor_data,key=lambda k:tensor_data[k]['mean_update']),
        traces={a:{'occurrence':float(x['gradients']['emb'].norm()),'state':float(x['gradients']['program_state'].norm()),
          'attention':float(x['sites']['att2'][:,:,-1,0].mean()),'active':float((x['sites']['mlp']>0).float().mean())} for a,x in trace.items()},
        gradients={i:float(x['net.W_U.weight'][:w.size].norm()) if 'net.W_U.weight' in x else 0. for i,x in grads.items()},
        cpu=time.process_time()-start))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['pilot','study']);p.add_argument('task',choices=['obj','auto'])
    p.add_argument('root',type=Path);p.add_argument('step',type=int)
    a=p.parse_args()
    if a.kind=='study':
        lock=json.loads((a.root.parent/'lock.json').read_text())
        assert all(sha(Path(__file__).parent/p)==v for p,v in lock.items())
    with ProcessPoolExecutor(8,mp_context=mp.get_context('fork')) as pool:
        args=[(a.root,a.kind,a.task,a.step,s,c) for s,c in names(a.kind,a.task)]
        list(pool.map(job,args))
    print(a.task,a.step,len(args),flush=True)
if __name__=='__main__':main()
