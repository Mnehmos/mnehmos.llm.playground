"""Experiment 30 model and 100-update checkpoint worker."""
import argparse
import hashlib
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from nri.model import TinyTyped

ARMS = "ABCDEF"
SEEDS = (830401, 830409)
STEPS = 600
SCALES = torch.tensor([1,5,8])
INV = torch.tensor([pow(int(s), -1, 13) for s in SCALES])
V = torch.arange(13)
U = torch.arange(3)
CV = torch.cartesian_prod(V,U,U)
v,u,d = CV.T
q = v*SCALES[u]%13
Y = q*INV[d]%13
FR = torch.cartesian_prod(V,U,V,U,U)
m,mu,a,au,fu = FR.T
M,A = m*SCALES[mu]%13, a*SCALES[au]%13
FY = M*A*INV[fu]%13
cv_tok = torch.stack([v,13+u,19+d,22+0*v,24+0*v],1)
force_tok = torch.stack([m,13+mu,a,16+au,19+fu],1)
held_c = (u==2)&((v>=9)|(d==1))
held_f = ((mu==2)&(m>=9))|((au==2)&(a>=9))
held_pair = (3*M+5*A)%7==0
conv_ids, force_ids = torch.where(~held_c)[0], torch.where(~(held_f|held_pair))[0]
dev_c = torch.where(held_c)[0]
force_groups = dict(practice=force_ids, held_form=torch.where(held_f)[0],
                    held_pair=torch.where(held_pair&~held_f)[0])
# A small additional learned task with balanced classes and the same shared trunk.
# Return the written mass or acceleration according to destination unit parity.
routing_y = torch.where(fu%2==0,m,a)
EARLY = (1,5,10,20,50)
ALPHA = .4


class Student(nn.Module):
    def __init__(self, seed, arm):
        super().__init__()
        torch.manual_seed(seed)
        self.arm = arm
        self.encoder = nn.Parameter(F.one_hot(SCALES[:,None]*V%13,13).float(),requires_grad=arm=="A")
        self.decoder = nn.Parameter(F.one_hot(INV[:,None]*V%13,13).float(),requires_grad=arm=="A")
        self.residual = TinyTyped("untyped",d=32,n_heads=4,d_mlp=64,n_types=1,vocab=25,seq_len=5,n_out=39)
        self.force = nn.Sequential(nn.Linear(29,96),nn.GELU(),nn.Linear(96,13))
        nn.init.zeros_(self.residual.W_U.weight)

    def trunk(self,tok):
        return self.residual(tok,torch.zeros_like(tok))

    def branch(self,logits):
        if self.arm=="E":
            return 0*logits
        if self.arm=="C":
            return ALPHA*logits.tanh()
        return logits

    def conversion(self, base, donor=None, raw=None):
        src = base if donor is None else donor
        h = self.encoder[u[src],v[src]]
        core = torch.einsum("ni,nij->nj",h,self.decoder[d[base]])
        res = self.trunk(cv_tok[base])[:,:13] if raw is None else raw
        return core,self.branch(res)

    def additional(self,ids,z=None):
        if z is None:
            z = self.trunk(force_tok[ids])
        x = self.encoder[mu[ids],m[ids]]
        y = self.encoder[au[ids],a[ids]]
        dest = F.one_hot(fu[ids],3).float()
        return self.force(torch.cat([x,y,dest],1))+z[:,13:26],z[:,26:]


def draw(gen, pool, n):
    return pool[torch.randint(len(pool),(n,),generator=gen)]


def rate(logits,target):
    return float(logits.argmax(1).eq(target).double().mean())


def split_swaps(base,donors):
    i,j = torch.cartesian_prod(base,donors).T
    target = q[j]*INV[d[i]]%13
    raw = v[j]*SCALES[u[i]]*INV[d[i]]%13
    distinct = (u[i]!=u[j])&(d[i]!=d[j])&(target!=Y[i])&(target!=Y[j])&(raw!=Y[i])&(raw!=Y[j])&(raw!=target)&(Y[i]!=Y[j])
    eq = (u[i]!=u[j])&(d[i]!=d[j])&(q[i]==q[j])
    return (i[distinct],j[distinct],target[distinct],raw[distinct]),(i[eq],j[eq])


def margin(logits,target):
    other = logits.masked_fill(F.one_hot(target,13).bool(),-float("inf"))
    return logits.gather(1,target[:,None]).squeeze(1)-other.max(1).values


@torch.no_grad()
def assess(s, early=False, panels=None, include_secondary=True):
    out = {}
    if panels is None:
        panels = dict(dev=dev_c) if early else dict(practiced=conv_ids,dev=dev_c)
    all_rows = torch.cat(list(panels.values())).unique()
    z = torch.zeros((len(CV), 13))
    trunk = torch.zeros((len(CV), 39))
    trunk[all_rows] = s.trunk(cv_tok[all_rows])
    z = trunk[:,:13]
    prog,res = torch.zeros_like(z),torch.zeros_like(z)
    prog[all_rows],res[all_rows] = s.conversion(all_rows,raw=z[all_rows])
    for name,base in panels.items():
        (i,j,want,raw),(k,l) = split_swaps(base,conv_ids)
        pr,rl = s.conversion(i,j,raw=z[i])
        preserved = sum(s.conversion(k,l,raw=z[k]))
        correct = (prog[i]+res[i]).argmax(1)==Y[i]
        control_donor = conv_ids[torch.randint(len(conv_ids),(len(i),),generator=torch.Generator().manual_seed(30))]
        rand_effect = sum(s.conversion(i,control_donor,raw=z[i]))
        logits = pr+rl
        patched_unrelated = trunk[i].clone()
        patched_unrelated[:,26:] = trunk[i.roll(1),26:]
        untouched = sum(s.conversion(i,raw=patched_unrelated[:,:13]))
        pred = logits.argmax(1)
        # Conversion has one intervened variable. Wrong-variable controls apply to force.
        responses = {key:float((pred==answer).double().mean())
                     for key,answer in (("intended",want),("unchanged",Y[i]),("raw_number",raw),("donor_answer",Y[j]))}
        responses["other"] = 1-sum(responses.values())
        out[name] = dict(n=len(i), accuracy=rate(prog[base]+res[base],Y[base]),
                         integrity=rate(prog[base],Y[base]), authority=responses["intended"],
                         authority_correct_base=rate(logits[correct],want[correct]) if correct.any() else None,
                         responses=responses,isolated=rate(pr,want),residual_only=rate(rl,want),
                         program_muted_accuracy=rate(res[base],Y[base]),equivalence=rate(preserved,Y[k]),
                         identity=rate(sum(s.conversion(base,base,raw=z[base])),Y[base]),
                         random=rate(rand_effect,want),unpatched=rate(prog[i]+res[i],want),
                         unrelated_route_patch=rate(untouched,want),
                         program_margin_min=float(margin(pr,want).min()),
                         full_margin_mean=float(margin(logits,want).mean()),
                         max_residual_spread=float((rl.max(1).values-rl.min(1).values).max()))
    if early or not include_secondary:
        return out
    f,r = (torch.cat(w) for w in zip(*(s.additional(ids) for ids in torch.arange(len(FR)).split(256))))
    out["force"] = {k:rate(f[ids],FY[ids]) for k,ids in force_groups.items()}
    out["routing"] = {k:rate(r[ids],routing_y[ids]) for k,ids in force_groups.items()}
    out["force_swaps"] = force_swaps(s)
    return out


def force_swaps(s):
    bases = force_groups["held_pair"][::3]
    donors = draw(torch.Generator().manual_seed(8124), force_ids, len(bases))
    canon_m = s.encoder[mu[bases],m[bases]]
    canon_a = s.encoder[au[bases],a[bases]]
    donor_m = s.encoder[mu[donors],m[donors]]
    donor_a = s.encoder[au[donors],a[donors]]
    dest = F.one_hot(fu[bases],3).float()
    bypass = s.trunk(force_tok[bases])[:,13:26]
    labels = (M[donors]*A[bases]*INV[fu[bases]]%13,
              M[bases]*A[donors]*INV[fu[bases]]%13,
              M[donors]*A[donors]*INV[fu[bases]]%13)
    original = FY[bases]
    keep = torch.stack([x!=y for x,y in itertools.combinations((original,*labels),2)]).all(0)
    inputs = ((donor_m,canon_a),(canon_m,donor_a),(donor_m,donor_a))
    out = {}
    for i,(key,(mass,accel)) in enumerate(zip(("mass","acceleration","both"),inputs)):
        logits = s.force(torch.cat((mass,accel,dest),1))+bypass
        pred = logits[keep].argmax(1)
        answers = dict(original=original,**dict(zip(("mass","acceleration","both"),labels)))
        rates = {k:float((pred==y[keep]).double().mean()) for k,y in answers.items()}
        out[key] = dict(n=int(keep.sum()), intended=rates[key], responses=rates,
                        other=1-sum(rates.values()))
    return out


def get_trace(s,tok, loss_function):
    net=s.residual
    emb=net.E_c(tok)
    x=emb+net.E_t(torch.zeros_like(tok))+net.E_p(torch.arange(tok.shape[1]))
    y1,p1=net.attn1(x,return_pattern=True)
    h1=x+y1
    y2,p2=net.attn2(h1,return_pattern=True)
    h2=h1+y2
    mlp=net.W_in(h2[:,-1]).relu()
    logits=net.W_U(h2[:,-1]+net.W_out(mlp))
    loss, interface=loss_function(logits)
    sites=dict(embedding=emb,embed_context=x,attention1=p1,attention2=p2,layer1=h1,layer2=h2,mlp=mlp,logits=logits,**interface)
    params={name:par for name,par in s.named_parameters() if par.requires_grad}
    measurements={name:tensor for name,tensor in (sites|params).items() if tensor.requires_grad}
    grads=torch.autograd.grad(loss,list(measurements.values()),allow_unused=True)
    return dict(activations={key:value.detach() for key,value in sites.items()},
                gradients={key:v.detach() for key,v in zip(measurements,grads) if v is not None},tokens=tok)


def trace_panel(s):
    (i,j,want,_),(k,l)=split_swaps(dev_c,conv_ids)
    # Two instances per role and trace, sampling indices deterministically.
    def conversion(base,donor,labels):
        def evaluate(logits):
            interface=s.encoder[u[donor],v[donor]].detach().requires_grad_(True)
            p=torch.einsum("ni,nij->nj",interface,s.decoder[d[base]])
            return F.cross_entropy(p+s.branch(logits[:,:13]),labels),dict(canonical=interface)
        return get_trace(s,cv_tok[base],evaluate)
    fp=force_ids[[27,151]]
    fd=force_groups["held_pair"][[19,93]]
    def f_eval(ids, route=False):
        def evaluate(logits):
            mm=s.encoder[mu[ids],m[ids]].detach().requires_grad_(True)
            aa=s.encoder[au[ids],a[ids]].detach().requires_grad_(True)
            dest=F.one_hot(fu[ids],3).float()
            force=s.force(torch.cat((mm,aa,dest),1))+logits[:,13:26]
            if route:
                return F.cross_entropy(logits[:,26:],routing_y[ids]),{}
            return F.cross_entropy(force,FY[ids]),dict(mass=mm,accel=aa)
        return get_trace(s,force_tok[ids],evaluate)
    p,dev=conv_ids[[7,42]],dev_c[[3,min(15,len(dev_c)-1)]]
    return dict(practice=conversion(p,p,Y[p]),unfamiliar=conversion(dev,dev,Y[dev]),
                equivalent=conversion(k[[0,9]],l[[0,9]],Y[k[[0,9]]]),
                intervention=conversion(i[[27,min(99,len(i)-1)]],j[[27,min(99,len(i)-1)]],want[[27,min(99,len(i)-1)]]),
                force=f_eval(fp),force_unfamiliar=f_eval(fd),routing=f_eval(fd,True))


def hash_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def job(args):
    root, seed, arm, stop = args
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    begin = time.process_time()
    s=Student(seed,arm)
    named=dict(s.named_parameters())
    init={k:v.detach().clone() for k,v in named.items()}
    opt=torch.optim.AdamW([p for p in named.values() if p.requires_grad],lr=.002)
    gen=torch.Generator().manual_seed(seed+701)
    folder=root/f"{seed}-{arm}"
    folder.mkdir(parents=True,exist_ok=True)
    if stop:
        previous=folder/f"{stop-100:04}.pt"
        gate=json.loads((root/f"review-{stop-100:04}.json").read_text())
        assert gate["hashes"][folder.name]==hash_file(previous)
        assert gate["notes"][folder.name].strip()
        ck=torch.load(previous,weights_only=True)
        s.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        gen.set_state(ck["generator"])
        torch.set_rng_state(ck["torch_rng"])
    prev={k:v.detach().clone() for k,v in named.items()}
    records,early=[],{}
    for step in range(stop-99,stop+1) if stop else ():
        c, f = draw(gen,conv_ids,64),draw(gen,force_ids,96)
        extra, donor = draw(gen,conv_ids,64),draw(gen,conv_ids,64)
        # All arms execute the extra conversion forward and receive matching index streams.
        # Arm F uses extra ordinary examples of identical size and coefficient.
        c_logit=sum(s.conversion(c))
        f_logit, r_logit=s.additional(f)
        e_logit=sum(s.conversion(extra,donor if arm=="D" else extra))
        e_y=q[donor]*INV[d[extra]]%13 if arm=="D" else Y[extra]
        losses=(F.cross_entropy(c_logit,Y[c]),F.cross_entropy(f_logit,FY[f]),
                F.cross_entropy(r_logit,routing_y[f]),F.cross_entropy(e_logit,e_y))
        opt.zero_grad()
        before={name:t.detach().clone() for name,t in named.items()}
        (sum(losses[:3])+(1. if arm in "DF" else 0.)*losses[3]).backward()
        opt.step()
        idx=torch.cat([c,f,extra,donor])
        records.append(dict(step=step,losses=[float(loss.detach()) for loss in losses],lr=opt.param_groups[0]["lr"],
                            fingerprint=hashlib.sha256(idx.numpy().tobytes()).hexdigest(),
                            token_exposure=torch.cat((cv_tok[c],force_tok[f],cv_tok[extra],cv_tok[donor])).flatten().bincount(minlength=25).tolist(),
                            tensors={k:[None if t.grad is None else float(t.grad.norm()),float((t.detach()-before[k]).norm())] for k,t in named.items()}))
        if step in EARLY:
            early[step]=assess(s,early=True)
    metrics=assess(s)
    if stop==0 or arm in "CE":
        assert all(g["integrity"]==g["authority"]==g["isolated"]==g["equivalence"]==1 for g in [metrics["dev"],metrics["practiced"]])
    if arm!="A":
        assert all(torch.equal(named[k],init[k]) for k in ("encoder","decoder"))
    trace=trace_panel(s)
    update={k:[float((p.detach()-init[k]).norm()),float((p.detach()-prev[k]).norm()),
               sum(r["tensors"][k][1] for r in records)/max(len(records),1)] for k,p in named.items()}
    def grad_pair(left,right):
        agree={}
        a,b=trace[left]["gradients"],trace[right]["gradients"]
        for k in named:
            if k in a and k in b:
                agree[k]=float(F.cosine_similarity(a[k].flatten(),b[k].flatten(),dim=0))
        return agree
    path=folder/f"{stop:04}.pt"
    assert not path.exists()
    tmp=path.with_suffix(".tmp")
    torch.save(dict(model=s.state_dict(),optimizer=opt.state_dict(),scheduler={"type":"constant","lr":.002},
                    generator=gen.get_state(),torch_rng=torch.get_rng_state(),trace=trace,
                    telemetry=records,step=stop),tmp)
    tmp.replace(path)
    summary={kind:{k:float(v.norm()) for k,v in t["gradients"].items()
                    if k in ("embedding","canonical","mass","accel")} for kind,t in trace.items()}
    card=dict(metrics=metrics,early=early,update=update,trace=summary,hash=hash_file(path),
              gradient_alignment={"conversion_force":grad_pair("practice","force"),
                                  "conversion_causal":grad_pair("practice","intervention")},
              cpu_seconds=time.process_time()-begin)
    path.with_suffix(".json").write_text(json.dumps(card,indent=2,allow_nan=False))
    return folder.name


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("step",type=int,choices=range(0,STEPS+1,100))
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    with ProcessPoolExecutor(4) as pool:
        print(args.step,list(pool.map(job,[(args.output,s,a,args.step) for s,a in itertools.product(SEEDS,ARMS)])))


if __name__=="__main__":
    main()
