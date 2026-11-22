import sys; sys.path.insert(0,'work')
from common import *
import torch
gen=torch.Generator().manual_seed(1)
vars_={'ab':lambda t:A(t)*11+B(t),'abop':lambda t:(A(t)*11+B(t))*3+OP(t)-11,'res':res,'a':A,'b':B,'op':OP}
def proj(name,r):
    v=torch.tensor([vars_[name](t) for t in L])
    vals=sorted(set(v.tolist()))
    M=torch.stack([H[v==k].mean(0) for k in vals]); M=M-H.mean(0)
    U,S,Vh=torch.linalg.svd(M,full_matrices=False)
    return Vh[:r].T, v
def outfn(t,name,s):
    a,b,o=A(t),B(t),OP(t)
    if name=='ab': a,b=s//11,s%11
    if name=='abop': a,b,o=s//33,(s//3)%11,s%3+11
    if name=='res': return s
    if name=='a': a=s
    if name=='b': b=s
    if name=='op': o=s
    return f(a,b,o)
N=4000
idx=torch.randperm(20000,generator=gen)[:N]; src=torch.randperm(20000,generator=gen)[:N]
for name,rs in [('ab',[10,20,30,40,50,60]),('abop',[20,40,50,60,64]),('res',[10]),('a',[10]),('b',[10]),('op',[2])]:
    for r in rs:
        U,v=proj(name,r); P=U@U.T
        hn=H[idx]-H[idx]@P+H[src]@P
        with torch.no_grad(): y=net.head_from_resid(hn).argmax(-1)
        tgt=torch.tensor([outfn(L[i],name,v[j].item()) for i,j in zip(idx.tolist(),src.tolist())])
        hm=H[idx]-H[idx]@P+H.mean(0)@P
        with torch.no_grad(): ym=net.head_from_resid(hm).argmax(-1)
        print(name,r,'ichg',round((y==tgt).float().mean().item(),3),'abl',round((ym==Y[idx]).float().mean().item(),3))
# same-op pairs interchange for a,b
for name in ['a','b','res']:
    U,v=proj(name,10); P=U@U.T
    ops=torch.tensor([OP(t) for t in L])
    m=ops[idx]==ops[src]
    i2,s2=idx[m],src[m]
    hn=H[i2]-H[i2]@P+H[s2]@P
    with torch.no_grad(): y=net.head_from_resid(hn).argmax(-1)
    tgt=torch.tensor([outfn(L[i],name,v[j].item()) for i,j in zip(i2.tolist(),s2.tolist())])
    print('sameop',name,round((y==tgt).float().mean().item(),3))
