import sys; sys.path.insert(0,'work')
from common import *
import torch
gen=torch.Generator().manual_seed(0)
vars_={'a':A,'b':B,'op':OP,'pos':oppos,'res':res}
def proj(name,r):
    v=torch.tensor([vars_[name](t) for t in L])
    vals=sorted(set(v.tolist()))
    M=torch.stack([H[v==k].mean(0) for k in vals]); M=M-H.mean(0)
    U,S,Vh=torch.linalg.svd(M,full_matrices=False)
    return Vh[:r].T, v, S
def outfn(t, name, srcval):
    a,b,o=A(t),B(t),OP(t)
    if name=='a': a=srcval
    if name=='b': b=srcval
    if name=='op': o=srcval
    if name=='res': return srcval
    if name=='pos': return f(a,b,o)
    return f(a,b,o)
N=4000
idx=torch.randperm(20000,generator=gen)[:N]; src=torch.randperm(20000,generator=gen)[:N]
for name in vars_:
    for r in [1,2,3,5,8,10]:
        U,v,S=proj(name,r)
        if r>len(S): continue
        P=U@U.T
        hn=H[idx]-H[idx]@P+H[src]@P
        with torch.no_grad(): y=net.head_from_resid(hn).argmax(-1)
        tgt=torch.tensor([outfn(L[i],name,v[j].item()) for i,j in zip(idx.tolist(),src.tolist())])
        acc=(y==tgt).float().mean().item()
        hm=H[idx]-H[idx]@P+H.mean(0)@P
        with torch.no_grad(): ym=net.head_from_resid(hm).argmax(-1)
        base=torch.tensor([res(L[i]) for i in idx.tolist()])
        nec=(ym==base).float().mean().item()
        print(name,r,'interchange',round(acc,3),'ablate_acc',round(nec,3))
    print(name,'sv',S[:12].numpy().round(2))
