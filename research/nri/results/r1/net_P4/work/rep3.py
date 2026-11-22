import sys; sys.path.insert(0,'work')
from common import *
import torch
gen=torch.Generator().manual_seed(2)
def ab3(t): return (A(t)*11+B(t))*3+OP(t)-11
v=torch.tensor([ab3(t) for t in L]); vals=sorted(set(v.tolist())); print('nclasses',len(vals))
M=torch.stack([H[v==k].mean(0) for k in vals])-H.mean(0)
U,S,Vh=torch.linalg.svd(M,full_matrices=False); print(S.numpy().round(1))
N=4000
idx=torch.randperm(20000,generator=gen)[:N]; src=torch.randperm(20000,generator=gen)[:N]
for r in [4,6,8,10,12,14,16,18,20,24]:
    P=Vh[:r].T@Vh[:r]
    hn=H[idx]-H[idx]@P+H[src]@P
    with torch.no_grad(): y=net.head_from_resid(hn).argmax(-1)
    tgt=Y[src]  # output_fn with abop from source = source's result
    hm=H[idx]-H[idx]@P+H.mean(0)@P
    with torch.no_grad(): ym=net.head_from_resid(hm).argmax(-1)
    print(r,round((y==tgt).float().mean().item(),3),'abl',round((ym==Y[idx]).float().mean().item(),3))
# residual variance outside top-20
P=Vh[:20].T@Vh[:20]; Hc=H-H.mean(0)
print('frac var in top20', ((Hc@P)**2).sum()/(Hc**2).sum())
