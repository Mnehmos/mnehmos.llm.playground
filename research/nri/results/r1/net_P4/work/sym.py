import sys; sys.path.insert(0,'work')
from common import *
import torch
ops=torch.tensor([OP(t) for t in L]); a=torch.tensor([A(t) for t in L]); b=torch.tensor([B(t) for t in L]); pos=torch.tensor([oppos(t) for t in L])
for o in [11,12,13]:
    m=ops==o; v=(a*11+b)[m]; Hm=H[m]
    M=torch.stack([Hm[v==k].mean(0) for k in range(121)])
    dsw=[(M[i*11+j]-M[j*11+i]).norm().item() for i in range(11) for j in range(11) if i!=j]
    dall=torch.cdist(M,M); 
    within=torch.cat([(Hm[v==k]-M[k]).norm(dim=1) for k in range(121)])
    print(o,'swap dist',sum(dsw)/len(dsw),'mean pair dist',dall.mean().item(),'within-class spread',within.mean().item())
# within class spread due to pos?
m=ops==11
for p in range(1,5):
    mm=m&(pos==p); print(p, mm.sum().item())
