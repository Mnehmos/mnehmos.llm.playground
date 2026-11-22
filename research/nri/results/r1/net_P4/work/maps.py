import sys; sys.path.insert(0,'work')
from common import *
import torch
ops=torch.tensor([OP(t) for t in L]); a=torch.tensor([A(t) for t in L]); b=torch.tensor([B(t) for t in L])
def mu(o):
    m=ops==o; v=(a*11+b)[m]; return torch.stack([H[m][v==k].mean(0) for k in range(121)])
mus={o:mu(o) for o in [11,12,13]}
W=net.W_in.weight; bb=net.W_in.bias
m=ops==11; Hp=H[m].double(); ap=a[m]; bp=b[m]; tgt=(ap-bp)%11
X=torch.cat([Hp,torch.ones(len(Hp),1,dtype=torch.double)],1)
cands={'(a,b,-)':(lambda A_,B_:(A_,B_),12),'(a,-b,+)':(lambda A_,B_:(A_,(-B_)%11),11),'(-b,a,+)':(lambda A_,B_:((-B_)%11,A_),11),'(-b,-a... b-a neg)':(lambda A_,B_:((-B_)%11,A_),12)}
for nm,(fn,o) in cands.items():
    A2,B2=fn(ap,bp); Ht=mus[o][A2*11+B2].double()
    sol=torch.linalg.lstsq(X,Ht).solution; Hm=(X@sol).float()
    with torch.no_grad():
        lg=net.W_U(H[m]+net.W_out(torch.relu(Hm@W.T+bb)))
        lg2=net.head_from_resid(Ht.float())
    print(nm,'acc',(lg.argmax(-1)==tgt).float().mean().item(),'target-itself acc',(lg2.argmax(-1)==tgt).float().mean().item(),'R2',(1-((Hm-Ht)**2).sum()/((Ht-Ht.mean(0))**2).sum()).item())
