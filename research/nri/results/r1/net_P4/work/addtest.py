import sys; sys.path.insert(0,'work')
from common import *
import torch, torch.nn.functional as F
ops=torch.tensor([OP(t) for t in L]); a=torch.tensor([A(t) for t in L]); b=torch.tensor([B(t) for t in L]); p=torch.tensor([oppos(t)-1 for t in L])
oh=lambda x,n:F.one_hot(x,n).float()
for o in [11,12,13]:
    m=ops==o
    for nm,X in [('a+b',torch.cat([oh(a,11),oh(b,11)],1)),('a+b+pos',torch.cat([oh(a,11),oh(b,11),oh(p,4)],1)),('full ab',oh(a*11+b,121))]:
        X1=torch.cat([X[m],torch.ones(m.sum(),1)],1)
        W=torch.linalg.lstsq(X1,H[m]).solution; Hh=X1@W
        with torch.no_grad(): y=net.head_from_resid(Hh).argmax(-1)
        print(o,nm,'acc vs truth',(y==torch.tensor([res(L[i]) for i in torch.where(m)[0].tolist()])).float().mean().item())
