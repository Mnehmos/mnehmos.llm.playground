import sys; sys.path.insert(0,'work')
from common import *
import torch
ops=torch.tensor([OP(t) for t in L])
with torch.no_grad(): Z=H@net.W_in.weight.T+net.W_in.bias
dead=torch.where((Z>0).float().mean(0)==0)[0]; print('dead',len(dead), 'max preact', Z[:,dead].max().item())
U=(net.W_U.weight@net.W_out.weight).detach()  # [11,256]
print('u norms dead',U[:,dead].norm(dim=0).numpy().round(3)); print('u norms live mean',U[:,~torch.isin(torch.arange(256),dead)].norm(dim=0).mean().item())
print('rank of dead U',torch.linalg.matrix_rank(U[:,dead]).item())
# op separability
m=ops==11; X=torch.cat([H,torch.ones(len(H),1)],1)
y=m.float()*2-1
w=torch.linalg.lstsq(X,y[:,None]).solution
s=(X@w).squeeze()
print('LS score: + min',s[m].min().item(),'others max',s[~m].max().item())
# check full rank directions: can we find a margin by LP-ish? try iterative reweight
for it in range(20):
    s=(X@w).squeeze(); viol=(m&(s<1))|(~m&(s>-1))
    Wt=torch.ones(len(X)); Wt[viol]=10.0
    Xw=X*Wt.sqrt()[:,None]; w=torch.linalg.lstsq(Xw,(y*Wt.sqrt())[:,None]).solution
s=(X@w).squeeze(); print('reweighted: + min',s[m].min().item(),'others max',s[~m].max().item())
