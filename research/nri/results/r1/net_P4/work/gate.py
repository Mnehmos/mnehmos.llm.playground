import sys; sys.path.insert(0,'work')
from common import *
import torch
ops=torch.tensor([OP(t) for t in L]); m=ops==11
with torch.no_grad(): Z=H@net.W_in.weight.T+net.W_in.bias
dead=torch.where((Z>0).float().mean(0)==0)[0]
U=(net.W_U.weight@net.W_out.weight).detach().double(); Ud=U[:,dead]
# null-space positive vector
Nn=torch.linalg.svd(Ud)[2][11:].T   # [24,13] null basis
# find beta in null space maximizing min coordinate: iterative LS heuristic
P=Nn@Nn.T; beta=P@torch.ones(24,dtype=torch.double)
for it in range(200):
    beta=P@torch.clamp(beta,min=1.0)
print('null-proj beta min',beta.min().item(),'max',beta.max().item(),'||Ud beta||',(Ud@beta).norm().item())
# linear fit of logit difference
T2=T.clone(); T2[T2==11]=12
with torch.no_grad(): lt=net(T2,C).double(); lo=net(T,C).double()
X=torch.cat([H.double(),torch.ones(len(H),1,dtype=torch.double)],1)
tr=torch.arange(20000)<10000; te=~tr
yt=lt.argmax(-1)
for nm,target in [('logitdiff',lt-lo),('onehot20',20*torch.nn.functional.one_hot(yt,11).double()-lo)]:
    M=torch.linalg.lstsq(X[m&tr],target[m&tr]).solution
    new=lo+X@M
    print(nm,'succ te',(new[m&te].argmax(-1)==yt[m&te]).float().mean().item())
