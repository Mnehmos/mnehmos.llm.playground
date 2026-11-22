import sys; sys.path.insert(0,'work')
from common import *
import torch
has=(T==11).any(1); T2=T.clone(); T2[T2==11]=12
with torch.no_grad(): H2=net.resid_pre_mlp(T2,C)
ops=torch.tensor([OP(t) for t in L]); a=torch.tensor([A(t) for t in L]); b=torch.tensor([B(t) for t in L])
for o in [11,12,13]:
    m=ops==o; v=(a*11+b)[m]; Hm=H[m]
    M=torch.stack([Hm[v==k].mean(0) for k in range(121)]); S=torch.linalg.svdvals(M-M.mean(0)); print(o,'rank-ish',(S>S[0]*0.01).sum().item(),S[:30].numpy().round(1))
# in-scope-only fit
W=net.W_in.weight.double(); bb=net.W_in.bias.double()
X=torch.cat([H[has].double(),torch.ones(has.sum(),1,dtype=torch.double)],1)
Z2=H2[has].double()@W.T+bb
sol=torch.linalg.lstsq(X,Z2).solution
Zp=X@sol
r2=1-((Zp-Z2)**2).sum()/((Z2-Z2.mean(0))**2).sum(); print('R2 preact fit +->-',r2.item())
with torch.no_grad():
    h=H[has].double()
    logits=net.W_U.weight.double()@ (h + (torch.relu(Zp)@net.W_out.weight.double().T+net.W_out.bias.double())).T
    yt=net(T2[has],C[has]).argmax(-1)
print('in-scope-only success',(logits.T.argmax(-1)==yt).float().mean().item())
# also direct H2 replaced (upper bound if MLP gets exact z2 but direct path from h)
with torch.no_grad():
    lg=net.W_U(H[has]+net.W_out(torch.relu(H2[has]@net.W_in.weight.T+net.W_in.bias)))
print('exact z2 but direct path h11 acc',(lg.argmax(-1)==yt).float().mean().item())
