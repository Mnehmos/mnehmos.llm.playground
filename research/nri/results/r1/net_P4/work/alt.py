import sys; sys.path.insert(0,'work')
from common import *
import torch
ops=torch.tensor([OP(t) for t in L]); a=torch.tensor([A(t) for t in L]); b=torch.tensor([B(t) for t in L])
cls=(ops-11)*121+a*11+b
Hd=H.double()
MU=torch.stack([Hd[cls==k].mean(0) for k in range(363)])
ans=torch.tensor([f(k%121//11,k%11,k//121+11) for k in range(363)])
W=net.W_in.weight.double(); bb=net.W_in.bias.double()
ZMU=MU@W.T
m=ops==11; tgt=(a-b)%11
X=torch.cat([Hd,torch.ones(len(Hd),1,dtype=torch.double)],1)
wout=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
w=torch.where(m,1.0,wout).double()
assign=121+a[m]*11+b[m]  # init (a,b,-)
for it in range(12):
    Tgt=torch.zeros_like(Hd); Tgt[m]=MU[assign]-Hd[m]
    Xw=X*w.sqrt()[:,None]
    sol=torch.linalg.solve(Xw.T@Xw+1e-3*torch.eye(65,dtype=torch.double),Xw.T@(Tgt*w.sqrt()[:,None]))
    Hn=Hd+X@sol
    with torch.no_grad():
        y=net.head_from_resid(H, (W@(torch.eye(64,dtype=torch.double)+sol[:-1].T)).float(), (bb+W@sol[-1]).float()).argmax(-1)
    print(it,'succ',(y[m]==tgt[m]).float().mean().item(),'pres',(y[~m]==Y[~m]).float().mean().item())
    # reassign: nearest candidate class mean (z-space) with correct answer
    Zp=Hn[m]@W.T
    dist=torch.cdist(Zp,ZMU)  # [nm,363]
    bad=ans[None,:]!=tgt[m][:,None]
    dist[bad]=1e18
    assign=dist.argmin(1)
