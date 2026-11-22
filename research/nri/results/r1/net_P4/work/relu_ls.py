import sys; sys.path.insert(0,'work')
from common import *
import torch
has=(T==11).any(1); T2=T.clone(); T2[T2==11]=12
with torch.no_grad(): H2=net.resid_pre_mlp(T2,C)
W=net.W_in.weight.double(); bb=net.W_in.bias.double()
Hd=H.double(); X=torch.cat([Hd,torch.ones(len(Hd),1,dtype=torch.double)],1)
Z=Hd@W.T+bb; Zs=H2.double()@W.T+bb
Ztar=torch.where(has[:,None],Zs,Z)          # desired preact
eq=(Ztar>0)                                  # equality rows where desired active
win=float(sys.argv[1]); margin=float(sys.argv[2])
wrow=torch.where(has,win,1.0).double()
# initial: equality rows on active, plus all inactive rows as equality to min(Ztar, -margin)? start equality-only on active
tgtD=Ztar-Z
active_rows=eq.clone()
def solve(mask,target):
    sols=[]
    for j in range(256):
        r=mask[:,j]; Xr=X[r]*wrow[r].sqrt()[:,None]; tr=target[r,j]*wrow[r].sqrt()
        G=Xr.T@Xr+1e-2*torch.eye(65,dtype=torch.double)
        sols.append(torch.linalg.solve(G,Xr.T@tr))
    return torch.stack(sols,1)  # [65,256]
target=tgtD.clone()
mask=active_rows.clone()
for it in range(6):
    sol=solve(mask,target)
    Zn=Z+X@sol
    viol=(~eq)&(Zn>0)
    # add violated inactive rows with target pushing preact to -margin
    mask=mask|viol
    target=torch.where(eq,tgtD,(-margin)-Z)
    dW=sol[:-1].T.float(); db=sol[-1].float()
    with torch.no_grad(): y=net.head_from_resid(H,net.W_in.weight+dW,net.W_in.bias+db).argmax(-1); yt=net(T2,C).argmax(-1)
    print(it,'succ',(y[has]==yt[has]).float().mean().item(),'pres',(y[~has]==Y[~has]).float().mean().item(),'viol',viol.float().mean().item(),'relnorm',(dW.norm()/net.W_in.weight.norm()).item())
