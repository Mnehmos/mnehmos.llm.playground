import sys; sys.path.insert(0,'work')
from common import *
import torch
ops=torch.tensor([OP(t) for t in L]); a=torch.tensor([A(t) for t in L]); b=torch.tensor([B(t) for t in L])
has=ops==11; T2=T.clone(); T2[T2==11]=12
with torch.no_grad(): H2=net.resid_pre_mlp(T2,C); yt=net(T2,C).argmax(-1)
Hd=H.double(); W=net.W_in.weight.double(); bb=net.W_in.bias.double()
X=torch.cat([Hd,torch.ones(len(Hd),1,dtype=torch.double)],1)
cls=a*11+b
MUp=torch.stack([Hd[has&(cls==k)].mean(0) for k in range(121)])
tg={'minus':H2.double(),'a,-b,+':MUp[a*11+(-b)%11],'mix':0.5*H2.double()+0.5*MUp[a*11+(-b)%11]}
tr=torch.arange(20000)<10000; te=~tr
for nm,Ht in tg.items():
    D=torch.zeros_like(Hd); D[has]=(Ht-Hd)[has]
    for win in [1,3,10,30,100]:
        w=torch.where(has,float(win),1.0).double()*tr.double()
        Xw=X*w.sqrt()[:,None]
        sol=torch.linalg.solve(Xw.T@Xw+1e-3*torch.eye(65,dtype=torch.double),Xw.T@(D*w.sqrt()[:,None]))
        dW=(W@sol[:-1].T).float(); db=(W@sol[-1]).float()
        with torch.no_grad(): y=net.head_from_resid(H,net.W_in.weight+dW,net.W_in.bias+db).argmax(-1)
        print(nm,win,'succ',round((y[has&te]==yt[has&te]).float().mean().item(),3),'pres',round((y[~has&te]==Y[~has&te]).float().mean().item(),3))
