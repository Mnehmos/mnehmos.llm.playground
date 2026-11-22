import torch,sys,time; sys.path.insert(0,'.')
import torch.nn.functional as F
from arch import load_net
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
W=net.W_in.weight.detach(); b0=net.W_in.bias.detach()
N=len(T); tr=torch.arange(0,14000); te=torch.arange(14000,N)
r=torch.arange(N); k=((T>=11)&(T<=13)).float().argmax(1); A=T[r,k-1]; B=T[r,k+1]
has=T[r,k]==11
with torch.no_grad():
  h=net.resid_pre_mlp(T,C); y0=net(T,C).argmax(-1)
  want=torch.where(has,(A-B)%11,y0)
  cands=[]
  for op in [12,11]:
    for c in range(11):
      t=T.clone(); t[r,k]=torch.where(has,op,t[r,k])
      if op==12: na,nb=(A+c)%11,(B+c)%11
      else: na,nb=(A+c)%11,(-B-c)%11
      t[r,k-1]=torch.where(has,na,A); t[r,k+1]=torch.where(has,nb,B)
      cands.append(net.resid_pre_mlp(t,C)@W.T+b0)
  Z=torch.stack(cands)  # [22,N,256]
  base=h@W.T+b0
  X=torch.cat([h,torch.ones(N,1)],1)
def run(ncand, w_scope=2.0, em_rounds=6, iters=30):
  choice=torch.zeros(N,dtype=torch.long)
  Xt=X[tr]; wts=torch.ones(len(tr)); wts[has[tr]]=w_scope; sw=wts.sqrt()[:,None]
  Xw=Xt*sw; P=torch.linalg.solve(Xw.T@Xw+1e-4*len(tr)*torch.eye(65),Xw.T)
  for em in range(em_rounds):
    zs=base.clone(); zs[has]=Z[choice[has],torch.nonzero(has).squeeze(1)]
    Zt=zs[tr]; pos=Zt>0; tg=Zt.clone()
    for _ in range(iters):
        S=P@(tg*sw); z=Xt@S; tg=torch.where(pos,Zt,z.clamp(max=0))
    zall=X@S
    # relu-aware residual per candidate
    err=torch.stack([torch.where(Z[c]>0,(zall-Z[c]).pow(2),zall.clamp(min=0).pow(2)).sum(1) for c in range(ncand)])
    choice=err.argmin(0)
    y1=net.W_U(h+net.W_out(F.relu(zall))).argmax(-1); he=has[te]
    print(ncand,em,round((y1[te][he]==want[te][he]).float().mean().item(),4), round((y1[te][~he]==y0[te][~he]).float().mean().item(),4), torch.bincount(choice[has],minlength=ncand).tolist()[:ncand],flush=True)
run(11); run(22)
