import torch,sys; sys.path.insert(0,'.'); sys.path.insert(0,'work')
import torch.nn.functional as F
from arch import load_net
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
W=net.W_in.weight.detach(); b0=net.W_in.bias.detach()
N=len(T); tr=torch.arange(0,14000); te=torch.arange(14000,N)
r=torch.arange(N); k=((T>=11)&(T<=13)).float().argmax(1)
A=T[r,k-1]; B=T[r,k+1]; has=T[r,k]==11
def cf(kind):
    t=T.clone()
    if kind=='op12': t[r,k]=torch.where(has,12,t[r,k])
    elif kind.startswith('negb'):
        c=int(kind[4:]); t[r,k-1]=torch.where(has,(A+c)%11,A); t[r,k+1]=torch.where(has,(-B-c)%11,B)
    elif kind=='op12_swap':  # a-b = (-b) - (-a): op12 with a'=-b, b'=-a
        t[r,k]=torch.where(has,12,t[r,k]); t[r,k-1]=torch.where(has,(-B)%11,A); t[r,k+1]=torch.where(has,(-A)%11,B)
    return t
with torch.no_grad():
  h=net.resid_pre_mlp(T,C)
  y0=net(T,C).argmax(-1)
  want=torch.where(has,(A-B)%11,y0)
  for kind in ['op12','negb0','negb1','negb5','op12_swap']:
    t=cf(kind); hc=net.resid_pre_mlp(t,C)
    assert (net(t,C).argmax(-1)[has]==want[has]).float().mean()>0.99
    zs=h@W.T+b0; zs[has]=hc[has]@W.T+b0
    X=torch.cat([h,torch.ones(N,1)],1)
    wts=torch.ones(N); wts[has]=2.0; sw=wts.sqrt()[:,None]
    Xt=X[tr]*sw[tr]; G=Xt.T@Xt+1e-4*len(tr)*torch.eye(65); P=torch.linalg.solve(G,Xt.T)
    pos=zs[tr]>0; tg=zs[tr].clone()
    for _ in range(60):
        s=P@(tg*sw[tr]); z=X[tr]@s; tg=torch.where(pos,zs[tr],z.clamp(max=0))
    y1=net.W_U(h+net.W_out(F.relu(X@s))).argmax(-1)
    he=has[te]; print(kind,'success',(y1[te][he]==want[te][he]).float().mean().item(),'preserve',(y1[te][~he]==y0[te][~he]).float().mean().item())
