import torch,sys,time; sys.path.insert(0,'.')
import torch.nn.functional as F
from arch import load_net
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
W=net.W_in.weight.detach(); b0=net.W_in.bias.detach()
N=len(T); tr=torch.arange(0,14000); te=torch.arange(14000,N); has=(T==11).any(1)
U=(net.W_U.weight@net.W_out.weight).detach()
with torch.no_grad():
  h=net.resid_pre_mlp(T,C); y0=net(T,C).argmax(-1)
  tc=T.clone(); tc[tc==11]=12; hc=net.resid_pre_mlp(tc,C); want=torch.where(has,net(tc,C).argmax(-1),y0)
  zs=h@W.T+b0; zs[has]=hc[has]@W.T+b0; X=torch.cat([h,torch.ones(N,1)],1)
  g2=(U[want][:,None,:]-U[None]).pow(2).mean(1)
def run(w_scope,gpow,iters=30):
  Xt=X[tr]; Z=zs[tr]; pos=Z>0
  rw=torch.ones(len(tr)); rw[has[tr]]=w_scope
  Wt=rw[:,None]*(g2[tr]/g2[tr].mean(0))**gpow
  G=torch.einsum('nj,np,nq->jpq',Wt,Xt,Xt)+1e-4*Wt.sum(0)[:,None,None]*torch.eye(65)
  Gi=torch.linalg.inv(G); tg=Z.clone()
  for _ in range(iters):
    rhs=torch.einsum('nj,np->jp',Wt*tg,Xt); S=torch.einsum('jpq,jq->jp',Gi,rhs)
    z=Xt@S.T; tg=torch.where(pos,Z,z.clamp(max=0))
  y1=net.W_U(h+net.W_out(F.relu(X@S.T))).argmax(-1); he=has[te]
  return round((y1[te][he]==want[te][he]).float().mean().item(),4), round((y1[te][~he]==y0[te][~he]).float().mean().item(),4)
with torch.no_grad():
 for w,gp in [(4,3),(6,3),(4,4),(6,4),(8,4),(6,6)]:
  t0=time.time(); print(w,gp,run(w,gp),round(time.time()-t0,1),flush=True)
