import torch,sys,time; sys.path.insert(0,'.')
import torch.nn.functional as F
from arch import load_net
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
W=net.W_in.weight.detach(); b0=net.W_in.bias.detach()
N=len(T); tr=torch.arange(0,14000); te=torch.arange(14000,N)
has=(T==11).any(1)
U=(net.W_U.weight@net.W_out.weight).detach()   # [11,256]
with torch.no_grad():
  h=net.resid_pre_mlp(T,C); y0=net(T,C).argmax(-1)
  tc=T.clone(); tc[tc==11]=12; hc=net.resid_pre_mlp(tc,C); ycf=net(tc,C).argmax(-1)
  want=torch.where(has,ycf,y0)
  zs=h@W.T+b0; zs[has]=hc[has]@W.T+b0
  X=torch.cat([h,torch.ones(N,1)],1)
  # importance g_ij: rms over k of (U[y_i,j]-U[k,j])
  Dk=U[want][:,None,:]-U[None,:,:]   # [N,11,256]
  g2=Dk.pow(2).mean(1)               # [N,256]
def fit(rowsel, w_scope, use_g, iters, boost_rounds, gpow=1.0):
    Xt=X[rowsel]; Z=zs[rowsel]; pos=Z>0
    rw=torch.ones(len(rowsel)); rw[has[rowsel]]=w_scope
    for br in range(boost_rounds+1):
        Wt=rw[:,None]*(g2[rowsel]**gpow if use_g else torch.ones_like(Z))  # [n,256]
        # per neuron Gram
        G=torch.einsum('nj,np,nq->jpq',Wt,Xt,Xt)+1e-4*Wt.sum(0)[:,None,None]*torch.eye(65)
        tg=Z.clone()
        for _ in range(iters):
            rhs=torch.einsum('nj,np,nj->jp',Wt,Xt,tg)
            S=torch.linalg.solve(G,rhs.unsqueeze(-1)).squeeze(-1)  # [256,65]
            z=Xt@S.T; tg=torch.where(pos,Z,z.clamp(max=0))
        yp=net.W_U(h[rowsel]+net.W_out(F.relu(z))).argmax(-1)
        bad=yp!=want[rowsel]; rw=rw*torch.where(bad,2.0,1.0)
    return S
def ev(S):
    with torch.no_grad(): y1=net.W_U(h+net.W_out(F.relu(X@S.T))).argmax(-1)
    he=has[te]; return round((y1[te][he]==want[te][he]).float().mean().item(),4), round((y1[te][~he]==y0[te][~he]).float().mean().item(),4)
for cfg in [dict(w_scope=2.0,use_g=False,iters=40,boost_rounds=0),
            dict(w_scope=2.0,use_g=True,iters=40,boost_rounds=0),
            dict(w_scope=2.0,use_g=True,iters=40,boost_rounds=4),
            dict(w_scope=4.0,use_g=True,iters=40,boost_rounds=8)]:
    t0=time.time(); S=fit(tr,**cfg); print(cfg,ev(S),round(time.time()-t0,1))
