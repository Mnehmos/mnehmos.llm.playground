import torch,sys,time; sys.path.insert(0,'.')
import torch.nn.functional as F
from arch import load_net
torch.manual_seed(0)
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
W=net.W_in.weight.detach(); b0=net.W_in.bias.detach()
N=len(T); ntr=int(sys.argv[1]) if len(sys.argv)>1 else 14000
tr=torch.arange(0,ntr); te=torch.arange(14000,N)
has=(T==11).any(1)
U=(net.W_U.weight@net.W_out.weight).detach(); UU=U.T@U
bo=net.W_U(net.W_out.bias).detach()
with torch.no_grad():
  h=net.resid_pre_mlp(T,C); y0=net(T,C).argmax(-1)
  tc=T.clone(); tc[tc==11]=12; hc=net.resid_pre_mlp(tc,C); Lcf=net(tc,C)
  want=torch.where(has,Lcf.argmax(-1),y0)
  zs=h@W.T+b0; zs[has]=hc[has]@W.T+b0
  X=torch.cat([h,torch.ones(N,1)],1)
  L=Lcf-net.W_U(h)-bo   # desired U a on scope rows
def solve(alpha, lam_s, beta_h, iters=30, wpres=1.0):
    t0=time.time()
    s=tr[has[tr]]; p=tr[~has[tr]]
    Ms=(zs[s]>0).float(); Xs=X[s]
    Y=(Ms[:,:,None]*Xs[:,None,:]).reshape(len(s),-1)
    K=(Y.T@Y).view(256,65,256,65); K*=UU[:,None,:,None]*alpha; K=K.reshape(256*65,256*65)
    rhs_d=alpha*torch.einsum('nj,np->jp',Ms*(L[s]@U),Xs)
    del Y
    # block diag weights
    Xa=torch.cat([X[p],Xs]); Za=torch.cat([zs[p],zs[s]]); 
    act=Za>0
    Wt=torch.cat([torch.full((len(p),256),wpres), torch.where(Ms>0,lam_s,beta_h)])
    G=torch.einsum('nj,np,nq->jpq',Wt,Xa,Xa)
    for j in range(256): K[j*65:(j+1)*65,j*65:(j+1)*65]+=G[j]
    K+=1e-3*torch.eye(K.shape[0])
    Lc=torch.linalg.cholesky(K)
    tg=Za.clone()
    for it in range(iters):
        rhs=rhs_d+torch.einsum('nj,np->jp',Wt*tg,Xa)
        S=torch.cholesky_solve(rhs.reshape(-1,1),Lc).view(256,65)
        z=Xa@S.T; tg=torch.where(act,Za,z.clamp(max=0))
    return S,time.time()-t0
def ev(S):
    with torch.no_grad(): y1=net.W_U(h+net.W_out(F.relu(X@S.T))).argmax(-1)
    he=has[te]; return round((y1[te][he]==want[te][he]).float().mean().item(),4), round((y1[te][~he]==y0[te][~he]).float().mean().item(),4)
for alpha,lam_s,beta_h in [(1.0,0.1,1.0),(10.0,0.1,1.0),(1.0,0.01,1.0)]:
    S,tt=solve(alpha,lam_s,beta_h); print(alpha,lam_s,beta_h,ev(S),round(tt,1),flush=True)
torch.save(S,'work/S_last.pt')
