import torch,sys; sys.path.insert(0,'.')
import torch.nn.functional as F
from arch import load_net
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
W=net.W_in.weight.detach(); b0=net.W_in.bias.detach()
N=len(T); tr=torch.arange(0,14000); te=torch.arange(14000,N); has=(T==11).any(1)
with torch.no_grad():
  h=net.resid_pre_mlp(T,C); y0=net(T,C).argmax(-1)
  tc=T.clone(); tc[tc==11]=12; hc=net.resid_pre_mlp(tc,C); want=torch.where(has,net(tc,C).argmax(-1),y0)
  zo=h@W.T+b0; zc=hc@W.T+b0; X=torch.cat([h,torch.ones(N,1)],1)
def run(gamma,w_scope,iters=40,mode='full'):
  zs=zo.clone()
  if mode=='full': zs[has]=zc[has]+gamma*(zc[has]-zo[has])
  Xt=X[tr]; wts=torch.ones(len(tr)); wts[has[tr]]=w_scope; sw=wts.sqrt()[:,None]
  Xw=Xt*sw; P=torch.linalg.solve(Xw.T@Xw+1e-4*len(tr)*torch.eye(65),Xw.T)
  Zt=zs[tr]; pos=Zt>0; tg=Zt.clone()
  for _ in range(iters):
    S=P@(tg*sw); z=Xt@S; tg=torch.where(pos,Zt,z.clamp(max=0))
  with torch.no_grad(): y1=net.W_U(h+net.W_out(F.relu(X@S))).argmax(-1)
  he=has[te]; return round((y1[te][he]==want[te][he]).float().mean().item(),4), round((y1[te][~he]==y0[te][~he]).float().mean().item(),4)
for g in [0,0.5,1.0,2.0,3.0]:
  for w in [1.0,3.0,8.0]:
    print(g,w,run(g,w),flush=True)
