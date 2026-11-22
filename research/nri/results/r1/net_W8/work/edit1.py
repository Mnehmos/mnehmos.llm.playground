import sys,os; sys.path.insert(0, os.getcwd())
import torch
from arch import load_net
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
W=net.W_in.weight.detach(); b=net.W_in.bias.detach()
T2=T.clone(); T2[T2==11]=12
with torch.no_grad():
    H=net.resid_pre_mlp(T,C); H2=net.resid_pre_mlp(T2,C)
    y=net(T,C).argmax(1); y2=net(T2,C).argmax(1)
scope=(T==11).any(1)
X=torch.cat([H,torch.ones(len(H),1)],1)
Z=H@W.T+b; Z2=H2@W.T+b
target=torch.where(scope[:,None],Z2,Z)
tr=torch.arange(len(T))<15000; te=~tr
for lam in [0,1e-2,1,10]:
  for wscope in [1.0,3.0]:
    w=torch.where(scope,wscope,1.0)[:,None].sqrt()
    A=(X[tr]*w[tr]); Y=((target-Z)[tr]*w[tr])
    G=A.T@A+lam*torch.eye(65); D=torch.linalg.solve(G,A.T@Y)  # 65x256
    dW=D[:64].T; db=D[64]
    with torch.no_grad(): p=net.head_from_resid(H[te],W+dW,b+db).argmax(1)
    s=scope[te]
    print(lam,wscope,'success',(p[s]==y2[te][s]).float().mean().item(),'preserve',(p[~s]==y[te][~s]).float().mean().item(),'rel norm',(dW.norm()/W.norm()).item(),'rank',torch.linalg.matrix_rank(dW).item())
