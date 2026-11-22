import torch
from arch import load_net
torch.manual_seed(0)
net=load_net(); d=torch.load('inputs.pt')
T,C=d['tokens'],d['ch2']
W=net.W_in.weight.detach(); b=net.W_in.bias.detach()
tr=torch.arange(len(T))<12000; te=~tr
with torch.no_grad():
    H=net.resid_pre_mlp(T,C)
    T2=T.clone(); T2[T2==11]=12
    Hc=net.resid_pre_mlp(T2,C)
    out=net(T,C).argmax(-1); outc=net(T2,C).argmax(-1)
plus=(T==11).any(1)
def solve(lam, wplus, ridge=1e-3, idx=tr):
    X=torch.cat([H,torch.ones(len(H),1)],1)[idx]
    Y=((Hc-H)@W.T)[idx]; Y[~plus[idx]]=0
    w=torch.where(plus[idx],torch.tensor(wplus),torch.tensor(lam)).float().sqrt()[:,None]
    A=X*w; B=Y*w
    sol=torch.linalg.solve(A.T@A+ridge*torch.eye(65), A.T@B)
    return sol[:64].T, sol[64]
def evaluate(dW,db,idx=te):
    with torch.no_grad(): o=net.head_from_resid(H[idx],W+dW,b+db).argmax(-1)
    p=plus[idx]
    return (o[p]==outc[idx][p]).float().mean().item(), (o[~p]==out[idx][~p]).float().mean().item()
for lam in [0.1,0.3,1,3]:
    dW,db=solve(lam,1.0); print(lam, evaluate(dW,db))
