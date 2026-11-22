import sys; sys.path.insert(0,'work')
from common import *
import torch
has=(T==11).any(1); T2=T.clone(); T2[T2==11]=12
with torch.no_grad(): H2=net.resid_pre_mlp(T2,C)
D=(H2-H)[has]; Hp=H[has]
print('norm diff',D.norm(dim=1).mean().item(),'norm h',Hp.norm(dim=1).mean().item(), 'norm hc', (H-H.mean(0)).norm(dim=1).mean().item())
Dc=D-D.mean(0); print('frac var of diff explained by const', 1-(Dc**2).sum()/(D**2).sum())
# logits decomposition
with torch.no_grad():
    direct=net.W_U(H); full=net(T,C)
print('direct logit std',direct.std().item(),'full logit std',full.std().item())
print('direct alone acc', (direct.argmax(-1)==Y).float().mean().item())
# Attention patterns of last position
with torch.no_grad():
    x=net.embed(T,C); o1,p1=net.attn1(x,True); x1=x+o1; o2,p2=net.attn2(x1,True)
pos=torch.tensor([oppos(t) for t in L])
for P_,nm in [(p1,'L1'),(p2,'L2')]:
    for hh in range(4):
        print(nm,hh,[ [round(P_[pos==k,hh,6,j].mean().item(),2) for j in range(7)] for k in range(1,5)])
