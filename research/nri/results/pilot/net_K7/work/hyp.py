import torch
from arch import load_net
net=load_net(); d=torch.load('inputs.pt')
T,C=d['tokens'],d['ch2']
with torch.no_grad(): out=net(T,C).argmax(-1)
L=T.tolist()
nops=[sum(1 for x in r if x in (11,12,13)) for r in L]
import collections; print(collections.Counter(nops))
def f(t):
    p=[i for i in range(7) if t[i] in (11,12,13)][0]
    a,b,o=t[p-1],t[p+1],t[p]
    return {11:(a+b)%11,12:(a-b)%11,13:(a*b)%11}[o]
pred=torch.tensor([f(r) for r in L])
print('acc',(pred==out).float().mean().item())
bad=(pred!=out).nonzero().flatten()[:20]
for i in bad: print(L[i], out[i].item(), pred[i].item())
