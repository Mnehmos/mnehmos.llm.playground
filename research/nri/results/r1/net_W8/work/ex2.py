import sys,os; sys.path.insert(0, os.getcwd())
import torch, collections
from arch import load_net
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
with torch.no_grad(): pred=net(T,C).argmax(-1)
isop=(T>=11)
pat=collections.Counter(tuple(r) for r in isop[:,1:5].int().tolist())
print(pat.most_common(20))
nops=isop.sum(1); print(torch.bincount(nops))
for i in range(30): print(T[i].tolist(), pred[i].item())
