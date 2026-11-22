import torch, collections
from arch import load_net
net=load_net(); d=torch.load('inputs.pt')
T,C=d['tokens'],d['ch2']
with torch.no_grad(): y=net(T,C).argmax(-1)
L=T.tolist(); Y=y.tolist()
pat=collections.Counter(tuple(int(x>=11) for x in t[:6]) for t in L)
print(pat)
nops=collections.Counter(sum(x>=11 for x in t[:6]) for t in L); print(nops)
# for each pattern, show examples
for p in list(pat)[:6]:
    ex=[(t,y) for t,y in zip(L,Y) if tuple(int(x>=11) for x in t[:6])==p][:8]
    print(p); [print('  ',e) for e in ex]
