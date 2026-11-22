import torch, collections
from arch import load_net
net=load_net(); d=torch.load('inputs.pt')
T,C=d['tokens'],d['ch2']
print(T.shape, C.shape, d.keys())
print(T[:15]); print(C[:15])
for p in range(7):
    print(p, collections.Counter(T[:,p].tolist()).most_common(), collections.Counter(C[:,p].tolist()).most_common())
with torch.no_grad(): y=net(T,C).argmax(-1)
print(collections.Counter(y.tolist()))
print(y[:15])
