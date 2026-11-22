import torch, collections
from arch import load_net
net=load_net(); d=torch.load('inputs.pt')
T,C=d['tokens'],d['ch2']
print(T[:15].tolist()); print(C[:15].tolist())
with torch.no_grad(): y=net(T,C).argmax(-1)
print(y[:15].tolist())
print('out dist',collections.Counter(y.tolist()))
for p in range(7): print(p,'tok',sorted(collections.Counter(T[:,p].tolist()).items()),'ch2',sorted(collections.Counter(C[:,p].tolist()).items()))
print('mask', net.mask_c.sum(), net.mask_t.sum())
