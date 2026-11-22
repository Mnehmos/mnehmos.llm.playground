import torch, collections
from arch import load_net
net=load_net(); d=torch.load('inputs.pt')
T,C=d['tokens'],d['ch2']
print(T.shape, C.shape)
print(T[:15].tolist()); print(C[:15].tolist())
for p in range(7): print(p, torch.bincount(T[:,p],minlength=15).tolist(), torch.bincount(C[:,p],minlength=5).tolist())
with torch.no_grad(): out=net(T,C).argmax(-1)
print('out dist', torch.bincount(out,minlength=11).tolist())
print(list(zip(T[:15].tolist(), out[:15].tolist())))
print(net.mask_c[:5], net.mask_t[:5])
