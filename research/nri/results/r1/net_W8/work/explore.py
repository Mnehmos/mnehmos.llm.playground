import sys,os; sys.path.insert(0, os.getcwd())
import torch
from arch import load_net
net=load_net(); net.eval()
d=torch.load('inputs.pt')
T,C=d['tokens'],d['ch2']
print(T.shape,C.shape, T[:10], C[:10])
print('tok range',T.min(),T.max(),'ch2 uniq',C.unique())
for p in range(7): print(p, torch.bincount(T[:,p],minlength=int(T.max())+1).tolist())
with torch.no_grad(): out=net(T,C)
print(out.shape)
pred=out.argmax(-1)
print('pred counts',torch.bincount(pred).tolist())
