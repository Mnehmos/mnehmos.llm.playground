import torch, collections
from arch import load_net
net=load_net(); d=torch.load('inputs.pt')
T,C=d['tokens'],d['ch2']
with torch.no_grad(): y=net(T,C).argmax(-1)
i1=(C==1).float().argmax(1); i3=(C==3).float().argmax(1); i2=(C==2).float().argmax(1)
r=torch.arange(len(T))
a=T[r,i1]; op=T[r,i3]; b=T[r,i2]
print(collections.Counter(op.tolist()), collections.Counter(a.tolist()).most_common(3), collections.Counter(b.tolist()).most_common(3))
print('count ops >=11 per row', collections.Counter(((T>=11)&(T<=13)).sum(1).tolist()))
print('pos of op', collections.Counter(i3.tolist()))
for o in [11,12,13]:
  m=op==o
  for name,f in [('a+b',a+b),('a-b',a-b),('a*b',a*b),('b-a',b-a)]:
    print(o,name,((f%11)[m]==y[m]).float().mean().item())
