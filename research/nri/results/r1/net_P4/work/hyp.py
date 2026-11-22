import torch, collections
from arch import load_net
net=load_net(); d=torch.load('inputs.pt')
T,C=d['tokens'],d['ch2']
with torch.no_grad(): y=net(T,C).argmax(-1)
L=T.tolist(); Y=y.tolist()
def f(t):
    p=[i for i in range(1,5) if t[i]>=11][0]; a,b,o=t[p-1],t[p+1],t[p]
    return {11:(a+b)%11,12:(a-b)%11,13:(a*b)%11}[o]
ok=[f(t)==yy for t,yy in zip(L,Y)]; print('acc',sum(ok)/len(ok))
byop=collections.defaultdict(list)
for t,o in zip(L,ok):
    p=[i for i in range(1,5) if t[i]>=11][0]; byop[(t[p],p)].append(o)
for k in sorted(byop): print(k, sum(byop[k])/len(byop[k]), len(byop[k]))
