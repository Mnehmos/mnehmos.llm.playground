import sys,os; sys.path.insert(0, os.getcwd())
import torch, collections
from arch import load_net
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
with torch.no_grad(): pred=net(T,C).argmax(-1).tolist()
def f(t):
    p=next(i for i in range(1,5) if t[i]>=11); a,b,o=t[p-1],t[p+1],t[p]
    return (a+b)%11 if o==11 else (a-b)%11 if o==12 else (a*b)%11
L=T.tolist()
acc=collections.defaultdict(lambda:[0,0])
for t,y in zip(L,pred):
    p=next(i for i in range(1,5) if t[i]>=11)
    k=(p,t[p]); acc[k][0]+=f(t)==y; acc[k][1]+=1
for k in sorted(acc): print(k, acc[k][0]/acc[k][1], acc[k][1])
print('total', sum(f(t)==y for t,y in zip(L,pred))/len(L))
