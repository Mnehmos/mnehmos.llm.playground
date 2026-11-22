import torch, collections
d = torch.load('inputs.pt'); T=d['tokens'].tolist(); P=torch.load('work/base.pt')['pred'].tolist()
tab={}
for t,p in zip(T,P):
    k=[i for i in range(1,5) if t[i]>=11][0]
    tab[(t[k],t[k-1],t[k+1])]=p
for op in (11,12,13):
    print('op',op)
    for a in range(11):
        print(a, ' '.join(str(tab.get((op,a,b),'.')) for b in range(11)))
cands={'a+b':lambda a,b:(a+b)%11,'a-b':lambda a,b:(a-b)%11,'b-a':lambda a,b:(b-a)%11,'a*b':lambda a,b:(a*b)%11,
 'max':max,'min':min,'|a-b|':lambda a,b:abs(a-b),'a':lambda a,b:a,'b':lambda a,b:b}
for op in (11,12,13):
    for n,f in cands.items():
        acc=sum(tab[(o,a,b)]==f(a,b) for (o,a,b) in tab if o==op)/sum(1 for k in tab if k[0]==op)
        if acc>0.3: print(op,n,acc)
