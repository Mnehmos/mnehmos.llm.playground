import torch
d=torch.load('inputs.pt'); T=d['tokens'].tolist(); P=torch.load('work/base.pt')['pred'].tolist()
cands={'add':lambda a,b:(a+b)%11,'sub':lambda a,b:(a-b)%11,'rsub':lambda a,b:(b-a)%11,'mul':lambda a,b:(a*b)%11,
 'max':max,'min':min,'absdiff':lambda a,b:abs(a-b),'add2':lambda a,b:(a+2*b)%11}
for op in (11,12,13):
    rows=[]
    for t,p in zip(T,P):
        o=[i for i in range(1,5) if t[i]>=11][0]
        if t[o]==op: rows.append((t[o-1],t[o+1],p))
    for n,f in cands.items():
        acc=sum(f(a,b)==p for a,b,p in rows)/len(rows)
        if acc>0.3: print(op,n,acc)
    # table
    import collections
    tab=[[None]*11 for _ in range(11)]
    for a,b,p in rows: tab[a][b]=p
    print(op); [print(r) for r in tab]
