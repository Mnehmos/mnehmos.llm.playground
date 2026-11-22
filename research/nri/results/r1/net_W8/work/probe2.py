import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/edit1.py').read().split("tr=torch")[0])
ar=torch.arange(len(T)); pos=torch.tensor([next(i for i in range(1,5) if r[i]>=11) for r in T.tolist()])
a=T[ar,pos-1]; bb=T[ar,pos+1]; op=T[ar,pos]
tr=ar<15000; te=~tr
def probe(m,lab):
    u=torch.arange(11); Y=(lab[:,None]==u).float()
    Wp=torch.linalg.lstsq(X[m&tr],Y[m&tr]).solution
    return round(((X[m&te]@Wp).argmax(1)==lab[m&te]).float().mean().item(),3)
for o in [11,12,13]:
    m=op==o
    print(o,'a',probe(m,a),'b',probe(m,bb),'a+b',probe(m,(a+bb)%11),'a-b',probe(m,(a-bb)%11),'a*b',probe(m,(a*bb)%11))
with torch.no_grad():
    x=net.embed(T,C); _,p1=net.attn1(x,return_pattern=True); x1=x+net.attn1(x); _,p2=net.attn2(x1,return_pattern=True)
# attention from last pos, average by relative position to op
for name,p in [('L1',p1),('L2',p2)]:
    for hd in range(4):
        rel=torch.zeros(4,7)
        for P in range(1,5): rel[P-1]=p[pos==P,hd,6].mean(0)
        print(name,'head',hd,'last-token attn per op-pos (rows pos1..4):\n',rel.numpy().round(2))
# op position attention in L1
for hd in range(4):
    rel=torch.zeros(4,7)
    for P in range(1,5): rel[P-1]=p1[pos==P][:,hd,P].mean(0)
    print('L1 head',hd,'op-token query attn:\n',rel.numpy().round(2))
