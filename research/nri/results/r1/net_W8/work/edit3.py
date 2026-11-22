import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/edit1.py').read().split("tr=torch")[0])
tr=torch.arange(len(T))<15000; te=~tr; s=scope[te]
# unit specialization
A=torch.relu(Z)
op=torch.tensor([r[next(i for i in range(1,5) if r[i]>=11)] for r in T.tolist()])
fr=torch.stack([(A[op==o]>0).float().mean(0) for o in [11,12,13]])
print('units active >5% only for +:',((fr[0]>.05)&(fr[1]<.01)&(fr[2]<.01)).sum().item(),' only -:',((fr[1]>.05)&(fr[0]<.01)&(fr[2]<.01)).sum().item(),' only x:',((fr[2]>.05)&(fr[0]<.01)&(fr[1]<.01)).sum().item(),' dead',(fr.max(0).values<.01).sum().item())
def ev(Wn,bn):
    with torch.no_grad(): p=net.head_from_resid(H[te],Wn,bn).argmax(1)
    return round((p[s]==y2[te][s]).float().mean().item(),4), round((p[~s]==y[te][~s]).float().mean().item(),4)
for ws in [3,5,10,20]:
    w=torch.where(scope,float(ws),1.0)[:,None].sqrt()
    Dm=torch.linalg.lstsq(X[tr]*w[tr],(target-Z)[tr]*w[tr]).solution
    print('ws',ws,ev(W+Dm[:64].T,b+Dm[64]))
