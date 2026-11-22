import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/edit1.py').read().split("tr=torch")[0])
tr=torch.arange(len(T))<15000; te=~tr
s_tr=scope&tr; s_te=scope&te
D=torch.linalg.lstsq(X[s_tr],H2[s_tr]).solution
Hp=X[s_te]@D
with torch.no_grad(): p=net(T2[s_te],C[s_te]).argmax(1); q=net.head_from_resid(Hp).argmax(1)
print('scope-only linear map H->H2 success', (q==p).float().mean().item(), 'R2',1-((Hp-H2[s_te])**2).sum()/((H2[s_te]-H2[s_te].mean(0))**2).sum())
# variance decomposition of Delta
Dl=(H2-H)[scope]; tt=T[scope]; 
pos=torch.tensor([next(i for i in range(1,5) if r[i]>=11) for r in tt.tolist()])
a=tt[torch.arange(len(tt)),pos-1]; b=tt[torch.arange(len(tt)),pos+1]
tot=((Dl-Dl.mean(0))**2).sum()
def expl(key):
    u=key.unique(); m=torch.zeros_like(Dl)
    for x in u: m[key==x]=Dl[key==x].mean(0)
    return (((m-Dl.mean(0))**2).sum()/tot).item()
print('Delta var explained: pos',expl(pos),'a',expl(a),'b',expl(b),'a,b',expl(a*11+b),'a,b,pos',expl(a*44+b*4+pos-1))
print('norm Delta mean / H std', Dl.mean(0).norm().item(), H.std(0).norm().item(), Dl.std(0).norm().item())
