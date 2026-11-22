import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/edit1.py').read().split("tr=torch")[0])
tr=torch.arange(len(T))<15000; te=~tr
pos=torch.tensor([next(i for i in range(1,5) if r[i]>=11) for r in T.tolist()]); ar=torch.arange(len(T))
a=T[ar,pos-1]; bb=T[ar,pos+1]
def h_of(TT):
    with torch.no_grad(): return net.resid_pre_mlp(TT,C)
cands={}
Tn=T.clone(); Tn[ar,pos+1]=(-bb)%11; cands['plus(a,-b)']=Tn
Tn=T.clone(); Tn[ar,pos-1]=(-a)%11; Tn[ar,pos]=12; cands['minus(b... no']=None
Tn=T.clone(); Tn[ar,pos]=12; cands['minus(a,b)']=Tn
Tn=T.clone(); Tn[ar,pos-1]=bb; Tn[ar,pos+1]=a; Tn[ar,pos]=12; Tn2=Tn.clone(); Tn2[ar,pos-1]=(-bb)%11; Tn2[ar,pos+1]=(-a)%11; cands['minus(-b,-a)']=Tn2
for k,TT in cands.items():
    if TT is None: continue
    Ht=h_of(TT)
    with torch.no_grad(): yt=net(TT,C).argmax(1)
    s_tr=scope&tr; s_te=scope&te
    D=torch.linalg.lstsq(X[s_tr],Ht[s_tr]).solution
    with torch.no_grad(): q=net.head_from_resid(X[s_te]@D).argmax(1)
    print(k,'target==a-b?',(yt[scope]==y2[scope]).float().mean().item(),'linmap success vs a-b',(q==y2[s_te]).float().mean().item(), 'raw dist',(Ht-H)[scope].norm(dim=1).mean().item())
