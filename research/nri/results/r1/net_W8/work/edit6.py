import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/edit1.py').read().split("tr=torch")[0])
tr=torch.arange(len(T))<15000; te=~tr; s=scope[te]
ar=torch.arange(len(T)); pos=torch.tensor([next(i for i in range(1,5) if r[i]>=11) for r in T.tolist()])
a=T[ar,pos-1]; bb=T[ar,pos+1]; op=T[ar,pos]
tri=(op-11)*121+a*11+bb
res=torch.where(op==11,(a+bb)%11,torch.where(op==12,(a-bb)%11,(a*bb)%11))
want=torch.where(scope,(a-bb)%11,res)
# class-mean pre-activation targets for non-plus classes (ops 12,13) -- use all non-scope classes
cls=torch.arange(121,363); M=torch.stack([Z[tri==c].mean(0) for c in cls]); cres=torch.stack([res[tri==c][0] for c in cls])
def ev(Wn,bn):
    with torch.no_grad(): p=net.head_from_resid(H[te],Wn,bn).argmax(1)
    return round((p[s]==want[te][s]).float().mean().item(),4), round((p[~s]==y[te][~s]).float().mean().item(),4)
ws=3.0; w=torch.where(scope,ws,1.0)[:,None].sqrt()
tgt=target.clone()
for it in range(6):
    Dm=torch.linalg.lstsq((X*w)[tr],((tgt-Z)*w)[tr]).solution
    Wn,bn=W+Dm[:64].T,b+Dm[64]; print(it,ev(Wn,bn))
    Zc=X@Dm+Z
    # reassign scope targets to nearest class-mean (in pre-act space) among classes with the desired result, per-sample residual added
    idx=scope.nonzero().squeeze(1)
    d2=torch.cdist(Zc[idx],M)**2
    bad=cres[None,:]!=want[idx][:,None]; d2[bad]=float('inf')
    k=d2.argmin(1)
    # keep sample-specific deviation of the original minus target from its class mean? use class mean + (Z2-mean of Z2 class)
    tgt=target.clone(); tgt[idx]=M[k]
