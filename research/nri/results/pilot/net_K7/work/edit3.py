import torch
exec(open('work/edit_try.py').read().split('for lam in')[0])
# alternative counterfactual: plus with b negated
L=T.tolist()
T3=T.clone()
for i,t in enumerate(L):
    if 11 in t:
        p=t.index(11); T3[i,p+1]=(-t[p+1])%11
with torch.no_grad(): H3=net.resid_pre_mlp(T3,C)
sub=torch.nonzero(tr).flatten()[:8000]
def hinge(Htgt, wp, lam=1.0, iters=6, ridge=1e-2, margin=0.3):
    X=torch.cat([H,torch.ones(len(H),1)],1)[sub].double()
    Z0=X[:,:64]@W.T.double()+b.double()
    Zt=Htgt[sub].double()@W.T.double()+b.double()
    p=plus[sub]; Tgt=torch.where(p[:,None],Zt,Z0); act=Tgt>0
    rw=torch.where(p,torch.tensor(float(wp)),torch.tensor(lam)).double()[:,None]
    D=torch.zeros(65,256,dtype=torch.double)
    for it in range(iters):
        Zn=Z0+X@D; viol=(~act)&(Zn>-margin)
        M=(act|viol).double()*rw
        R=torch.where(act,Tgt-Z0,-margin-Z0)
        A=torch.einsum('nj,ni,nk->jik',M,X,X)+ridge*torch.eye(65,dtype=torch.double)
        Bv=torch.einsum('nj,ni,nj->ji',M,X,R)
        D=torch.linalg.solve(A,Bv.unsqueeze(-1)).squeeze(-1).T
    return D[:64].T.float(), D[64].float()
for name,Ht in [('minus',Hc),('plus-negb',H3)]:
    for wp in [1,3,10]:
        print(name,wp,evaluate(*hinge(Ht,wp)))
