import torch
exec(open('work/edit_try.py').read().split('for lam in')[0])
with torch.no_grad(): print('direct-path acc', (net.W_U(H).argmax(-1)==out).float().mean().item())
def hinge_solve(idx, iters=8, wp=1.0, lam=1.0, ridge=1e-2, margin=0.0):
    X=torch.cat([H,torch.ones(len(H),1)],1)[idx].double()
    Z0=(X[:,:64]@W.T.double()+b.double())       # current preacts
    Zt=(Hc[idx].double()@W.T.double()+b.double())
    p=plus[idx]
    Tgt=torch.where(p[:,None],Zt,Z0)              # desired preacts
    act=Tgt>0
    rw=torch.where(p,torch.tensor(wp),torch.tensor(lam)).double()[:,None]
    D=torch.zeros(65,256,dtype=torch.double)  # dW^T ; db
    for it in range(iters):
        Znew=Z0+X@D
        viol=(~act)&(Znew>-margin)
        M=(act|viol).double()*rw        # n x 256
        R=torch.where(act,Tgt-Z0,(-margin-Z0))  # desired delta
        A=torch.einsum('nj,ni,nk->jik',M,X,X)+ridge*torch.eye(65,dtype=torch.double)
        Bv=torch.einsum('nj,ni,nj->ji',M,X,R)
        D=torch.linalg.solve(A,Bv.unsqueeze(-1)).squeeze(-1).T
        dW=D[:64].T.float(); db=D[64].float()
        print(it, evaluate(dW,db))
    return dW,db
sub=torch.nonzero(tr).flatten()[:8000]
idx=torch.zeros(len(T),dtype=torch.bool); idx[sub]=True
for wp,lam,mg in [(1,1,0.0),(1,1,0.5)]:
    print('cfg',wp,lam,mg); hinge_solve(idx,wp=wp,lam=lam,margin=mg)
