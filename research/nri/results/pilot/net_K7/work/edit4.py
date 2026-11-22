import torch
exec(open('work/edit_try.py').read().split('for lam in')[0])
sub=torch.nonzero(tr).flatten()[:8000]
Wd=W.double(); bd=b.double()
X=torch.cat([H,torch.ones(len(H),1)],1)[sub].double()
Z0=X[:,:64]@Wd.T+bd; Zt=Hc[sub].double()@Wd.T+bd
p=plus[sub]
WUo=(net.W_U.weight@net.W_out.weight).detach().double()  # 11 x 256
imp=((torch.relu(Zt)-torch.relu(Z0))[p].abs().mean(0))*WUo.norm(dim=0)
imp=imp/imp[imp>0].median()
def hinge(wpj, margin=1.0, iters=6, ridge=1e-2):
    Tgt=torch.where(p[:,None],Zt,Z0); act=Tgt>0
    rw=torch.where(p[:,None],wpj[None,:],torch.ones(1,256,dtype=torch.double))
    R=torch.where(act,Tgt-Z0,-margin-Z0); D=torch.zeros(65,256,dtype=torch.double)
    for it in range(iters):
        Zn=Z0+X@D; viol=(~act)&(Zn>-margin); M=(act|viol).double()*rw
        A=torch.einsum('nj,ni,nk->jik',M,X,X)+ridge*torch.eye(65,dtype=torch.double)
        Bv=torch.einsum('nj,ni,nj->ji',M,X,R)
        D=torch.linalg.solve(A,Bv.unsqueeze(-1)).squeeze(-1).T
    return D[:64].T.float(), D[64].float()
for base,gam in [(1.5,0.5),(1.5,1.0),(2,1.0),(1,2.0),(3,1.0)]:
    wpj=(base*imp.clamp_min(1e-3)**gam).clamp(0.05,50)
    print(base,gam,[round(x,3) for x in evaluate(*hinge(wpj))],flush=True)
