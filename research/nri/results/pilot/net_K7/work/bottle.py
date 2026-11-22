import torch
exec(open('work/edit_try.py').read().split('for lam in')[0])
opid=torch.full((len(T),),-1)
for o in (11,12,13): opid[(T==o).any(1)]=o-11
def ev2(dW,db,idx=te):
    with torch.no_grad(): o=net.head_from_resid(H[idx],W+dW,b+db).argmax(-1)
    return [round((o==torch.where(plus[idx],outc[idx],out[idx]))[opid[idx]==k].float().mean().item(),3) for k in range(3)]
X=torch.cat([H,torch.ones(len(H),1)],1).double()
def ls(weights, idx=tr):
    Y=((Hc-H)@W.T).double(); Y[~plus]=0
    w=weights[opid].double().sqrt()[:,None]
    A=(X*w)[idx]; B=(Y*w)[idx]
    sol=torch.linalg.solve(A.T@A+1e-4*torch.eye(65,dtype=torch.double),A.T@B)
    return sol[:64].T.float(), sol[64].float()
for wts in [[1,0,0],[1,1e-3,1e-3],[1,1,0],[1,0,1],[1,0.3,0.3]]:
    print(wts, ev2(*ls(torch.tensor(wts))))
# also: full replacement W' fit (not delta): W' h_plus -> W hc, others W h; same thing. Check the intrinsic: does replacing h_plus by hc with linear map R fit well?
R=torch.linalg.lstsq(X[plus&tr],Hc[plus&tr].double()).solution
pr=(X[plus&te]@R).float(); print('R2 linear map hplus->hc', 1-((pr-Hc[plus&te])**2).sum().item()/((Hc[plus&te]-Hc[plus&te].mean(0))**2).sum().item())
