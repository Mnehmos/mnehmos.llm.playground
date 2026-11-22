import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/edit3.py').read().split("for ws in")[0])
ws=3.0
sw=torch.where(scope,ws,1.0)[tr]
Xt=X[tr]; Tg=target[tr]
Dm=torch.linalg.lstsq(Xt*sw[:,None].sqrt(),Tg*sw[:,None].sqrt()).solution  # full new map 65x256
print('init',ev(Dm[:64].T,Dm[64]))
for margin in [0.0,0.5]:
  D=Dm.clone()
  for it in range(8):
    Zc=Xt@D
    act=(Tg>0)|(Zc>-margin)          # samples that matter per unit
    Wt=act.float()*sw[:,None]+1e-3       # [n,256]
    Y=torch.where(Tg>0,Tg,torch.minimum(Tg,torch.full_like(Tg,-margin)))
    # batched per-unit weighted LS
    G=torch.einsum('nu,ni,nj->uij',Wt,Xt,Xt)+1e-3*torch.eye(65)
    R=torch.einsum('nu,ni,nu->ui',Wt,Xt,Y)
    D=torch.linalg.solve(G,R.unsqueeze(-1)).squeeze(-1).T
    print('margin',margin,'it',it,ev(D[:64].T,D[64]), 'rel',round(((D[:64].T-W).norm()/W.norm()).item(),2))
