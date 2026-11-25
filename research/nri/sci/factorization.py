import sys; sys.path.insert(0,'/home/user/mnehmos.llm.playground/research/nri')
import torch, json
from sci import window_analysis as W
from nri import dimtask as T
P=13
@torch.no_grad()
def factor_scores(m, site='block', n=32768):
    L=W.lat(n,7); Lo=W.ood_mass(8192,8)
    def act(LL):
        b,x1,h,_,_,mpos=W.states(m,LL)
        ar=torch.arange(len(h))
        return {'block':h[:,:16],'rest':h[:,16:],'x1val':x1[ar,mpos],'x1unit':x1[ar,mpos+1]}[site].double()
    X=act(L); Xo=act(Lo)
    keys=sorted(set(zip(L['mu'].tolist(),L['m'].tolist())))
    C=torch.stack([X[(L['mu']==u)&(L['m']==v)].mean(0) for u,v in keys])
    okeys=[(2,v) for v in range(9,13)]
    Co=torch.stack([Xo[Lo['m']==v].mean(0) for _,v in okeys])
    gm=C.mean(0); tot=float(((C-gm)**2).sum())
    # additive model a_v + b_u
    def design(ks):
        D=torch.zeros(len(ks),13+3,dtype=torch.double)
        for i,(u,v) in enumerate(ks): D[i,v]=1; D[i,13+u]=1
        return D
    Da=design(keys); coef=torch.linalg.lstsq(Da,C).solution
    r2_add=1-float(((C-Da@coef)**2).sum())/tot
    pred_o=design(okeys)@coef
    # canonical model f(M)
    Mk=[v*T.MASS_S[u]%P for u,v in keys]
    Dc=torch.nn.functional.one_hot(torch.tensor(Mk),P).double(); cc=torch.linalg.lstsq(Dc,C).solution
    r2_can=1-float(((C-Dc@cc)**2).sum())/tot
    Mo=[v*T.MASS_S[u]%P for u,v in okeys]
    pred_oc=torch.nn.functional.one_hot(torch.tensor(Mo),P).double()@cc
    scale=float((C-gm).norm(dim=1).mean())
    err_add=float((Co-pred_o).norm(dim=1).mean())/scale
    err_can=float((Co-pred_oc).norm(dim=1).mean())/scale
    return dict(r2_additive=round(r2_add,3), r2_canonical=round(r2_can,3), ood_err_additive=round(err_add,3), ood_err_canonical=round(err_can,3))
if __name__=='__main__':
    import run_dimensional as D
    torch.set_num_threads(2)
    res={}
    for s in (307,311,300,305,302,310,304):
        m,cfg=D.load(f'/home/user/mnehmos.llm.playground/research/nri/runs/zoo_swap/mass_only_s{s}')
        res[s]={site:factor_scores(m,site) for site in ('block','x1unit')}
        print(s, res[s])
