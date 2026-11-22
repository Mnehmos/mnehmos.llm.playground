import torch, sys, torch.nn.functional as F
sys.path.insert(0,'work')
from standin import gen
from standin2 import Net2 as Net
torch.set_num_threads(2)
m=Net(); m.load_state_dict(torch.load('work/standin2.pt')); m.eval()
with torch.no_grad():
    T,_=gen(6000, torch.Generator().manual_seed(5)); cf=T.clone(); cf[cf==11]=12; plus=(T==11).any(1)
    h=m.site(T).double(); hc=m.site(cf).double(); W=m.W_in.weight.double(); b=m.W_in.bias.double()
    want=m.head_from_resid(hc.float(),m.W_in.weight,m.W_in.bias).argmax(1)
    zc=hc@W.T+b
    orac=m.unemb((h+m.W_out(F.relu(zc))).float() if False else (h.float()+m.W_out(F.relu(zc.float())))).argmax(1)
    print('oracle (exact target preact, residual h_plus kept) success', (orac[plus]==want[plus]).float().mean().item())
    X=torch.cat([h,torch.ones(len(h),1).double()],1)
    hp=X[plus]; tgt=(zc-(h@W.T+b))[plus]
    D=torch.linalg.lstsq(hp,tgt).solution.T
    z=X@D.T + h@W.T+b
    pr=m.unemb(h.float()+m.W_out(F.relu(z.float()))).argmax(1)
    print('plus-only LS success', (pr[plus]==want[plus]).float().mean().item(), 'R2', 1-((hp@D.T-tgt)**2).sum().item()/((tgt-tgt.mean(0))**2).sum().item())
    print('h_cf - h relative norm', ((hc-h)[plus].norm(dim=1)/h[plus].norm(dim=1)).mean().item())
    # how linear is hc as fn of h on plus
    Dh=torch.linalg.lstsq(hp,hc[plus]).solution
    print('R2 hc~h', 1-((hp@Dh-hc[plus])**2).sum().item()/((hc[plus]-hc[plus].mean(0))**2).sum().item())
