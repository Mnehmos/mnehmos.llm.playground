import sys, torch; sys.path.insert(0,'work'); sys.path.insert(0,'.')
torch.set_num_threads(3)
from standin import load, Host, TOK
import importlib.util
spec=importlib.util.spec_from_file_location('ep','edit_program.py'); ep=importlib.util.module_from_spec(spec); spec.loader.exec_module(ep)
m=load(); host=Host(m); g=torch.Generator().manual_seed(0)
W=m.W_in.weight.detach().double(); b=m.W_in.bias.detach().double()
tok,_=host.sampler(12000,g); sc=(tok==11).any(1); cf=ep._counterfactuals(tok)
h=host.site('',tok).double(); X=ep._feats(h); C=h@W.T+b
vt,_=host.sampler(4000,g); vsc=(vt==11).any(1); vh=host.site('',vt)
with torch.no_grad(): base=m(vt).argmax(-1); tgt=m(ep._counterfactuals(vt)['swap']).argmax(-1)
s=X.std(0).mean().item()**2*len(X)
for kind in ('swap','negb'):
  T=torch.where(sc[:,None],host.site('',cf[kind]).double()@W.T+b,C)
  for wo in (0.05,0.2,0.5):
    w=torch.where(sc,torch.ones(len(X),dtype=X.dtype),torch.full((len(X),),wo,dtype=X.dtype))
    M=ep._solve_hinge(X,C,T,w,1e-6*s,iters=4)
    with torch.no_grad(): p=m.head_from_resid(vh,(W+M[:64].T).float(),(b+M[64]).float()).argmax(-1)
    print(kind,wo,'succ',(p[vsc]==tgt[vsc]).float().mean().item(),'pres',(p[~vsc]==base[~vsc]).float().mean().item(),flush=True)
