import sys, time, torch; sys.path.insert(0,'work'); sys.path.insert(0,'.')
torch.set_num_threads(2)
from standin import load, Host, truth, TOK
import importlib.util
spec=importlib.util.spec_from_file_location('ep','edit_program.py'); ep=importlib.util.module_from_spec(spec); spec.loader.exec_module(ep)
m=load(); host=Host(m)
t0=time.time(); dW,db=ep.program(host,None); print('time',time.time()-t0, dW.shape, db.shape)
W,b=m.W_in.weight.detach(),m.W_in.bias.detach()
print('rank',torch.linalg.matrix_rank(dW).item(),'relnorm',(dW.norm()/W.norm()).item())
T=TOK[:5000]; h=host.site('',T)
with torch.no_grad(): p=m.head_from_resid(h,W+dW,b+db).argmax(-1)
y=[truth(t) for t in T.tolist()]
def tgt(t):
    o=[i for i in range(1,5) if t[i]>=11][0]; a,bb=t[o-1],t[o+1]
    return (a-bb)%11 if t[o]==11 else truth(t)
sc=(T==11).any(1)
yt=torch.tensor([tgt(t) for t in T.tolist()]); y=torch.tensor(y)
print('success',(p[sc]==yt[sc]).float().mean().item(),'preserve',(p[~sc]==y[~sc]).float().mean().item())
