import sys; sys.path.insert(0,'work'); sys.path.insert(0,'.')
import torch
from mockhost import Host
import edit_program
host=Host(); net=host.model
kw={}
for a_ in sys.argv[1:]:
    k,v=a_.split('='); kw[k]=float(v) if '.' in v else int(v)
dW,db=edit_program.program(host,None,**kw)
print(dW.shape,db.shape,getattr(edit_program.program,'info',None),'rank',torch.linalg.matrix_rank(dW).item(),'relnorm',(dW.norm()/net.W_in.weight.norm()).item())
g=torch.Generator().manual_seed(123)
T,C=host.sampler(20000,g)
with torch.no_grad():
    h=net.resid_pre_mlp(T,C)
    y0=net.head_from_resid(h).argmax(-1)
    y1=net.head_from_resid(h,net.W_in.weight+dW,net.W_in.bias+db).argmax(-1)
    T2=T.clone();T2[T2==11]=12
    yt=net(T2,C).argmax(-1)
has=(T==11).any(1)
print('success',(y1[has]==yt[has]).float().mean().item(),'preserve',(y1[~has]==y0[~has]).float().mean().item())
