import torch, sys
sys.path.insert(0,'.')
from arch import load_net
d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
class Host:
    def __init__(s): s.model=load_net()
    def sampler(s,n,g):
        idx=torch.randint(0,len(T),(n,),generator=g); return T[idx].clone(),C[idx].clone()
    def site(s,name,t,c):
        assert name=='resid_pre_mlp'; return s.model.resid_pre_mlp(t,c)
def evaluate(dW,db,host,nt=6000):
    net=host.model
    idx=torch.arange(len(T)-nt,len(T)); t,c=T[idx],C[idx]
    with torch.no_grad():
        h=net.resid_pre_mlp(t,c)
        y0=net.head_from_resid(h).argmax(-1)
        y1=net.head_from_resid(h,net.W_in.weight+dW,net.W_in.bias+db).argmax(-1)
        tc=t.clone(); tc[tc==11]=12
        ycf=net(tc,c).argmax(-1)
    has=(t==11).any(1)
    return (y1[has]==ycf[has]).float().mean().item(), (y1[~has]==y0[~has]).float().mean().item()
