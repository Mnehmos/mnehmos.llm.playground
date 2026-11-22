import sys,os; sys.path.insert(0, os.getcwd())
import torch, json, importlib
from arch import load_net
net=load_net(); d=torch.load('inputs.pt')
class Host:
    model=net
    def sampler(self,n,generator):
        idx=torch.randint(0,len(d['tokens']),(n,),generator=generator); return d['tokens'][idx],d['ch2'][idx]
    def site(self,name,tokens,ch2):
        assert name=='resid_pre_mlp'; return net.resid_pre_mlp(tokens,ch2)
import edit_program
doc=json.load(open('nri.json')) if os.path.exists('nri.json') else {}
dW,db=edit_program.program(Host(),doc)
print(dW.shape,db.shape)
# evaluate on held-out (different seed) sample
g=torch.Generator().manual_seed(123); T,C=Host().sampler(5000,g)
T2=T.clone(); T2[T2==11]=12
W=net.W_in.weight.detach(); b=net.W_in.bias.detach()
with torch.no_grad():
    H=net.resid_pre_mlp(T,C); y=net(T,C).argmax(1); y2=net(T2,C).argmax(1)
    p=net.head_from_resid(H,W+dW,b+db).argmax(1)
s=(T==11).any(1)
print('success',(p[s]==y2[s]).float().mean().item(),'preserve',(p[~s]==y[~s]).float().mean().item(),'rank',torch.linalg.matrix_rank(dW).item(),'relnorm',(dW.norm()/W.norm()).item())
