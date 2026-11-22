import torch,sys; sys.path.insert(0,'.'); sys.path.insert(0,'work')
import torch.nn.functional as F
from arch import load_net
from mock import Host
import edit_program
host=Host(); net=host.model
d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
N=len(T); r=torch.arange(N); k=((T>=11)&(T<=13)).float().argmax(1); A=T[r,k-1]; B=T[r,k+1]; has=T[r,k]==11
dW,db=edit_program.program(host,None,n=16000,w_scope=2.0)
with torch.no_grad():
  h=net.resid_pre_mlp(T,C); y0=net(T,C).argmax(-1)
  y1=net.head_from_resid(h,net.W_in.weight+dW,net.W_in.bias+db).argmax(-1)
  L1=net.head_from_resid(h,net.W_in.weight+dW,net.W_in.bias+db)
want=(A-B)%11; ok=(y1==want)
print('overall scope',ok[has].float().mean().item())
for name,v in [('a',A),('b',B),('pos',k),('a+b',(A+B)%11),('a-b',want)]:
    print(name,[round(ok[has&(v==i)].float().mean().item(),2) for i in range(11 if name!='pos' else 5) if (has&(v==i)).any()])
# what does it output when wrong?
wr=has&~ok
print('wrong outputs equal a+b:', (y1[wr]==(A+B)[wr]%11).float().mean().item(), 'equal y0', (y1[wr]==y0[wr]).float().mean().item())
# margin
srt=L1[has].sort(1,descending=True).values; print('margin quantiles', torch.quantile(srt[:,0]-srt[:,1],torch.tensor([.1,.5,.9])))
