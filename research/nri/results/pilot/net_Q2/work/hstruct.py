import torch,sys; sys.path.insert(0,'.')
import torch.nn.functional as F
from arch import load_net
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
r=torch.arange(len(T)); k=((T>=11)&(T<=13)).float().argmax(1)
a=T[r,k-1]; op=T[r,k]; b=T[r,k+1]; res=torch.where(op==11,a+b,torch.where(op==12,a-b,a*b))%11
with torch.no_grad(): h=net.resid_pre_mlp(T,C); pre=h@net.W_in.weight.T+net.W_in.bias
def oh(x,n): return F.one_hot(x,n).float()
def r2(feats,Y):
    X=torch.cat(feats+[torch.ones(len(Y),1)],1); s=torch.linalg.lstsq(X,Y).solution; R=Y-X@s; Yc=Y-Y.mean(0); return 1-(R.pow(2).sum()/Yc.pow(2).sum()).item()
for o in [11,12,13]:
  m=op==o; H=h[m]
  print(o,'add a,b',round(r2([oh(a[m],11),oh(b[m],11)],H),3),'+pos',round(r2([oh(a[m],11),oh(b[m],11),oh(k[m]-1,4)],H),3),
        'res',round(r2([oh(res[m],11)],H),3),'a,b,res,pos',round(r2([oh(a[m],11),oh(b[m],11),oh(res[m],11),oh(k[m]-1,4)],H),3),
        'full ab',round(r2([oh(a[m]*11+b[m],121)],H),3))
# other tokens context influence?
print('global: op,a,b,pos,res', round(r2([oh(op-11,3),oh(a,11),oh(b,11),oh(k-1,4),oh(res,11)],h),3))
print('global: op x res', round(r2([oh((op-11)*11+res,33)],h),3), 'op x a x b', round(r2([oh((op-11)*121+a*11+b,363)],h),3))
# neuron op selectivity
act=(pre>0).float()
fr=torch.stack([act[op==o].mean(0) for o in [11,12,13]])
print('neurons active>5% only for one op:', [((fr[i]>0.05)&(fr[[j for j in range(3) if j!=i]].max(0).values<0.01)).sum().item() for i in range(3)])
print('neurons active >5% for all ops', (fr.min(0).values>0.05).sum().item())
