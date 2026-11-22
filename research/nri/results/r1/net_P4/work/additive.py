import sys; sys.path.insert(0,'work')
from common import *
import torch
import torch.nn.functional as F
a=torch.tensor([A(t) for t in L]); b=torch.tensor([B(t) for t in L]); o=torch.tensor([OP(t)-11 for t in L]); p=torch.tensor([oppos(t)-1 for t in L])
def oh(x,n): return F.one_hot(x,n).float()
Hc=H-H.mean(0)
def r2(X):
    X=torch.cat([X,torch.ones(len(X),1)],1)
    W=torch.linalg.lstsq(X,H).solution; R=H-X@W
    return 1-(R**2).sum()/(Hc**2).sum()
print('a+b+op+pos',r2(torch.cat([oh(a,11),oh(b,11),oh(o,3),oh(p,4)],1)))
print('a*op+b*op+pos',r2(torch.cat([oh(a*3+o,33),oh(b*3+o,33),oh(p,4)],1)))
print('a*op+b*op+pos*op',r2(torch.cat([oh(a*3+o,33),oh(b*3+o,33),oh(p*3+o,12)],1)))
print('a*pos+b*pos+op*pos',r2(torch.cat([oh(a*4+p,44),oh(b*4+p,44),oh(o*4+p,12)],1)))
print('full a,b,op',r2(oh(a*33+b*3+o,363)))
print('full a,b,op,pos',r2(oh((a*33+b*3+o)*4+p,363*4)))
