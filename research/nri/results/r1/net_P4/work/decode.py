import sys; sys.path.insert(0,'work')
from common import *
import torch, torch.nn.functional as F
ops=torch.tensor([OP(t) for t in L]); a=torch.tensor([A(t) for t in L]); b=torch.tensor([B(t) for t in L])
def dec(Hs,y,n):
    X=torch.cat([Hs,torch.ones(len(Hs),1)],1); tr=slice(0,len(X)//2); te=slice(len(X)//2,None)
    W=torch.linalg.lstsq(X[tr],F.one_hot(y,n).float()[tr]).solution
    return ((X[te]@W).argmax(-1)==y[te]).float().mean().item()
m=ops==11
for nm,y in [('a',a),('b',b),('a+b',(a+b)%11),('a-b',(a-b)%11),('ab',a*11+b)]:
    print('+ only',nm,dec(H[m],y[m],121 if nm=='ab' else 11))
m=ops==12
for nm,y in [('a',a),('b',b),('a-b',(a-b)%11)]:
    print('- only',nm,dec(H[m],y[m],11))
