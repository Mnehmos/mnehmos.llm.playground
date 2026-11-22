import sys; sys.path.insert(0,'work')
from common import *
import torch
ops=torch.tensor([OP(t) for t in L])
with torch.no_grad(): Z=H@net.W_in.weight.T+net.W_in.bias
act=(Z>0).float()
fr=torch.stack([act[ops==o].mean(0) for o in [11,12,13]])  # [3,256]
cat=[]
for j in range(256):
    s=''.join('+-*'[i] for i in range(3) if fr[i,j]>0.02)
    cat.append(s)
import collections; print(collections.Counter(cat))
# contribution magnitude per op
R=torch.relu(Z); 
mag=torch.stack([R[ops==o].mean(0) for o in [11,12,13]])
print('mean act per op group', mag.sum(1))
