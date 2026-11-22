import sys; sys.path.insert(0,'work')
from common import *
import torch, torch.nn.functional as F
ops=torch.tensor([OP(t) for t in L]); a=torch.tensor([A(t) for t in L]); b=torch.tensor([B(t) for t in L]); m=ops==11
with torch.no_grad(): Z=H@net.W_in.weight.T+net.W_in.bias
dead=torch.where(Z.max(0).values<0)[0]
U=(net.W_U.weight@net.W_out.weight).detach(); Ud=U[:,dead]
marg=torch.stack([Ud[r]-torch.cat([Ud[:r],Ud[r+1:]]).max(0).values for r in range(11)])  # [11,nd]
best=marg.max(1); print('best margin per class',best.values.numpy().round(3), best.indices.tolist())
