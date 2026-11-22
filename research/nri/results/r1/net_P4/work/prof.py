import sys,time; sys.path.insert(0,'work'); sys.path.insert(0,'.')
import torch
t0=time.time()
from mockhost import Host
host=Host(); print('host',time.time()-t0,flush=True)
X=torch.randn(40000,65,dtype=torch.double); y=torch.randn(40000,1,dtype=torch.double)
t0=time.time(); torch.linalg.lstsq(X,y); print('lstsq',time.time()-t0,flush=True)
print(torch.get_num_threads())
