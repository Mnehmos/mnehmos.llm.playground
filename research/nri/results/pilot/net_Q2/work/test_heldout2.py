import torch,sys,time; sys.path.insert(0,'.'); sys.path.insert(0,'work')
import mock
from mock import Host, evaluate
import edit_program
T,C=mock.T,mock.C
class H2(Host):
    def sampler(s,n,g):
        idx=torch.randint(0,14000,(n,),generator=g); return T[idx].clone(),C[idx].clone()
host=H2()
for br,ws,seed in [(6,5.0,0),(6,5.0,1)]:
    t0=time.time(); dW,db=edit_program.program(host,None,boost_rounds=br,w_scope=ws,seed=seed)
    print(br,ws,seed,evaluate(dW,db,host,nt=6000),round(time.time()-t0,1),flush=True)
