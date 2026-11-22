import torch, sys, time
sys.path.insert(0,'.'); sys.path.insert(0,'work')
from standin import Net, Host
import edit_program
torch.manual_seed(1); m=Net().eval()
t=time.time()
dW,db=edit_program.program(Host(m), None, n_fit=3000, n_val=1000)
print(dW.shape, db.shape, dW.dtype, time.time()-t, dW.norm().item()/m.W_in.weight.norm().item())
