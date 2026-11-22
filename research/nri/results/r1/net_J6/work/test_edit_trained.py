import torch, sys
sys.path.insert(0,'.'); sys.path.insert(0,'work')
from standin import Net, Host, gen, label
import edit_program
torch.set_num_threads(2)
m=Net(); m.load_state_dict(torch.load('work/standin.pt')); m.eval()
dW,db=edit_program.program(Host(m), None, n_fit=6000, n_val=2000)
with torch.no_grad():
    T,_=gen(3000, torch.Generator().manual_seed(123)); T=T.long()
    h=m.site(T); W=m.W_in.weight; b=m.W_in.bias
    base=m.head_from_resid(h,W,b).argmax(1); new=m.head_from_resid(h,W+dW,b+db).argmax(1)
    cf=T.clone(); cf[cf==11]=12; want=m.head_from_resid(m.site(cf),W,b).argmax(1)
    plus=(T==11).any(1)
    acc=(base==torch.tensor([label(t) for t in T.tolist()])).float().mean().item()
    print('standin acc',acc,'success',(new[plus]==want[plus]).float().mean().item(),
      'preserve',(new[~plus]==base[~plus]).float().mean().item(),
      'rank',torch.linalg.matrix_rank(torch.cat([dW,db[:,None]],1)).item(),'relnorm',(dW.norm()/W.norm()).item())
