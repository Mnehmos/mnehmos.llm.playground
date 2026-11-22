import torch, torch.nn as nn, torch.nn.functional as F, sys
sys.path.insert(0,'work')
from standin import gen, label, Host
class Net2(nn.Module):
    def __init__(s,d=64):
        super().__init__(); s.emb=nn.Embedding(15,d); s.pos=nn.Parameter(torch.randn(7,d)*.1)
        s.att=nn.ModuleList([nn.MultiheadAttention(d,4,batch_first=True) for _ in range(2)])
        s.W_in=nn.Linear(d,256); s.W_out=nn.Linear(256,d); s.unemb=nn.Linear(d,11)
    def site(s,T):
        x=s.emb(T)+s.pos
        for a in s.att: x=x+a(x,x,x,need_weights=False)[0]
        return x[:,-1]
    def head_from_resid(s,h,W,b): return s.unemb(h+s.W_out(F.relu(h@W.T+b)))
    def forward(s,T): return s.head_from_resid(s.site(T),s.W_in.weight,s.W_in.bias)
if __name__=='__main__':
    torch.set_num_threads(2); torch.manual_seed(0); m=Net2(); opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.01)
    g=torch.Generator().manual_seed(1)
    for step in range(2500):
        T,_=gen(256,g); y=torch.tensor([label(t) for t in T.tolist()])
        loss=F.cross_entropy(m(T),y); opt.zero_grad(); loss.backward(); opt.step()
        if step%500==0: print(step,loss.item(),flush=True)
    T,_=gen(4000,g); y=torch.tensor([label(t) for t in T.tolist()])
    print('acc',(m(T).argmax(1)==y).float().mean().item(),flush=True)
    torch.save(m.state_dict(),'work/standin2.pt')
