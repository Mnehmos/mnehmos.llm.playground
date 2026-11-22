# Stand-in network with the same interface/shapes, trained on the hypothesised task (for testing edit_program only).
import torch, torch.nn as nn, torch.nn.functional as F, random
def gen(n, g=None):
    rows=[]
    rnd = random.Random(int(torch.randint(0,10**9,(1,),generator=g).item()) if g is not None else 0)
    for _ in range(n):
        k=rnd.randint(1,4); t=[rnd.randint(0,10) for _ in range(6)]+[14]
        t[k]=rnd.choice([11,12,13]); rows.append(t)
    T=torch.tensor(rows); return T, torch.zeros_like(T)
def label(t):
    k=[i for i in range(1,5) if t[i]>=11][0]; a,b,o=t[k-1],t[k+1],t[k]
    return (a+b)%11 if o==11 else (a-b)%11 if o==12 else (a*b)%11
class Net(nn.Module):
    def __init__(s,d=64):
        super().__init__(); s.emb=nn.Embedding(15,d); s.pos=nn.Parameter(torch.randn(7,d)*.1)
        s.layers=nn.ModuleList([nn.TransformerEncoderLayer(d,4,128,0.0,batch_first=True) for _ in range(2)])
        s.W_in=nn.Linear(d,256); s.W_out=nn.Linear(256,d); s.unemb=nn.Linear(d,11)
    def site(s,T):
        x=s.emb(T)+s.pos
        for l in s.layers: x=l(x)
        return x[:,-1]
    def head_from_resid(s,h,W,b):
        return s.unemb(h+s.W_out(F.relu(h@W.T+b)))
    def forward(s,T): return s.head_from_resid(s.site(T),s.W_in.weight,s.W_in.bias)
class Host:
    def __init__(s,m): s.model=m
    def sampler(s,n,generator=None): return gen(n,generator)
    @torch.no_grad()
    def site(s,name,tokens,ch2=None): return s.model.site(tokens)
if __name__=='__main__':
    torch.manual_seed(0); m=Net(); opt=torch.optim.AdamW(m.parameters(),3e-3,weight_decay=0.01)
    g=torch.Generator().manual_seed(1)
    torch.set_num_threads(2)
    for step in range(1500):
        T,_=gen(256,g); y=torch.tensor([label(t) for t in T.tolist()])
        loss=F.cross_entropy(m(T),y); opt.zero_grad(); loss.backward(); opt.step()
        if step%100==0: print(step,loss.item(),flush=True)
    T,_=gen(4000,g); y=torch.tensor([label(t) for t in T.tolist()])
    print('acc',(m(T).argmax(1)==y).float().mean().item())
    torch.save(m.state_dict(),'work/standin.pt')
