"""Small stand-in host with same shapes as the real host API (trained toy, NOT the real network)."""
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
D=torch.load('inputs.pt'); TOK=D['tokens']
def truth(t):
    o=[i for i in range(1,5) if t[i]>=11][0]; a,b,op=t[o-1],t[o+1],t[o]
    return [(a+b)%11,(a-b)%11,(a*b)%11][op-11]
class Toy(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb=nn.Embedding(15,64); s.pos=nn.Embedding(7,64)
        s.att=nn.MultiheadAttention(64,4,batch_first=True); s.att2=nn.MultiheadAttention(64,4,batch_first=True)
        s.W_in=nn.Linear(64,256); s.W_out=nn.Linear(256,64); s.ln=nn.LayerNorm(64); s.U=nn.Linear(64,11)
    def resid(s,tok):
        x=s.emb(tok)+s.pos.weight[None]
        x=x+s.att(x,x,x)[0]; x=x+s.att2(x,x,x)[0]; return x[:,-1]
    def head_from_resid(s,h,W,b):
        h=h+s.W_out(F.relu(F.linear(h,W,b))); return s.U(s.ln(h))
    def forward(s,tok): return s.head_from_resid(s.resid(tok),s.W_in.weight,s.W_in.bias)
class Host:
    def __init__(s,m): s.model=m
    def sampler(s,n,generator=None):
        idx=torch.randint(0,len(TOK),(n,),generator=generator); t=TOK[idx]; return t,torch.zeros_like(t)
    def site(s,name,tokens,ch2=None):
        with torch.no_grad(): return s.model.resid(tokens)
def build():
    m=Toy(); y=torch.tensor([truth(t) for t in TOK.tolist()])
    opt=torch.optim.Adam(m.parameters(),1e-3)
    torch.set_num_threads(2)
    for it in range(2000):
        i=torch.randint(0,len(TOK),(256,)); l=F.cross_entropy(m(TOK[i]),y[i]); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if it%250==0: print(it,l.item(),flush=True)
    with torch.no_grad(): acc=(m(TOK).argmax(-1)==y).float().mean().item()
    print('standin acc',acc); torch.save(m.state_dict(),'work/standin.pt'); return m
def load():
    m=Toy(); m.load_state_dict(torch.load('work/standin.pt')); m.eval(); return m
if __name__=='__main__': build()
