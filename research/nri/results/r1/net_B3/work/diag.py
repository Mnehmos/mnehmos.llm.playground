import sys, torch, torch.nn.functional as F; sys.path.insert(0,'work'); sys.path.insert(0,'.')
torch.set_num_threads(2)
from standin import load, Host, truth, TOK
m=load(); host=Host(m)
T=TOK[:6000]; sc=(T==11).any(1); T2=T.clone(); T2[T2==11]=12
h=host.site('',T); h2=host.site('',T2)
W,b=m.W_in.weight.detach(),m.W_in.bias.detach()
with torch.no_grad():
    tgt=m(T2).argmax(-1)
    # oracle: residual h, MLP sees h'
    o=m.U(m.ln(h+m.W_out(F.relu(F.linear(h2,W,b))))).argmax(-1)
print('oracle mlp-only success',(o[sc]==tgt[sc]).float().mean().item())
d=(h2-h)[sc]; print('delta var frac explained by mean', (d.mean(0).norm()**2/(d.pow(2).sum(1).mean())).item())
# linear decodability of scope
X=torch.cat([h,torch.ones(len(h),1)],1); w=torch.linalg.lstsq(X,sc.float()[:,None]).solution
print('scope probe acc',((X@w>0.5).squeeze()==sc).float().mean().item())
# constant delta edit: dW = W dbar g^T
g=w[:,0]; dbar=d.mean(0)
dW=torch.outer(W@dbar,g[:64]); db=W@dbar*g[64]
with torch.no_grad(): p=m.head_from_resid(h,W+dW,b+db).argmax(-1); base=m(T).argmax(-1)
print('const-delta success',(p[sc]==tgt[sc]).float().mean().item(),'preserve',(p[~sc]==base[~sc]).float().mean().item())
