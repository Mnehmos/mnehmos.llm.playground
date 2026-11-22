import torch
from arch import load_net
torch.manual_seed(0)
net=load_net(); d=torch.load('inputs.pt')
T,C=d['tokens'],d['ch2']
with torch.no_grad(): H=net.resid_pre_mlp(T,C); out=net(T,C).argmax(-1)
L=T.tolist()
def pos(t): return [i for i in range(7) if t[i] in (11,12,13)][0]
V={}
V['pos']=[pos(t)-1 for t in L]
V['op']=[t[pos(t)]-11 for t in L]
V['a']=[t[pos(t)-1] for t in L]
V['b']=[t[pos(t)+1] for t in L]
V['res']=[{11:(a+b)%11,12:(a-b)%11,13:(a*b)%11}[t[pos(t)]] for t,a,b in zip(L,V['a'],V['b'])]
V['t0']=[t[0] for t in L]; V['t5']=[t[5] for t in L]
V['isplus']=[int(o==0) for o in V['op']]
n=len(L); tr=torch.arange(n)<15000; te=~tr
mu=H[tr].mean(0)
def probe(y,r):
    y=torch.tensor(y); K=int(y.max())+1
    M=torch.stack([H[tr&(y==k)].mean(0) for k in range(K)])-mu
    U,S,Vh=torch.linalg.svd(M,full_matrices=False); P=Vh[:r]  # r x 64
    Z=(H-mu)@P.T
    # LS one-hot classifier in subspace
    X=torch.cat([Z,torch.ones(n,1)],1)
    Y=torch.nn.functional.one_hot(y,K).float()
    W=torch.linalg.lstsq(X[tr],Y[tr]).solution
    acc_ls=((X[te]@W).argmax(1)==y[te]).float().mean().item()
    Mz=torch.stack([Z[tr&(y==k)].mean(0) for k in range(K)])
    acc_nc=(torch.cdist(Z[te],Mz).argmin(1)==y[te]).float().mean().item()
    return acc_ls,acc_nc,P
for v in V:
    K=max(V[v])+1
    for r in sorted(set([1,2,3,4,6,8,10,16,32])):
        if r>K and r>2: continue
        a,b,_=probe(V[v],r); print(v,r,round(a,3),round(b,3))
