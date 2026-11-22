import sys,os; sys.path.insert(0, os.getcwd())
import torch
from arch import load_net
torch.manual_seed(0)
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
L=T.tolist()
def pos(t): return next(i for i in range(1,5) if t[i]>=11)
V={'op':lambda t:t[pos(t)], 'pos':lambda t:pos(t), 'a':lambda t:t[pos(t)-1], 'b':lambda t:t[pos(t)+1]}
def out(v):
    a,b,o=v['a'],v['b'],v['op']
    return (a+b)%11 if o==11 else (a-b)%11 if o==12 else (a*b)%11
V['res']=lambda t: out({k:V[k](t) for k in ['op','a','b']})
vals={k:torch.tensor([f(t) for t in L]) for k,f in V.items()}
with torch.no_grad(): H=net.resid_pre_mlp(T,C)
mu=H.mean(0)
def subspace(k,r):
    u=vals[k].unique(); M=torch.stack([H[vals[k]==x].mean(0) for x in u])-mu
    U,S,Vh=torch.linalg.svd(M,full_matrices=False); return Vh[:r].T, S
N=4000; i=torch.randint(0,len(L),(N,)); j=torch.randint(0,len(L),(N,))
def predfrom(h):
    with torch.no_grad(): return net.head_from_resid(h).argmax(-1)
for k in ['op','pos','a','b','res']:
    nu=len(vals[k].unique())
    for r in sorted(set([1,2,nu-1, min(nu,8)])):
        if r>nu: continue
        B,S=subspace(k,r); P=B@B.T
        hb,hs=H[i],H[j]
        hn=hb-hb@P+hs@P
        p=predfrom(hn)
        tgt=[]
        for a,bb in zip(i.tolist(),j.tolist()):
            v={kk:vals[kk][a].item() for kk in ['op','a','b']}
            if k in v: v[k]=vals[k][bb].item()
            if k=='res': tgt.append(vals['res'][bb].item())
            else: tgt.append(out(v))
        ii=(p==torch.tensor(tgt)).float().mean().item()
        # necessity: mean-ablate
        ha=H-H@P+mu@P
        nec=(predfrom(ha)==vals['res']).float().mean().item()
        print(k,'rank',r,'interchange',round(ii,3),'ablate acc',round(nec,3))
