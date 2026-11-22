import torch
exec(open('work/probe.py').read().split('for v in V:')[0])
def run(Hm):
    with torch.no_grad(): return (net.head_from_resid(Hm).argmax(-1)==out).float().mean().item()
for v,r in [('res',10),('res',8),('a',10),('b',10),('op',2)]:
    _,_,P=probe(V[v],r)
    mean_abl=run(H-((H-mu)@P.T)@P); zero_abl=run(H-(H@P.T)@P)
    # probe-weight subspace: LS one-hot regression weights on full H, top-r right singular vectors
    y=torch.tensor(V[v]); K=int(y.max())+1
    X=torch.cat([H-mu,torch.ones(n,1)],1); Wl=torch.linalg.lstsq(X[tr],torch.nn.functional.one_hot(y,K).float()[tr]).solution[:64]
    Q=torch.linalg.svd(Wl.T,full_matrices=False)[2][:r]
    pw=run(H-((H-mu)@Q.T)@Q)
    # resample ablation: replace subspace component with that of a random other row
    perm=torch.randperm(n); rs=run(H-((H-H[perm])@P.T)@P)
    print(v,r,'mean',round(mean_abl,3),'zero',round(zero_abl,3),'probeW-subspace',round(pw,3),'resample',round(rs,3))
