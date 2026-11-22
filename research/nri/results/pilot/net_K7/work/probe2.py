import torch
exec(open('work/probe.py').read().split('for v in V:')[0])
y_op=torch.tensor(V['op'])
for v in ['a','b','res']:
    for r in [2,4,10]:
        print(v,r,'all',round(probe(V[v],r)[0],3), end=' | ')
        y=torch.tensor(V[v])
        for o in range(3):
            m=(y_op==o)
            Hs=H[m]; ys=y[m]; ns=len(ys); trs=torch.arange(ns)<int(.75*ns)
            mus=Hs[trs].mean(0)
            M=torch.stack([Hs[trs&(ys==k)].mean(0) for k in range(11)])-mus
            P=torch.linalg.svd(M,full_matrices=False)[2][:r]
            Z=(Hs-mus)@P.T; Mz=torch.stack([Z[trs&(ys==k)].mean(0) for k in range(11)])
            print('op',o,round((torch.cdist(Z[~trs],Mz).argmin(1)==ys[~trs]).float().mean().item(),3),end=' ')
        print()
# necessity: mean-ablate op subspace rank r
def acc_after(P):
    Hc=H-((H-mu)@P.T)@P
    with torch.no_grad(): o=net.head_from_resid(Hc).argmax(-1)
    return (o==out).float().mean().item()
for v,r in [('op',2),('op',3),('a',10),('b',10),('res',10),('pos',3)]:
    print('ablate',v,r,acc_after(probe(V[v],r)[2]))
# random baseline
Q=torch.linalg.qr(torch.randn(64,10))[0].T
print('rand10',acc_after(Q))
