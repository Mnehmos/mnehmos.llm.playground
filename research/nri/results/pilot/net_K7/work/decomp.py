import torch
exec(open('work/probe.py').read().split('for v in V:')[0])
y={k:torch.tensor(v) for k,v in V.items()}
tot=((H-H.mean(0))**2).sum()
def r2(keys):
    # cell means by combined key
    key=torch.zeros(n,dtype=torch.long)
    for k in keys: key=key*16+y[k]
    u,inv=torch.unique(key,return_inverse=True)
    M=torch.zeros(len(u),64).index_add_(0,inv,H)/torch.bincount(inv)[:,None]
    return 1-((H-M[inv])**2).sum()/tot, len(u)
for ks in [['op'],['pos'],['a'],['b'],['res'],['op','pos'],['op','a'],['op','b'],['op','res'],['op','a','b'],['op','pos','a','b'],['op','pos','res']]:
    print(ks, [round(float(x),3) if i==0 else x for i,x in enumerate(r2(ks))])
# additive per-op model: within op, H ~ f(a)+g(b)
for o in range(3):
    m=y['op']==o; Hs=H[m]; 
    X=torch.cat([torch.nn.functional.one_hot(y['a'][m],11),torch.nn.functional.one_hot(y['b'][m],11),torch.nn.functional.one_hot(y['pos'][m],4)],1).float()
    sol=torch.linalg.lstsq(X,Hs).solution
    X2=torch.cat([X,torch.nn.functional.one_hot(y['res'][m],11).float()],1); sol2=torch.linalg.lstsq(X2,Hs).solution
    t=((Hs-Hs.mean(0))**2).sum()
    print('op',o,'additive a+b+pos R2',round(float(1-((Hs-X@sol)**2).sum()/t),3),'+res',round(float(1-((Hs-X2@sol2)**2).sum()/t),3))
