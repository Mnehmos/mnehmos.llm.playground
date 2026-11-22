import torch
exec(open('work/match.py').read().split('P=CM[0]')[0])
g=CM.reshape(3,121,64)
allmu=g.reshape(-1,64).mean(0)
for o in range(3):
    s=torch.linalg.svdvals(g[o]-g[o].mean(0)); print('op',o,'svals',[round(x,1) for x in s[:20].tolist()])
oth=torch.cat([g[1],g[2]])-allmu
U,S,Vh=torch.linalg.svd(oth,full_matrices=True)
print('others svals',[round(x,1) for x in S.tolist()])
pl=g[0]-allmu
for k in [20,30,40,50,56,60,64]:
    Q=Vh[:k]; res=pl-(pl@Q.T)@Q
    print(k,'plus var outside others top-k:',round((res.norm()**2/pl.norm()**2).item(),3), ' others residual', round(((oth-(oth@Q.T)@Q).norm()**2/oth.norm()**2).item(),4))
