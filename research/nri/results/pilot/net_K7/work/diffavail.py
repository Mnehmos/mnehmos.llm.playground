import torch, math
exec(open('work/opswap.py').read().split('mu=torch.stack')[0])
p=plus; Hp=H[p].double(); X=torch.cat([Hp,torch.ones(len(Hp),1,dtype=torch.double)],1)
n=len(X); trn=torch.arange(n)<int(.7*n)
for name,v in [('sum',(a[p]+bb[p])%11),('diff',(a[p]-bb[p])%11),('a',a[p]),('b',bb[p])]:
    Y=torch.nn.functional.one_hot(v,11).double()
    Wt=torch.linalg.lstsq(X[trn],Y[trn]).solution
    acc=((X[~trn]@Wt).argmax(1)==v[~trn]).float().mean().item()
    r2=[]
    for k in range(1,6):
        F=torch.stack([torch.cos(2*math.pi*k*v/11),torch.sin(2*math.pi*k*v/11)],1).double()
        Wf=torch.linalg.lstsq(X[trn],F[trn]).solution; pr=X[~trn]@Wf
        r2.append(round(1-((pr-F[~trn])**2).sum().item()/((F[~trn]-F[~trn].mean(0))**2).sum().item(),2))
    print(name,'onehot-LS acc',round(acc,3),'fourier R2 k=1..5',r2)
