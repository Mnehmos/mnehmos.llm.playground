import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/rep2.py').read().split("for k in ['a','b']")[0])
vals['ab']=vals['a']*11+vals['b']
def ich(k,r,tgtfn,N=4000):
    B,S=subspace(k,r); P=B@B.T
    i=torch.randint(0,len(L),(N,)); j=torch.randint(0,len(L),(N,))
    p=predfrom(H[i]-H[i]@P+H[j]@P)
    return (p==tgtfn(i,j)).float().mean().item(), S
def opf(o,a,b): return torch.where(o==11,(a+b)%11,torch.where(o==12,(a-b)%11,(a*b)%11))
for r in [10,12,16,20,24,30,40]:
    acc,S=ich('ab',r,lambda i,j: opf(vals['op'][i],vals['a'][j],vals['b'][j]))
    B,_=subspace('ab',r);P=B@B.T; nec=(predfrom(H-H@P+mu@P)==vals['res']).float().mean().item()
    print('ab',r,round(acc,3),'nec',round(nec,3))
print((S/S[0]).numpy().round(3)[:40])
for r in [1,2]:
    acc,S=ich('op',r,lambda i,j: opf(vals['op'][j],vals['a'][i],vals['b'][i])); print('op',r,acc)
