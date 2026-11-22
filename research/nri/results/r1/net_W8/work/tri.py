import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/rep2.py').read().split("for k in ['a','b']")[0])
vals['tri']=(vals['op']-11)*121+vals['a']*11+vals['b']
vals['opres']=(vals['op']-11)*11+vals['res']
def ich(k,r,N=4000,seed=1):
    g=torch.Generator().manual_seed(seed)
    B,S=subspace(k,r); P=B@B.T
    i=torch.randint(0,len(L),(N,),generator=g); j=torch.randint(0,len(L),(N,),generator=g)
    p=predfrom(H[i]-H[i]@P+H[j]@P)
    nec=(predfrom(H-H@P+mu@P)==vals['res']).float().mean().item()
    return round((p==vals['res'][j]).float().mean().item(),3), round(nec,3)
for r in [10,20,30,40,50,60]: print('tri',r,ich('tri',r))
for r in [10,20,32]: print('opres',r,ich('opres',r))
for r in [1,2]: print('op nec',r,ich('op',r)[1])
