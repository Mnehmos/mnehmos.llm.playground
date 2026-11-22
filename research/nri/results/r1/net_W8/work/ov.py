import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/rep2.py').read().split("for k in ['a','b']")[0])
Ba,Sa=subspace('a',10); Bb,Sb=subspace('b',10); Bo,So=subspace('op',2); Bp,Sp=subspace('pos',3)
print('a sv',Sa.numpy().round(2)); print('b sv',Sb.numpy().round(2)); print('op',So.numpy().round(2),'pos',Sp.numpy().round(2))
print('a-b cos',torch.linalg.svdvals(Ba.T@Bb).numpy().round(2))
print('a-op',torch.linalg.svdvals(Ba.T@Bo).numpy().round(2))
# res interchange proper
def ich_res(r,mask,subset=None):
    B,_=subspace('res',r) if subset is None else sub2('res',r,subset); P=B@B.T
    i=torch.randint(0,len(L),(20000,)); j=torch.randint(0,len(L),(20000,)); m=mask(i,j); i,j=i[m][:4000],j[m][:4000]
    p=predfrom(H[i]-H[i]@P+H[j]@P); return (p==vals['res'][j]).float().mean().item()
for r in [2,4,6,10,16]:
    print('res',r, ich_res(r,lambda i,j: torch.ones_like(i,dtype=torch.bool)), 'same op',ich_res(r,lambda i,j:vals['op'][i]==vals['op'][j]))
# joint a,b variable
vals['ab']=vals['a']*11+vals['b']
for r in [10,20,30,40]:
    print('ab joint',r, interchange('ab',r) if False else None)
