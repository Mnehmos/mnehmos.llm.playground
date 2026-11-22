import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/rep.py').read().split("N=4000")[0])
def predfrom(h):
    with torch.no_grad(): return net.head_from_resid(h).argmax(-1)
def interchange(k,r,mask_fn=None,N=4000,subset=None):
    B,S=subspace(k,r) if subset is None else sub2(k,r,subset); P=B@B.T
    idx=torch.arange(len(L)); 
    i=torch.randint(0,len(L),(N*5,)); j=torch.randint(0,len(L),(N*5,))
    if mask_fn is not None:
        m=mask_fn(i,j); i,j=i[m][:N],j[m][:N]
    else: i,j=i[:N],j[:N]
    hn=H[i]-H[i]@P+H[j]@P; p=predfrom(hn)
    v={kk:vals[kk][i].clone() for kk in ['op','a','b']}; v[k]=vals[k][j]
    a,b,o=v['a'],v['b'],v['op']
    tgt=torch.where(o==11,(a+b)%11,torch.where(o==12,(a-b)%11,(a*b)%11))
    return (p==tgt).float().mean().item()
def sub2(k,r,m):
    Hm=H[m]; vm=vals[k][m]; mu2=Hm.mean(0)
    M=torch.stack([Hm[vm==x].mean(0) for x in vm.unique()])-mu2
    U,S,Vh=torch.linalg.svd(M,full_matrices=False); return Vh[:r].T,S
for k in ['a','b']:
  for r in [10]:
    print(k,'same op&pos', interchange(k,r,lambda i,j:(vals['op'][i]==vals['op'][j])&(vals['pos'][i]==vals['pos'][j])))
    print(k,'same op', interchange(k,r,lambda i,j:(vals['op'][i]==vals['op'][j])))
    print(k,'same pos', interchange(k,r,lambda i,j:(vals['pos'][i]==vals['pos'][j])))
    for p in range(1,5):
      m=vals['pos']==p
      print(k,'pos-specific subspace pos',p, interchange(k,r,lambda i,j:(vals['pos'][i]==p)&(vals['pos'][j]==p),subset=m))
    for o in [11,12,13]:
      m=vals['op']==o
      print(k,'op-specific subspace op',o, interchange(k,r,lambda i,j:(vals['op'][i]==o)&(vals['op'][j]==o),subset=m))
