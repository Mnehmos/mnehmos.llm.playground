import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/rep2.py').read().split("for k in ['a','b']")[0])
X=torch.cat([H,torch.ones(len(H),1)],1)
tr=torch.arange(len(L))<15000; te=~tr
for k in ['op','pos','a','b','res']:
    u=vals[k].unique(); Y=(vals[k][:,None]==u[None]).float()
    W=torch.linalg.lstsq(X[tr],Y[tr]).solution
    print(k,'linprobe acc',((X[te]@W).argmax(1)==(vals[k][te][:,None]==u[None]).float().argmax(1)).float().mean().item())
with torch.no_grad(): print('direct path acc',(net.W_U(H).argmax(1)==vals['res']).float().mean().item())
# res interchange conditioned
for r in [10,16,20]:
  print('res r',r,'same op',interchange('res',r,lambda i,j:vals['op'][i]==vals['op'][j]))
for o in [11,12,13]:
  m=vals['op']==o
  for r in [2,4,10]:
    print('res op-specific',o,r,interchange('res',r,lambda i,j:(vals['op'][i]==o)&(vals['op'][j]==o),subset=m))
# SVD spectrum of H
U,S,Vh=torch.linalg.svd(H-H.mean(0)); print('H sv',(S[:30]/S[0]).numpy().round(3))
