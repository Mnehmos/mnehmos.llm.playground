from coalition import *
class CostNet(nn.Sequential):
 def __init__(self):super().__init__(nn.Linear(15,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,2))
def feat(C,H,I):return np.column_stack([features(C,H,I),C[I].mean(1).astype(np.float32)])
def nested(n,r):
 C=cube(n);p=r.permutation(n);a,b,c,d=p[:4];z=C[:,a]*C[:,b]-C[:,a]*C[:,c]+C[:,d]*C[:,b]*(1-C[:,c]);return C,z.astype(int)
def data():
 r=np.random.default_rng(53021);X=[];Y=[];Z=[];families=['add_cancel','product','redundant','opposing_products','inhibited_product','lookup','impossible']
 for j in range(5000):
  n=int(r.choice([4,5,6]));C,y=build(n,families[j%len(families)],r);neg=np.flatnonzero(y==0);k=min(len(neg),int(r.integers(1,9)));H=[0]+r.choice(neg[neg!=0],k-1,replace=False).tolist();I=np.setdiff1d(np.arange(len(C)),H);hit=y[I]!=0;cost=np.zeros(len(I));pos=np.flatnonzero(y)
  for ii,i in enumerate(I):
   if hit[ii]:cost[ii]=min(int(q).bit_count() for q in pos if q&i==q)
  X.append(feat(C,H,I));Y.append(hit);Z.append(cost)
 return t.from_numpy(np.concatenate(X)),t.from_numpy(np.concatenate(Y).astype(np.float32)),t.from_numpy(np.concatenate(Z).astype(np.float32))
def traincost(seed,X,Y,Z,folder):
 folder.mkdir(exist_ok=False);t.manual_seed(seed);g=t.Generator().manual_seed(seed+1000);net=CostNet();o=t.optim.AdamW(net.parameters(),lr=.002);parent='training_seed_53021'
 for step in range(1001):
  if step%100==0:
   p=folder/f's{step:04}.pt';t.save(dict(model=net.state_dict(),optimizer=o.state_dict(),rng=g.get_state(),torch_rng=t.get_rng_state(),step=step,parent=parent,seed=seed),p);parent=sha(p)
  if step==1000:break
  I=t.randint(len(X),(256,),generator=g);a,b=net(X[I]).T;l=F.binary_cross_entropy_with_logits(a,Y[I]);hit=Y[I]>0
  if hit.any():l=l+.1*F.mse_loss(b[hit],Z[I][hit])
  o.zero_grad();l.backward();o.step()
 return net

def search(C,y,kind,seed,net=None,pen=0,full=False):
 rng=np.random.default_rng(seed);H=[0];D={0:0};N=len(C)
 for j in range(16):
  I=np.setdiff1d(np.arange(N),H)
  if j==0 and full:i=N-1
  elif kind=='random':i=int(rng.choice(I))
  else:
   X=features(C,H,I)
   if kind=='farthest':s=X[:,3]+.001*X[:,5]
   elif kind=='old':
    with t.no_grad():s=net(t.from_numpy(X)).flatten().numpy()
   else:
    with t.no_grad():a,b=net(t.from_numpy(feat(C,H,I))).T;s=(a-pen*b).numpy()
   i=int(I[rng.choice(np.flatnonzero(s==s.max()))])
  D[i]=int(y[i]);H.append(i)
  if D[i]!=0:return i,D
 return None,D

def main():
 X,Y,Z=data();print('training',len(X),flush=True);old=[];new=[]
 for s in [53011,53012]:
  n=Selector();n.load_state_dict(t.load(ROOT/f'net_{s}/s1000.pt',weights_only=True)['model']);old.append(n.eval())
 for s in [53031,53032]:new.append(traincost(s,X,Y,Z,ROOT/f'cost_{s}').eval())
 (ROOT/'COST_LOCK.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('cost_*/*.pt')},indent=2))
 rows=[];r=np.random.default_rng(53921);ix=0
 for n in [8,10]:
  for f in ['add_cancel','product','redundant','opposing_products','inhibited_product','lookup','impossible','nested']:
   for j in range(24):
    C,y=nested(n,r) if f=='nested' else build(n,f,r);ix+=1;exists=bool(np.any(y));globalmin=min((int(i).bit_count() for i in np.flatnonzero(y)),default=-1)
    methods=[('random','random',None,0,False),('full_random','random',None,0,True),('full_farthest','farthest',None,0,True)]+[(f'full_old_{s}','old',net,0,True) for s,net in zip([53011,53012],old)]+[(f'new_{s}_p{p}_f{int(full)}','new',net,p,full) for s,net in zip([53031,53032],new) for p in [0,.5] for full in [False,True]]
    for name,kind,net,pen,full in methods+[('exhaustive','exhaustive',None,0,False)]:
     w,D=first(C,y,'exhaustive',54200+ix) if kind=='exhaustive' else search(C,y,kind,54200+ix,net,pen,full);found=w is not None;sc=len(D);w,D,_=minimize(w,D,y);sz=w.bit_count() if found else -1;rows.append(dict(task=ix,n=n,family=f,method=name,exists=int(exists),found=int(found),search_calls=sc,total_calls=len(D),loss64=len(D)+64*int(exists and not found),size=sz,globalmin=globalmin,globally_minimal=int(found and sz==globalmin)))
   print(n,f,'done',flush=True)
 with open(ROOT/'portfolio_scores.csv','w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
