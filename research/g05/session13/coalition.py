from observer import *

def build(n,f,r):
 C=cube(n);p=r.permutation(n);k=int(r.integers(2,n//2+1));I=p[:k];J=p[k:k+int(r.integers(2,n-k+1))];u=C[:,I].prod(1);v=C[:,J].prod(1)
 if f=='add_cancel':
  w=int(r.integers(1,6)); y=w*(C[:,p[0]]-C[:,p[1]])
 elif f=='product':y=u
 elif f=='redundant':y=1-np.maximum(1-u,1-v) # all alternative paths must switch
 elif f=='opposing_products':y=u-v
 elif f=='inhibited_product':y=u*(1-v)
 elif f=='lookup':
  y=np.zeros(len(C),dtype=int);y[r.choice(np.arange(1,len(C)),int(r.choice([1,2,4,8])),replace=False)]=1
 elif f=='impossible':y=np.zeros(len(C),dtype=int)
 else:raise ValueError(f)
 assert y[0]==0;return C,y.astype(int)

def first(C,y,method,seed,net=None,prior=None):
 if method in ['random','farthest','bayes','neural']:
  found,obs=eval_one(C,np.zeros_like(y),y,0,method,np.random.default_rng(seed),net,prior);return (obs[-1][0] if found else None),{i:b for i,a,b in obs}
 rng=np.random.default_rng(seed);n=C.shape[1];single=[1<<j for j in rng.permutation(n)]
 if method=='singletons':I=single
 elif method=='full_single':
  init=[len(C)-1]+single;I=init+list(rng.permutation(np.setdiff1d(np.arange(1,len(C)),init)))[:16-len(init)]
 elif method=='exhaustive':I=sorted(range(1,len(C)),key=lambda i:i.bit_count())
 else:raise ValueError(method)
 D={0:0}
 for i in I:
  i=int(i);D[i]=int(y[i])
  if D[i]!=0:return i,D
 return None,D

def minimize(w,D,y):
 if w is None:return None,D,0
 def ask(i):
  if i not in D:D[i]=int(y[i])
  return D[i]
 orig=w;again=True
 while again:
  again=False
  for j in range(w.bit_length()):
   if (w>>j)&1 and ask(w^(1<<j))!=0:
    w^=1<<j;again=True;break
 # one-step deletion alone is not sufficient under nonmonotonicity. Audit all strict subsets in size order.
 subsets=[0];q=w
 while q:
  q=(q-1)&w
  if q:subsets.append(q)
 for q in sorted(subsets,key=lambda x:x.bit_count()):
  if ask(q)!=0:w=q;break
 assert y[w]!=0 and all(y[i]==0 for i in range(len(y)) if i!=w and i&w==i)
 return w,D,orig.bit_count()-w.bit_count()

def main():
 nets=[]
 for s in [53011,53012]:
  net=Selector();net.load_state_dict(t.load(ROOT/f'net_{s}/s1000.pt',weights_only=True)['model']);net.eval();nets.append(net)
 rows=[];traces=[];r=np.random.default_rng(53911);ix=0
 for n in [6,8]:
  prior=threshold_prior(cube(n))
  for family in ['add_cancel','product','redundant','opposing_products','inhibited_product','lookup','impossible']:
   for _ in range(20):
    C,y=build(n,family,r);exists=bool(np.any(y!=0));globalmin=min((int(i).bit_count() for i in np.flatnonzero(y)),default=-1);ix+=1
    for method,net in [('singletons',None),('full_single',None),('random',None),('farthest',None),('bayes',None),('neural_53011',nets[0]),('neural_53012',nets[1]),('exhaustive',None)]:
     w,D=first(C,y,'neural' if net is not None else method,54100+ix,net,prior);cost=len(D);found=w is not None;single=int(any(y[1<<j]!=0 for j in range(n)));full=int(y[-1]!=0);w,D,reduction=minimize(w,D,y);size=w.bit_count() if found else -1
     assert not found or exists and size>=globalmin
     rows.append(dict(task=ix,n=n,family=family,method=method,exists=int(exists),single_visible=single,full_visible=full,found=int(found),search_calls=cost,total_calls=len(D),size=size,globalmin=globalmin,globally_minimal=int(found and size==globalmin),inclusion_minimal=int(found),reduction=reduction));traces.append(dict(task=ix,method=method,witness=w,observations=D))
   print(n,family,'done',flush=True)
 with open(ROOT/'coalition_scores.csv','w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (ROOT/'coalition_traces.json').write_text(json.dumps(traces,separators=(',',':')))
if __name__=='__main__':main()
