import itertools,json,csv,hashlib
from pathlib import Path
import numpy as np
import torch as t
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
t.set_num_threads(1);t.use_deterministic_algorithms(True)
KNOWN=['product','cancellation','branch','saturation','redundancy']; FAMILIES=KNOWN+['parity','two_clause','lookup','impossible']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cube(n):return ((np.arange(2**n)[:,None]>>np.arange(n))&1).astype(np.int8)
def task(n,f,rng):
 C=cube(n);k=int(rng.integers(2,n+1));ix=rng.choice(n,k,replace=False);pol=rng.integers(0,2,k);h=(C[:,ix]==pol).sum(1);q=int(rng.integers(1,k+1));a,b=sorted(rng.choice(6,2,replace=False));base=int(rng.integers(-5,6))+C@rng.integers(-3,4,n)
 if f=='product':fun=lambda z:base+z*(h==k)
 elif f=='cancellation':fun=lambda z:base+z-z*(h==0)
 elif f=='branch':fun=lambda z:base+np.where(h>=q,z,0)
 elif f=='saturation':fun=lambda z:base+np.minimum(z+6*np.maximum(0,q-h),6)
 elif f=='redundancy':fun=lambda z:base+np.maximum(z,6*(h<q))
 elif f=='parity':fun=lambda z:base+z*((h%2)==(q%2))
 elif f=='two_clause':
  j=int(rng.integers(1,k));gate=((C[:,ix[:j]]==pol[:j]).all(1)|(C[:,ix[j:]]==pol[j:]).all(1));fun=lambda z:base+z*gate
 elif f=='lookup':
  gate=np.zeros(len(C),bool);gate[rng.choice(len(C),int(rng.choice([1,2,4,8])),replace=False)]=True;fun=lambda z:base+z*gate
 else:fun=lambda z:base+z-z
 A,B=fun(a),fun(b);y=A!=B;assert (~y).any();return C,A.astype(np.int64),B.astype(np.int64),y

def features(C,H,I):
 Z=C[I];P=C[H];d=(Z[:,None,:]!=P[None,:,:]).mean(-1);p=P.mean(0);m=np.where(Z==1,1-p,p)
 return np.stack([np.full(len(Z),len(H)/len(C)),np.full(len(Z),np.log2(len(H)+1)/5),d[:,0],d.min(1),d.max(1),d.mean(1),d.std(1),m.min(1),m.max(1),m.mean(1),m.std(1),(m==0).mean(1),(m==1).mean(1),np.full(len(Z),C.shape[1]/8)],1).astype(np.float32)
class Selector(nn.Sequential):
 def __init__(self):super().__init__(nn.Linear(14,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
def training():
 r=np.random.default_rng(53001);xs=[];ys=[]
 for i in range(6000):
  C,A,B,y=task(int(r.choice([4,5])),KNOWN[i%5],r);neg=np.flatnonzero(~y);H=r.choice(neg,min(len(neg),int(r.integers(1,9))),replace=False);I=np.setdiff1d(np.arange(len(C)),H);xs.append(features(C,H,I));ys.append(y[I])
 return t.from_numpy(np.concatenate(xs)),t.from_numpy(np.concatenate(ys).astype('float32'))
def train(seed,X,Y,folder):
 folder.mkdir(exist_ok=False);t.manual_seed(seed);g=t.Generator().manual_seed(seed+1000);net=Selector();o=t.optim.AdamW(net.parameters(),lr=.002);parent='training_seed_53001';rows=[]
 for step in range(1001):
  if step%100==0:
   p=folder/f's{step:04}.pt';t.save(dict(model=net.state_dict(),optimizer=o.state_dict(),rng=g.get_state(),torch_rng=t.get_rng_state(),step=step,parent=parent,seed=seed),p);parent=sha(p)
  if step==1000:break
  I=t.randint(len(X),(256,),generator=g);l=F.binary_cross_entropy_with_logits(net(X[I]).flatten(),Y[I]);o.zero_grad();l.backward();o.step();rows.append(float(l.detach()))
 return net,rows

def threshold_prior(C):
 arr=[]
 for ss in itertools.product([-1,0,1],repeat=C.shape[1]):
  w=np.asarray(ss);k=np.count_nonzero(w);v=C@w+np.count_nonzero(w==-1)
  arr.extend(v>=q for q in range(1,k+1))
 return np.stack(arr) # each signed threshold is a prior hypothesis

def eval_one(C,A,B,start,method,rng,net=None,prior=None):
 H=[start];obs=[(start,int(A[start]),int(B[start]))];mask=np.ones(len(C),bool);mask[start]=False
 for _ in range(16):
  I=np.flatnonzero(mask)
  if not len(I):break
  if method=='random':i=int(rng.choice(I))
  else:
   feat=features(C,H,I)
   if method=='farthest':score=feat[:,3]+1e-3*feat[:,5]
   elif method=='bayes':
    p=prior[~prior[:,H].any(1)];score=p[:,I].mean(0) if len(p) else feat[:,3]+1e-3*feat[:,5]
   else:
    with t.no_grad():score=net(t.from_numpy(feat)).flatten().numpy()
   i=int(I[rng.choice(np.flatnonzero(score==score.max()))])
  aa,bb=int(A[i]),int(B[i]);obs.append((i,aa,bb));H.append(i);mask[i]=False
  if aa!=bb:return 1,obs
 return 0,obs

def main():
 X,Y=training();print('train rows',len(X),flush=True);nets=[]
 for s in [53011,53012]:
  net,l=train(s,X,Y,ROOT/f'net_{s}');nets.append(net);print(s,l[-1],flush=True)
 def eq(a,b):
  if isinstance(a,t.Tensor):return t.equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
  return a==b
 replay=[]
 for s in [53011,53012]:
  train(s,X,Y,ROOT/f'replay_{s}')
  for i in range(0,1001,100):
   a=t.load(ROOT/f'net_{s}/s{i:04}.pt',weights_only=True);b=t.load(ROOT/f'replay_{s}/s{i:04}.pt',weights_only=True);assert all(eq(a[k],b[k]) for k in ['model','optimizer','rng','torch_rng']);replay.append(dict(seed=s,step=i,bitwise=True))
 (ROOT/'replays.json').write_text(json.dumps(replay));(ROOT/'TRAIN_LOCK.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in ROOT.glob('net_*/*.pt')},indent=2))
 rows=[];traces=[];r=np.random.default_rng(53901);taskid=0
 for n in [6,8]:
  C=cube(n);prior=threshold_prior(C);print('prior',n,prior.shape,flush=True)
  for fam in FAMILIES:
   for trial in range(24):
    C,A,B,y=task(n,fam,r);start=int(r.choice(np.flatnonzero(~y)));taskid+=1
    for method,net in [('random',None),('farthest',None),('bayes',None),('neural_53011',nets[0]),('neural_53012',nets[1])]:
     hit,obs=eval_one(C,A,B,start,'neural' if net is not None else method,np.random.default_rng(54000+taskid),net,prior);assert not hit or obs[-1][1]!=obs[-1][2];rows.append(dict(task=taskid,n=n,family=fam,method=method,available=int(y.sum()),found=hit,pairs=len(obs),scalar_calls=2*len(obs)));traces.append(dict(task=taskid,method=method,observations=obs))
   print(n,fam,'done',flush=True)
 with open(ROOT/'scores.csv','w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (ROOT/'traces.json').write_text(json.dumps(traces,separators=(',',':')))
if __name__=='__main__':main()
