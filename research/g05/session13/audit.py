from portfolio import *
import pandas as pd

def eq(a,b):
 if isinstance(a,t.Tensor):return t.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b

def main():
 for f in ['TRAIN_LOCK.json','COST_LOCK.json','PARENT_LOCK.json']:
  x=json.loads((ROOT/f).read_text());assert all(sha(ROOT/p)==h for p,h in x.items());print(f,len(x))
 X,Y,Z=data();replay=[]
 for s in [53031,53032]:
  traincost(s,X,Y,Z,ROOT/f'cost_replay_{s}')
  for i in range(0,1001,100):
   a=t.load(ROOT/f'cost_{s}/s{i:04}.pt',weights_only=True);b=t.load(ROOT/f'cost_replay_{s}/s{i:04}.pt',weights_only=True);assert all(eq(a[k],b[k]) for k in ['model','optimizer','rng','torch_rng']);replay.append(dict(seed=s,step=i,bitwise=True))
 (ROOT/'cost_replays.json').write_text(json.dumps(replay));a=pd.read_csv(ROOT/'scores.csv');b=pd.read_csv(ROOT/'coalition_scores.csv');c=pd.read_csv(ROOT/'portfolio_scores.csv');z=pd.read_csv(ROOT/'reference_scores.csv');r=pd.read_csv(ROOT/'reference_selection.csv');assert len(a)==2160 and len(b)==2240 and len(c)==5376 and len(z)==165 and len(r)==60;assert not a[a.available==0].found.any();assert not b[b.exists==0].found.any();assert not c[c.exists==0].found.any();assert (r.selected_coverage==5).all()
 for f,prefix in [('TRAIN_LOCK.json','net'),('COST_LOCK.json','cost')]:
  x=json.loads((ROOT/f).read_text());
  for s in [53011,53012] if prefix=='net' else [53031,53032]:
   pp=sorted((ROOT/f'{prefix}_{s}').glob('s*.pt'));assert len(pp)==11
   for u,v in zip(pp,pp[1:]):assert t.load(v,weights_only=True)['parent']==sha(u)
 # All saved trace queries regenerated from registered seeds; no witness claim without observed unequal output.
 aa=json.loads((ROOT/'traces.json').read_text());rr=np.random.default_rng(53901);tables={};idx=0
 for n in [6,8]:
  for f in FAMILIES:
   for j in range(24):
    C,A,B,y=task(n,f,rr);start=int(rr.choice(np.flatnonzero(~y)));idx+=1;tables[idx]=(A,B,start)
 for rec in aa:
  A,B,start=tables[rec['task']];obs=rec['observations'];assert obs[0][0]==start and len({i for i,*_ in obs})==len(obs);assert all(A[i]==a and B[i]==b for i,a,b in obs)
 bb=json.loads((ROOT/'coalition_traces.json').read_text());rr=np.random.default_rng(53911);tables={};idx=0
 for n in [6,8]:
  for f in ['add_cancel','product','redundant','opposing_products','inhibited_product','lookup','impossible']:
   for j in range(20):
    C,y=build(n,f,rr);idx+=1;tables[idx]=y
 for rec in bb:
  y=tables[rec['task']];D={int(i):v for i,v in rec['observations'].items()};assert all(y[i]==v for i,v in D.items());w=rec['witness']
  if w is not None:assert D[w]!=0 and all(i in D and D[i]==0 for i in range(len(y)) if i!=w and i&w==i)
 print('verified',len(aa),len(bb),'traces',len(replay),'cost replay checkpoints')
if __name__=='__main__':main()
