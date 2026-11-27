# S12 is a read-only dependency, use exact source and model files.
import sys,os,json,csv,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;P=HERE.parent/'nri_G05_session12';S11=HERE.parent/'nri_G05_session11'
sys.dont_write_bytecode=True;sys.path.insert(0,str(P))
from protected import *
import observe as ob
ob.MF=setzero(folds(39,53111),39);ob.CF=setzero(folds(117,53112),117)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Oracle:
 def __init__(self,z):
  self.z=z;self.allow=t.zeros(9126,dtype=t.bool);self.allow[key(aslat(z))]=True;self.cache={};self.rows=0;self.unique=0
 def lat(self,D):
  a={f:self.z[D[:,i],i] for i,f in enumerate(FIELDS)};a['order']=self.z[D[:,5],5];return a
 def valid(self,D):return self.allow[key(self.lat(D))]
 def call(self,D):
  a=self.lat(D);k=key(a);assert self.allow[k].all();self.rows+=len(k);idx=[];miss=[]
  for i,v in enumerate(k.tolist()):
   if v not in self.cache:self.cache[v]=None;idx.append(i);miss.append(v)
  if idx:
   y=T.build({f:q[t.tensor(idx)] for f,q in a.items()}).y
   for v,w in zip(miss,y.tolist()):self.cache[v]=w
  self.unique=len(self.cache);return t.tensor([self.cache[v] for v in k.tolist()])

def design(r,I,site):
 D=t.full((len(I),6),int(r),dtype=t.long);D[:,site]=I;return D

def choose(oracle,search,zero,trial):
 g=t.Generator().manual_seed(53190+trial);zref=zero[t.randint(len(zero),(1,),generator=g)[0]];candidates=t.cat([zref.reshape(1),search[t.randperm(len(search),generator=g)[:15]]]);pan=[];chosen=None
 for j,r in enumerate(candidates):
  d0=t.full((1,6),int(r),dtype=t.long);y=oracle.call(d0);counts=[]
  for site in range(5):
   D=design(r,search,site);D=D[oracle.valid(D)];D=D[t.randperm(len(D),generator=g)[:8]];out=oracle.call(D);counts.append(int(t.cat([y,out]).unique().numel()))
  pan.append(dict(trial=trial,candidate=j,ref=int(r),output=int(y[0]),coverage=sum(v>=2 for v in counts),distinct=counts,rows=oracle.rows,unique=oracle.unique))
  if j>=1 and all(v>=2 for v in counts):chosen=j;break
 if chosen is None:chosen=max(range(len(pan)),key=lambda i:(pan[i]['coverage'],sum(pan[i]['distinct']),-i))
 return pan,chosen

def main():
 allpan=[];refs=[];scores=[];locks={}
 for s,mode,step,p in paths():
  if step!=6000:continue
  name=f'{s}_{mode}';z=grid(s);a=aslat(z);mf,cf=split(a);search=t.where((mf!=0)&(cf!=0))[0];reserve=t.where((mf==0)|(cf==0))[0]
  # only external stress setup has zero-conditioned sampling. Its label cost not treated as free discovery.
  Y=T.build(a).y;zero=search[Y[search]==0];assert len(zero)>0
  selections=[]
  for trial in range(12):
   oracle=Oracle(z);pan,i=choose(oracle,search,zero,trial);allpan.extend(dict(backbone=name,**v) for v in pan);refs.append(dict(backbone=name,trial=trial,zero_ref=pan[0]['ref'],random_ref=pan[1]['ref'],selected_ref=pan[i]['ref'],selected_index=i,zero_coverage=pan[0]['coverage'],random_coverage=pan[1]['coverage'],selected_coverage=pan[i]['coverage'],rows=pan[-1]['rows'],unique=pan[-1]['unique']));selections.append((pan,i))
  # Lock before evaluating head behavior. Use first trial only for full audit.
  pan,i=selections[0];(HERE/f'reference_lock_{name}.json').write_text(json.dumps(selections));
  m=TinyTyped('mass_only',**MODEL_KW);locks[str(p)]=digest(p);m.load_state_dict(t.load(p,weights_only=True)['model']);m.eval().requires_grad_(False);raw=views(features(m,z),'early');N=len(z);D0=t.arange(N)[:,None].expand(-1,5)
  heads=[]
  for f in sorted((P/'compiled').glob(name+'_*.pt')):heads.append(('S12',f))
  for f in sorted((S11/'compiled'/name).glob('conditioned_*.pt')):heads.append(('S11',f))
  for stage,f in heads:
   locks[str(f)]=digest(f);ck=t.load(f,weights_only=True);net=Relative(1) if stage=='S12' else Direct(0);net.load_state_dict(ck['model']);net.eval();_,_,ordinary=infer(net,raw,D0);good=ordinary==Y
   for strategy,j in [('zero',0),('random',1),('diversity',i)]:
    r=pan[j]['ref'];oracle=Oracle(z);seen=set();err=set();masked=set();counts=[];total=bad=masked_n=masked_bad=0;zero_n=zero_bad=0
    for site in range(5):
     D=design(r,reserve,site);D=D[oracle.valid(D)];yt=oracle.call(D);_,_,yp=infer(net,raw,D[:,:5]);ok=yp==yt;donor=D[:,site];masks=(Y[donor]==0)&good[donor]&(yt!=0);zeros=yt==0;counts.append(int((~ok).sum()));total+=len(D);bad+=int((~ok).sum());masked_n+=int(masks.sum());masked_bad+=int((masks&~ok).sum());zero_n+=int(zeros.sum());zero_bad+=int((zeros&~ok).sum());seen.update(donor.tolist());err.update(donor[~ok].tolist());masked.update(donor[masks&~ok].tolist())
    scores.append(dict(backbone=name,stage=stage,model=f.stem,strategy=strategy,reference=r,reference_correct=int(good[r]),ref_oracle=int(Y[r]),pilot_coverage=pan[j]['coverage'],ordinary_n=N,ordinary_errors=int((~good).sum()),reserve_n=len(reserve),swap_n=total,swap_errors=bad,masked_n=masked_n,masked_errors=masked_bad,zero_n=zero_n,zero_errors=zero_bad,unique_donor_errors=len(err),unique_masked_donor_errors=len(masked),site_errors=';'.join(map(str,counts)),unique_oracle=oracle.unique))
  print(name,'heads',len(heads),'refs',refs[-1],flush=True)
 for p,h in locks.items():assert digest(Path(p))==h
 for file,rows in [('reference_pilots.csv',allpan),('reference_selection.csv',refs),('reference_scores.csv',scores)]:
  with open(HERE/file,'w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (HERE/'PARENT_LOCK.json').write_text(json.dumps(locks,indent=2))
if __name__=='__main__':main()
