import sys,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sources/nri_G05_session09'))
from metric import *
FIELDS=['m','mu','a','au','fu']; NS=[13,3,13,3,3]; ROLES=[0,0,1,1,1]
ALL=t.tensor(list(itertools.product(range(13),range(3),range(13),range(3),range(3),range(2))))
def bad(a):
 return ((a['mu']==1)&t.isin(a['m'],t.tensor([6,9,10])))|((a['mu']==2)&t.isin(a['m'],t.tensor([2,4,6])))|((a['au']==1)&t.isin(a['a'],t.tensor([6,9,10])))|((a['au']==2)&t.isin(a['a'],t.tensor([2,4,6])))
def features(m,z):
 rr=[reps(m,aslat(b)) for b in z.split(512)];return {k:t.cat([v[k] for v in rr]) for k in rr[0]}
def views(r,st):
 if st=='early':return [r['earlyM'][:,:64],r['earlyM'][:,64:128],r['earlyC'][:,:64],r['earlyC'][:,64:128],r['earlyC'][:,128:192]]
 if st=='late':return [r['lateM'][:,:16]]*2+[r['lateC'][:,:48]]*3
 return [r['earlyM'][:,:128],r['earlyC']]
def norm(X,ms,rs):
 st=[(x[ms[j]].mean(0),x[ms[j]].std(0).clamp_min(1e-4)) for x,j in zip(X,rs)];return [(x-u)/v for x,(u,v) in zip(X,st)],st
class Phase(nn.Module):
 def __init__(self,base):
  super().__init__();self.v=nn.Parameter(t.rand(12)*6.283185307);self.u=nn.Parameter(t.rand(3,2)*6.283185307);self.register_buffer('angles',t.arange(12)*6.283185307/12)
  if base is not None:
   with t.no_grad():self.v.copy_(self.angles[base]);self.u.mul_(.05)
 def logits(self,ff):
  m,mu,a,au,fu=ff;u=F.pad(self.u,(1,0));z=[self.v[(m-1).clamp_min(0)]+u[0,mu],self.v[(a-1).clamp_min(0)]+u[1,au]+u[2,fu]];return [10*t.cos(x[:,None]-self.angles) for x in z]
 def predict(self,ff):return [t.where(v==0,0,INV[l.argmax(1)+1]) for v,l in zip([ff[0],ff[2]],self.logits(ff))]
def uniq(a,ix,fields):
 code=sum(a[k][ix]*b for k,b in fields);o=t.argsort(code,stable=True);q=code[o];return ix[o[t.cat([t.ones(1,dtype=t.bool),q[1:]!=q[:-1]])]]
def fit_phase(a,ms,M,C,seed,kind,root,ph):
 im=uniq(a,t.where(ms[0]&(a['m']!=0))[0],[('m',3),('mu',1)]);ic=uniq(a,t.where(ms[1]&(a['a']!=0))[0],[('a',9),('au',3),('fu',1)])
 base=t.tensor([int(G[M[t.where(ms[0]&(a['m']==v)&(a['mu']==0))[0][0]]]-1) for v in range(1,13)])
 t.manual_seed(seed);p=Phase(base if kind=='inferred' else None);o=t.optim.AdamW(p.parameters(),lr=.06,weight_decay=0);root.mkdir(parents=True);parent=ph
 for k in range(401):
  if k%100==0:
   dst=root/f'h{k:04}.pt';t.save(dict(model=p.state_dict(),optimizer=o.state_dict(),torch_rng=t.get_rng_state(),step=k,parent_sha256=parent,im=im,ic=ic,kind=kind,seed=seed),dst);parent=sha(dst)
  if k==400:break
  l=F.cross_entropy(p.logits([a[f][im] for f in FIELDS])[0],G[M[im]]-1)+F.cross_entropy(p.logits([a[f][ic] for f in FIELDS])[1],G[C[ic]]-1);o.zero_grad();l.backward();o.step()
 return p

def fit_heads(X,stats,targets,ms,rs,ns,seed,kind,root,ph):
 t.manual_seed(seed);hs=nn.ModuleList([nn.Linear(x.shape[1],n) if kind!='flat' else nn.Sequential(nn.Linear(x.shape[1],32),nn.ReLU(),nn.Linear(32,n)) for x,n in zip(X,ns)]);o=t.optim.AdamW(hs.parameters(),lr=.003);g=t.Generator().manual_seed(seed+100000);I=[t.where(x)[0] for x in ms];root.mkdir(parents=True);parent=ph
 for k in range(401):
  if k%100==0:
   dst=root/f'h{k:04}.pt';t.save(dict(model=hs.state_dict(),optimizer=o.state_dict(),rng=g.get_state(),torch_rng=t.get_rng_state(),step=k,parent_sha256=parent,normalization=stats,seed=seed,kind=kind),dst);parent=sha(dst)
  if k==400:break
  js=[i[t.randint(len(i),(128,),generator=g)] for i in I];l=sum(F.cross_entropy(h(x[js[j]]),y[js[j]]) for h,x,y,j in zip(hs,X,targets,rs));o.zero_grad();l.backward();o.step()
 return hs

def sparse(a,ms,M,C):
 vals=t.tensor([int(M[t.where(ms[0]&(a['m']==v)&(a['mu']==0))[0][0]]) for v in range(13)]);inv=t.tensor([0]+[pow(i,-1,13) for i in range(1,13)]);sh=[]
 for role,field,unit,extra,fac in [(0,'m','mu',None,M),(1,'a','au',('fu',0),C),(1,'a','fu',('au',0),C)]:
  u=[1]
  for j in [1,2]:
   mask=ms[role]&(a[field]!=0)&(a[unit]==j)
   if extra:mask&=a[extra[0]]==extra[1]
   i=t.where(mask)[0][0];u.append(int(fac[i]*inv[vals[a[field][i]]]%13))
  sh.append(t.tensor(u))
 return vals,sh

def run(s,mode,p):
 root=ROOT/'runs'/f'{s}_{mode}';root.mkdir(parents=True,exist_ok=False);ph=sha(p);m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(p,weights_only=True)['model']);m.eval().requires_grad_(False)
 z=grid(s);z=z[~bad(aslat(z))];a=aslat(z);r=features(m,z);M,A,S=T.canon(a);C=A*S%13;ms=newmasks(a)[0];t.save(dict(z=z,masks=ms,parent_sha256=ph),root/'train_domain.pt');models={};stats={};phases={};protos=[]
 for st in ['early','late','flat']:
  rs=ROLES if st!='flat' else [0,1];X,stt=norm(views(r,st),ms,rs);stats[st]=stt
  for seed in [51101,51102]:models[(st,seed)]=fit_heads(X,stt,[a[f] for f in FIELDS] if st!='flat' else [M,C],ms,rs,NS if st!='flat' else [13,13],seed,st,root/f'{st}_{seed}',ph)
  if st=='early':
   for x,y,j,n in zip(X,[a[f] for f in FIELDS],ROLES,NS):protos.append(t.stack([x[ms[j]&(y==v)].mean(0) for v in range(n)]))
 for kind in ['random','inferred']:
  for seed in [51101,51102]:phases[(kind,seed)]=fit_phase(a,ms,M,C,seed,kind,root/f'phase_{kind}_{seed}',ph)
 vals,sh=sparse(a,ms,M,C);t.save(dict(vals=vals,shifts=sh,protos=protos,stats=stats),root/'controls.pt');lock={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*.pt'))};(root/'LOCKED.json').write_text(json.dumps(lock,indent=2));assert sha(p)==ph
 # locked before full qualification
 a=aslat(ALL);r=features(m,ALL);M,A,S=T.canon(a);C=A*S%13;Y=M*C%13;orig=t.zeros(9126,dtype=t.bool);orig[key(aslat(grid(s)))]=True;orig=orig[key(a)];nb=bad(a);held=newmasks(a)[2];splits={'context':orig&~nb&held[0]&held[1],'repair_only':orig&nb,'historical_unseen':~orig,'all_familiar':orig};H=r['H'];L=native(m,H);q=group(L);preds={};fields={}
 with t.no_grad():
  for (st,seed),net in models.items():
   X=[(x-u)/v for x,(u,v) in zip(views(r,st),stats[st])];pr=[h(x).argmax(1) for h,x in zip(net,X)]
   if st=='flat':preds[f'flat_{seed}']=pr
   else:fields[f'{st}_{seed}']=pr
  X=[(x-u)/v for x,(u,v) in zip(views(r,'early'),stats['early'])];fields['early_proto']=[t.cdist(x,c).argmin(1) for x,c in zip(X,protos)];fields['oracle']=[a[f] for f in FIELDS]
  for name,ff in fields.items():
   for (kind,seed),p2 in phases.items():
    if name not in ['oracle','early_proto'] and not name.endswith(str(seed)):continue
    preds[f'{name}_phase_{kind}_{seed}']=p2.predict(ff)
   mm,mu,aa,au,fu=ff;preds[f'{name}_sparse_exact']=[vals[mm]*sh[0][mu]%13,vals[aa]*sh[1][au]*sh[2][fu]%13]
 rows=[];fl=[]
 for split,mask in splits.items():
  i=t.where(mask)[0];g=t.Generator().manual_seed(51190);nz=i[(M[i]!=0)&(C[i]!=0)];u=nz[t.randint(len(nz),(4096,),generator=g)];v=nz[t.randint(len(nz),(4096,),generator=g)];y=M[u]*C[v]%13;jl=native(m,t.cat([H[u,:16],H[v,16:]],1));jq=group(jl)
  for name,ff in fields.items():fl.append(dict(seed=s,mode=mode,split=split,method=name,n=len(i),**{f:int((x[i]==a[f][i]).sum()) for f,x in zip(FIELDS,ff)},all_fields=int(t.stack([x[i]==a[f][i] for f,x in zip(FIELDS,ff)]).all(0).sum())))
  rows.append(dict(seed=s,mode=mode,split=split,method='native',n=len(i),ordinary=int((L[i].argmax(1)==Y[i]).sum()),joint_n=len(y),joint=int((jl.argmax(1)==y).sum()),group_ordinary=int((q[i]==Q[Y[i]]).sum()),group_joint=int((jq==Q[y]).sum()),repair_ordinary=-1,repair_joint=-1,abstain_ordinary=-1,abstain_joint=-1,m_correct=-1,c_correct=-1))
  for name,(mm,cc) in preds.items():
   op=mm[i]*cc[i]%13;jp=mm[u]*cc[v]%13;ro=CRT[q[i],Q4[op]];rj=CRT[jq,Q4[jp]]
   rows.append(dict(seed=s,mode=mode,split=split,method=name,n=len(i),ordinary=int((op==Y[i]).sum()),joint_n=len(y),joint=int((jp==y).sum()),group_ordinary=int((Q[op]==Q[Y[i]]).sum()),group_joint=int((Q[jp]==Q[y]).sum()),repair_ordinary=int((ro==Y[i]).sum()),repair_joint=int((rj==y).sum()),abstain_ordinary=int((ro<0).sum()),abstain_joint=int((rj<0).sum()),m_correct=int((mm[i]==M[i]).sum()),c_correct=int((cc[i]==C[i]).sum())))
 dumpcsv(root/'scores.csv',rows);dumpcsv(root/'fields.csv',fl);assert sha(p)==ph;print(s,mode,'done',flush=True)
if __name__=='__main__':
 for s,mode,step,p in paths():
  if step==6000 and (len(sys.argv)==1 or str(s)+'_'+mode==sys.argv[1]):run(s,mode,p)
