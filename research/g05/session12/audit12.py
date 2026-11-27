from protected import *
import observe as ob, anchor as an, gauge as ga, precondition as pc, protected as pr
from zipfile import ZipFile
from study import eq
from collections import defaultdict

def verify():
 prior=ROOT.parent/'nri_G05_session11_semantic_invariance_core.zip';src=[]
 with ZipFile(prior) as z:
  for f in sorted(ROOT.glob('sources/**/*')):
   if f.is_file() and '__pycache__' not in f.parts:
    assert f.read_bytes()==z.read('nri_G05_session11/'+str(f.relative_to(ROOT))),f;src.append(f)
  for name in ['run.py','direct.py','condition.py']:assert (ROOT/name).read_bytes()==z.read('nri_G05_session11/'+name);src.append(ROOT/name)
 parents={f'{s}_{mode}':p for s,mode,step,p in paths() if step==6000};total=0;locks=0;matched=0;stages={};rows=[];queries=[]
 for stage in ['runs12','anchor','gauge','precondition_invalid','precondition','protected']:
  ns=0;runs=0;upd=0
  for name,p in parents.items():
   d=ROOT/stage/name;lock=json.loads((d/'LOCKED.json').read_text());assert all(sha(d/k)==v for k,v in lock.items());locks+=len(lock);ph=sha(p);td=t.load(d/'train_domain.pt',weights_only=True);assert td['parent_sha256']==ph;z=td['z'];a=aslat(z);s=int(name.split('_')[0]);a0=aslat(grid(s));allow=t.zeros(len(ALL),dtype=t.bool);allow[key(a0)]=True;assert allow[key(a)].all();rng={};M,A,S=T.canon(a);C=A*S%13;N=len(M);J=t.where((M*C%13)!=0)[0];P=[t.where((M!=0)&(C==0))[0],t.where((M==0)&(C!=0))[0],t.where(M!=0)[0],t.where(C!=0)[0]]
   for sub in sorted(d.iterdir()):
    if not sub.is_dir():continue
    fs=sorted(sub.glob('h*.pt'));parent=ph;runs+=1;upd+=int(fs[-1].stem[1:]);seed=int(sub.name.split('_')[-1]);arm=sub.name.rsplit('_',1)[0]
    for f in fs:
     c=t.load(f,weights_only=True);assert c['parent_sha256']==parent and c['backbone_sha256']==ph and c['step']==int(f.stem[1:]);parent=sha(f);ns+=1;rk=(seed,c['step']);
     if rk in rng:assert t.equal(c['rng'],rng[rk]);matched+=1
     else:rng[rk]=c['rng']
    # Reproduce the sampled oracle target indices for the first three stages, no holdout training.
    if stage in ['runs12','anchor','gauge']:
     gg=t.Generator().manual_seed(seed+100000);forbidden=0;count=0
     for k in range(int(fs[-1].stem[1:])):
      if stage=='runs12':
       D=t.randint(N,(128,5),generator=gg);base=D.clone();pick=[pool[t.randint(len(pool),(32,),generator=gg)] for pool in P]
       if arm in ['ordinary','factor']:D=base[:,0,None].expand(-1,5).clone()
       elif arm=='role':D=base[:,[0,0,2,2,2]].clone()
       elif arm=='expose':D[:32,0]=pick[0];D[:32,2]=pick[3];D[32:64,2]=pick[1];D[32:64,0]=pick[2]
      elif stage=='anchor':
       i=t.randint(N,(128,),generator=gg);j=J[t.randint(len(J),(1,),generator=gg)][0];D=j.expand(128,5).clone();D[:,k%5]=i
       if arm in ['ordinary','factor']:D=i[:,None].expand(-1,5).clone()
      else:
       i=t.randint(N,(128,),generator=gg);j=td['reference'];D=t.full((128,5),j,dtype=t.long);D[:,k%5]=i
      lat={f:a[f][D[:,i]] for i,f in enumerate(FIELDS)};lat['order']=a['order'][D[:,0]];forbidden+=int((~allow[key(lat)]).sum());count+=len(D)
     assert t.equal(gg.get_state(),c['rng']);queries.append(dict(stage=stage,backbone=name,arm=arm,seed=seed,target_rows=count,forbidden=forbidden,unique_prompts=-1))
   if stage in ['precondition','precondition_invalid','protected']:
    j=td['reference'];D=t.full((N,5),j,dtype=t.long);ks=[]
    for site in range(5):
     d0=D.clone();d0[:,site]=t.arange(N);lat={f:a[f][d0[:,i]] for i,f in enumerate(FIELDS)};lat['order']=a['order'][j].expand(N);ks.append(key(lat))
    k=t.cat(ks);bad=int((~allow[k]).sum());queries.append(dict(stage=stage,backbone=name,arm='full_design',seed=0,target_rows=len(k)+1,forbidden=bad,unique_prompts=int(t.cat([k,key(a)[j:j+1]]).unique().numel())))
    if stage=='protected':
     assert bad==0
     for f in sorted(d.glob('*32_*.pt')):
      ck=t.load(f,weights_only=True);kk=t.cat([k[i*N+ix] for i,ix in enumerate(ck['indices'])]);assert allow[kk].all();queries.append(dict(stage=stage,backbone=name,arm=f.stem.rsplit('_',1)[0],seed=int(f.stem.split('_')[-1]),target_rows=ck['observations'],forbidden=0,unique_prompts=int(t.cat([kk,key(a)[j:j+1]]).unique().numel())))
  stages[stage]=dict(checkpoints=ns,runs=runs,updates=upd);total+=ns
 # audit fresh replay runs against every saved model/optimizer/RNG state
 for stage,arm,seed in [('runs12','expose',52101),('anchor','ratio',52201),('gauge','local_reg',52301),('precondition','white',52401),('protected','raw',52501),('protected','white',52501)]:
  name='32002_spontaneous';p=parents[name];ph=sha(p);d=ROOT/stage/name;td=t.load(d/'train_domain.pt',weights_only=True);a=aslat(td['z']);m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(p,weights_only=True)['model']);m.eval().requires_grad_(False);x=[apply(v,c) for v,c in zip(views(features(m,td['z']),'early'),td['transform'])];M,A,S=T.canon(a);C=A*S%13;ff=field_factors(a);root=ROOT/'replays'/stage/f'{arm}_{seed}'
  if stage=='runs12':ob.fit(x,a,ff,M,C,root,ph,seed,arm)
  elif stage=='anchor':an.fit2(x,ff,M,C,root,ph,seed,arm)
  elif stage=='gauge':ga.fit3(x,ff,M,C,a,root,ph,seed,arm,t.tensor(td['reference']))
  elif stage=='precondition':j=t.tensor(td['reference']);pc.fit4(pc.designs(x,ff,M,C,j),M[j]*C[j]%13,root,ph,seed,arm)
  else:
   j=t.tensor(td['reference']);allow=t.zeros(len(ALL),dtype=t.bool);allow[key(aslat(grid(32002)))]=True;pr.fit4(pr.designs(x,ff,M,C,j,a=a,allowed=allow),M[j]*C[j]%13,root,ph,seed,arm)
  for f in sorted(root.glob('h*.pt')):
   q=t.load(f,weights_only=True);c=t.load(d/f'{arm}_{seed}'/f.name,weights_only=True);ks=['model','optimizer','rng','torch_rng']+(['learned'] if 'learned' in c else []);assert all(eq(q[k],c[k]) for k in ks),(stage,arm,f);rows.append(dict(stage=stage,arm=arm,seed=seed,step=c['step'],bitwise=True))
 dumpcsv(ROOT/'replay_verification.csv',rows);dumpcsv(ROOT/'oracle_audit.csv',queries);(ROOT/'verification.json').write_text(json.dumps(dict(source_files=len(src),primary_checkpoints=total,locked_entries=locks,matched_rng=matched,bitwise_replays=len(rows),stages=stages),indent=2));print((ROOT/'verification.json').read_text())
if __name__=='__main__':verify()
