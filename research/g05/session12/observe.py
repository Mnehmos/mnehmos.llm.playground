from condition import *
import hashlib
STEPS=800; B=128; SEEDS=[52101,52102]; ARMS=['ordinary','role','site','expose','factor']
# All splits constructed from the complete literal-context vocabulary, not the truncated grid.
def folds(n,seed):
 p=t.randperm(n,generator=t.Generator().manual_seed(seed));f=t.empty_like(p);f[p]=t.arange(n)%5;return f
MF=folds(39,52111); CF=folds(117,52112)
def split(a):
 m=MF[a['m']*3+a['mu']];c=CF[(a['a']*3+a['au'])*3+a['fu']];return m,c
# One product factor for each source field (uses documented task conversions, not a model target).
def field_factors(a):
 one=t.ones_like(a['m']);zero=t.zeros_like(one)
 def canon(m,mu,acc,au,fu):return T.canon(dict(m=m,mu=mu,a=acc,au=au,fu=fu,order=zero))
 _,_,_=canon(one,zero,one,zero,zero)
 mv,_,_=canon(a['m'],zero,one,zero,zero);mu,_,_=canon(one,a['mu'],one,zero,zero)
 _,av,_=canon(one,zero,a['a'],zero,zero);_,au,_=canon(one,zero,one,a['au'],zero);_,_,fu=canon(one,zero,one,zero,a['fu'])
 return [mv,mu,av,au,fu]
def target(ff,D):
 z=ff[0][D[:,0]]
 for i in range(1,5):z=z*ff[i][D[:,i]]%13
 return z
def output_loss(net,xx,y):
 _,z,p=net(xx);v=p[0]
 for w in p[1:]:v=mul(v,w)
 logits=v@net.code.T*10; nz=y!=0
 lognz=F.logsigmoid(-z[0])+F.logsigmoid(-z[1]); logzero=t.logaddexp(F.logsigmoid(z[0]),F.logsigmoid(-z[0])+F.logsigmoid(z[1]))
 return -t.where(nz,lognz,logzero).mean()+(F.cross_entropy(logits[nz],G[y[nz]]-1) if nz.any() else logits.sum()*0)
def factor_loss(net,xx,m,c):
 l,z,_=net(xx);loss=0
 for a,b,y in zip(l,z,[m,c]):
  nz=y!=0;loss+=F.binary_cross_entropy_with_logits(b,(~nz).float())+(F.cross_entropy(a[nz],G[y[nz]]-1) if nz.any() else a.sum()*0)
 return loss

def fit(x,a,ff,M,C,root,ph,seed,arm,replay=False):
 t.manual_seed(seed);net=Direct(0);opt=t.optim.AdamW(net.parameters(),lr=.003);g=t.Generator().manual_seed(seed+100000);N=len(M);root.mkdir(parents=True,exist_ok=False);parent=ph
 P=[t.where((M!=0)&(C==0))[0],t.where((M==0)&(C!=0))[0],t.where(M!=0)[0],t.where(C!=0)[0]]
 assert all(len(p)>0 for p in P);exposure=t.zeros(5,dtype=t.long);nonzero=0;losses=[]
 for step in range(STEPS+1):
  if step%100==0:
   dst=root/f'h{step:04}.pt';t.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=g.get_state(),torch_rng=t.get_rng_state(),step=step,parent_sha256=parent,backbone_sha256=ph,seed=seed,arm=arm,nonzero=nonzero,exposure=exposure.clone()),dst);parent=sha(dst)
  if step==STEPS:break
  D=t.randint(N,(B,5),generator=g);base=D.clone();pick=[p[t.randint(len(p),(B//4,),generator=g)] for p in P] # matched RNG even if not used
  if arm in ['ordinary','factor']:D=base[:,0,None].expand(-1,5).clone()
  elif arm=='role':D=base[:,[0,0,2,2,2]].clone()
  elif arm=='expose':
   D[:B//4,0]=pick[0];D[:B//4,2]=pick[3];D[B//4:B//2,2]=pick[1];D[B//4:B//2,0]=pick[2]
  y=target(ff,D);nz=y!=0;nonzero+=int(nz.sum());exposure[0]+=int((nz&(C[D[:,0]]==0)).sum());exposure[2]+=int((nz&(M[D[:,2]]==0)).sum());
  for i in [1,3,4]:exposure[i]+=int((nz&((M*C%13)[D[:,i]]==0)).sum())
  xx=[q[D[:,i]] for i,q in enumerate(x)]
  if arm=='factor':loss=factor_loss(net,xx,M[D[:,0]],C[D[:,2]])
  else:loss=output_loss(net,xx,y)
  opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
 (root/'training.json').write_text(json.dumps(dict(nonzero=nonzero,exposure=exposure.tolist(),last100=sum(losses[-100:])/100)))
 return net

def panels(ff,Y,I,seed,both=False,n=4096):
 g=t.Generator().manual_seed(seed);out=[]
 while sum(len(d) for d in out)<n:
  D=I[t.randint(len(I),(16384,5),generator=g)];y=target(ff,D);s=D.sort(1).values;ok=(s[:,1:]!=s[:,:-1]).all(1)&(y!=0)&(Y[D]!=y[:,None]).all(1)
  if both:ok &= ((Y[D[:,2]]==0)|(Y[D[:,0]]==0))
  out.append(D[ok])
 return t.cat(out)[:n]

def infer(net,x,D):
 with t.no_grad():m,c=net.predict([q[D[:,i]] for i,q in enumerate(x)])
 return m,c,m*c%13

def align(net,x,M,C,I):
 D=I[:,None].expand(-1,5);m,c,_=infer(net,x,D);inv=t.tensor([0]+[pow(i,-1,13) for i in range(1,13)]);mask=(m!=0)&(M[I]!=0);a=(M[I][mask]*inv[m[mask]]%13);k=int(t.bincount(a,minlength=13).argmax()) if len(a) else 1
 return k,pow(k,-1,13) if k else 1

def evaluate(net,x,ff,M,C,Y,train,groups,seed):
 k,l=align(net,x,M,C,train);rows=[]
 for name,I in groups.items():
  D=I[:,None].expand(-1,5);m,c,y=infer(net,x,D);mm=m*k%13;cc=c*l%13
  rm=(C[I]==0)&(M[I]!=0);rc=(M[I]==0)&(C[I]!=0);n=len(I)
  row=dict(split=name,n=n,ordinary=int((y==Y[I]).sum()),m_correct=int((mm==M[I]).sum()),c_correct=int((cc==C[I]).sum()),raw_m=int((m==M[I]).sum()),raw_c=int((c==C[I]).sum()),masked_m_n=int(rm.sum()),masked_m_correct=int((mm[rm]==M[I][rm]).sum()),masked_c_n=int(rc.sum()),masked_c_correct=int((cc[rc]==C[I][rc]).sum()),gauge=k)
  # Fresh five-site draws, nonzero and distinct source answers. Two-role independently drawn, nonzero.
  g=t.Generator().manual_seed(seed);J=I[(M[I]!=0)];K=I[(C[I]!=0)];u=J[t.randint(len(J),(4096,),generator=g)];v=K[t.randint(len(K),(4096,),generator=g)];_,_,out=infer(net,x,t.stack([u,u,v,v,v],1));row.update(joint_n=4096,joint=int((out==M[u]*C[v]%13).sum()))
  for masked in [False,True]:
   D=panels(ff,Y,I,seed+int(masked),masked);_,_,out=infer(net,x,D);row['site_masked' if masked else 'site']=int((out==target(ff,D)).sum())
  row['site_n']=4096;rows.append(row)
 return rows

def main(which):
 for s,mode,step,p in paths():
  if step!=6000 or f'{s}_{mode}'!=which:continue
  root=ROOT/'runs12'/which;root.mkdir(parents=True,exist_ok=False);ph=sha(p);m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(p,weights_only=True)['model']);m.eval().requires_grad_(False)
  a0=aslat(grid(s));m0,c0=split(a0);z=grid(s)[(m0!=0)&(c0!=0)];a=aslat(z);raw=views(features(m,z),'early');M,A,S=T.canon(a);C=A*S%13;ff=field_factors(a);assert t.equal(target(ff,t.arange(len(z))[:,None].expand(-1,5)),M*C%13)
  q=transform(t.cat([raw[0],raw[2]]),t.cat([a['m'],a['a']]),12,'conditioned',52100);u=[transform(raw[i],a[f],2,'conditioned',52100+i) for i,f in [(1,'mu'),(3,'au'),(4,'fu')]];st=[q,u[0],q,u[1],u[2]];x=[apply(v,c) for v,c in zip(raw,st)];t.save(dict(z=z,transform=st,parent_sha256=ph),root/'train_domain.pt')
  models=[]
  for arm in ARMS:
   for seed in SEEDS:
    d=root/f'{arm}_{seed}';models.append((arm,seed,fit(x,a,ff,M,C,d,ph,seed,arm)));print(which,arm,seed,'done',flush=True)
  # Freeze all learned states before constructing evaluation features and donor panels.
  lock={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*.pt'))};(root/'LOCKED.json').write_text(json.dumps(lock,indent=2));assert sha(p)==ph
  a=aslat(ALL);raw=views(features(m,ALL),'early');x=[apply(v,c) for v,c in zip(raw,st)];M,A,S=T.canon(a);C=A*S%13;Y=M*C%13;ff=field_factors(a);orig=t.zeros(len(ALL),dtype=t.bool);orig[key(a0)]=True;orig=orig[key(a)];mf,cf=split(a);train=t.where(orig&(mf!=0)&(cf!=0))[0];assert len(train)==len(z);groups={'train':train,'one_context_unseen':t.where(orig&((mf==0)^(cf==0)))[0],'both_context_unseen':t.where(orig&(mf==0)&(cf==0))[0],'historical_unseen':t.where(~orig)[0]};rows=[]
  for arm,seed,net in models:
   for row in evaluate(net,x,ff,M,C,Y,train,groups,52190):rows.append(dict(backbone=which,arm=arm,seed=seed,**row))
  dumpcsv(root/'scores.csv',rows);assert sha(p)==ph
if __name__=='__main__':main(sys.argv[1])
