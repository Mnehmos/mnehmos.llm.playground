import observe as ob
from observe import *
from anchor import setzero,witness
from gauge import Relative
ob.MF=setzero(folds(39,52511),39);ob.CF=setzero(folds(117,52512),117)
ST=100; seeds=[52501,52502]; ranks=[12,2,12,2,2]

def designs(x,ff,M,C,j,sub=None,a=None,allowed=None):
 ya=M[j]*C[j]%13;inv=t.tensor([0]+[pow(i,-1,13) for i in range(1,13)]);code=Direct(0).code.double();N=len(M);xs=[];ys=[];targets=[]
 for site in range(5):
  I=t.arange(N) if sub is None else sub[site];D=j.expand(len(I),5).clone();D[:,site]=I;lat={f:a[f][D[:,i]] for i,f in enumerate(FIELDS)};lat['order']=a['order'][j].expand(len(I));assert allowed[key(lat)].all(), 'forbidden oracle query';y=T.build(lat).y;assert t.equal(y,target(ff,D));nz=y!=0;X=F.pad(x[site][I,:ranks[site]].double(),(0,1),value=1.);Y=code[(G[y*inv[ya]%13]-1).clamp_min(0)];xs.append(X);ys.append(Y);targets.append(nz)
 ds=[(t.cat([xs[i][targets[i]] for i in [0,2]]),t.cat([ys[i][targets[i]] for i in [0,2]])),(t.cat([xs[i] for i in [0,2]]),t.cat([(~targets[i]).double()*2-1 for i in [0,2]])[:,None])]+[(xs[i],ys[i]) for i in [1,3,4]]
 return ds

def pack(weights,ya):
 net=Relative(ya)
 with t.no_grad():
  for h in [net.val,*net.units]:h.weight.zero_();h.bias.zero_()
  w=weights[0].float();net.val.weight[:2,:12].copy_(w[:-1].T);net.val.bias[:2].copy_(w[-1]);w=weights[1].float();net.val.weight[2,:12].copy_(w[:-1,0]);net.val.bias[2].copy_(w[-1,0])
  for h,w in zip(net.units,weights[2:]):
   w=w.float();h.weight[:,:2].copy_(w[:-1].T);h.bias.copy_(w[-1])
 return net

def prep(ds,kind):
 out=[]
 for X,Y in ds:
  C=X.T@X/len(X);v,V=t.linalg.eigh(C);assert float(v.min())>1e-12;W=(V*v.rsqrt())@V.T if kind=='white' else t.eye(X.shape[1],dtype=t.double);Z=X@W;lr=.5/float(t.linalg.eigvalsh(Z.T@Z/len(Z)).max());out.append((Z,Y,W,lr,float(v.max()/v.min())))
 return out

def fit4(ds,ya,root,ph,seed,kind):
 root.mkdir(parents=True,exist_ok=False);t.manual_seed(seed);g=t.Generator().manual_seed(seed);z=prep(ds,kind);W0=[t.randn(X.shape[1],Y.shape[1],generator=g,dtype=t.double)*.05 for X,Y in ds];weights=nn.ParameterList([nn.Parameter(t.linalg.solve(W,w)) for (_,_,W,_,_),w in zip(z,W0)]);o=t.optim.SGD([dict(params=[w],lr=lr) for w,(_,_,_,lr,_) in zip(weights,z)]);parent=ph;rows=[]
 for step in range(ST+1):
  if step%10==0:
   dest=root/f'h{step:04}.pt';fused=[W@w.detach() for (_,_,W,_,_),w in zip(z,weights)];net=pack(fused,ya);t.save(dict(model=net.state_dict(),learned=weights.state_dict(),optimizer=o.state_dict(),rng=g.get_state(),torch_rng=t.get_rng_state(),step=step,parent_sha256=parent,backbone_sha256=ph,seed=seed,kind=kind),dest);parent=sha(dest)
   for k,((X,Y),w) in enumerate(zip(ds,fused)):rows.append(dict(step=step,block=k,condition=z[k][-1],lr=z[k][-2],mse=float((X@w-Y).square().mean())))
  if step==ST:break
  loss=sum(.5*(X@w-Y).square().sum()/len(X) for w,(X,Y,W,lr,cn) in zip(weights,z));o.zero_grad();loss.backward();o.step()
 dumpcsv(root/'training.csv',rows);return net

def pivots(X,k,seed):
 # design-only row selection, without querying output or labels. After rank exhausted, random fill.
 X=F.pad(X.double(),(0,1),value=1.);D=X.clone();I=[];g=t.Generator().manual_seed(seed)
 for _ in range(k):
  norms=D.square().sum(1);norms[I]=-1;i=int(norms.argmax())
  if float(norms[i])<=1e-12:
   p=t.randperm(len(X),generator=g).tolist();I += [x for x in p if x not in I][:k-len(I)];break
  I.append(i);v=D[i]/D[i].norm();D-= (D@v)[:,None]*v[None]
 return t.tensor(I)

def solve(ds,ya):return pack([t.linalg.pinv(X,rtol=1e-8)@Y for X,Y in ds],ya)

def main(which):
 for s,mode,step,p in paths():
  if step!=6000 or f'{s}_{mode}'!=which:continue
  root=ROOT/'protected'/which;root.mkdir(parents=True,exist_ok=False);ph=sha(p);m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(p,weights_only=True)['model']);m.eval().requires_grad_(False)
  a0=aslat(grid(s));m0,c0=split(a0);z=grid(s)[(m0!=0)&(c0!=0)];a=aslat(z);raw=views(features(m,z),'early');M,A,S=T.canon(a);C=A*S%13;ff=field_factors(a);assert t.equal(target(ff,t.arange(len(z))[:,None].expand(-1,5)),M*C%13)
  q=transform(t.cat([raw[0],raw[2]]),t.cat([a['m'],a['a']]),12,'conditioned',52500);u=[transform(raw[i],a[f],2,'conditioned',52500+i) for i,f in [(1,'mu'),(3,'au'),(4,'fu')]];st=[q,u[0],q,u[1],u[2]];x=[apply(v,c) for v,c in zip(raw,st)];allowed=t.zeros(len(ALL),dtype=t.bool);allowed[key(a0)]=True;valid=t.ones(13,dtype=t.bool);
  for v in range(13):
   valid[v]=all(bool(((a0['m']==v)&(a0['mu']==u)).any() and ((a0['a']==v)&(a0['au']==u)).any()) for u in range(3))
  J=t.where((a['m']==a['a'])&(a['mu']==0)&(a['au']==0)&(a['m']!=0)&valid[a['m']])[0];assert len(J)>0;j=J[t.randint(len(J),(1,),generator=t.Generator().manual_seed(52500))[0]];assert allowed[key(a)[j]];ya=M[j]*C[j]%13;ds=designs(x,ff,M,C,j,a=a,allowed=allowed);t.save(dict(z=z,transform=st,parent_sha256=ph,reference=int(j)),root/'train_domain.pt');models=[]
  for arm in ['raw','white']:
   for seed in seeds:models.append((arm,seed,fit4(ds,ya,root/f'{arm}_{seed}',ph,seed,arm)));print(which,arm,seed,'done',flush=True)
  net=solve(ds,ya);t.save(dict(model=net.state_dict(),backbone_sha256=ph),root/'least_squares.pt');models.append(('least_squares',0,net))
  for arm in ['random32','pivot32']:
   for seed in seeds:
    sub=[t.randint(len(M),(32,),generator=t.Generator().manual_seed(seed+i)) if arm=='random32' else pivots(x[i][:,:ranks[i]],32,seed+i) for i in range(5)];d=designs(x,ff,M,C,j,sub,a=a,allowed=allowed);net=solve(d,ya);t.save(dict(model=net.state_dict(),indices=sub,backbone_sha256=ph,observations=sum(len(u.unique()) for u in sub)+1),root/f'{arm}_{seed}.pt');models.append((arm,seed,net))
  lock={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*.pt'))};(root/'LOCKED.json').write_text(json.dumps(lock,indent=2));assert sha(p)==ph
  a=aslat(ALL);raw=views(features(m,ALL),'early');x=[apply(v,c) for v,c in zip(raw,st)];M,A,S=T.canon(a);C=A*S%13;Y=M*C%13;ff=field_factors(a);orig=t.zeros(len(ALL),dtype=t.bool);orig[key(a0)]=True;orig=orig[key(a)];mf,cf=split(a);train=t.where(orig&(mf!=0)&(cf!=0))[0];assert len(train)==len(z);groups={'train':train,'one_context_unseen':t.where(orig&((mf==0)^(cf==0)))[0],'both_context_unseen':t.where(orig&(mf==0)&(cf==0))[0],'historical_unseen':t.where(~orig)[0]};rows=[];wr=[]
  for arm,seed,net in models:
   for row in evaluate(net,x,ff,M,C,Y,train,groups,52590):rows.append(dict(backbone=which,arm=arm,seed=seed,**row))
   for name,I in groups.items():
    for row in witness(net,x,ff,M,C,Y,train,I):wr.append(dict(backbone=which,arm=arm,seed=seed,split=name,**row))
  dumpcsv(root/'scores.csv',rows);dumpcsv(root/'witness.csv',wr);assert sha(p)==ph
if __name__=='__main__':main(sys.argv[1])
