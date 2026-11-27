import observe as ob
from observe import *
ST=1000; seeds=[52201,52202]; arms=['ordinary','swap','ratio','factor']
def setzero(f,n):
 # move index 0 to fold0 if necessary, swapping with a nonzero index to retain counts.
 if f[0]!=0:
  i=t.where((f==0)&(t.arange(n)>=3))[0][0];tmp=f[0].clone();f[0]=0;f[i]=tmp
 return f
ob.MF=setzero(folds(39,52211),39);ob.CF=setzero(folds(117,52212),117)
def fit2(x,ff,M,C,root,ph,seed,arm):
 t.manual_seed(seed);net=Direct(0);opt=t.optim.AdamW(net.parameters(),lr=.003);g=t.Generator().manual_seed(seed+100000);N=len(M);root.mkdir(parents=True,exist_ok=False);parent=ph;J=t.where((M*C%13)!=0)[0];inv=t.tensor([0]+[pow(i,-1,13) for i in range(1,13)]);seen=set();calls=0;losses=[]
 for step in range(ST+1):
  if step%100==0:
   dst=root/f'h{step:04}.pt';t.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=g.get_state(),torch_rng=t.get_rng_state(),step=step,parent_sha256=parent,backbone_sha256=ph,seed=seed,arm=arm,calls=calls,anchors=len(seen)),dst);parent=sha(dst)
  if step==ST:break
  i=t.randint(N,(B,),generator=g);j=J[t.randint(len(J),(1,),generator=g)][0];site=step%5;D=j.expand(B,5).clone();D[:,site]=i;ordinary=i[:,None].expand(-1,5);ya=M[j]*C[j]%13;y=target(ff,D);xa=[q[j:j+1] for q in x];anchor_y=ya.reshape(1)
  if arm=='ordinary':loss=output_loss(net,[q[i] for q in x],M[i]*C[i]%13);calls+=B
  elif arm=='factor':loss=factor_loss(net,[q[i] for q in x],M[i],C[i]);calls+=B*2
  else:
   seen.add(int(j));calls+=B+1
   if arm=='swap':loss=output_loss(net,[q[D[:,k]] for k,q in enumerate(x)],y)+output_loss(net,xa,anchor_y)
   else:
    _,az,ap=net(xa);_,z,p=net([q[D[:,k]] for k,q in enumerate(x)]);r=mul(p[site],ap[site]*t.tensor([1.,-1.]));nz=y!=0;logits=r@net.code.T*10;loss=F.cross_entropy(logits[nz],G[y[nz]*inv[ya]%13]-1) if nz.any() else logits.sum()*0
    if site in [0,2]:loss+=F.binary_cross_entropy_with_logits(z[site//2],(~nz).float())
    loss+=output_loss(net,xa,anchor_y)
  opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
 (root/'training.json').write_text(json.dumps(dict(calls=calls,unique_anchors=len(seen),last100=sum(losses[-100:])/100)));return net

def witness(net,x,ff,M,C,Y,train,I):
 # All five sites evaluated once per source. Fixed anchor chosen from training only.
 j=train[(Y[train]!=0)][0];D0=j.expand(1,5);_,_,anchor_pred=infer(net,x,D0);ya=Y[j];inv=t.tensor([0]+[pow(i,-1,13) for i in range(1,13)]);rows=[]
 with t.no_grad():
  ap,az=net.pieces([q[j:j+1] for q in x]);allp,allz=net.pieces([q[I] for q in x])
  for site in range(5):
   D=j.expand(len(I),5).clone();D[:,site]=I;truth=target(ff,D);_,_,pred=infer(net,x,D);rp=mul(allp[site],ap[site]*t.tensor([1.,-1.]));ph=INV[(rp@net.code.T).argmax(1)+1];ph=ph*ya%13
   if site in [0,2]:ph=t.where(allz[site//2]>0,0,ph)
   # Relative phase diagnostic uses correct reference output rather than other sites' predictions.
   mask=(Y[I]==0)&(truth!=0);rows.append(dict(site=site,n=len(I),correct=int((pred==truth).sum()),ratio_correct=int((ph==truth).sum()),masked_n=int(mask.sum()),masked_correct=int((pred[mask]==truth[mask]).sum()),masked_ratio_correct=int((ph[mask]==truth[mask]).sum()),anchor_true=int(ya),anchor_pred=int(anchor_pred[0])))
 return rows

def main(which):
 for s,mode,step,p in paths():
  if step!=6000 or f'{s}_{mode}'!=which:continue
  root=ROOT/'anchor'/which;root.mkdir(parents=True,exist_ok=False);ph=sha(p);m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(p,weights_only=True)['model']);m.eval().requires_grad_(False)
  a0=aslat(grid(s));m0,c0=split(a0);z=grid(s)[(m0!=0)&(c0!=0)];a=aslat(z);raw=views(features(m,z),'early');M,A,S=T.canon(a);C=A*S%13;ff=field_factors(a);assert t.equal(target(ff,t.arange(len(z))[:,None].expand(-1,5)),M*C%13)
  q=transform(t.cat([raw[0],raw[2]]),t.cat([a['m'],a['a']]),12,'conditioned',52200);u=[transform(raw[i],a[f],2,'conditioned',52200+i) for i,f in [(1,'mu'),(3,'au'),(4,'fu')]];st=[q,u[0],q,u[1],u[2]];x=[apply(v,c) for v,c in zip(raw,st)];t.save(dict(z=z,transform=st,parent_sha256=ph),root/'train_domain.pt');models=[]
  for arm in arms:
   for seed in seeds:models.append((arm,seed,fit2(x,ff,M,C,root/f'{arm}_{seed}',ph,seed,arm)));print(which,arm,seed,'done',flush=True)
  lock={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*.pt'))};(root/'LOCKED.json').write_text(json.dumps(lock,indent=2));assert sha(p)==ph
  a=aslat(ALL);raw=views(features(m,ALL),'early');x=[apply(v,c) for v,c in zip(raw,st)];M,A,S=T.canon(a);C=A*S%13;Y=M*C%13;ff=field_factors(a);orig=t.zeros(len(ALL),dtype=t.bool);orig[key(a0)]=True;orig=orig[key(a)];mf,cf=split(a);train=t.where(orig&(mf!=0)&(cf!=0))[0];assert len(train)==len(z);groups={'train':train,'one_context_unseen':t.where(orig&((mf==0)^(cf==0)))[0],'both_context_unseen':t.where(orig&(mf==0)&(cf==0))[0],'historical_unseen':t.where(~orig)[0]};rows=[];wr=[]
  for arm,seed,net in models:
   for row in evaluate(net,x,ff,M,C,Y,train,groups,52290):rows.append(dict(backbone=which,arm=arm,seed=seed,**row))
   for name,I in groups.items():
    for row in witness(net,x,ff,M,C,Y,train,I):wr.append(dict(backbone=which,arm=arm,seed=seed,split=name,**row))
  dumpcsv(root/'scores.csv',rows);dumpcsv(root/'witness.csv',wr);assert sha(p)==ph
if __name__=='__main__':main(sys.argv[1])
