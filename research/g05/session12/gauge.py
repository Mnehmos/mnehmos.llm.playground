import observe as ob
from observe import *
from anchor import setzero,witness
ST=800; seeds=[52301,52302]; arms=['fixed_swap','fixed_ratio','local_ce','local_reg']
ob.MF=setzero(folds(39,52311),39);ob.CF=setzero(folds(117,52312),117)
class Relative(Direct):
 def __init__(self,ya):
  super().__init__(0);self.register_buffer('refout',t.tensor(int(ya),dtype=t.long))
 def predict(self,xx):
  a,b=super().predict(xx);return a,b*self.refout%13

def fit3(x,ff,M,C,a,root,ph,seed,arm,j):
 ya=M[j]*C[j]%13;local=arm.startswith('local');t.manual_seed(seed);net=Relative(ya) if local else Direct(0);opt=t.optim.AdamW(net.parameters(),lr=.003);g=t.Generator().manual_seed(seed+100000);N=len(M);root.mkdir(parents=True,exist_ok=False);parent=ph;inv=t.tensor([0]+[pow(i,-1,13) for i in range(1,13)]);seen=t.zeros((5,N),dtype=t.bool);calls=1;losses=[]
 xa=[q[j:j+1] for q in x];anchor_y=ya.reshape(1)
 for step in range(ST+1):
  if step%100==0:
   dst=root/f'h{step:04}.pt';t.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=g.get_state(),torch_rng=t.get_rng_state(),step=step,parent_sha256=parent,backbone_sha256=ph,seed=seed,arm=arm,reference=int(j),calls=calls,unique_pairs=int(seen.sum())),dst);parent=sha(dst)
  if step==ST:break
  i=t.randint(N,(B,),generator=g);site=step%5;D=j.expand(B,5).clone();D[:,site]=i;y=target(ff,D);seen[site,i]=True;calls+=B;nz=y!=0;rt=G[y[nz]*inv[ya]%13]-1;xx=[q[D[:,k]] for k,q in enumerate(x)]
  if arm=='fixed_swap':loss=output_loss(net,xx,y)+output_loss(net,xa,anchor_y)
  elif arm=='fixed_ratio':
   ap,az=net.pieces(xa);p,z=net.pieces(xx);r=mul(p[site],ap[site]*t.tensor([1.,-1.]));logits=r@net.code.T*10;loss=F.cross_entropy(logits[nz],rt) if nz.any() else logits.sum()*0
   if site in [0,2]:loss+=F.binary_cross_entropy_with_logits(z[site//2],(~nz).float())
   loss+=output_loss(net,xa,anchor_y)
  else:
   h=net.val(x[site][i]) if site in [0,2] else net.units[[1,3,4].index(site)](x[site][i]);z=h[:,:2];loss=z.sum()*0
   if nz.any():
    if arm=='local_reg':loss=F.mse_loss(z[nz],net.code[rt])
    else:loss=F.cross_entropy(F.normalize(z[nz],dim=1,eps=1e-6)@net.code.T*10,rt)
   if site in [0,2]:loss+=F.binary_cross_entropy_with_logits(h[:,2],(~nz).float())
  opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
 (root/'training.json').write_text(json.dumps(dict(calls=calls,unique_pairs=int(seen.sum()),reference=int(j),reference_y=int(ya),last100=sum(losses[-100:])/100)));return net

def main(which):
 for s,mode,step,p in paths():
  if step!=6000 or f'{s}_{mode}'!=which:continue
  root=ROOT/'gauge'/which;root.mkdir(parents=True,exist_ok=False);ph=sha(p);m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(p,weights_only=True)['model']);m.eval().requires_grad_(False)
  a0=aslat(grid(s));m0,c0=split(a0);z=grid(s)[(m0!=0)&(c0!=0)];a=aslat(z);raw=views(features(m,z),'early');M,A,S=T.canon(a);C=A*S%13;ff=field_factors(a);assert t.equal(target(ff,t.arange(len(z))[:,None].expand(-1,5)),M*C%13)
  q=transform(t.cat([raw[0],raw[2]]),t.cat([a['m'],a['a']]),12,'conditioned',52300);u=[transform(raw[i],a[f],2,'conditioned',52300+i) for i,f in [(1,'mu'),(3,'au'),(4,'fu')]];st=[q,u[0],q,u[1],u[2]];x=[apply(v,c) for v,c in zip(raw,st)];J=t.where((a['m']==a['a'])&(M!=0)&(C!=0))[0];j=J[t.randint(len(J),(1,),generator=t.Generator().manual_seed(52300))[0]];t.save(dict(z=z,transform=st,parent_sha256=ph,reference=int(j)),root/'train_domain.pt');models=[]
  for arm in arms:
   for seed in seeds:models.append((arm,seed,fit3(x,ff,M,C,a,root/f'{arm}_{seed}',ph,seed,arm,j)));print(which,arm,seed,'done',flush=True)
  # Deterministic least-squares diagnostic for local relative phase targets (full train site support).
  ya=M[j]*C[j]%13;net=Relative(ya);inv=t.tensor([0]+[pow(i,-1,13) for i in range(1,13)]);D=j.expand(len(M),5).clone();rows=[]
  for sites,h in [([0,2],net.val)]+[([i],hh) for i,hh in zip([1,3,4],net.units)]:
   xx=[];yy=[];zg=[];zz=[]
   for site in sites:
    d=D.clone();d[:,site]=t.arange(len(M));y=target(ff,d);nz=y!=0;rt=G[y[nz]*inv[ya]%13]-1;xx.append(x[site][nz]);yy.append(net.code[rt]);zg.append(x[site]);zz.append((~nz).float()*2-1)
   def lst(x,y):
    x=t.cat(x).double();y=t.cat(y).double();X=F.pad(x,(0,1),value=1.);return (t.linalg.pinv(X,rtol=1e-8)@y).float()
   with t.no_grad():
    w=lst(xx,yy);h.weight[:2].copy_(w[:-1].T);h.bias[:2].copy_(w[-1]);
    if len(sites)==2:
     w=lst(zg,[v[:,None] for v in zz]);h.weight[2].copy_(w[:-1,0]);h.bias[2].copy_(w[-1,0])
  t.save(dict(model=net.state_dict(),reference=int(j),backbone_sha256=ph),root/'least_squares.pt');models.append(('least_squares',0,net))
  lock={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*.pt'))};(root/'LOCKED.json').write_text(json.dumps(lock,indent=2));assert sha(p)==ph
  a=aslat(ALL);raw=views(features(m,ALL),'early');x=[apply(v,c) for v,c in zip(raw,st)];M,A,S=T.canon(a);C=A*S%13;Y=M*C%13;ff=field_factors(a);orig=t.zeros(len(ALL),dtype=t.bool);orig[key(a0)]=True;orig=orig[key(a)];mf,cf=split(a);train=t.where(orig&(mf!=0)&(cf!=0))[0];assert len(train)==len(z);groups={'train':train,'one_context_unseen':t.where(orig&((mf==0)^(cf==0)))[0],'both_context_unseen':t.where(orig&(mf==0)&(cf==0))[0],'historical_unseen':t.where(~orig)[0]};rows=[];wr=[]
  for arm,seed,net in models:
   for row in evaluate(net,x,ff,M,C,Y,train,groups,52390):rows.append(dict(backbone=which,arm=arm,seed=seed,**row))
   for name,I in groups.items():
    for row in witness(net,x,ff,M,C,Y,train,I):wr.append(dict(backbone=which,arm=arm,seed=seed,split=name,**row))
  dumpcsv(root/'scores.csv',rows);dumpcsv(root/'witness.csv',wr);assert sha(p)==ph
if __name__=='__main__':main(sys.argv[1])
