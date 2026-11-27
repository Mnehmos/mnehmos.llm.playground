from run import *

def bad2(a):
 return ((a['mu']==1)&t.isin(a['m'],t.tensor([1,7,12])))|((a['mu']==2)&t.isin(a['m'],t.tensor([3,8,11])))|((a['au']==1)&t.isin(a['a'],t.tensor([1,7,12])))|((a['au']==2)&t.isin(a['a'],t.tensor([3,8,11])))
def mul(x,y):return t.stack([x[:,0]*y[:,0]-x[:,1]*y[:,1],x[:,0]*y[:,1]+x[:,1]*y[:,0]],1)
class Direct(nn.Module):
 def __init__(self,width):
  super().__init__();f=lambda d,n:nn.Linear(d,n) if width==0 else nn.Sequential(nn.Linear(d,width),nn.ReLU(),nn.Linear(width,n));self.val=f(64,3);self.units=nn.ModuleList([f(64,2) for _ in range(3)]);a=t.arange(12)*6.283185307/12;self.register_buffer('code',t.stack([a.cos(),a.sin()],1))
 def pieces(self,xx):
  m,mu,a,au,fu=xx;v1=self.val(m);v2=self.val(a);u=[F.normalize(h(x),dim=1,eps=1e-6) for h,x in zip(self.units,[mu,au,fu])];return [F.normalize(v1[:,:2],dim=1,eps=1e-6),u[0],F.normalize(v2[:,:2],dim=1,eps=1e-6),u[1],u[2]],[v1[:,2],v2[:,2]]
 def forward(self,xx):
  p,z=self.pieces(xx);return [mul(p[0],p[1])@self.code.T*10,mul(mul(p[2],p[3]),p[4])@self.code.T*10],z,p
 def predict(self,xx):
  l,z,_=self(xx);return [t.where(a>0,0,INV[b.argmax(1)+1]) for a,b in zip(z,l)]
def fit(x,a,ms,M,C,root,ph,seed,width,reg):
 t.manual_seed(seed);net=Direct(width);o=t.optim.AdamW(net.parameters(),lr=.003);g=t.Generator().manual_seed(seed+100000);g2=t.Generator().manual_seed(seed+200000);I=[t.where(k)[0] for k in ms];root.mkdir(parents=True);parent=ph
 # partner tables by field equality and matching role training masks
 pools=[[t.where(ms[j]&(a[f]==v))[0] for v in range(n)] for f,j,n in zip(FIELDS,ROLES,NS)]
 for step in range(1001):
  if step%100==0:
   dst=root/f'h{step:04}.pt';t.save(dict(model=net.state_dict(),optimizer=o.state_dict(),rng=g.get_state(),partner_rng=g2.get_state(),torch_rng=t.get_rng_state(),step=step,parent_sha256=parent,seed=seed,width=width,reg=reg),dst);parent=sha(dst)
  if step==1000:break
  ii=[j[t.randint(len(j),(128,),generator=g)] for j in I];xx=[q[ii[j]] for q,j in zip(x,ROLES)];ls,z,p=net(xx);loss=0
  for l,k,y in zip(ls,z,[M[ii[0]],C[ii[1]]]):
   nz=y!=0;loss+=F.cross_entropy(l[nz],G[y[nz]]-1)+F.binary_cross_entropy_with_logits(k,(~nz).float())
  if reg:
   partners=[]
   for j,f,pool in zip(ROLES,FIELDS,pools):
    labels=a[f][ii[j]];idx=t.empty_like(labels)
    for v in labels.unique():
     k=labels==v;P=pool[int(v)];idx[k]=P[t.randint(len(P),(int(k.sum()),),generator=g2)]
    partners.append(idx)
   p2,z2=net.pieces([q[j] for q,j in zip(x,partners)]);loss+=.5*sum((u-v).square().mean() for u,v in zip(p,p2))
  o.zero_grad();loss.backward();o.step()
 return net

def run2(s,mode,p):
 root=ROOT/'direct'/f'{s}_{mode}';root.mkdir(parents=True);ph=sha(p);m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(p,weights_only=True)['model']);m.eval().requires_grad_(False)
 z=grid(s);z=z[~bad2(aslat(z))];a=aslat(z);r=features(m,z);M,A,S=T.canon(a);C=A*S%13;ms=newmasks(a)[0];x,stats=norm(views(r,'early'),ms,ROLES)
 t.save(dict(z=z,masks=ms,normalization=stats,parent_sha256=ph),root/'train_domain.pt');models={}
 for width in [0,32]:
  for reg in [False,True]:
   for seed in [51201,51202]:
    name=f'w{width}_reg{int(reg)}_{seed}';models[name]=fit(x,a,ms,M,C,root/name,ph,seed,width,reg)
 # matched flat baseline (separate role heads), same schedule, no literal labels
 xf,stf=norm(views(r,'flat'),ms,[0,1]);flat={}
 for seed in [51201,51202]:
  t.manual_seed(seed);net=nn.ModuleList([nn.Sequential(nn.Linear(q.shape[1],32),nn.ReLU(),nn.Linear(32,13)) for q in xf]);o=t.optim.AdamW(net.parameters(),lr=.003);g=t.Generator().manual_seed(seed+100000);I=[t.where(k)[0] for k in ms];d=root/f'flat_{seed}';d.mkdir();parent=ph
  for step in range(1001):
   if step%100==0:
    dst=d/f'h{step:04}.pt';t.save(dict(model=net.state_dict(),optimizer=o.state_dict(),rng=g.get_state(),torch_rng=t.get_rng_state(),step=step,parent_sha256=parent,normalization=stf),dst);parent=sha(dst)
   if step==1000:break
   ii=[j[t.randint(len(j),(128,),generator=g)] for j in I];loss=sum(F.cross_entropy(h(q[j]),y[j]) for h,q,y,j in zip(net,xf,[M,C],ii));o.zero_grad();loss.backward();o.step()
  flat[f'flat_{seed}']=net
 lock={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*.pt'))};(root/'LOCKED.json').write_text(json.dumps(lock,indent=2));assert sha(p)==ph
 a=aslat(ALL);r=features(m,ALL);M,A,S=T.canon(a);C=A*S%13;Y=M*C%13;orig=t.zeros(9126,dtype=t.bool);orig[key(aslat(grid(s)))]=True;orig=orig[key(a)];nb=bad2(a);held=newmasks(a)[2];splits={'context':orig&~nb&held[0]&held[1],'repair_only':orig&nb,'historical_unseen':~orig,'all_familiar':orig};H=r['H'];L=native(m,H);q=group(L);x=[(v-u)/sd for v,(u,sd) in zip(views(r,'early'),stats)];xf=[(v-u)/sd for v,(u,sd) in zip(views(r,'flat'),stf)]
 with t.no_grad():
  preds={name:net.predict(x) for name,net in models.items()};preds.update({name:[h(q).argmax(1) for h,q in zip(net,xf)] for name,net in flat.items()})
 rows=[]
 for split,mask in splits.items():
  i=t.where(mask)[0];g=t.Generator().manual_seed(51290);nz=i[(M[i]!=0)&(C[i]!=0)];u=nz[t.randint(len(nz),(4096,),generator=g)];v=nz[t.randint(len(nz),(4096,),generator=g)];y=M[u]*C[v]%13;jl=native(m,t.cat([H[u,:16],H[v,16:]],1));jq=group(jl)
  for name,(mm,cc) in preds.items():
   op=mm[i]*cc[i]%13;jp=mm[u]*cc[v]%13;ro=CRT[q[i],Q4[op]];rj=CRT[jq,Q4[jp]]
   rows.append(dict(seed=s,mode=mode,split=split,method=name,n=len(i),ordinary=int((op==Y[i]).sum()),joint_n=len(y),joint=int((jp==y).sum()),repair_ordinary=int((ro==Y[i]).sum()),repair_joint=int((rj==y).sum()),m_correct=int((mm[i]==M[i]).sum()),c_correct=int((cc[i]==C[i]).sum())))
 dumpcsv(root/'scores.csv',rows);assert sha(p)==ph;print(s,mode,'done',flush=True)
if __name__=='__main__':
 for s,mode,step,p in paths():
  if step==6000 and (len(sys.argv)==1 or str(s)+'_'+mode==sys.argv[1]):run2(s,mode,p)
