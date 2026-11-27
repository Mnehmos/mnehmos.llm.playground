from direct import *
def bad3(a):
 return ((a['mu']==1)&t.isin(a['m'],t.tensor([2,5,11])))|((a['mu']==2)&t.isin(a['m'],t.tensor([1,7,10])))|((a['au']==1)&t.isin(a['a'],t.tensor([2,5,11])))|((a['au']==2)&t.isin(a['a'],t.tensor([1,7,10])))
def transform(x,y,rank,kind,seed):
 x=x.double();mean=x.mean(0);sd=x.std(0).clamp_min(1e-4);X=(x-mean)/sd
 if kind=='standard':W=t.eye(64,dtype=t.double)
 elif kind=='random':W=t.linalg.qr(t.randn(64,rank,generator=t.Generator().manual_seed(seed),dtype=t.double))[0]
 else:
  cent=t.stack([X[y==i].mean(0) for i in range(int(y.max())+1)]);D=X-cent[y];C=D.T@D/len(D);v,V=t.linalg.eigh(C);W0=(V*(v.clamp_min(0)+max(float(v.mean())*.001,1e-6)).rsqrt())@V.T;B=(cent-cent.mean(0))@W0;_,U=t.linalg.eigh(B.T@B/len(B));W=W0@U[:,-rank:]
 Z=X@W;mu=Z.mean(0);ss=Z.std(0).clamp_min(1e-4)
 return (mean,sd,W,mu,ss)
def apply(x,s):
 u,sd,W,mu,ss=s;z=((x.double()-u)/sd@W-mu)/ss;return F.pad(z.float(),(0,64-z.shape[1]))
def run3(s,mode,p):
 root=ROOT/'condition'/f'{s}_{mode}';root.mkdir(parents=True);ph=sha(p);m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(p,weights_only=True)['model']);m.eval().requires_grad_(False)
 z=grid(s);z=z[~bad3(aslat(z))];a=aslat(z);r=features(m,z);M,A,S=T.canon(a);C=A*S%13;ms=newmasks(a)[0];raw=views(r,'early');t.save(dict(z=z,masks=ms,parent_sha256=ph),root/'train_domain.pt');models={};trans={}
 for kind in ['standard','random','conditioned']:
  q=transform(t.cat([raw[0][ms[0]],raw[2][ms[1]]]),t.cat([a['m'][ms[0]],a['a'][ms[1]]]),12,kind,51300);u=[transform(raw[i][ms[j]],a[f][ms[j]],2,kind,51300+i) for i,j,f in [(1,0,'mu'),(3,1,'au'),(4,1,'fu')]];st=[q,u[0],q,u[1],u[2]];trans[kind]=st;x=[apply(v,c) for v,c in zip(raw,st)];t.save(st,root/f'{kind}_transform.pt')
  for seed in [51301,51302]:models[(kind,seed)]=fit(x,a,ms,M,C,root/f'{kind}_{seed}',ph,seed,0,False)
 lock={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*.pt'))};(root/'LOCKED.json').write_text(json.dumps(lock,indent=2));assert sha(p)==ph
 a=aslat(ALL);r=features(m,ALL);M,A,S=T.canon(a);C=A*S%13;Y=M*C%13;orig=t.zeros(9126,dtype=t.bool);orig[key(aslat(grid(s)))]=True;orig=orig[key(a)];nb=bad3(a);held=newmasks(a)[2];splits={'context':orig&~nb&held[0]&held[1],'repair_only':orig&nb,'historical_unseen':~orig,'all_familiar':orig};H=r['H'];L=native(m,H);q=group(L);raw=views(r,'early')
 with t.no_grad():preds={f'{kind}_{seed}':net.predict([apply(v,c) for v,c in zip(raw,trans[kind])]) for (kind,seed),net in models.items()}
 rows=[]
 for split,mask in splits.items():
  i=t.where(mask)[0];g=t.Generator().manual_seed(51390);nz=i[(M[i]!=0)&(C[i]!=0)];u=nz[t.randint(len(nz),(4096,),generator=g)];v=nz[t.randint(len(nz),(4096,),generator=g)];y=M[u]*C[v]%13;jl=native(m,t.cat([H[u,:16],H[v,16:]],1));jq=group(jl)
  for name,(mm,cc) in preds.items():
   op=mm[i]*cc[i]%13;jp=mm[u]*cc[v]%13;ro=CRT[q[i],Q4[op]];rj=CRT[jq,Q4[jp]]
   rows.append(dict(seed=s,mode=mode,split=split,method=name,n=len(i),ordinary=int((op==Y[i]).sum()),joint_n=len(y),joint=int((jp==y).sum()),repair_ordinary=int((ro==Y[i]).sum()),repair_joint=int((rj==y).sum()),m_correct=int((mm[i]==M[i]).sum()),c_correct=int((cc[i]==C[i]).sum())))
 dumpcsv(root/'scores.csv',rows);assert sha(p)==ph;print(s,mode,'done',flush=True)
if __name__=='__main__':
 for s,mode,step,p in paths():
  if step==6000 and (len(sys.argv)==1 or str(s)+'_'+mode==sys.argv[1]):run3(s,mode,p)
