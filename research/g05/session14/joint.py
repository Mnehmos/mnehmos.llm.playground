from topology import *
from time import perf_counter
MASK9=np.array([[(i>>j)&1 for j in range(9)] for i in range(512)],dtype=np.float32);K9=MASK9.sum(1).astype(int);SUB9=[[j for j in range(i) if j&i==j] for i in range(512)]
def setup(m,b,d):
 tok=T.build(aslat(t.cat([b,d]))).tokens;truth=T.build(aslat(t.cat([b,d]))).y
 with t.no_grad():
  e=m.embed(tok,t.zeros_like(tok));a=parts(m.attn1,e);x=a.sum(1);c=parts(m.attn2,x)[:,:,-1];l=m(tok,t.zeros_like(tok));a0,a1=a.chunk(2);c0,c1=c.chunk(2);l0,l1=l.chunk(2)
 return a0,a1-a0,c0,c1,l0,l1,truth.chunk(2)
def evaluate(m,a,da,c1,mask):
 # batch size n pairs, each with a mask, or broadcast one pair over many masks
 q=a+(da*mask[:,:5,None,None]).sum(1)[:,None] if False else a.sum(1)+(da*mask[:,:5,None,None]).sum(1)
 cp=parts(m.attn2,q)[:,:,-1];h=cp[:,4]+t.where(mask[:,5:,None].bool(),c1[:,:4],cp[:,:4]).sum(1);return m.head_from_resid(h)
def features9(m,a,da,c0,c1,l0,l1):
 start=perf_counter();n=len(a);x=a.sum(1).detach().requires_grad_();h=c0.sum(1).detach().requires_grad_();lo=out(m,x,1);lh=out(m,h,2);d2=(c1-c0)[:,:4];effects=[]
 for j in range(13):
  g=t.autograd.grad(lo[:,j].sum(),x,retain_graph=j<12)[0];gh=t.autograd.grad(lh[:,j].sum(),h,retain_graph=j<12)[0];effects.append(t.cat([(da*g[:,None]).flatten(2).sum(2),(d2*gh[:,None]).sum(2)],1))
 eff=t.stack(effects,-1).detach();mask=t.tensor(MASK9);lin=l0[:,None]+t.einsum('mk,bkc->bmc',mask,eff);target=l1.argmax(1);one=F.one_hot(target,13).bool();lm=(lin*one[:,None]).sum(-1)-lin.masked_fill(one[:,None],-t.inf).max(-1).values;bm=lm[:,0];dn=t.cat([da.flatten(2).norm(dim=2),d2.norm(dim=2)],1);ms=mask.bool()[None];single=lm[:,[1<<i for i in range(9)]]-bm[:,None];s=single[:,None].expand(-1,512,-1);selmax=s.masked_fill(~ms,-1e9).max(-1).values;selmin=s.masked_fill(~ms,1e9).min(-1).values;selmax[:,0]=selmin[:,0]=0;den=dn.sum(1,keepdim=True).clamp_min(1e-8);sumdn=t.einsum('mk,bk->bm',mask,dn)/den;maxdn=dn[:,None].expand(-1,512,-1).masked_fill(~ms,0).max(-1).values/den;scale=(l0.max(1).values-l0.min(1).values).clamp_min(1e-4)[:,None];feat=t.stack([mask.sum(1)[None].expand(n,-1)/9,bm[:,None].expand_as(lm)/scale,lm/scale,(lm-bm[:,None])/scale,selmax/scale,selmin/scale,sumdn,maxdn,(lin.argmax(-1)==target[:,None]).float(),lm[:,-1,None].expand_as(lm)/scale,t.full_like(lm,.75),(l1.max(1).values-l1.min(1).values)[:,None].expand_as(lm)/scale],-1)
 return feat,lm.numpy(),perf_counter()-start

def policy_order(lin,pr,i):
 order=np.arange(1,511);rng=np.random.default_rng(54300+i);rnd=rng.permutation(order);D={'random':rnd,'size_first':rnd[np.argsort(K9[rnd],kind='stable')],'linear_margin':order[np.argsort(-lin[order],kind='stable')],'linear_cost':order[np.argsort(-(lin[order]-lin[0])/(2.**K9[order]),kind='stable')]}
 for j,p in enumerate(pr):
  D[f'neural{j+1}']=order[np.argsort(-p[order],kind='stable')];D[f'neural_cost{j+1}']=order[np.argsort(-p[order]/(2.**K9[order]),kind='stable')]
 return D

def online(m,a,da,c1,l0,l1,method,net,mu,sd):
 source=int(l0.argmax(1));target=int(l1.argmax(1));cache={0:False,511:True};cnt=0;start=perf_counter();featsec=0.
 def query(Q):
  Q=[int(q) for q in Q if int(q) not in cache]
  if not Q:return
  with t.no_grad():
   L=evaluate(m,a.expand(len(Q),-1,-1,-1),da.expand(len(Q),-1,-1,-1),c1.expand(len(Q),-1,-1),t.tensor(MASK9[Q]));cache.update(zip(Q,(L.argmax(1)==target).tolist()))
 if method=='exhaustive':query(range(1,511));cert=min([q for q in cache if cache[q]],key=lambda q:(K9[q],q))
 else:
  found=-1
  if method=='neural_fallback':
   # c0 natural base computed directly from known layer1 components, common extraction excluded
   with t.no_grad():c0=parts(m.attn2,a.sum(1))[:,:,-1]
   feat,lin,featsec=features9(m,a,da,c0,c1,l0,l1)
   with t.no_grad():p=net((feat-mu)/sd).sigmoid().flatten().numpy();order=np.arange(1,511);order=order[np.argsort(-p[order]/(2.**K9[order]),kind='stable')]
   for q in order[:8]:
    query([q]);
    if cache[q]:found=int(q);break
  if found>=0:
   query(SUB9[found]);cert=min([q for q in cache if q&found==q and cache[q]],key=lambda q:(K9[q],q))
  else:
   for k in range(1,10):
    ids=np.flatnonzero(K9==k);query(ids);wins=[q for q in ids if cache[q]]
    if wins:cert=int(wins[0]);break
 elapsed=perf_counter()-start
 assert cache[cert] and all(q in cache and not cache[q] for q in SUB9[cert]);return cert,len(cache)-2,elapsed,featsec

def main():
 train=t.load(ROOT/'train.pt',weights_only=True);mu,sd=train['mu'],train['sd'];nets=[]
 for seed in [54051,54052]:
  n=Net();n.load_state_dict(t.load(ROOT/f'net{seed}/s0600.pt',weights_only=True)['model']);n.eval();nets.append(n)
 rows=[];atlas=[];on=[];traces=[];parents=json.loads((ROOT/'PARENT_LOCK.json').read_text());i0=0
 for s,mode,step,p in paths():
  if step!=6000:continue
  i0+=1;name=f'{s}_{mode}';assert parents[str(p.relative_to(PARENT))]==sha(p);m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(p,weights_only=True)['model']);m.eval().requires_grad_(False);b,d,f=pair_grid(s,54210+i0);b,d,f=b[:96],d[:96],f[:96];a,da,c0,c1,l0,l1,(y0,y1)=setup(m,b,d);feat,lin,ft=features9(m,a,da,c0,c1,l0,l1);target=l1.argmax(1);source=l0.argmax(1);hit=[];start=perf_counter()
  with t.no_grad():
   for k in range(512):hit.append((evaluate(m,a,da,c1,t.tensor(MASK9[k])[None].expand(len(a),-1)).argmax(1)==target).numpy())
   pr=[n((feat-mu)/sd).sigmoid().squeeze(-1).numpy() for n in nets]
  hit=np.stack(hit,1);elapsed=perf_counter()-start;assert hit[:,511].all() and hit[:,31].all() and np.array_equal(hit[:,0],(source==target).numpy());keep=(source!=target).numpy();mins=[];jointonly=0;crossbetter=0
  for i in np.flatnonzero(keep):
   h=hit[i];minimum=min(K9[np.flatnonzero(h)]);mins.append(minimum);layerone=min(K9[np.flatnonzero(h[:32])]);layertwo=np.flatnonzero(h & ((np.arange(512)&31)==0));l2min=min(K9[layertwo]) if len(layertwo) else 10;crossbetter+=minimum<min(layerone,l2min);jointonly+=not h[[1<<j for j in range(9)]].any()
   for meth,order in policy_order(lin[i],[x[i] for x in pr],i).items():
    q={0,511};found=-1
    for j in order[:8]:
     q.add(int(j));
     if h[j]:found=int(j);break
    cert=-1
    if found>=0:
     q.update(SUB9[found]);cert=min([k for k in q if k&found==k and h[k]],key=lambda k:(K9[k],k));assert not h[SUB9[cert]].any()
    rows.append(dict(backbone=name,i=int(i),method=meth,success=int(cert>=0),queries=len(q)-2,penalized=len(q)-2+512*int(cert<0),optimal=int(cert>=0 and K9[cert]==minimum),global_min=int(minimum),cert_size=int(K9[cert]) if cert>=0 else -1));traces.append([name,int(i),meth,sorted(q),cert])
   if not name.startswith('34001') and i<24:
    for meth in ['exhaustive','size_batches','neural_fallback']:
     cert,nq,seconds,fs=online(m,a[i:i+1],da[i:i+1],c1[i:i+1],l0[i:i+1],l1[i:i+1],meth,nets[0],mu,sd);assert h[cert] and not h[SUB9[cert]].any();on.append(dict(backbone=name,i=int(i),method=meth,queries=nq,seconds=seconds,gradient_feature_seconds=fs,certificate_size=int(K9[cert]),global_min=int(minimum),optimal=int(K9[cert]==minimum)))
  np.savez_compressed(ROOT/f'joint_{name}.npz',base=b.numpy(),donor=d.numpy(),field=f,hit=hit,features=feat.numpy(),source=source.numpy(),target=target.numpy(),truth0=y0.numpy(),truth1=y1.numpy());atlas.append(dict(backbone=name,pairs=len(b),distinct=int(keep.sum()),no_single=int(jointonly),cross_layer_improves=int(crossbetter),mean_min=float(np.mean(mins)),min1=mins.count(1),min2=mins.count(2),min3=mins.count(3),min4=mins.count(4),min5=mins.count(5),all_512_seconds=elapsed,feature_seconds=ft));print(atlas[-1],flush=True);assert sha(p)==parents[str(p.relative_to(PARENT))]
 dump(ROOT/'joint_atlas.csv',atlas);dump(ROOT/'joint_policy.csv',rows);dump(ROOT/'online.csv',on);(ROOT/'joint_traces.json').write_text(json.dumps(traces));print('DONE',len(rows),len(on),flush=True)
if __name__=='__main__':main()
