import sys, math, time, json, csv, hashlib
from pathlib import Path
import numpy as np
import torch as t
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'nri_G05_session12'
sys.path.insert(0,str(PARENT/'sources/nri_G05_session09'))
from experiment import paths,grid,aslat,T,TinyTyped,MODEL_KW,sha
from experiment import group
MASK=np.array([[(i>>j)&1 for j in range(5)] for i in range(32)],dtype=np.float32)
K=MASK.sum(1).astype(int); SUB=[[j for j in range(32) if j!=i and j&i==j] for i in range(32)]
t.set_num_threads(1);t.use_deterministic_algorithms(True)
def parts(att,x):
 b,l,d=x.shape;q,k,v=att.qkv(x).view(b,l,3,att.h,att.dh).unbind(2);q,k,v=[u.transpose(1,2) for u in (q,k,v)]
 h=((q@k.transpose(-1,-2)/math.sqrt(att.dh)).softmax(-1)@v).transpose(1,2)
 c=t.einsum('blhi,dhi->bhld',h,att.o.weight.reshape(d,att.h,att.dh))
 return t.cat([c,(x+att.o.bias)[:,None]],1)
def pair_grid(s,seed):
 z=grid(s);a=z.numpy();rng=np.random.default_rng(seed);lookup={tuple(row):i for i,row in enumerate(a)};ii=[];jj=[];field=[]
 for _ in range(384):
  i=int(rng.integers(len(a)));f=int(rng.integers(5));opts=[]
  for val in range([13,3,13,3,3][f]):
   if val==a[i,f]:continue
   b=a[i].copy();b[f]=val;j=lookup.get(tuple(b));
   if j is not None:opts.append(j)
  if not opts:raise RuntimeError('empty donor')
  ii.append(i);jj.append(int(rng.choice(opts)));field.append(f)
 return z[ii],z[jj],np.array(field)
def out(m,x,layer):
 if layer==1:x=(x+m.attn2(x))[:,-1]
 return m.head_from_resid(x)
def collect(m,b,d,layer):
 tok=T.build(aslat(t.cat([b,d]))).tokens;true=T.build(aslat(t.cat([b,d]))).y
 with t.no_grad():
  emb=m.embed(tok,t.zeros_like(tok));x=m.resid_layer1(tok,t.zeros_like(tok));p=parts(m.attn1,emb) if layer==1 else parts(m.attn2,x)[:,:,-1];native=m(tok,t.zeros_like(tok));a0,a1=native.chunk(2);n=len(b);p0,p1=p.chunk(2);base=p0.sum(1);delta=p1-p0
  direct=out(m,base,layer);donor=out(m,p1.sum(1),layer)
  assert t.equal(direct.argmax(1),a0.argmax(1)) and t.equal(donor.argmax(1),a1.argmax(1)); err=max(float((direct-a0).abs().max()),float((donor-a1).abs().max()))
  start=time.perf_counter();exact=[]
  for i in range(32):exact.append(out(m,base+delta[:,MASK[i].astype(bool)].sum(1),layer))
  exact=t.stack(exact,1);exact_time=time.perf_counter()-start
  assert t.equal(exact[:,0].argmax(1),a0.argmax(1)) and t.equal(exact[:,31].argmax(1),a1.argmax(1))
 # true local derivative, no mixed-mask oracle outcomes
 start=time.perf_counter();base=base.detach().requires_grad_();log=out(m,base,layer);J=[]
 for j in range(13):
  g=t.autograd.grad(log[:,j].sum(),base,retain_graph=j<12)[0];J.append((delta*g[:,None]).flatten(2).sum(2))
 local=t.stack(J,-1).detach();lin=a0[:,None]+t.einsum('mk,bkc->bmc',t.tensor(MASK),local);linear_time=time.perf_counter()-start
 target=a1.argmax(1);source=a0.argmax(1);one=F.one_hot(target,13).bool();margin=lambda x: (x*one[:,None]).sum(-1)-x.masked_fill(one[:,None],-t.inf).max(-1).values
 lm=margin(lin);bm=lm[:,0];actual=exact.argmax(-1);hit=actual==target[:,None]
 dn=delta.flatten(2).norm(dim=2);size=t.tensor(K,dtype=t.float32)[None].expand(n,-1);ms=t.tensor(MASK).bool()[None];single=lm[:,[1,2,4,8,16]]-bm[:,None];sumdn=t.einsum('mk,bk->bm',t.tensor(MASK),dn)/dn.sum(1,keepdim=True).clamp_min(1e-8);maxdn=dn[:,None].expand(-1,32,-1).masked_fill(~ms,0).max(-1).values/dn.sum(1,keepdim=True).clamp_min(1e-8);s=single[:,None].expand(-1,32,-1);selmax=s.masked_fill(~ms,-1e9).max(-1).values;selmin=s.masked_fill(~ms,1e9).min(-1).values;selmax[:,0]=selmin[:,0]=0;gain=lm-bm[:,None];scale=(a0.max(1).values-a0.min(1).values).clamp_min(1e-4)[:,None];
 feat=t.stack([size/5,bm[:,None].expand_as(lm)/scale,lm/scale,gain/scale,selmax/scale,selmin/scale,sumdn,maxdn,(lin.argmax(-1)==target[:,None]).float(),lm[:,31,None].expand_as(lm)/scale,t.full_like(lm,layer/2), (a1.max(1).values-a1.min(1).values)[:,None].expand_as(lm)/scale],-1)
 return dict(features=feat.numpy(),hit=hit.numpy(),linear=lm.numpy(),pred=actual.numpy(),source=source.numpy(),target=target.numpy(),truth0=true[:n].numpy(),truth1=true[n:].numpy(),endpoint_err=err,exact_seconds=exact_time,gradient_seconds=linear_time)
def mlp(m,b,d):
 with t.no_grad():
  tok=T.build(aslat(t.cat([b,d]))).tokens;h=m.resid_pre_mlp(tok,t.zeros_like(tok));h0,h1=h.chunk(2);z0,z1=F.relu(m.W_in(h)).chunk(2);l0=m.head_from_resid(h0);l1=m.head_from_resid(h1);src=l0.argmax(1);tar=l1.argmax(1);W=m.W_U.weight@m.W_out.weight;diff=z1-z0;gain=diff*(W[tar]-W[src]);rows=[];rng=np.random.default_rng(54099)
  for i in range(len(b)):
   if src[i]==tar[i]:continue
   margin=float(l0[i,tar[i]]-l0[i,src[i]]);g=gain[i];order=t.argsort(g,descending=True);cum=margin+g[order].cumsum(0);k=t.where(cum>0)[0];n=int(k[0])+1 if len(k) else -1
   # full target verification, residual fixed; exact neuron-only patch
   full=l0[i]+W@diff[i];minpred=-1;randompred=-1
   if n>=0:
    l=l0[i]+W[:,order[:n]]@diff[i,order[:n]];minpred=int(l.argmax()==tar[i]);rr=t.tensor(rng.permutation(256)[:n]);randompred=int((l0[i]+W[:,rr]@diff[i,rr]).argmax()==tar[i]);assert float(cum[n-1])>0 and (n==1 or float(cum[n-2])<=0)
   rows.append(dict(i=i,pairwise_min=n,sorted_target=minpred,random_target=randompred,all_neurons_target=int(full.argmax()==tar[i]),base_target_margin=margin))
 return rows
class Net(nn.Sequential):
 def __init__(self):super().__init__(nn.Linear(12,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
def eq(a,b):
 if isinstance(a,t.Tensor):return t.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b
def train(X,Y,seed,folder):
 folder.mkdir();t.manual_seed(seed);g=t.Generator().manual_seed(seed+1000);net=Net();opt=t.optim.AdamW(net.parameters(),lr=.002);p='training_features'
 for step in range(601):
  if step%100==0:
   dest=folder/f's{step:04}.pt';t.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=g.get_state(),torch_rng=t.get_rng_state(),step=step,parent=p),dest);p=sha(dest)
  if step==600:break
  i=t.randint(len(X),(256,),generator=g);loss=F.binary_cross_entropy_with_logits(net(X[i]).flatten(),Y[i]);opt.zero_grad();loss.backward();opt.step()
 return net

def dump(path,rows):
 with open(path,'w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 lock={p:h for h,p in [x.strip().split(maxsplit=1) for x in (PARENT/'PARENT_MANIFEST.sha256').read_text().splitlines()]};data=[];parents={};neur=[];atlas=[];tim=[]
 for s,mode,step,p in paths():
  if step!=6000:continue
  name=f'{s}_{mode}';rel=str(p.relative_to(PARENT));assert sha(p)==lock[rel];parents[rel]=sha(p);m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(p,weights_only=True)['model']);m.eval().requires_grad_(False);b,d,field=pair_grid(s,54010+len(parents));np.savez_compressed(ROOT/f'{name}_pairs.npz',base=b.numpy(),donor=d.numpy(),field=field)
  for layer in [1,2]:
   v=collect(m,b,d,layer);np.savez_compressed(ROOT/f'{name}_layer{layer}.npz',**v,field=field);tim.append(dict(backbone=name,layer=layer,endpoint_maxerr=v['endpoint_err'],exact_seconds=v['exact_seconds'],gradient_seconds=v['gradient_seconds']));keep=v['source']!=v['target'];h=v['hit'];mins=[];nonmon=0;nosingle=0;multi=0;allonly=0
   for i in np.flatnonzero(keep):
    win=np.flatnonzero(h[i]);mi=[int(j) for j in win if not h[i,SUB[j]].any()];assert mi;mins.append(min(K[mi]));nosingle+=min(K[mi])>1;allonly+=min(K[mi])==5;multi+=len(mi)>1;nonmon+=any(not h[i,j] for k in win for j in range(32) if k&j==k)
   atlas.append(dict(backbone=name,layer=layer,pairs=len(b),distinct_native=int(keep.sum()),donor_task_correct=int((v['target']==v['truth1']).sum()),both_native_task_correct=int(((v['target']==v['truth1'])&(v['source']==v['truth0'])&keep).sum()),no_single=int(nosingle),full_only=int(allonly),multiple_minimal=int(multi),nonmonotone=int(nonmon),mean_min=float(np.mean(mins)),min1=mins.count(1),min2=mins.count(2),min3=mins.count(3),min4=mins.count(4),min5=mins.count(5)));data.append((name,layer,v));print(name,layer,atlas[-1],flush=True)
  neur.extend(dict(backbone=name,**r) for r in mlp(m,b,d));assert sha(p)==parents[rel]
 (ROOT/'PARENT_LOCK.json').write_text(json.dumps(parents,indent=2));dump(ROOT/'atlas.csv',atlas);dump(ROOT/'neuron_scores.csv',neur);dump(ROOT/'timing.csv',tim)
 # only training lineage masks, exclude endpoints and unchanged native targets
 trainrows=[(v['features'][v['source']!=v['target'],1:31].reshape(-1,12),v['hit'][v['source']!=v['target'],1:31].reshape(-1)) for n,l,v in data if n.startswith('34001')];X=t.tensor(np.concatenate([x for x,y in trainrows]));Y=t.tensor(np.concatenate([y for x,y in trainrows]),dtype=t.float32);mu=X.mean(0);sd=X.std(0).clamp_min(1e-5);X=(X-mu)/sd;t.save(dict(mu=mu,sd=sd,X=X,Y=Y),ROOT/'train.pt');nets=[];rep=[]
 for seed in [54051,54052]:
  net=train(X,Y,seed,ROOT/f'net{seed}');nets.append(net);train(X,Y,seed,ROOT/f'replay{seed}');
  for step in range(0,601,100):
   a=t.load(ROOT/f'net{seed}/s{step:04}.pt',weights_only=True);b=t.load(ROOT/f'replay{seed}/s{step:04}.pt',weights_only=True);a.pop('parent');b.pop('parent');assert eq(a,b);rep.append([seed,step])
 (ROOT/'replay.json').write_text(json.dumps(rep));rows=[];traces=[]
 for name,layer,v in data:
  if name.startswith('34001'):continue
  with t.no_grad():scores=[net((t.tensor(v['features'])-mu)/sd).squeeze(-1).sigmoid().numpy() for net in nets]
  for i in np.flatnonzero(v['source']!=v['target']):
   h=v['hit'][i];rng=np.random.default_rng(54100+i+layer*1000);order=np.arange(1,31);rnd=rng.permutation(order);minimum=min(K[np.flatnonzero(h)]);methods={'random':rnd,'smallest_first':rnd[np.argsort(K[rnd],kind='stable')],'linear_margin':order[np.argsort(-v['linear'][i,order],kind='stable')],'linear_cost':order[np.argsort(-(v['linear'][i,order]-v['linear'][i,0])/(2.**K[order]),kind='stable')]}
   for j,p in enumerate(scores):
    methods[f'neural{j+1}']=order[np.argsort(-p[i,order],kind='stable')];methods[f'neural_cost{j+1}']=order[np.argsort(-p[i,order]/(2.**K[order]),kind='stable')]
   methods['full_certify']=np.array([31]);
   for method,ordr in methods.items():
    queried={0,31};found=-1;proposals=0
    for q in ordr[:8]:
     q=int(q);queried.add(q);proposals+=1
     if h[q]:found=q;break
    cert=-1
    if found>=0:
     queried.update(SUB[found]);w=[k for k in queried if k&found==k and h[k]];cert=min(w,key=lambda q:(K[q],q));assert not h[SUB[cert]].any()
    rows.append(dict(backbone=name,layer=layer,i=int(i),method=method,success=int(cert>=0),calls=len(queried)-2,proposals=proposals,certificate_size=int(K[cert]) if cert>=0 else -1,global_min=int(minimum),optimal=int(cert>=0 and K[cert]==minimum),penalized=len(queried)-2+32*int(cert<0),donor_correct=int(v['target'][i]==v['truth1'][i])));traces.append([name,layer,int(i),method,sorted(queried),cert])
 dump(ROOT/'policy_scores.csv',rows);(ROOT/'traces.json').write_text(json.dumps(traces));print('DONE',len(rows),flush=True)
if __name__=='__main__':main()
