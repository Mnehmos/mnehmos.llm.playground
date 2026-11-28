import sys, math, time, json, csv, hashlib, copy
from pathlib import Path
import numpy as np
import torch as t
import torch.nn.functional as F
from scipy.stats import spearmanr
R=Path(__file__).resolve().parent
P=R.parent/'g05_s15_parents/nri_G05_session12'
sys.path.insert(0,str(P/'sources/nri_G05_session09'))
from experiment import paths,grid,aslat,T,TinyTyped,MODEL_KW,sha,group

t.set_num_threads(1); t.use_deterministic_algorithms(True)
def pieces(att,x):
 b,l,d=x.shape;q,k,v=att.qkv(x).view(b,l,3,att.h,att.dh).unbind(2);q,k,v=[u.transpose(1,2) for u in (q,k,v)]
 h=((q@k.transpose(-1,-2)/math.sqrt(att.dh)).softmax(-1)@v).transpose(1,2)
 return t.einsum('blhi,dhi->bhld',h,att.o.weight.reshape(d,att.h,att.dh))
def score(m,b,y):
 with t.no_grad():
  l=m(b,t.zeros_like(b));p=l.argmax(1);return dict(loss=float(F.cross_entropy(l,y)),sign=float(F.softplus(l.gather(1,((-y)%13)[:,None]).flatten()-l.gather(1,y[:,None]).flatten()).mean()),exact=float((p==y).float().mean()),quotient=float((group(l)==t.minimum(y,(-y)%13)).float().mean()))
def signloss(m,b,y):
 l=m(b,t.zeros_like(b));return F.softplus(l.gather(1,((-y)%13)[:,None]).flatten()-l.gather(1,y[:,None]).flatten()).mean()
def params(m,layer,h):
 att=getattr(m,'attn'+str(layer));d=m.d;dh=att.dh
 ix=t.cat([t.arange(k*d+h*dh,k*d+(h+1)*dh) for k in range(3)])
 jj=t.arange(h*dh,(h+1)*dh)
 return [(att.qkv.weight,(ix,slice(None))), (att.qkv.bias,ix), (att.o.weight,(slice(None),jj))]
def flat(gs,ps):return t.cat([g[k].reshape(-1) for g,(_,k) in zip(gs,ps)])
def adjust(ps,delta):
 n=0
 with t.no_grad():
  for p,k in ps:
   z=p[k];q=delta[n:n+z.numel()].reshape_as(z);p[k]=z+q;n+=z.numel()
 assert n==len(delta)
def patch(m,b,d,y):
 with t.no_grad():
  z=t.cat([b,d]);e=m.embed(z,t.zeros_like(z));x=e+m.attn1(e);p1=pieces(m.attn1,e);p2=pieces(m.attn2,x);n=len(b);base=m(b,t.zeros_like(b));out=[]
  def margin(l):return l.gather(1,y[:,None]).flatten()-l.gather(1,((-y)%13)[:,None]).flatten()
  for lay,p in [(1,p1),(2,p2)]:
   for h in range(4):
    diff=p[n:,h]-p[:n,h]
    if lay==1:
     a=x[:n]+diff;res=(a+m.attn2(a))[:,-1]
    else:res=(x[:n]+m.attn2(x[:n]))[:,-1]+diff[:,-1]
    l=m.head_from_resid(res);out.append((lay,h,float((l.argmax(1)==y).float().mean()),float((margin(l)-margin(base)).mean())))
 return out

def main():
 locks=json.loads((R/'PARENT_LOCK.json').read_text());rows=[];meta=[];summ=[]
 for s,mode,step,path in paths():
  if step!=6000:continue
  name=f'{s}_{mode}';assert sha(path)==locks[str(path.relative_to(P))];m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(path,weights_only=True)['model']);m.eval();z=grid(s);batch=T.build(aslat(z));x,y=batch.tokens,batch.y;M,A,S=T.canon(aslat(z));C=A*S%13
  with t.no_grad():pred=t.cat([m(b,t.zeros_like(b)).argmax(1) for b in x.split(512)])
  rng=np.random.default_rng(55101+s);unique,idx=np.unique(z[:,:5].numpy(),axis=0,return_inverse=True);train=t.tensor(np.isin(idx,rng.permutation(len(unique))[:int(.6*len(unique))]));err=(y!=0)&(pred==(-y)%13);good=pred==y
  pools=[t.where(q)[0].numpy() for q in [train&err,train&good,~train&err,~train&good]]
  if min(map(len,pools))<128:meta.append(dict(name=name,status='skip_insufficient',pool_counts=list(map(len,pools))));continue
  ti,ki,vi,ri=[t.tensor(rng.choice(p,128,replace=False)) for p in pools];donors=[];fallback=0
  for i in ti:
   choices=t.where(train&good&(M==M[i])&(C==C[i]))[0];
   if not len(choices):choices=t.where(train&good&(y==y[i]))[0];fallback+=1
   assert len(choices);donors.append(int(rng.choice(choices.numpy())))
  di=t.tensor(donors);assert (y[di]==y[ti]).all();assert not (train[vi]|train[ri]).any();
  t.save(dict(z=z,train=ti,preserve=ki,exam=vi,retention=ri,donor=di,split=train),R/f'{name}_panel.pt');meta.append(dict(name=name,status='run',pool_counts=list(map(len,pools)),fallback=fallback))
  pat=patch(m,x[ti],x[di],y[ti]);initial=copy.deepcopy(m.state_dict());base_e=score(m,x[vi],y[vi]);base_r=score(m,x[ri],y[ri]);
  for layer,h,hit,pm in pat:
   for arm in ['target','protected']:
    m.load_state_dict(initial);ps=params(m,layer,h);pars=[p for p,k in ps];gn0=None;start=time.perf_counter();proj=0;total_disp=0.;maxerr=0.
    for k in range(21):
     if k in [0,1,5,20]:
      es=score(m,x[vi],y[vi]);rs=score(m,x[ri],y[ri]);rows.append(dict(backbone=name,layer=layer,head=h,arm=arm,step=k,patch_hit=hit,patch_margin=pm,grad_norm=gn0,exam_sign=es['sign'],exam_ce=es['loss'],exam_exact=es['exact'],exam_quotient=es['quotient'],retention_exact=rs['exact'],retention_quotient=rs['quotient'],retention_ce=rs['loss'],sign_gain=base_e['sign']-es['sign'],retention_drift=rs['loss']-base_r['loss'],projected_steps=proj,displacement=total_disp,cpu_seconds=time.perf_counter()-start))
     if k==20:break
     g=flat(t.autograd.grad(signloss(m,x[ti],y[ti]),pars),ps);gn=float(g.norm());
     if k==0:gn0=gn;rows[-1]['grad_norm']=gn
     v=-g
     if arm=='protected':
      loss=F.cross_entropy(m(x[ki],t.zeros_like(x[ki])),y[ki]);a=flat(t.autograd.grad(loss,pars),ps);dot=v@a
      if dot>0:v=v-dot*a/(a@a).clamp_min(1e-20);proj+=1
     d=v*(.04/v.norm().clamp_min(1e-12));before=[p[k].detach().clone() for p,k in ps];adjust(ps,d);actual=t.cat([(p[k]-b).flatten() for (p,k),b in zip(ps,before)]);total_disp+=float(actual.norm());maxerr=max(maxerr,abs(float(actual.norm())-.04));
    final=m.state_dict();allowed={id(p):k for p,k in ps}
    for key,p in m.named_parameters():
     diff=p.detach()-initial[key];
     if id(p) in allowed:diff=diff.clone();diff[allowed[id(p)]]=0
     assert not diff.count_nonzero(),key
    assert maxerr<1e-5,maxerr
    t.save(dict(model=final,parent_sha=sha(path),layer=layer,head=h,arm=arm,steps=20,displacement=total_disp),R/f'{name}_L{layer}H{h}_{arm}.pt')
  assert sha(path)==locks[str(path.relative_to(P))];print(name,meta[-1],flush=True)
  for arm in ['target','protected']:
   for k in [1,20]:
    rr=[r for r in rows if r['backbone']==name and r['arm']==arm and r['step']==k];best=max(r['sign_gain'] for r in rr)
    for f in ['patch_hit','patch_margin','grad_norm']:
     chosen=max(rr,key=lambda r:r[f]);summ.append(dict(backbone=name,arm=arm,step=k,selector=f,spearman=float(spearmanr([r[f] for r in rr],[r['sign_gain'] for r in rr]).statistic),gain=chosen['sign_gain'],oracle_gain=best,regret=best-chosen['sign_gain'],exam_exact=chosen['exam_exact'],retention_exact=chosen['retention_exact'],retention_quotient=chosen['retention_quotient']))
  for file,data in [('scores.csv',rows),('summary.csv',summ)]:
   with open(R/file,'w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
  (R/'panels.json').write_text(json.dumps(meta,indent=2))
 (R/'panels.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
