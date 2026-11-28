from pilot import *

def pair_metrics(m,b,y):
 with t.no_grad():
  l=m(b,t.zeros_like(b));p=l.argmax(1);n=len(b)//2;q=group(l)==t.minimum(y,(-y)%13)
  marg=l.gather(1,y[:,None]).flatten()-l.gather(1,((-y)%13)[:,None]).flatten();good=p==y
  return dict(sign=float(F.softplus(-marg).mean()),exact=float(good.float().mean()),both=float((good[:n]&good[n:]).float().mean()),correct=float(good[:n].float().mean()),wrong=float(good[n:].float().mean()),quotient=float(q.float().mean()),distinction=float(((marg[:n]+marg[n:])/2).mean()),bias_abs=float(((marg[:n]-marg[n:])/2).abs().mean()))
def obj(m,b,y,arm):
 l=m(b,t.zeros_like(b));marg=l.gather(1,y[:,None]).flatten()-l.gather(1,((-y)%13)[:,None]).flatten();n=len(b)//2
 return F.softplus(-marg).mean() if arm=='balanced' else F.softplus(-(marg[:n]+marg[n:])/2).mean()
def quotientloss(m,b,y):
 l=m(b,t.zeros_like(b));g=t.cat([l[:,:1],t.logaddexp(l[:,1:7],l[:,[12,11,10,9,8,7]])],1);return F.cross_entropy(g,t.minimum(y,(-y)%13))
def gflat(loss,pars,ps):return flat(t.autograd.grad(loss,pars),ps)
def make_panel(m,s,name,z,x,y):
 with t.no_grad():pred=t.cat([m(b,t.zeros_like(b)).argmax(1) for b in x.split(512)])
 prev=t.load(R/f'{name}_panel.pt',weights_only=True);assert t.equal(z,prev['z']);used=t.cat([prev[k] for k in ['train','preserve','exam','retention','donor']]);Z=z.numpy();
 def gid(row):return (min(int(row[0]),int(-row[0])%13),*map(int,row[1:5]))
 ban={gid(Z[i]) for i in used.tolist()};lookup={tuple(r):i for i,r in enumerate(Z)};pairs={}
 for i,r in enumerate(Z):
  if y[i]==0 or pred[i]!=y[i]:continue
  b=r.copy();b[0]=int(-b[0])%13;j=lookup.get(tuple(b));k=gid(r)
  if j is None or k in ban or pred[j]!=y[i] or pred[j]==y[j]:continue
  assert y[j]==(-y[i])%13;pairs.setdefault(k,[]).append((i,j))
 if len(pairs)<192:return None,len(pairs)
 rng=np.random.default_rng(55200+s);keys=sorted(pairs);chosen=rng.choice(len(keys),192,replace=False);pp=[pairs[keys[i]][int(rng.integers(len(pairs[keys[i]])))] for i in chosen]
 panels={k:t.tensor(pp[i:i+64]).T.flatten() for k,i in [('train',0),('cal',64),('exam',128)]};
 assert len({gid(Z[i]) for v in panels.values() for i in v.tolist()})==192
 # independent correct same-target donors from train pairs, prefer same canonical coordinates
 M,A,S=T.canon(aslat(z));C=A*S%13;ti=panels['train'];good=ti[:64];wrong=ti[64:];ds=[];fallback=0
 # all eligible target values need not occur on correct half of 64 pairs. Permit extra familiar native-correct donor only outside banned and cal/exam groups.
 blocked=ban|{gid(Z[i]) for k in ['cal','exam'] for i in panels[k].tolist()};allow=t.tensor([gid(r) not in blocked for r in Z]);
 for i in wrong:
  c=t.where(allow&(pred==y)&(M==M[i])&(C==C[i]))[0]
  if not len(c):c=t.where(allow&(pred==y)&(y==y[i]))[0];fallback+=1
  assert len(c);ds.append(int(rng.choice(c.numpy())))
 panels['donor']=t.tensor(ds);panels['fallback']=fallback;panels['z']=z
 return panels,len(pairs)
def main():
 locks=json.loads((R/'PARENT_LOCK.json').read_text());rows=[];meta=[];summary=[];out=R/'pair_states';out.mkdir(exist_ok=True)
 for s,mode,step,path in paths():
  if step!=6000:continue
  name=f'{s}_{mode}';assert sha(path)==locks[str(path.relative_to(P))];
  if not (R/f'{name}_panel.pt').exists():continue
  m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(path,weights_only=True)['model']);m.eval();z=grid(s);ba=T.build(aslat(z));x,y=ba.tokens,ba.y
  panel,n=make_panel(m,s,name,z,x,y)
  if panel is None:meta.append(dict(backbone=name,status='insufficient',groups=n));continue
  t.save(panel,out/f'{name}_panel.pt');meta.append(dict(backbone=name,status='run',groups=n,donor_fallback=panel['fallback']));ti,ci,ei=[panel[k] for k in ['train','cal','exam']];di=panel['donor'];pat=patch(m,x[ti[64:]],x[di],y[ti[64:]]);initial=copy.deepcopy(m.state_dict());base=pair_metrics(m,x[ei],y[ei]);
  for layer,h,hit,pm in pat:
   for arm in ['balanced','pair','pair_protected']:
    m.load_state_dict(initial);ps=params(m,layer,h);pars=[p for p,k in ps];start=time.perf_counter();proj=0;total=0.;gn0=transfer=None
    for k in range(21):
     if k in [0,1,5,20]:
      es=pair_metrics(m,x[ei],y[ei]);row=dict(backbone=name,layer=layer,head=h,arm=arm,step=k,patch_hit=hit,patch_margin=pm,grad_norm=gn0,transfer=transfer,sign_gain=base['sign']-es['sign'],**es,projected_steps=proj,displacement=total,cpu_seconds=time.perf_counter()-start);rows.append(row)
      if k>0:t.save(dict(parent_sha=sha(path),layer=layer,head=h,arm=arm,step=k,weights=[p[key].detach().clone() for p,key in ps],displacement=total),out/f'{name}_L{layer}H{h}_{arm}_{k}.pt')
     if k==20:break
     g=gflat(obj(m,x[ti],y[ti],arm),pars,ps);v=-g
     if arm=='pair_protected':
      a=gflat(quotientloss(m,x[ti],y[ti]),pars,ps);dot=v@a
      if dot>0:v=v-dot*a/(a@a).clamp_min(1e-20);proj+=1
     d=v*(.04/v.norm().clamp_min(1e-12))
     if k==0:
      gn0=float(g.norm());cg=gflat(obj(m,x[ci],y[ci],'balanced'),pars,ps);transfer=float(-cg@d);row.update(grad_norm=gn0,transfer=transfer)
     before=[p[key].detach().clone() for p,key in ps];adjust(ps,d);act=t.cat([(p[key].detach()-b).flatten() for (p,key),b in zip(ps,before)]);assert abs(float(act.norm())-.04)<1e-5;total+=float(act.norm())
    allowed={id(p):k for p,k in ps}
    for key,p in m.named_parameters():
     diff=p.detach()-initial[key]
     if id(p) in allowed:diff=diff.clone();diff[allowed[id(p)]]=0
     assert not diff.count_nonzero(),key
  assert sha(path)==locks[str(path.relative_to(P))];print(name,meta[-1],flush=True)
  for arm in ['balanced','pair','pair_protected']:
   for k in [1,20]:
    rr=[r for r in rows if r['backbone']==name and r['arm']==arm and r['step']==k];best=max(r['sign_gain'] for r in rr)
    for f in ['patch_hit','patch_margin','grad_norm','transfer']:
     ch=max(rr,key=lambda r:r[f]);summary.append(dict(backbone=name,arm=arm,step=k,selector=f,spearman=float(spearmanr([r[f] for r in rr],[r['sign_gain'] for r in rr]).statistic),gain=ch['sign_gain'],oracle_gain=best,regret=best-ch['sign_gain'],exact=ch['exact'],both=ch['both'],quotient=ch['quotient']))
  for file,data in [('pair_scores.csv',rows),('pair_summary.csv',summary)]:
   with open(R/file,'w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
  (R/'pair_panels.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
