from pair import *

def main():
 locks=json.loads((R/'PARENT_LOCK.json').read_text());rows=[];corr=[]
 for s,mode,step,path in paths():
  if step!=6000:continue
  name=f'{s}_{mode}';pp=R/'pair_states'/f'{name}_panel.pt'
  if not pp.exists():continue
  assert sha(path)==locks[str(path.relative_to(P))];m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(path,weights_only=True)['model']);m.double().eval();panel=t.load(pp,weights_only=True);ba=T.build(aslat(panel['z']));x,y=ba.tokens,ba.y
  I={k:panel[k] for k in ['train','cal','exam']};base={k:float(obj(m,x[i],y[i],'balanced').detach()) for k,i in I.items()};initial=copy.deepcopy(m.state_dict())
  for layer in [1,2]:
   for h in range(4):
    m.load_state_dict(initial);ps=params(m,layer,h);pars=[p for p,k in ps];g={k:gflat(obj(m,x[i],y[i],'balanced'),pars,ps) for k,i in I.items()};v=-g['train']/g['train'].norm().clamp_min(1e-30)
    cos=lambda a,b:float(g[a]@g[b]/(g[a].norm()*g[b].norm()).clamp_min(1e-30))
    for radius in [.0001,.001,.01,.04]:
     m.load_state_dict(initial);d=v*radius;adjust(ps,d);pred={k:float(-a@d) for k,a in g.items()};actual={k:base[k]-float(obj(m,x[i],y[i],'balanced').detach()) for k,i in I.items()};rows.append(dict(backbone=name,layer=layer,head=h,radius=radius,cos_train_cal=cos('train','cal'),cos_train_exam=cos('train','exam'),cos_cal_exam=cos('cal','exam'),**{f'pred_{k}':v for k,v in pred.items()},**{f'actual_{k}':v for k,v in actual.items()}))
  assert sha(path)==locks[str(path.relative_to(P))];print(name,flush=True)
  for rad in [.0001,.001,.01,.04]:
   rr=[r for r in rows if r['backbone']==name and r['radius']==rad];
   for a,b in [('pred_train','actual_train'),('pred_cal','actual_cal'),('pred_exam','actual_exam'),('pred_cal','actual_exam'),('actual_cal','actual_exam')]:
    u=np.array([r[a] for r in rr]);v=np.array([r[b] for r in rr]);corr.append(dict(backbone=name,radius=rad,predictor=a,target=b,spearman=float(spearmanr(u,v).statistic),sign_agree=float(np.mean(np.sign(u)==np.sign(v))),mae=float(np.mean(np.abs(u-v))),mean_actual_abs=float(np.mean(np.abs(v)))))
 for file,data in [('local_scores.csv',rows),('local_summary.csv',corr)]:
  with open(R/file,'w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
if __name__=='__main__':main()
