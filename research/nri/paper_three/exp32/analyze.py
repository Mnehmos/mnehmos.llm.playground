"""Registered seed analysis and result cards independent of model inference."""
from pathlib import Path
import argparse,csv,itertools,json
from collections import defaultdict
from statistics import mean
import numpy as np
from experiment import sha,names

CONTRASTS={
 'obj':(('satisfied','ce'),('scale','ce'),('routed','ce'),('soft','ordinary')),
 'auto':(('trainable','frozen'),('bounded','frozen'),('isolated','frozen'),('soft','frozen'))}

def read(p):return json.loads(p.read_text())

def analyze(root):
    rng=np.random.default_rng(832076)
    ii=rng.integers(0,12,size=(10000,12))
    sign=np.asarray(list(itertools.product((-1,1),repeat=12)))
    results={};rows=[]
    for task in ('obj','auto'):
      where=root/task
      public=read(where/'public.json');pri=read(where/'released.json')
      assert sha(where/'released.json')==public['commitment']
      sets=[set(public['train']),set(public['dev']),set(pri['hidden']),set(pri['final'])]
      assert len(set.union(*sets))==sum(len(x) for x in sets)
      scores={k:v['final'] for k,v in read(where/'scores.json').items()}
      matched=defaultdict(dict); group=defaultdict(list); forecast=defaultdict(list)
      for s,a in names('study',task):
        k=f'{s}-{a}';d=scores[k];matched[a][s]=d
        t100=read(where/k/'0100.json')
        middles=[read(where/k/f'0{t}.json')['metrics'] for t in (200,300)]
        qualification=lambda p:p['authority']>=.95 and p['ordinary']>=.95 and p['isolated']>=.95
        qualifies=qualification(d) and all(qualification(p) for p in middles) and (task=='auto' or d['outside']>=.9)
        m20,m50=[t100['early'][str(t)]['effective_median'] for t in (20,50)]
        signal=m50+(50/30)*(m50-m20)
        actual=d['authority']<.8
        for x in ('all','unrestricted') if a not in ('bounded','isolated') else ('all',):
            forecast[x].append((int(signal<=0),int(t100['early']['50']['authority']<.8),int(actual)))
        group[a].append(dict(authority=d['authority'],ordinary=d['ordinary'],isolated=d['isolated'],
                             unpatched=d['unpatched'],conditional=d['conditional'],outside=d.get('outside'),practical=qualifies))
        rows.append(dict(task=task,seed=s,arm=a,practical=qualifies,forecast_margin=signal,**d))
      contrasts={}
      for a,b in CONTRASTS[task]:
        x=np.array([matched[a][s]['authority']-matched[b][s]['authority'] for s in sorted(matched[a])])
        m=x.mean();p=(np.abs((sign*x).mean(1))>=abs(m)-1e-12).mean()
        contrasts[f'{a}-{b}']={'mean':float(m),'ci':np.quantile(x[ii].mean(1),[.025,.975]).tolist(),'raw_p':float(p),'paired':x.tolist()}
      prior=0.
      for j,k in enumerate(sorted(contrasts,key=lambda k:contrasts[k]['raw_p'])):
        prior=min(1,max(prior,(4-j)*contrasts[k]['raw_p']))
        contrasts[k]['holm']=prior
      forecast_counts={}
      for who,pred in forecast.items():
        forecast_counts[who]={n:{f'{i}:{j}':sum(p[k]==i and p[-1]==j for p in pred)
                             for i,j in itertools.product((0,1),repeat=2)} for k,n in enumerate(('slope','simple'))}
      summ={a:{k:mean(y[k] for y in v if y[k] is not None) for k in v[0] if any(y[k] is not None for y in v)}
            for a,v in group.items()}
      results[task]={'endpoint':summ,'counts':{k:next(iter(scores.values()))[k] for k in ('n','bases','unique_pairs')},
                     'contrasts':contrasts,'forecast':forecast_counts,
                     'host_custody':'same-machine separated process'}
    fields=sorted(set().union(*(x.keys() for x in rows)))
    with (root/'seed_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator=chr(10));w.writeheader();w.writerows(rows)
    (root/'analysis.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    analyze(p.parse_args().root)
