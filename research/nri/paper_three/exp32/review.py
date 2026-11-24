"""Review gate for each experimental cohort."""
from collections import Counter
from pathlib import Path
from statistics import mean
import json
import sys
from experiment import names,put,sha

kind,task,where,num,*record=sys.argv[1:]
step=int(num);root=Path(where)
groups={}
cards={}
for seed,arm in names(kind,task):
    tag=f'{seed}-{arm}'; p=root/tag/f'{step:04}.pt'
    x=json.loads(p.with_suffix('.json').read_text())
    assert x['hash']==sha(p)
    groups.setdefault(arm,[]).append(x)
    cards[tag]=x
for arm,x in groups.items():
    fields=('authority','ordinary','outside','route','isolated','conditional','residual_max_spread','effective_median')
    print(arm,{k:round(mean(r['metrics'][k] for r in x if r['metrics'].get(k) is not None),4)
               for k in fields if any(r['metrics'].get(k) is not None for r in x)},
          'ordinary gradient',round(mean(r['gradients'].get('0',0) for r in x),4),
          'traces',[(k,round(mean(r['traces']['ordinary'][k] for r in x),3)) for k in ('occurrence','state','attention','active')],
          'top',Counter(r['max_update'] for r in x).most_common(1),
          'early',[round(mean(r['early'][t]['authority'] for r in x),4) for t in x[0]['early']])
if record:
    note=sys.stdin.read().strip()
    assert len(note.split())>=55
    put(root/f'review-{step:04}.json',dict(step=step,assessment=note,hashes={k:x['hash'] for k,x in cards.items()},
         per_network={k:{f:x[f] for f in ('metrics','traces','gradients','max_update')} for k,x in cards.items()}))
