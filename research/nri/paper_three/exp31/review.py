"""Inspect and bind substantive cohort reviews to all authenticated cards."""
import argparse
import json
from pathlib import Path
import sys
from statistics import mean
from collections import Counter

from runner import jobs,sha,save_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('study')
    p.add_argument('step',type=int)
    p.add_argument('directory',type=Path)
    p.add_argument('--record',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    cards={}
    for k in jobs(a.study,a.smoke):
        path=a.directory/k/f'{a.step:04}.pt'
        x=json.loads(path.with_suffix('.json').read_text())
        assert x['hash']==sha(path)
        cards[k]=x
    groups={}
    for k,x in cards.items():
        group=k.split('-',1)[-1] if a.study!='prehost' else 'hosts'
        groups.setdefault(group,[]).append((k,x))
    evidence={}
    for group,rows in groups.items():
        def average(fn): return round(mean(fn(x) for _,x in rows),4)
        def top(x):
            return max(x['updates'],key=x['updates'].get)
        if a.study=='prehost':
            evidence[group]=dict(host=average(lambda x:x['metrics']['host']),per_host={k:x['metrics']['host'] for k,x in rows})
        else:
            dev=lambda x:x['metrics']['dev']
            vals=[dev(x)['authority'] for _,x in rows]
            top_site=Counter(top(x) for _,x in rows).most_common(1)[0]
            trace=next(iter(rows[0][1]['geometry']))
            evidence[group]=dict(authority=round(mean(vals),4),authority_range=[round(min(vals),4),round(max(vals),4)],
                                 integrity=average(lambda x:dev(x)['integrity']),
                                 isolated=average(lambda x:dev(x)['isolated']),
                                 ordinary=average(lambda x:dev(x)['accuracy']),
                                 spread=average(lambda x:dev(x).get('max_residual_spread',dev(x).get('spread_max'))),
                                 max_update=top_site[0],
                                 objective0_readout=average(lambda x:x['objective_readout'].get('0',0)),
                                 attention=average(lambda x:x['geometry'][trace]['attention2_last_to_first']),
                                 mlp=average(lambda x:x['geometry'][trace]['mlp_active']))
            if a.study=='repair' and a.step==0:
                evidence[group]['edit']=average(lambda x:x['metadata']['relative_conversion_readout_delta'])
            if a.step==100:
                evidence[group]['early']=[average(lambda x:x['early'][str(t)]['dev']['authority']) for t in (1,5,10,20,50)]
    for group,value in evidence.items():
        print(group, value,flush=True)
    if a.record:
        assessment=sys.stdin.read().strip()
        assert len(assessment.split())>=55
        save_json(a.directory/f'review-{a.step:04}.json',dict(
            hashes={k:x['hash'] for k,x in cards.items()},study=a.study,step=a.step,assessment=assessment,evidence=evidence,
            run_findings={k:dict(metric=x['metrics'],top_effective=max(x['updates'],key=x['updates'].get),
                                 gradients=x['trace'],objective_readout=x['objective_readout'],lineage=x.get('lineage',{})) for k,x in cards.items()}))


if __name__=='__main__':
    main()
