"""Hash-bound review receipts with measurements for each network and an authored synthesis."""
import argparse
import json
from pathlib import Path
import sys
from statistics import mean

from confirm import SEEDS, STEPS
import study_model as model


def inspect(root,step,seeds):
    runnames=[f'{s}-{a}' for s in seeds for a in model.ARMS]
    cards={}
    for run in runnames:
        ck=root/run/f'{step:04}.pt'
        card=json.loads(ck.with_suffix('.json').read_text())
        assert card['hash']==model.hash_file(ck)
        cards[run]=card
    return cards


def summarize(cards):
    out={}
    for a in model.ARMS:
        c=[v for k,v in cards.items() if k.endswith(a)]
        metrics=lambda key,field:[i['metrics'][key][field] for i in c]
        authority=metrics('dev','authority')
        attn=max(c,key=lambda x:x['trace']['practice'].get('embedding',0))
        updates=[(max(i['update'],key=lambda k:i['update'][k][2])) for i in c]
        out[a]={'dev_authority':mean(authority),'range':[min(authority),max(authority)],
                'practiced':mean(metrics('practiced','accuracy')),
                'integrity':mean(metrics('dev','integrity')),'isolated':mean(metrics('dev','isolated')),
                'residual_spread':mean(metrics('dev','max_residual_spread')),
                'routing':mean(metrics('routing','held_form')), 'force':mean(metrics('force','held_pair')),
                'max_occurrence':attn['trace']['practice'].get('embedding',0),
                'top_effective_updates':{k:updates.count(k) for k in sorted(set(updates))}}
        if c[0]['early']:
            out[a]['early']={t:mean(i['early'][t]['dev']['authority'] for i in c) for t in c[0]['early']}
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('step',type=int,choices=STEPS)
    p.add_argument('--record',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    cards=inspect(a.root,a.step,(830617,) if a.smoke else SEEDS)
    summary=summarize(cards)
    if not a.record:
        print(json.dumps(summary,indent=2))
        for arm in model.ARMS:
            subset={k:v for k,v in cards.items() if k.endswith(arm)}
            low=min(subset,key=lambda k:subset[k]['metrics']['dev']['authority'])
            high=max(subset,key=lambda k:subset[k]['metrics']['dev']['authority'])
            print(arm,'extremes',low,high)
        return
    review=sys.stdin.read().strip()
    assert len(review.split())>=60, 'A substantive authored assessment is mandatory'
    notes={}
    for k,v in cards.items():
        d=v['metrics']['dev']
        top=max(v['update'],key=lambda name:v['update'][name][2])
        g=v['trace']['practice']
        notes[k]=f"Observed dev authority {d['authority']:.5f}; isolated {d['isolated']:.5f}; max logit spread {d['max_residual_spread']:.5f}; strongest mean effective update {top} {v['update'][top][2]:.6f}; sampled practice occurrence norm {g.get('embedding',0):.6f}."
    with (a.root/f'review-{a.step:04}.json').open('x') as f:
        json.dump(dict(step=a.step,assessment=review,hashes={k:v['hash'] for k,v in cards.items()},
                       notes=notes,cohort_summary=summary),f,indent=2)


if __name__=='__main__':
    main()
