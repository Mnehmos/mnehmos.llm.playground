from pathlib import Path
import csv
import json
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

root=Path(sys.argv[1])
margin_path=Path(sys.argv[2])
out=Path(sys.argv[3])
colors=['#a73d40','#d08b25','#556d28','#255a9c','#855188','#39777d']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
def read(p):return json.loads(p.read_text())
rng=np.random.default_rng(832109)
bootstrap=rng.integers(0,12,(4000,12))
def plot_series(ax,ts,vals,color,label,levels=True):
    x=np.asarray(vals)
    mid=x.mean(0)
    ax.plot(ts,mid,label=label,color=color,lw=1.65)
    if levels:
        a,b=np.quantile(x[bootstrap].mean(1),[.025,.975],axis=0)
        ax.fill_between(ts,a,b,color=color,alpha=.12,lw=0)
    ax.set_xlabel('AdamW updates')
def save(fig,name):
    for fmt in ('png','pdf','svg'):
        fig.savefig(out/f'{name}.{fmt}',dpi=210)
    plt.close(fig)

groups={}
for task in ('obj','auto'):
    groups[task]=defaultdict(list)
    for p in sorted((root/task).glob('83*-*')):
        a=p.name.split('-')[1]
        z={t:read(p/f'{t:04}.json') for t in (0,100,200,300)}
        met={t:y['metrics'] for t,y in z.items()}
        met.update({int(t):x for t,x in z[100]['early'].items()})
        groups[task][a].append((met,z))

fig,ax=plt.subplots(1,3,figsize=(10,3.45),layout='constrained')
names=['ce','routed','satisfied','scale','ordinary','soft']
for c,a in zip(colors,names):
    run=groups['obj'][a];t=sorted(run[0][0])
    plot_series(ax[0],t,[[m[x]['effective_median'] for x in t] for m,_ in run],c,a)
    j=[0,100,200,300]
    plot_series(ax[1],j,[[max(1e-5,m[x]['gradients'].get('0',0)) for x in j] for _,m in run],c,a)
    plot_series(ax[2],t,[[m[x]['authority'] for x in t] for m,_ in run],c,a)
ax[0].axhline(0,color='.45',lw=.8,ls=':')
ax[0].set(title='A   Effective margin',ylabel='Seed median across swaps',xscale='symlog')
ax[1].set(title='B   Natural readout gradient',ylabel='Norm (floored at $10^{-5}$)',yscale='log')
ax[2].set(title='C   Objective-specific authority',ylabel='Correct swaps',ylim=(-.04,1.04))
ax[2].legend(loc='center right',frameon=False,fontsize=7)
save(fig,'Experiment_32_Objectives')

fig,ax=plt.subplots(1,3,figsize=(10,3.4),layout='constrained')
scores=read(root/'obj'/'scores.json')
for c,a in zip(colors,names):
    data=[v['final'] for k,v in scores.items() if k.endswith('-'+a)]
    x=[r['authority'] for r in data];y=[r['outside'] for r in data]
    ax[0].scatter(x,y,color=c,s=12,alpha=.24)
    ax[0].scatter(np.mean(x),np.mean(y),color=c,s=42,label=a)
ax[0].axvline(.95,ls=':',color='.5',lw=.8);ax[0].axhline(.9,ls=':',color='.5',lw=.8)
ax[0].set(xlabel='Sealed causal authority',ylabel='Flagged-task accuracy',title='A   Preservation and learning',xlim=(-.03,1.04))
ax[0].legend(loc='lower left',fontsize=6,frameon=False,ncol=2)
for c,a in zip(colors,('frozen','trainable','bounded','isolated','soft')):
    rr=groups['auto'][a];t=sorted(rr[0][0])
    plot_series(ax[1],t,[[m[x]['authority'] for x in t] for m,_ in rr],c,a)
ax[1].set(ylabel='Correct state swaps',title='B   Five-state automaton')
ax[1].legend(loc='lower left',fontsize=6,frameon=False,ncol=2)
mv=defaultdict(list)
with margin_path.open() as f:
  for r in csv.DictReader(f):
    if r['panel']=='swap_held' and r['study'] in ('e30','host') and r['run'].endswith('-B'):
        mv[(r['study'],int(r['step']))].append(r)
ts=list(range(0,601,100))
for label,col in [('e30','#a73d40'),('host','#255a9c')]:
    for field,ls in [('program_median','--'),('effective_median','-')]:
        y=[np.mean([float(x[field]) for x in mv[(label,t)]]) for t in ts]
        ax[2].plot(ts,y,color=col,ls=ls,lw=1.45,label=('Initial' if label=='e30' else 'Trained host')+(' program' if ls=='--' else ' effective'))
ax[2].axhline(0,color='.5',lw=.8,ls=':')
ax[2].set(xlabel='Updates after installation',ylabel='Seed median margin',title='C   Archived freezing-only runs')
ax[2].legend(fontsize=6,frameon=False,loc='lower left')
save(fig,'Experiment_32_Transfer')
