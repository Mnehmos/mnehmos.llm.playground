import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root=Path('conversion-run')
rows=list(csv.DictReader((root/'trajectory.csv').open()))
xs=[int(r['step']) for r in rows]
fig,ax=plt.subplots(2,1,sharex=True,figsize=(10,6),layout='constrained')
for key,label,c in [
  ('practiced','Practiced conversion','#174974'),
  ('swap_value','Layer-one value swap','#c24d00'),
  ('reverse','Reverse probe','#5a2d82')]:
 ax[0].plot(xs,[float(r[key]) for r in rows],label=label,color=c,lw=2)
for key,label,c in [
  ('dev/convert_ood_value','Held-out unit–value conversion','#c24d00'),
  ('dev/convert_newfam','New conversion frame','#174974'),
  ('dev/force_id_trainfam','Familiar-frame force','#377349')]:
 ax[1].plot(xs,[float(r[key]) for r in rows],label=label,color=c,lw=1.8)
for a in ax:
 a.yaxis.set_major_formatter(PercentFormatter(1))
 a.grid(axis='y',alpha=.25);a.spines[['top','right']].set_visible(False)
 a.set_ylabel('Correct')
 a.legend(loc='best',frameon=False,fontsize=9)
ax[0].set_ylim(-.03,1.05);ax[1].set_ylim(-.01,.36)
ax[-1].set_xlabel('Optimizer updates')
fig.suptitle('One trainee across 61 scientific checkpoints',x=.04,ha='left',fontsize=15)
fig.savefig(root/'trajectory.png',dpi=160)
