"""Independent historical replay for the initialization pilot."""
import csv
import importlib.util
from pathlib import Path
import sys
import torch
from margins import decomposition
torch.set_num_threads(1)
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('prior',root/'experiment.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
rows=[]
with torch.no_grad():
 for cp in sorted((root/'runs').glob('*/*.pt')):
  seed,arm=cp.parent.name.split('-')
  s=prior.Student(int(seed),arm)
  s.load_state_dict(torch.load(cp,weights_only=True)['model'])
  i,j,y,_=prior.SWAPS
  val,unit,dest=prior.CV.T
  q=s.encoder[unit[j],val[j]]
  ph=torch.einsum('ni,nij->nj',q,s.decoder[dest[i]])
  z=s.residual(prior.CV_TOK[i],torch.zeros_like(prior.CV_TOK[i]))[:,:13]
  r=z*0 if arm=='C' else .2*z.tanh() if arm=='D' else z
  p,c,m,lo,hi,valid=decomposition(ph,r,y)
  rows.append(dict(run=cp.parent.name,step=int(cp.stem),n=len(i),program_median=float(p.min(1).values.median()),
                 competing_median=float(c.max(1).values.median()),effective_median=float(m.min(1).values.median()),
                 any_possible_attenuation=float((valid&(hi>0)&(lo<1)).float().mean())))
w=csv.DictWriter(sys.stdout,fieldnames=list(rows[0]))
w.writeheader();w.writerows(rows)
