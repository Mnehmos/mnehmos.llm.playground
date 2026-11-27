from topology import *
import pandas as pd
from collections import Counter
# Independently audit stored intervention certificates against exhaustive maps.
N=0;success=0;certs=0
for name,layer,i,method,queried,cert in json.loads((ROOT/'traces.json').read_text()):
 # lazily cache atlas arrays
 if N==0:cache={}
 k=(name,layer)
 if k not in cache:
  with np.load(ROOT/f'{name}_layer{layer}.npz') as v:cache[k]=v['hit']
 h=cache[k][i];assert not h[0] and h[31]
 assert len(set(queried))==len(queried)
 if cert>=0:
  assert h[cert] and cert in queried
  for q in range(cert):
   if q&cert==q:assert not h[q] and q in queried
  certs+=1
 N+=1
J=0;JC=0;cache={}
for name,i,method,queried,cert in json.loads((ROOT/'joint_traces.json').read_text()):
 if name not in cache:
  with np.load(ROOT/f'joint_{name}.npz') as v:cache[name]=v['hit']
 h=cache[name][i];assert not h[0] and h[511];assert len(set(queried))==len(queried)
 if cert>=0:
  assert h[cert] and cert in queried
  for q in range(cert):
   if q&cert==q:assert q in queried and not h[q]
  JC+=1
 J+=1
# All original archive dependencies in the S12 parent manifest (source and exact weights).
locks=[line.strip().split(maxsplit=1) for line in (PARENT/'PARENT_MANIFEST.sha256').read_text().splitlines()];bad=[]
for h,p in locks:
 f=PARENT/p
 if not f.exists() or sha(f)!=h:bad.append(p)
assert not bad,bad
# Primary and replay checkpoints, chain integrity, optimizer/RNG equality.
checks=0
for seed in [54051,54052]:
 for prefix in ['net','replay']:
  parent='training_features'
  for step in range(0,601,100):
   f=ROOT/f'{prefix}{seed}/s{step:04}.pt';q=t.load(f,weights_only=True);assert q['parent']==parent and q['step']==step;parent=sha(f)
 for step in range(0,601,100):
  a=t.load(ROOT/f'net{seed}/s{step:04}.pt',weights_only=True);b=t.load(ROOT/f'replay{seed}/s{step:04}.pt',weights_only=True);a.pop('parent');b.pop('parent');assert eq(a,b);checks+=1
out=dict(five_port_transactions=N,five_port_certificates=certs,nine_port_transactions=J,nine_port_certificates=JC,parent_manifest_entries=len(locks),primary_states=14,replay_states=14,bitwise_replay_comparisons=checks,online_transactions=len(pd.read_csv(ROOT/'online.csv')))
(ROOT/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
