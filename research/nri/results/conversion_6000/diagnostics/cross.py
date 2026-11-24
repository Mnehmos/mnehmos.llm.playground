"""Causal comparison across two distinct mass-conversion routes."""
import sys,json
from pathlib import Path
import torch
from sem import train as T, data as D, lang as L
from sci.conversion import examples
@torch.no_grad()
def main(path):
 st=T.load_state(path);m=st["model"].eval()
 directions=[(1,0),(0,1),(2,0)]
 rows=[r for route in directions for r in examples(st["vocab"],route,range(13)) if L.tuple_split(r["ast"])=="train"]
 b=D.encode_batch(rows,st["vocab"])
 c=m.trunk(b["tok"],ret=True)
 i,j=torch.cartesian_prod(torch.arange(len(rows)),torch.arange(len(rows))).T.tolist()
 keep=[];targets=[]
 for a,d in zip(i,j):
  A,B=rows[a]["ast"],rows[d]["ast"]
  if A["u"]==B["u"]:continue
  claims=[int(L.answer(z)) for z in (A,dict(A,v=B["v"]),dict(A,u=B["u"]),dict(A,v=B["v"],u=B["u"]),B)]
  # Isolate four intervention targets. The full donor answer can coincide when target units agree.
  if len(set(claims[:4]))==4:keep.append((a,d));targets.append(claims)
 i,j=torch.tensor(keep).T;y=torch.tensor(targets);key=b["tok"][i]!=D.PAD
 def head(x):
  x=x+m.attn2(m.ln2a(x),key);x=x+m.mlp2(m.ln2m(x));x=x+m.attn3(m.ln3a(x),key)
  return m.head(x[:,-1]).argmax(1)
 out={}
 for name,names in [('none',[]),('value',['src_val']),('unit',['src_unit']),('joint',['src_val','src_unit'])]:
  h=c['x1'][i].clone()
  for role in names:
   p=(b['role']==L.ROLE_ID[role]).long().argmax(1)
   h[torch.arange(len(i)),p[i]]=c['x1'][j,p[j]]
  z=head(h)
  out[name]={k:int((z==y[:,c]).sum()) for c,k in enumerate(['base','base_route_donor_number','base_number_donor_unit','donor_number_and_unit','full_donor'])}
 return {'n':len(i),'matches':out}
if __name__=='__main__':
 print(json.dumps(main(Path(sys.argv[1])),indent=2))
