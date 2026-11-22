import json, torch, sys; sys.path.insert(0,'.')
doc=json.load(open('nri.json')); T=torch.load('inputs.pt')['tokens'].tolist()
P=torch.load('work/base.pt')['pred'].tolist()
V={x['id']:eval(x['binding'],{'__builtins__':{'range':range,'len':len}}) for x in doc['variables']}
vals=[{k:f(t) for k,f in V.items()} for t in T]
assert all(isinstance(x,int) for v in vals for x in v.values())
for mm in doc['mechanisms']:
    f=eval(mm['output_fn']); print(mm['id'],'behavior acc',sum(f(v)==p for v,p in zip(vals,P))/len(P))
for c in doc['claims']:
    if c['kind']=='edit':
        sc=eval(c['scope']); ef=eval(c['edit_fn']); print('scope frac',sum(map(sc,vals))/len(vals), 'edit_fn ok', [ef(v) for v in vals[:3]])
print([ (c['id'],c['kind'],c['predicate']) for c in doc['claims']])
# the edit target (a-b) equals what the net does with 11->12
from query import predict
T2=[[12 if x==11 else x for x in t] for t,v in zip(T[:3000],vals) if v['V_op']==0]
p2=predict(T2)['pred']; vv=[{k:f(t) for k,f in V.items()} for t in T2]
ef=eval([c for c in doc['claims'] if c['kind']=='edit'][0]['edit_fn'])
orig=[t for t,v in zip(T[:3000],vals) if v['V_op']==0]; vo=[{k:f(t) for k,f in V.items()} for t in orig]
print('edit_fn matches net on swapped inputs',sum(ef(a)==p for a,p in zip(vo,p2))/len(p2))
