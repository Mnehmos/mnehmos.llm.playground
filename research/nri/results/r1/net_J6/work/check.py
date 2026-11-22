import json, torch, sys
sys.path.insert(0,'.')
from query import predict
doc=json.load(open('nri.json')); T=torch.load('inputs.pt')['tokens'].tolist()
P=torch.load('work/base.pt')['pred'].tolist()
B={v['id']:eval(v['binding']) for v in doc['variables']}
V=[{k:f(t) for k,f in B.items()} for t in T]
for m in doc['mechanisms']:
    f=eval(m['output_fn']); print(m['id'], sum(f(v)==p for v,p in zip(V,P))/len(P))
c=[c for c in doc['claims'] if c['kind']=='edit'][0]
sc=eval(c['scope']); ef=eval(c['edit_fn'])
ins=[i for i,v in enumerate(V) if sc(v)]
# black-box check of edit target: output on 11->12 swapped inputs equals edit_fn
sw=[[12 if x==11 else x for x in T[i]] for i in ins]
pp=predict(sw)['pred']
print('edit target agreement', sum(ef(V[i])==p for i,p in zip(ins,pp))/len(ins), len(ins))
