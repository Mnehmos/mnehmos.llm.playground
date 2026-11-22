import sys, json, torch; sys.path.insert(0,'work'); sys.path.insert(0,'.')
from mockhost import Host
import edit_program
doc=json.load(open('nri.json'))
T=torch.load('inputs.pt')['tokens'].tolist()
B={v['id']:eval(v['binding'],{"__builtins__":{"range":range}}) for v in doc['variables']}
host=Host(); net=host.model
C=torch.load('inputs.pt')['ch2']
with torch.no_grad(): Y=net(torch.tensor(T),C).argmax(-1).tolist()
for mech in doc['mechanisms']:
    fn=eval(mech['output_fn'],{"__builtins__":{}})
    acc=sum(fn({k:B[k](t) for k in B})==y for t,y in zip(T,Y))/len(T); print(mech['id'],'acc',acc)
for c in doc['claims']:
    if c['kind']=='edit':
        sc=eval(c['scope']); ef=eval(c['edit_fn'])
        vv=[{k:B[k](t) for k in B} for t in T[:200]]; [sc(v) for v in vv]; [ef(v) for v in vv]
dW,db=edit_program.program(host,doc)
print(dW.shape,db.shape)
