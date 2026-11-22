import json, torch, sys
sys.path.insert(0,'.')
from arch import load_net
doc=json.load(open('nri.json'))
d=torch.load('inputs.pt'); T=d['tokens']; C=d['ch2']; L=T.tolist()
B={'__builtins__':{'range':range,'len':len,'min':min,'max':max,'abs':abs,'sum':sum,'int':int}}
f={v['id']:eval(v['binding'],B) for v in doc['variables']}
for r in L[:5]: print(r,{k:fn(r) for k,fn in f.items()})
vals=[{k:fn(r) for k,fn in f.items()} for r in L]
net=load_net()
with torch.no_grad(): out=net(T,C).argmax(-1).tolist()
for m in doc['mechanisms']:
    of=eval(m['output_fn'],B); print(m['id'],'acc',sum(of(v)==o for v,o in zip(vals,out))/len(out))
for c in doc['claims']:
    if c['kind']=='edit':
        s=eval(c['scope'],B); e=eval(c['edit_fn'],B)
        print('scope frac',sum(s(v) for v in vals)/len(vals), 'edit_fn sample',[e(v) for v in vals[:5]])
print('types ok', all(isinstance(x,int) for v in vals[:100] for x in v.values()))
