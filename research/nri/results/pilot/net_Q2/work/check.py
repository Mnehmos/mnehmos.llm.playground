import json,torch,sys; sys.path.insert(0,'.')
from arch import load_net
doc=json.load(open('nri.json'))
d=torch.load('inputs.pt'); T=d['tokens'].tolist(); C=d['ch2']
safe={'__builtins__':{'range':range,'len':len,'abs':abs,'min':min,'max':max,'int':int,'sum':sum}}
fs={v['id']:eval(v['binding'],safe) for v in doc['variables']}
print([ {k:f(T[i]) for k,f in fs.items()} for i in range(3)])
V=[{k:f(t) for k,f in fs.items()} for t in T]
of=eval(doc['mechanisms'][0]['output_fn'],safe)
net=load_net()
with torch.no_grad(): y=net(d['tokens'],C).argmax(-1).tolist()
print('behavior',sum(of(v)==yy for v,yy in zip(V,y))/len(y), 'res var match', sum(v['V_res']==of(v) for v in V)/len(V))
sc=eval(doc['claims'][-1]['scope'],safe); ef=eval(doc['claims'][-1]['edit_fn'],safe)
print('scope frac',sum(sc(v) for v in V)/len(V), ef(V[0]))
