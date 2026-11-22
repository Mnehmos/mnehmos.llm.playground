import random, json, sys; sys.path.insert(0,'.')
from query import predict
doc=json.load(open('nri.json')); V={x['id']:eval(x['binding']) for x in doc['variables']}; f=eval(doc['mechanisms'][0]['output_fn'])
random.seed(1); R=[]
for _ in range(5000):
    t=[random.randrange(11) for _ in range(6)]+[14]; k=random.randrange(1,5); t[k]=random.choice([11,12,13]); R.append(t)
p=predict(R)['pred']
print('fresh random acc',sum(f({k:g(t) for k,g in V.items()})==q for t,q in zip(R,p))/len(R))
