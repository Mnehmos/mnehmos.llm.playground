import torch, json, collections
from query import predict
d = torch.load('inputs.pt')
print(d.keys())
T = d['tokens']; print(T.shape, T.dtype)
print({k:(v.shape if hasattr(v,'shape') else v) for k,v in d.items()})
for p in range(7):
    print(p, collections.Counter(T[:,p].tolist()).most_common())
out = predict(T.tolist())
pred = out['pred']
torch.save({'pred':torch.tensor(pred),'prob':torch.tensor(out['prob'])}, 'work/base.pt')
print(collections.Counter(pred))
for i in range(30): print(T[i].tolist(), pred[i], max(out['prob'][i]))
