import sys; sys.path.insert(0,".")
import torch, json, collections
from query import predict
d=torch.load('inputs.pt')
print({k:(v.shape,v.dtype) for k,v in d.items()})
T=d['tokens']
out=predict(T.tolist())
pred=torch.tensor(out['pred'])
torch.save({'pred':pred,'prob':torch.tensor(out['prob'])},'work/base.pt')
for i in range(25): print(T[i].tolist(), pred[i].item(), round(max(out['prob'][i]),3))
for p in range(7): print('pos',p,collections.Counter(T[:,p].tolist()).most_common())
print('pred',sorted(collections.Counter(pred.tolist()).items()))
