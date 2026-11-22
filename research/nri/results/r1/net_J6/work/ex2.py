import torch, collections
d = torch.load('inputs.pt'); T=d['tokens']; b=torch.load('work/base.pt'); P=b['pred']
isop = (T[:,1:5]>=11)
print('nops', collections.Counter(isop.sum(1).tolist()))
print('oppos', collections.Counter([tuple(r) for r in isop.int().tolist()]).most_common(20))
# rows with one op
for k in range(1,5):
    m = (isop.sum(1)==1)&isop[:,k-1]
    print('op at pos',k, m.sum().item())
