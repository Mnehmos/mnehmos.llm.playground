import torch, collections
d=torch.load('inputs.pt'); T=d['tokens']; P=torch.load('work/base.pt')['pred']
isop=(T>=11)&(T<=13)
pat=collections.Counter(''.join('o' if x else 'n' for x in r) for r in isop[:,:6].tolist())
print(pat.most_common())
c=collections.Counter(tuple(r) for r in T[:,:6].tolist())
print('unique',len(c))
