import torch
exec(open('work/probe.py').read().split('for v in V:')[0])
y={k:torch.tensor(v) for k,v in V.items()}
CM=torch.zeros(3,11,11,64)
for o in range(3):
  for a in range(11):
    for bb in range(11):
      m=(y['op']==o)&(y['a']==a)&(y['b']==bb); CM[o,a,bb]=H[m].mean(0)
P=CM[0]-CM[0].mean((0,1)); Mn=CM[1]-CM[1].mean((0,1)); Mu=CM[2]-CM[2].mean((0,1))
print('norms', P.norm(dim=-1).mean().item(), Mn.norm(dim=-1).mean().item())
# for each minus cell, nearest plus cell
D=torch.cdist(Mn.reshape(121,64),P.reshape(121,64))
nn_=D.argmin(1); import collections
rel=collections.Counter()
for i in range(121):
    a,bb=divmod(i,11); a2,b2=divmod(nn_[i].item(),11)
    rel[((a2-a)%11,(b2+bb)%11,(b2-bb)%11)]+=1
print(rel.most_common(8))
# compare candidate relations
def err(f):
    return sum((Mn[a,bb]-P[f(a,bb)]).norm().item() for a in range(11) for bb in range(11))/121
print('same',err(lambda a,b:(a,b)),'neg b',err(lambda a,b:(a,(-b)%11)),'swap',err(lambda a,b:(b,a)))
# Is h function of (a,b) via additive? check minus res structure: does h_minus depend on (a,b) or on a-b?
