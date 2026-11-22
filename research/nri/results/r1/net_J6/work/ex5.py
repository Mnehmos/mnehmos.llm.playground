import torch, random, sys
sys.path.insert(0,'.')
from query import predict
T=torch.load('inputs.pt')['tokens'].tolist(); b=torch.load('work/base.pt'); P=b['pred'].tolist(); PR=b['prob']
print('min maxprob', PR.max(1).values.min().item(), 'mean', PR.max(1).values.mean().item())
rnd=random.Random(0); rows=[]; ref=[]
for i in range(3000):
    t=list(T[i]); k=[j for j in range(1,5) if t[j]>=11][0]
    for j in range(6):
        if j not in (k-1,k,k+1): t[j]=rnd.randint(0,10)
    rows.append(t); ref.append(P[i])
pp=predict(rows)['pred']; print('pred unchanged after resampling non-adjacent digits', sum(a==b for a,b in zip(pp,ref))/len(ref))
# move operator triple to other positions
rows=[];ref=[]
for i in range(3000):
    t=T[i]; k=[j for j in range(1,5) if t[j]>=11][0]; k2=rnd.randint(1,4)
    u=[rnd.randint(0,10) for _ in range(6)]+[14]; u[k2-1],u[k2],u[k2+1]=t[k-1],t[k],t[k+1]
    rows.append(u); ref.append(P[i])
pp=predict(rows)['pred']; print('pred unchanged after moving triple', sum(a==b for a,b in zip(pp,ref))/len(ref))
